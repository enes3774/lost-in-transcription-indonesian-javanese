#!/usr/bin/env python3
"""Fine-tune Qwen/Qwen3-ASR-1.7B on SAPC2 dysarthric speech.

    python sapc/qwen_train.py --config sapc/conf/qwen3_asr_1p7b_sapc.yaml
    python sapc/qwen_train.py --config ... --smoke     # correctness checks only
    python sapc/qwen_train.py --config ... --bench     # size the batch, then exit

Run with the Qwen venv (/kaggle/working/venv_qwen/bin/python), never /opt/venv:
Qwen3-ASR needs transformers >= 5.13 and NeMo pins 4.57.x, so the two runs cannot
share an interpreter. The venv uses --system-site-packages, so /opt/venv stays
exactly as the Parakeet supervisor expects it.

What is deliberately shared with the Parakeet runs, so the numbers compare:

  * the same Train.jsonl, the same per-epoch repetition cap (20 per distinct
    text_ref2, reseeded each epoch), and the same duration-sorted batching --
    all via sapc/manifest_ops.py, at the same seed, so epoch N here contains the
    same recordings as epoch N there;
  * the same two validation sets, with checkpoint selection on the
    text-disjoint one.

What deliberately differs:

  * the target is `text_ref1`, not the spoken-form `text_train_ref1`. Parakeet's
    BPE vocabulary has no digits, so "25" tokenized to <unk> and numbers had to
    be spelled out. Qwen3-ASR inherits Qwen3's tokenizer and writes digits
    natively. This costs nothing at scoring time: the official normalizer maps
    spelled-out numbers *to* digits, so "twenty five past two" and "25 past 2"
    are the same string by the time WER is computed (verified against
    utils/normalizer/text_normalizer_hf.py).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Before torch/transformers: every cache- and temp-shaped path has to land on the
# big volume. The container overlay lives on a Windows drive whose virtual disk
# only grows, so a temp file deleted a second later still costs permanent space
# -- that is what exhausted the host twice (MINER1.md).
_CACHE = os.environ.setdefault("SAPC_CACHE_ROOT", "/kaggle/working/cache")
os.environ.setdefault("HF_HOME", f"{_CACHE}/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_CACHE}/hf/hub")
os.environ.setdefault("TORCH_HOME", f"{_CACHE}/torch")
for _v in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_v] = f"{_CACHE}/tmp"
os.makedirs(f"{_CACHE}/tmp", exist_ok=True)

# NCCL's cuMem allocator is unsupported under WSL2 and every collective dies with
# "Cuda failure 999". Harmless when single-GPU. Must precede NCCL init.
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lit_metrics import score as score_hyps  # noqa: E402

# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return RANK == 0


def log(msg: str):
    if is_main():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def barrier():
    if WORLD > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class BatchDataset(Dataset):
    """One item = one ready-made batch of manifest rows.

    manifest_ops.write_sorted_batches has already decided batch membership by
    sorting on duration, so the loader must not regroup: it reads consecutive
    runs of `batch_size` rows exactly as written. Automatic batching is disabled
    in the DataLoader, which makes each worker collate a whole batch.
    """

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


def to_device(batch, device, dtype):
    """Move a batch to the GPU, casting the mel features to the model's dtype.

    The processor always returns float32 `input_features`, but the model is
    loaded in bfloat16, so the first conv2d raises "Input type (float) and bias
    type (c10::BFloat16) should be the same". Only floating-point tensors are
    cast -- input_ids and the masks must stay integral.
    """
    out = {}
    for k, v in batch.items():
        v = v.to(device, non_blocking=True)
        if v.is_floating_point():
            v = v.to(dtype)
        out[k] = v
    return out


def read_batches(path: str, batch_size: int):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


def load_audio(path: str, max_seconds: float) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
        sr = 16000
    limit = int(max_seconds * 16000)
    return wav[:limit] if wav.shape[0] > limit else wav


class Collator:
    """Turn manifest rows into a training batch.

    The chat template renders as

        <|im_start|>system\\n<|im_end|>\\n
        <|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n
        <|im_start|>assistant\\n{transcript}<|im_end|>\\n

    and the processor's own `output_labels` masks audio and padding but leaves
    the scaffold unmasked, which would train the model to emit its own prompt.
    _mask_prompt clips the labels back to the assistant turn, so the loss covers
    the transcript and its terminating <|im_end|> and nothing else. --smoke
    asserts this, because getting it wrong costs a run without ever erroring.

    The assistant turn is prefilled with "language English<asr_text>", which is
    how the model's own apply_transcription_request forces a language.
    Qwen3-ASR does language identification before transcription and natively
    emits "language <NAME><asr_text>{transcript}". Without the prefill the model
    spends tokens re-deriving "English" on every utterance and we would be
    training it out of its own output format; prefilled, it emits the transcript
    and nothing else.
    """

    def __init__(self, processor, text_key: str, max_seconds: float,
                 language: str = "English", train: bool = True):
        self.proc = processor
        self.text_key = text_key
        self.max_seconds = max_seconds
        self.train = train
        # Match official Qwen SFT: the assistant turn is not prefilled with a
        # forced language. The model learns the complete None-language prefix,
        # then the transcript, and inference uses the same unforced prompt.
        self.header = "<|im_start|>assistant\n"
        self.target_prefix = f"language {language}<asr_text>"
        self.prompt = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            + self.header
        )
        self.header_ids = processor.tokenizer(
            self.header, add_special_tokens=False
        )["input_ids"]

    def _mask_prompt(self, input_ids, labels):
        """Mask everything up to and including the prefilled assistant header."""
        n = len(self.header_ids)
        for b in range(input_ids.size(0)):
            row = input_ids[b].tolist()
            cut = -1
            for i in range(len(row) - n, -1, -1):
                if row[i:i + n] == self.header_ids:
                    cut = i + n
                    break
            if cut < 0:
                # No assistant turn: drop the row from the loss entirely rather
                # than train on the scaffold.
                labels[b, :] = -100
            else:
                labels[b, :cut] = -100
        return labels

    def __call__(self, rows):
        audio = [load_audio(r["audio_filepath"], self.max_seconds) for r in rows]
        if self.train:
            texts = [self.prompt + self.target_prefix + r[self.text_key].strip() + "<|im_end|>\n" for r in rows]
            out = self.proc(
                text=texts, audio=audio, sampling_rate=16000,
                output_labels=True, return_tensors="pt", padding=True,
            )
            out["labels"] = self._mask_prompt(out["input_ids"], out["labels"])
        else:
            out = self.proc(
                text=[self.prompt] * len(rows), audio=audio, sampling_rate=16000,
                return_tensors="pt", padding=True,
            )
        return dict(out)


def prepare_epoch_batches(cfg, epoch: int):
    """Build a duration-efficient, source-balanced epoch from the full clean pool."""
    rows = [json.loads(l) for l in open(cfg.paths.train_manifest, encoding="utf-8")]
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    rng = random.Random(cfg.sampling.seed + epoch)
    total = int(cfg.sampling.epoch_samples)
    weights = {str(k): float(v) for k, v in cfg.sampling.weights.items()}
    selected, counts = [], {}
    names = list(weights)
    assigned = 0
    for i, name in enumerate(names):
        k = total - assigned if i == len(names) - 1 else round(total * weights[name])
        assigned += k
        pool = by_source.get(name, [])
        if not pool:
            raise ValueError(f"sampling source {name!r} is absent from {cfg.paths.train_manifest}")
        if k <= len(pool):
            take = rng.sample(pool, k)
        else:
            take = list(pool) + rng.choices(pool, k=k-len(pool))
        selected.extend(take)
        counts[name] = len(take)
    # Duration-homogeneous batches greatly reduce audio padding; batch order is shuffled.
    selected.sort(key=lambda r: r["duration"])
    bs = int(cfg.data.batch_size)
    batches = [selected[i:i+bs] for i in range(0, len(selected), bs)]
    rng.shuffle(batches)
    lo, hi = cfg.data.min_duration, cfg.data.max_duration
    batches = [[r for r in b if lo <= r["duration"] <= hi] for b in batches]
    batches = [b for b in batches if b]
    if WORLD > 1:
        batches = batches[:len(batches)//WORLD*WORLD][RANK::WORLD]
    log(f"[weighted-epoch] epoch={epoch} rows={sum(map(len,batches))} sources={counts} batches={len(batches)}")
    return batches


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model(cfg, device):
    """Load Qwen3-ASR and decide what learns.

    The domain shift here is acoustic, not linguistic: dysarthric articulation
    against a model that already transcribes ordinary English well. So the audio
    tower and the projector train fully, while the 1.7B language model gets LoRA
    only.

    Full fine-tuning the language model would be actively harmful. The corpus has
    297,769 recordings of just 24,891 distinct prompts, 74% of them Digital
    Assistant Commands -- an extremely peaked text distribution. A 1.7B decoder
    at an unconstrained learning rate collapses its language prior onto that
    distribution, while held-out text need not overlap the training text. The
    text-disjoint validation set exists to catch exactly this; LoRA plus the
    repetition cap is the defence.
    """
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        cfg.model.path,
        dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
    )

    if cfg.freeze.lora.enabled:
        from peft import LoraConfig, get_peft_model
        # Match on the fully-qualified name: `q_proj` alone also exists in the
        # audio tower, which is a full fine-tune here and must not pick up
        # adapters instead.
        pattern = (
            r"model\.language_model\.layers\.\d+\."
            r"(self_attn|mlp)\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        )
        model = get_peft_model(model, LoraConfig(
            r=cfg.freeze.lora.r,
            lora_alpha=cfg.freeze.lora.alpha,
            lora_dropout=cfg.freeze.lora.dropout,
            bias="none",
            target_modules=pattern,
        ))
        # get_peft_model freezes every base weight. Audio/projector adaptation is
        # explicitly configurable so conservative LoRA and aggressive runs can
        # be compared without changing code.
        train_audio = bool(cfg.freeze.get("train_audio_tower", True))
        train_projector = bool(cfg.freeze.get("train_projector", True))
        for name, p in model.named_parameters():
            if "audio_tower" in name:
                p.requires_grad_(train_audio)
            elif "multi_modal_projector" in name:
                p.requires_grad_(train_projector)
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # With a frozen embedding table the encoder output is the only thing
        # carrying grad into the checkpointed blocks; without this the backward
        # pass finds nothing to differentiate.
        model.enable_input_require_grads()

    model.to(device)

    # Separate learning rates per component. The audio tower is doing the actual
    # domain adaptation and needs the most movement; LoRA adapters start at zero
    # and tolerate a higher rate than a full fine-tune would.
    groups, seen = [], set()

    def collect(match, lr, tag):
        ps = [p for n, p in model.named_parameters()
              if p.requires_grad and match(n) and id(p) not in seen]
        for p in ps:
            seen.add(id(p))
        if ps:
            groups.append({"params": ps, "lr": lr, "name": tag,
                           "n": sum(p.numel() for p in ps)})

    collect(lambda n: "audio_tower" in n, cfg.optim.lr_encoder, "audio_tower")
    collect(lambda n: "multi_modal_projector" in n, cfg.optim.lr_projector, "projector")
    collect(lambda n: "lora_" in n, cfg.optim.lr_lora, "lora")
    collect(lambda n: True, cfg.optim.lr_decoder, "decoder")

    total = sum(p.numel() for p in model.parameters())
    train = sum(g["n"] for g in groups)
    log(f"[params] {total / 1e9:.3f} B total, {train / 1e6:.1f} M trainable "
        f"({100 * train / total:.1f}%)")
    for g in groups:
        log(f"         {g['name']:<14} {g['n'] / 1e6:8.1f} M  lr={g['lr']:g}")
    return model, groups


def build_optimizer(cfg, groups):
    opt = torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"]} for g in groups],
        betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay,
    )
    warm, total = cfg.optim.warmup_steps, cfg.trainer.max_steps
    floor = cfg.optim.min_lr_ratio

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(model, proc, cfg, manifest: str, tag: str, limit: int = 0):
    """Greedy-decode a validation manifest and score it the way the leaderboard does.

    Returns CER (the Track 1 primary metric) and WER, both min-of-two-references
    after the official normalizer, plus the unnormalized WER against ref1 so the
    curve is readable next to the Parakeet logs, which report exactly that.
    """
    model.eval()
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    rows.sort(key=lambda r: r["duration"])  # homogeneous batches, less padding

    # Batched generation needs left padding, or the short rows in a batch start
    # decoding from pad tokens and produce garbage.
    side = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"
    collate = Collator(proc, cfg.data.text_key, cfg.data.val_max_seconds,
                       cfg.data.language, train=False)
    device = next(model.parameters()).device
    hyps = []
    bs = cfg.data.val_batch_size
    try:
        for i in range(0, len(rows), bs):
            batch = to_device(collate(rows[i:i + bs]), device, model.dtype)
            out = model.generate(
                **batch, max_new_tokens=cfg.data.max_new_tokens,
                do_sample=False, num_beams=1,
            )
            gen = out[:, batch["input_ids"].shape[1]:]
            # The processor's own parser rather than batch_decode: it strips a
            # "language X<asr_text>" prefix if the model emits one anyway, and
            # applies Qwen's degenerate-repetition fix.
            hyps += [proc.decode(g, return_format="transcription_only").strip()
                     for g in gen]
    finally:
        proc.tokenizer.padding_side = side
        model.train()

    m = score_hyps(hyps, [r["text_ref1"] for r in rows], [r["text_ref2"] for r in rows])
    m["raw_wer"] = score_hyps(
        hyps, [r["text_ref1"] for r in rows], [r["text_ref1"] for r in rows],
        normalize=False,
    )["wer"]
    log(f"[val:{tag}] cer={m['cer']:.4f}  wer={m['wer']:.4f}  raw_wer={m['raw_wer']:.4f}"
        f"  ({len(rows)} utts)")
    # Report each validation domain/category independently. `eval_group` is used
    # for competition ind/javind labels; robust validation falls back to source.
    grouped = {}
    labels = [str(r.get("eval_group") or r.get("source") or "unknown") for r in rows]
    if len(set(labels)) > 1:
        for group in sorted(set(labels)):
            idx = [i for i, value in enumerate(labels) if value == group]
            gm = score_hyps([hyps[i] for i in idx],
                            [rows[i]["text_ref1"] for i in idx],
                            [rows[i]["text_ref2"] for i in idx])
            grouped[group] = gm
            log(f"[val:{tag}:{group}] cer={gm['cer']:.4f}  wer={gm['wer']:.4f}"
                f"  ({len(idx)} utts)")
    m["groups"] = grouped
    for r, h in list(zip(rows, hyps))[:3]:
        log(f"    ref: {r['text_ref1'][:100]}")
        log(f"    hyp: {h[:100]}")
    return m


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def trainable_state(model):
    """Only the trainable tensors: the frozen 1.4 B decoder is on disk already.

    Saves ~330 MB per checkpoint instead of ~4 GB, which matters because the
    container's virtual disk never shrinks.
    """
    return {n: p.detach().to(torch.bfloat16).cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(model, opt, sched, cfg, step, epoch, metrics, path):
    """Single resumable checkpoint; overwritten at every validation."""
    tmp = f"{path}.tmp"
    torch.save({
        "trainable": trainable_state(model), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "step": step, "epoch": epoch,
        "metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True),
    }, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write cannot corrupt the resume
    log(f"[ckpt] {path}  step={step} epoch={epoch} "
        f"cer={metrics.get('cer', float('nan')):.4f}")


def save_model_only(model, cfg, step, epoch, metrics, path):
    """Compact best/final checkpoint without optimizer moments."""
    tmp = f"{path}.tmp"
    torch.save({"trainable": trainable_state(model), "step": step, "epoch": epoch,
                "metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True)}, tmp)
    os.replace(tmp, path)
    log(f"[model-ckpt] {path}  step={step} wer={metrics.get('wer', float('nan')):.4f}")


def load_model_only(model, path):
    """Initialize trainable weights from another stage, resetting optimizer/step."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ck["trainable"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"initial checkpoint mismatch: {result.unexpected_keys[:5]}")
    log(f"[init] model weights from {path}; optimizer and schedule reset")


def load_ckpt(model, opt, sched, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ck["trainable"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint holds keys this model does not have: "
            f"{result.unexpected_keys[:5]} -- wrong config?"
        )
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    log(f"[resume] {path}  step={ck['step']} epoch={ck['epoch']}")
    return ck["step"], ck["epoch"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true", help="verify labels/decoding, then exit")
    ap.add_argument("--bench", action="store_true", help="measure step time and VRAM, then exit")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    torch.manual_seed(cfg.seed + RANK)
    random.seed(cfg.seed + RANK)
    np.random.seed(cfg.seed + RANK)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if WORLD > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    exp = Path(cfg.paths.exp_dir)
    if is_main():
        exp.mkdir(parents=True, exist_ok=True)
    proc = Qwen3ASRProcessor.from_pretrained(cfg.model.path)
    model, groups = build_model(cfg, device)
    last = exp / "last.ckpt"
    init_ckpt = cfg.trainer.get("init_checkpoint", None)
    if init_ckpt and not last.exists():
        load_model_only(model, str(init_ckpt))
    opt, sched = build_optimizer(cfg, groups)

    step, start_epoch, best = 0, 0, float("inf")
    if last.exists() and cfg.trainer.resume_if_exists:
        step, start_epoch = load_ckpt(model, opt, sched, str(last))
        bp = exp / "best.json"
        if bp.exists():
            best = json.loads(bp.read_text())["wer"]

    if WORLD > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], find_unused_parameters=False,
        )
    net = model.module if WORLD > 1 else model

    collate = Collator(proc, cfg.data.text_key, cfg.data.max_duration,
                       cfg.data.language, train=True)

    if args.smoke:
        run_smoke(cfg, proc, net, collate, device)
        return
    if args.bench:
        run_bench(cfg, proc, net, model, opt, collate, device)
        return

    dtype = net.dtype
    n_val = 0
    metrics_log = exp / "metrics.jsonl"
    t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0

    for epoch in range(start_epoch, cfg.trainer.max_epochs):
        batches = prepare_epoch_batches(cfg, epoch)
        loader = DataLoader(
            BatchDataset(batches), batch_size=None, shuffle=False,
            num_workers=cfg.data.num_workers, collate_fn=collate,
            # Page-locked host memory is a host-wide resource and is scarce under
            # WSL2; three Parakeet runs died in the pin-memory thread with
            # cudaHostAlloc "out of memory" while the GPU had tens of GB free.
            pin_memory=False,
            persistent_workers=cfg.data.num_workers > 0,
            prefetch_factor=4 if cfg.data.num_workers > 0 else None,
        )
        log(f"=== epoch {epoch}: {len(batches)} batches/rank, resuming at step {step}")

        for batch in loader:
            batch = to_device(batch, device, dtype)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], cfg.trainer.gradient_clip_val
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            seen += batch["input_ids"].size(0)
            loss_acc += loss.item()
            n_acc += 1

            if step % cfg.trainer.log_every_n_steps == 0:
                dt = time.time() - t0
                log(f"step {step:>6}/{cfg.trainer.max_steps}  ep {epoch}  "
                    f"loss {loss_acc / max(n_acc, 1):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                    f"{dt / max(n_acc, 1):.3f} s/step  {seen / dt:.1f} utt/s  "
                    f"vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} G")
                if is_main():
                    with open(metrics_log, "a") as f:
                        f.write(json.dumps({"step": step, "epoch": epoch,
                                            "loss": loss_acc / max(n_acc, 1),
                                            "lr": sched.get_last_lr()[0],
                                            "s_per_step": dt / max(n_acc, 1)}) + "\n")
                t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0

            if step % cfg.trainer.val_every_n_steps == 0 or step >= cfg.trainer.max_steps:
                barrier()
                if is_main():
                    n_val += 1
                    m = validate(net, proc, cfg, cfg.paths.val_unseen, "unseen")
                    # The 2,998-utt "all" set costs ~3x the text-disjoint one and
                    # selects nothing -- it exists to compare against the earlier
                    # Parakeet numbers. Decoding both every time would cost more
                    # wall-clock than the training itself, so it runs periodically.
                    m_all = None
                    if n_val % cfg.trainer.val_all_every == 0:
                        m_all = validate(net, proc, cfg, cfg.paths.val_all, "all")
                    with open(metrics_log, "a") as f:
                        f.write(json.dumps({"step": step, "epoch": epoch,
                                            "val_unseen": m, "val_all": m_all}) + "\n")
                    save_ckpt(net, opt, sched, cfg, step, epoch, m, str(last))
                    # Selection is on group-disjoint internal validation and competition WER.
                    if m["wer"] < best:
                        best = m["wer"]
                        save_model_only(net, cfg, step, epoch, m, str(exp / "best.ckpt"))
                        (exp / "best.json").write_text(json.dumps(
                            {"cer": m["cer"], "wer": m["wer"], "step": step, "epoch": epoch}))
                barrier()
                t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0

            if step >= cfg.trainer.max_steps:
                if is_main() and not bool(cfg.trainer.get("keep_last_after_completion", False)):
                    # best.ckpt is model-only and sufficient for conversion/evaluation.
                    # Drop Adam moments after a clean run to avoid tens of GB per full-FT experiment.
                    last.unlink(missing_ok=True)
                    log("[ckpt] removed optimizer-bearing last.ckpt after clean completion")
                log(f"max_steps={cfg.trainer.max_steps} reached")
                return


def run_smoke(cfg, proc, model, collate, device):
    """Check the three things that would silently corrupt this training."""
    batches = prepare_epoch_batches(cfg, 0)
    rows = batches[0][:4]
    b = collate(rows)
    log(f"[smoke] shapes: {({k: tuple(v.shape) for k, v in b.items()})}")

    # 1. the loss must cover the transcript and nothing else
    for i, r in enumerate(rows):
        kept = [t for t, l in zip(b["input_ids"][i].tolist(), b["labels"][i].tolist())
                if l != -100]
        got = proc.tokenizer.decode(kept).strip()
        want = (collate.target_prefix + r[cfg.data.text_key].strip() + "<|im_end|>").strip()
        log(f"[smoke] labels {'OK ' if got == want else 'BAD'}: {got[:80]!r}")
        assert got == want, f"label masking wrong\n got: {got!r}\nwant: {want!r}"

    # 2. a forward pass produces a finite, plausible loss
    b = to_device(b, device, model.dtype)
    with torch.no_grad():
        loss = model(**b).loss
    log(f"[smoke] loss={loss.item():.4f}  (an untrained head would be ~ln(V)=11.9; "
        f"a pretrained ASR model on in-domain text should be well under 2)")
    assert torch.isfinite(loss), "non-finite loss"

    # 3. batched generation must match one-at-a-time generation, or left-padding
    #    is silently corrupting the short rows in every batch
    saved = cfg.data.val_batch_size
    m_batched = validate(model, proc, cfg, cfg.paths.val_unseen, "batched", limit=8)
    cfg.data.val_batch_size = 1
    m_single = validate(model, proc, cfg, cfg.paths.val_unseen, "unbatched", limit=8)
    cfg.data.val_batch_size = saved
    log(f"[smoke] padding check: batched cer={m_batched['cer']:.4f} vs "
        f"unbatched cer={m_single['cer']:.4f}")
    log("[smoke] all checks passed")


def run_bench(cfg, proc, net, model, opt, collate, device):
    """Time a real step at both a typical and the worst-case batch."""
    batches = prepare_epoch_batches(cfg, 0)
    longest = max(batches, key=lambda b: max(r["duration"] for r in b))
    typical = batches[len(batches) // 2]
    for tag, rows in (("typical", typical), ("longest", longest)):
        torch.cuda.reset_peak_memory_stats()
        b = to_device(collate(rows), device, net.dtype)
        secs = max(r["duration"] for r in rows)
        t = time.time()
        for i in range(4):
            if i == 1:  # first step includes autotuning; time the rest
                torch.cuda.synchronize()
                t = time.time()
            model(**b).loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 3
        log(f"[bench] {tag:<8} bs={len(rows)} max_dur={secs:5.1f}s  "
            f"{dt:.3f} s/step  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    log(f"[bench] {len(batches)} batches/epoch at batch_size={cfg.data.batch_size}")


if __name__ == "__main__":
    main()
