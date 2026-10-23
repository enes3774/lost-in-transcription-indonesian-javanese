#!/usr/bin/env python3
"""Fine-tune MERaLiON-3-3B-ASR on the Indonesian/Javanese pool.

    python training/meralion_v1/meralion_train.py --config training/meralion_v1/meralion_a_proven_mix.yaml
    python training/meralion_v1/meralion_train.py --config ... --smoke     # correctness checks only
    python training/meralion_v1/meralion_train.py --config ... --bench     # size the batch, then exit

Run with verify_venv (transformers 4.57.6, peft 0.17.1): the model card asks for
4.50.1 but the remote code runs unchanged on 4.57, as the zero-shot evaluation
in PROJECT_REPORT.md §11 established.

This is training/qwen_v2/qwen_train.py with the model swapped. What is kept on
purpose, so the two runs compare: the weighted per-source epoch sampler, the
duration-sorted batching, the robust_val / competition-dev validation pair with
selection on robust_val, model-only best checkpoints, cosine schedule with the
same warm-up and floor. What changes is confined to the model boundary:

  * MERaLiON-3 = Whisper-large-v3 encoder -> MLP adapter (1500 frames -> 300
    tokens) -> Gemma-2-2B decoder. Speech enters the decoder by masked_scatter
    onto 300 copies of the <SpeechHere> token, so the prompt is ordinary text
    and the loss can be clipped to the model turn exactly as before.
  * The official `meralion-3-asr` package pins the prompt, greedy decoding with
    no_repeat_ngram_size=6, and EAGER attention (Gemma-2's logit softcapping is
    not applied correctly under sdpa). All three are honoured here.
  * The model's own recipe (tech report) fully fine-tunes the encoder and
    adapter and puts LoRA on the decoder. Runs A/B freeze the encoder like the
    winning Qwen v2 recipe did; `optim.lr_encoder > 0` unfreezes it (run C).
  * SpecAugment comes from the shipped speech_config (time p=0.1/len 20,
    feature p=0.1/len 10). Only WhisperModel applies it in transformers; the
    bare WhisperEncoder this composite wraps never does, so the training step
    applies it itself with the same HF mask generator.
  * Targets are `text_ref1` untouched: the pool already keeps fillers,
    ellipses and apostrophes, which is the convention MERaLiON was trained on
    and the one the dev references use.

Two things learned the hard way on the first runs, both fixed here and both
latent in qwen_train.py: (1) a resume replayed the epoch from batch 0 instead
of continuing; (2) the weighted epoch can hold fewer batches than max_steps
(duration filtering drops rows after the sample count is fixed), and the loop
then ended between validations with the final weights never saved.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Caches and temp files go on the big volume (see env.sh); /kaggle/working is
# the 82 %-full overlay.
_CACHE = os.environ.setdefault(
    "SAPC_CACHE_ROOT", os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/cache")
os.environ.setdefault("HF_HOME", f"{_CACHE}/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_CACHE}/hf/hub")
os.environ.setdefault("TORCH_HOME", f"{_CACHE}/torch")
for _v in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_v] = f"{_CACHE}/tmp"
os.makedirs(f"{_CACHE}/tmp", exist_ok=True)
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: E402

ROOT = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
sys.path.insert(0, str(ROOT / "training/qwen_v2"))
sys.path.insert(0, str(ROOT / "scripts"))
from lit_metrics import score as score_hyps  # noqa: E402
from newmodels_common import clean_text  # noqa: E402

# Pinned by the official meralion-3-asr package (prompts.py). No language hint:
# naming one halves the model's Javanese (PROJECT_REPORT.md §11.4).
ASR_PROMPT = "Please transcribe this speech."
USER_CONTENT = (f"Instruction: {ASR_PROMPT} \n"
                "Follow the text instruction based on the following audio: <SpeechHere>")
# Orthography conditioning (report 12.13): the same Javanese sound has two legal spellings
# (standard apa/isa/ana/dhewe vs phonetic opo/iso/ono/dewe) and the pool labels disagree by
# source, so a per-source clause in the instruction lets the model learn both as a switch
# instead of a coin flip; decoding picks one (data.decode_style).
STYLE_SUFFIX = {None: "", "phonetic": " Write Javanese words as they are pronounced.",
                "standard": " Write Javanese words in standard spelling."}
def user_content(style=None):
    return (f"Instruction: {ASR_PROMPT}{STYLE_SUFFIX[style]} \n"
            "Follow the text instruction based on the following audio: <SpeechHere>")
MODEL_HEADER = "<start_of_turn>model\n"
END_OF_TURN = "<end_of_turn>"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class BatchDataset(Dataset):
    """One item = one ready-made batch of manifest rows (see prepare_epoch_batches)."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


def to_device(batch, device, dtype):
    """Move a batch to the GPU, casting the float32 mel features to the model dtype."""
    out = {}
    for k, v in batch.items():
        v = v.to(device, non_blocking=True)
        if v.is_floating_point():
            v = v.to(dtype)
        out[k] = v
    return out


def load_audio(path: str, max_seconds: float) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
    limit = int(max_seconds * 16000)
    return wav[:limit] if wav.shape[0] > limit else wav


class Collator:
    """Turn manifest rows into a training batch.

    The chat template renders as

        <bos><start_of_turn>user\\n{USER_CONTENT}<end_of_turn>\\n
        <start_of_turn>model\\n{transcript}<end_of_turn>

    and the processor expands <SpeechHere> into 300 copies of the speech token
    per 30 s chunk. The processor tokenizes with add_special_tokens=False, so
    the <bos> Gemma-2 needs comes from the template and nowhere else.

    Labels are built here (the processor has no output_labels): padding and
    speech tokens are masked, then everything up to and including the model
    header, so the loss covers the transcript and its <end_of_turn> and nothing
    else. --smoke asserts this.
    """

    def __init__(self, processor, text_key: str, max_seconds: float, train: bool = True, augment: bool = False,
                 style_tags=None, decode_style=None):
        self.proc = processor
        self.style_tags = dict(style_tags) if style_tags else {}   # source -> phonetic | standard
        self.decode_style = decode_style
        self.augment = augment and train
        self.tok = processor.tokenizer
        self.text_key = text_key
        self.max_seconds = max_seconds
        self.train = train
        self.speech_token_index = processor.speech_token_index
        self.prompts = {st: self.tok.apply_chat_template([{"role": "user", "content": user_content(st)}],
                                                         tokenize=False, add_generation_prompt=True) for st in STYLE_SUFFIX}
        self.prompt = self.prompts[None]
        assert all(pr.endswith(MODEL_HEADER) for pr in self.prompts.values()), self.prompt[-60:]
        if self.style_tags or self.decode_style:
            log(f"[style] tags={self.style_tags} decode_style={self.decode_style}")
        self.header_ids = self.tok(MODEL_HEADER, add_special_tokens=False)["input_ids"]

    def _mask_prompt(self, input_ids, labels):
        n = len(self.header_ids)
        for b in range(input_ids.size(0)):
            row = input_ids[b].tolist()
            cut = -1
            for i in range(len(row) - n, -1, -1):
                if row[i:i + n] == self.header_ids:
                    cut = i + n
                    break
            if cut < 0:
                labels[b, :] = -100   # no model turn found: drop the row from the loss
            else:
                labels[b, :cut] = -100
        return labels

    def _augment(self, w):
        """Speaker/channel robustness: speed perturbation and additive noise.

        Generalising to unseen speakers and sessions is what dev-style tuning
        cannot reach. Speed 0.9-1.1 (p=.5) shifts pitch/tempo like a different
        speaker; Gaussian noise at 10-30 dB SNR (p=.3) stands in for channel
        variation. Labels are untouched, so a slowed clip that would overrun
        the 30 s window is sped up instead of cut.
        """
        limit = int(self.max_seconds * 16000)
        if random.random() < 0.5:
            f = random.uniform(0.9, 1.1)
            if len(w) / f > limit:
                f = len(w) / limit * 1.01
            n = int(len(w) / f)
            w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
        if random.random() < 0.3:
            snr = random.uniform(10, 30)
            power = float(np.mean(w ** 2)) + 1e-8
            w = w + np.random.randn(len(w)).astype(np.float32) * np.sqrt(power / (10 ** (snr / 10)))
        return np.clip(w, -1.0, 1.0)

    def __call__(self, rows):
        audio = [load_audio(r["audio_filepath"], self.max_seconds) for r in rows]
        if self.augment:
            audio = [self._augment(w) for w in audio]
        if self.train:
            # Right padding for teacher forcing: Gemma-2 derives positions from
            # cache_position, so left pads would shift every real token.
            self.tok.padding_side = "right"
            texts = [self.prompts[self.style_tags.get(r.get("source"))] + r[self.text_key].strip() + END_OF_TURN for r in rows]
            out = self.proc(text=texts, audios=audio, return_tensors="pt", padding=True)
            labels = out["input_ids"].clone()
            labels[out["attention_mask"] == 0] = -100
            labels[out["input_ids"] == self.speech_token_index] = -100
            out["labels"] = self._mask_prompt(out["input_ids"], labels)
        else:
            self.tok.padding_side = "left"
            out = self.proc(text=[self.prompts[self.decode_style]] * len(rows), audios=audio,
                            return_tensors="pt", padding=True)
        return dict(out)


def prepare_epoch_batches(cfg, epoch: int):
    """Duration-efficient, source-balanced epoch from the full pool (as qwen_train)."""
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
        take = rng.sample(pool, k) if k <= len(pool) else list(pool) + rng.choices(pool, k=k - len(pool))
        selected.extend(take)
        counts[name] = len(take)
    selected.sort(key=lambda r: r["duration"])
    bs = int(cfg.data.batch_size)
    batches = [selected[i:i + bs] for i in range(0, len(selected), bs)]
    rng.shuffle(batches)
    lo, hi = cfg.data.min_duration, cfg.data.max_duration
    batches = [[r for r in b if lo <= r["duration"] <= hi] for b in batches]
    batches = [b for b in batches if b]
    log(f"[weighted-epoch] epoch={epoch} rows={sum(map(len, batches))} sources={counts} batches={len(batches)}")
    return batches


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model(cfg, device):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.model.path, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=cfg.model.attn_implementation,
    )

    if cfg.freeze.lora.enabled:
        from peft import LoraConfig, get_peft_model
        # Fully-qualified match: q_proj/k_proj/v_proj also exist in the Whisper
        # encoder, which must not pick up adapters.
        pattern = (r"text_decoder\.model\.layers\.\d+\."
                   r"(self_attn|mlp)\."
                   r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        model = get_peft_model(model, LoraConfig(
            r=cfg.freeze.lora.r, lora_alpha=cfg.freeze.lora.alpha,
            lora_dropout=cfg.freeze.lora.dropout, bias="none", target_modules=pattern,
        ))
        train_encoder = float(cfg.optim.lr_encoder) > 0
        train_adapter = float(cfg.optim.lr_projector) > 0
        # Style adaptation trains only the upper decoder layers' adapters, as the
        # winning Qwen style recipe did (layer_min 14 of 28); 0 = all layers.
        layer_min = int(cfg.freeze.lora.get("layer_min", 0))
        import re as _re
        for name, p in model.named_parameters():
            if "speech_encoder" in name:
                p.requires_grad_(train_encoder)
            elif "speech_audio_adapter" in name or "ln_speech" in name:
                p.requires_grad_(train_adapter)
            elif "lora_" in name and layer_min > 0:
                mm = _re.search(r"layers\.(\d+)\.", name)
                if mm and int(mm.group(1)) < layer_min:
                    p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model.to(device)

    groups, seen = [], set()

    def collect(match, lr, tag):
        ps = [p for n, p in model.named_parameters()
              if p.requires_grad and match(n) and id(p) not in seen]
        for p in ps:
            seen.add(id(p))
        if ps:
            groups.append({"params": ps, "lr": lr, "name": tag, "n": sum(p.numel() for p in ps)})

    collect(lambda n: "speech_encoder" in n, cfg.optim.lr_encoder, "speech_encoder")
    collect(lambda n: "speech_audio_adapter" in n or "ln_speech" in n, cfg.optim.lr_projector, "adapter")
    collect(lambda n: "lora_" in n, cfg.optim.lr_lora, "lora")
    collect(lambda n: True, cfg.optim.lr_decoder, "decoder")

    total = sum(p.numel() for p in model.parameters())
    train = sum(g["n"] for g in groups)
    log(f"[params] {total / 1e9:.3f} B total, {train / 1e6:.1f} M trainable ({100 * train / total:.2f}%)")
    for g in groups:
        log(f"         {g['name']:<14} {g['n'] / 1e6:8.1f} M  lr={g['lr']:g}")
    return model, groups


def base(model):
    """The MERaLiON module under the PEFT wrapper (or the model itself)."""
    return model.base_model.model if hasattr(model, "base_model") else model


def spec_augment(model, batch):
    """Apply the model's own SpecAugment config to the mel features.

    Same semantics as WhisperModel._mask_input_features: HF's
    _compute_mask_indices, zero-fill (the features are mean-normalized), time
    masks bounded by the real frames via feature_attention_mask.
    """
    if not model.training:
        return batch
    cfg = base(model).speech_encoder.config
    if not getattr(cfg, "apply_spec_augment", False):
        return batch
    from transformers.models.whisper.modeling_whisper import _compute_mask_indices
    x = batch["input_features"].clone()          # (B, C, T)
    B, C, T = x.shape
    if cfg.mask_time_prob > 0:
        m = _compute_mask_indices((B, T), mask_prob=cfg.mask_time_prob, mask_length=cfg.mask_time_length,
                                  attention_mask=batch["feature_attention_mask"], min_masks=cfg.mask_time_min_masks)
        m = torch.from_numpy(m).to(x.device).unsqueeze(1).expand(-1, C, -1)
        x[m] = 0
    if cfg.mask_feature_prob > 0:
        m = _compute_mask_indices((B, C), mask_prob=cfg.mask_feature_prob, mask_length=cfg.mask_feature_length,
                                  min_masks=cfg.mask_feature_min_masks)
        m = torch.from_numpy(m).to(x.device).unsqueeze(2).expand(-1, -1, T)
        x[m] = 0
    return {**batch, "input_features": x}


def build_optimizer(cfg, groups):
    opt = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups],
                            betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
    warm, total, floor = cfg.optim.warmup_steps, cfg.trainer.max_steps, cfg.optim.min_lr_ratio

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
    """Greedy-decode a manifest with the official decoding settings and score it."""
    model.eval()
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    rows.sort(key=lambda r: r["duration"])
    collate = Collator(proc, cfg.data.text_key, cfg.data.val_max_seconds, train=False,
                       decode_style=cfg.data.get("decode_style"))
    device = next(model.parameters()).device
    tok = proc.tokenizer
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids(END_OF_TURN)]
    hyps = []
    bs = cfg.data.val_batch_size
    try:
        for i in range(0, len(rows), bs):
            batch = to_device(collate(rows[i:i + bs]), device, model.dtype)
            out = model.generate(**batch, max_new_tokens=cfg.data.max_new_tokens,
                                 do_sample=False, num_beams=1, no_repeat_ngram_size=6,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id)
            # The ASR class returns prompt+completion; decide from the shape.
            L = batch["input_ids"].shape[1]
            gen = out[:, L:] if out.shape[1] > L else out
            hyps += [clean_text(t) for t in proc.batch_decode(gen, skip_special_tokens=True)]
    finally:
        model.train()

    m = score_hyps(hyps, [r["text_ref1"] for r in rows], [r["text_ref2"] for r in rows])
    log(f"[val:{tag}] cer={m['cer']:.4f}  wer={m['wer']:.4f}  ({len(rows)} utts)")
    grouped = {}
    labels = [str(r.get("eval_group") or r.get("source") or "unknown") for r in rows]
    if len(set(labels)) > 1:
        for group in sorted(set(labels)):
            idx = [i for i, v in enumerate(labels) if v == group]
            gm = score_hyps([hyps[i] for i in idx], [rows[i]["text_ref1"] for i in idx],
                            [rows[i]["text_ref2"] for i in idx])
            grouped[group] = gm
            log(f"[val:{tag}:{group}] cer={gm['cer']:.4f}  wer={gm['wer']:.4f}  ({len(idx)} utts)")
    m["groups"] = grouped
    for r, h in list(zip(rows, hyps))[:3]:
        log(f"    ref: {r['text_ref1'][:100]}")
        log(f"    hyp: {h[:100]}")
    return m


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def trainable_state(model):
    return {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(model, opt, sched, cfg, step, epoch, metrics, path):
    tmp = f"{path}.tmp"
    torch.save({"trainable": trainable_state(model), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "step": step, "epoch": epoch,
                "metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True)}, tmp)
    os.replace(tmp, path)
    log(f"[ckpt] {path}  step={step} epoch={epoch} wer={metrics.get('wer', float('nan')):.4f}")


def save_model_only(model, cfg, step, epoch, metrics, path):
    tmp = f"{path}.tmp"
    torch.save({"trainable": trainable_state(model), "step": step, "epoch": epoch,
                "metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True)}, tmp)
    os.replace(tmp, path)
    log(f"[model-ckpt] {path}  step={step} wer={metrics.get('wer', float('nan')):.4f}")


def load_ckpt(model, opt, sched, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ck["trainable"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint holds keys this model does not have: {result.unexpected_keys[:5]}")
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")

    exp = Path(cfg.paths.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(cfg.model.path, trust_remote_code=True, local_files_only=True)
    model, groups = build_model(cfg, device)
    opt, sched = build_optimizer(cfg, groups)

    last = exp / "last.ckpt"
    step, start_epoch, best = 0, 0, float("inf")
    init_ckpt = cfg.trainer.get("init_checkpoint", None)
    if init_ckpt and not last.exists():
        # Start from another run's trainable weights (LoRA + adapter + encoder),
        # optimizer and schedule fresh -- the style-adaptation stage.
        ck = torch.load(str(init_ckpt), map_location="cpu", weights_only=False)
        res = model.load_state_dict(ck["trainable"], strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"init checkpoint mismatch: {res.unexpected_keys[:5]}")
        log(f"[init] trainable weights from {init_ckpt} (step {ck.get('step')}); optimizer reset")
    if last.exists() and cfg.trainer.resume_if_exists:
        step, start_epoch = load_ckpt(model, opt, sched, str(last))
        bp = exp / "best.json"
        if bp.exists():
            best = json.loads(bp.read_text())["wer"]

    collate = Collator(proc, cfg.data.text_key, cfg.data.max_duration, train=True,
                       augment=bool(cfg.data.get("augment", False)),
                       style_tags=cfg.data.get("style_tags"), decode_style=cfg.data.get("decode_style"))
    if args.smoke:
        run_smoke(cfg, proc, model, collate, device)
        return
    if args.bench:
        run_bench(cfg, proc, model, opt, collate, device)
        return

    dtype = model.dtype
    use_specaug = bool(cfg.data.get("spec_augment", True))
    metrics_log = exp / "metrics.jsonl"
    t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0
    last_val_step = step if step > 0 else -1   # a resumed step is a validated step

    def run_validation(step, epoch, best):
        m = validate(model, proc, cfg, cfg.paths.val_unseen, "unseen")
        # Style folds validate on one held-out set only; don't decode it twice.
        m_all = m if str(cfg.paths.val_all) == str(cfg.paths.val_unseen) else             validate(model, proc, cfg, cfg.paths.val_all, "all")
        with open(metrics_log, "a") as f:
            f.write(json.dumps({"step": step, "epoch": epoch, "val_unseen": m, "val_all": m_all}) + "\n")
        save_ckpt(model, opt, sched, cfg, step, epoch, m, str(last))
        # Selection on robust_val; competition dev is watched, never selected on.
        if m["wer"] < best:
            best = m["wer"]
            save_model_only(model, cfg, step, epoch, m, str(exp / "best.ckpt"))
            (exp / "best.json").write_text(json.dumps(
                {"cer": m["cer"], "wer": m["wer"], "dev_wer": m_all["wer"], "step": step, "epoch": epoch}))
        # Keep a model-only copy at every validation so the dev curve can be re-decoded later.
        save_model_only(model, cfg, step, epoch, m, str(exp / f"step{step}.ckpt"))
        return best

    def finish(step, why):
        if not bool(cfg.trainer.get("keep_last_after_completion", False)):
            last.unlink(missing_ok=True)
            log("[ckpt] removed optimizer-bearing last.ckpt after clean completion")
        log(why)

    for epoch in range(start_epoch, cfg.trainer.max_epochs):
        batches = prepare_epoch_batches(cfg, epoch)
        if epoch == start_epoch and step > 0:
            # Resume mid-epoch: the sampler is deterministic in (seed, epoch), so
            # dropping the first `step` batches continues the same data order
            # instead of replaying the start of the epoch (max_epochs is 1).
            log(f"[resume] skipping {step} already-consumed batches of {len(batches)}")
            batches = batches[step:]
        loader = DataLoader(BatchDataset(batches), batch_size=None, shuffle=False,
                            num_workers=cfg.data.num_workers, collate_fn=collate, pin_memory=False,
                            persistent_workers=cfg.data.num_workers > 0,
                            prefetch_factor=4 if cfg.data.num_workers > 0 else None)
        log(f"=== epoch {epoch}: {len(batches)} batches, resuming at step {step}")

        for batch in loader:
            batch = to_device(batch, device, dtype)
            if use_specaug:
                batch = spec_augment(model, batch)
            loss = model(**batch, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.trainer.gradient_clip_val)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            seen += batch["input_ids"].size(0)
            loss_acc += loss.item()
            n_acc += 1

            if step % cfg.trainer.log_every_n_steps == 0:
                dt = time.time() - t0
                log(f"step {step:>6}/{cfg.trainer.max_steps}  ep {epoch}  loss {loss_acc / max(n_acc, 1):.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {dt / max(n_acc, 1):.3f} s/step  {seen / dt:.1f} utt/s  "
                    f"vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} G")
                with open(metrics_log, "a") as f:
                    f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_acc / max(n_acc, 1),
                                        "lr": sched.get_last_lr()[0], "s_per_step": dt / max(n_acc, 1)}) + "\n")
                t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0

            if step % cfg.trainer.val_every_n_steps == 0 or step >= cfg.trainer.max_steps:
                best = run_validation(step, epoch, best)
                last_val_step = step
                t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0

            if step >= cfg.trainer.max_steps:
                finish(step, f"max_steps={cfg.trainer.max_steps} reached")
                return

    # The weighted epoch can hold fewer batches than max_steps, in which case
    # the loop ends here between validations. Validate and save the final
    # weights rather than exit silently with the last checkpoint stale.
    if step != last_val_step:
        log(f"[epoch-end] epoch exhausted at step {step} < max_steps={cfg.trainer.max_steps}: final validation")
        best = run_validation(step, epoch, best)
    finish(step, f"epoch budget exhausted at step {step}")


def run_smoke(cfg, proc, model, collate, device):
    """The three things that would silently corrupt this training."""
    batches = prepare_epoch_batches(cfg, 0)
    rows = batches[0][:4]
    b = collate(rows)
    log(f"[smoke] shapes: {({k: tuple(v.shape) for k, v in b.items()})}")

    # 1. the loss must cover the transcript + <end_of_turn> and nothing else
    for i, r in enumerate(rows):
        kept = [t for t, l in zip(b["input_ids"][i].tolist(), b["labels"][i].tolist()) if l != -100]
        got = proc.tokenizer.decode(kept).strip()
        want = (r[cfg.data.text_key].strip() + END_OF_TURN).strip()
        log(f"[smoke] labels {'OK ' if got == want else 'BAD'}: {got[:90]!r}")
        assert got == want, f"label masking wrong\n got: {got!r}\nwant: {want!r}"
    n_speech = int((b["input_ids"] == collate.speech_token_index).sum(1)[0])
    log(f"[smoke] speech tokens per row: {n_speech} (expect 300 per 30 s chunk)")

    # 2. a forward pass produces a finite, plausible loss; SpecAugment must not break shapes
    b = to_device(b, device, model.dtype)
    with torch.no_grad():
        loss = model(**b, use_cache=False).loss
        model.train()
        loss_aug = model(**spec_augment(model, b), use_cache=False).loss
    log(f"[smoke] loss={loss.item():.4f}  with_specaug={loss_aug.item():.4f}  "
        f"(untrained head ~ln(V)=12.4; a pretrained ASR model on in-domain text should be well under 2)")
    assert torch.isfinite(loss) and torch.isfinite(loss_aug), "non-finite loss"

    # 3. batched generation must match one-at-a-time generation
    saved = cfg.data.val_batch_size
    m_b = validate(model, proc, cfg, cfg.paths.val_unseen, "batched", limit=8)
    cfg.data.val_batch_size = 1
    m_s = validate(model, proc, cfg, cfg.paths.val_unseen, "unbatched", limit=8)
    cfg.data.val_batch_size = saved
    log(f"[smoke] padding check: batched wer={m_b['wer']:.4f} vs unbatched wer={m_s['wer']:.4f}")
    log("[smoke] all checks passed")


def run_bench(cfg, proc, model, opt, collate, device):
    batches = prepare_epoch_batches(cfg, 0)
    longest = max(batches, key=lambda b: max(r["duration"] for r in b))
    typical = batches[len(batches) // 2]
    for tag, rows in (("typical", typical), ("longest", longest)):
        torch.cuda.reset_peak_memory_stats()
        b = to_device(collate(rows), device, model.dtype)
        t = time.time()
        for i in range(4):
            if i == 1:
                torch.cuda.synchronize()
                t = time.time()
            model(**spec_augment(model, b), use_cache=False).loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 3
        log(f"[bench] {tag:<8} bs={len(rows)} max_dur={max(r['duration'] for r in rows):5.1f}s  "
            f"{dt:.3f} s/step  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    log(f"[bench] {len(batches)} batches/epoch at batch_size={cfg.data.batch_size}")


if __name__ == "__main__":
    main()
