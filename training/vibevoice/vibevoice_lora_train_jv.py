"""LoRA fine-tune of VibeVoice-ASR-HF (7B, MIT) on this track's pool, with dev eval.

VibeVoice declares `jv` and `id` and hears the Javanese that SeaLLMs rewrites into Indonesian, but zero shot it
is 0.3053 on dev and adds nothing on the Javanese-heavy clips (0.2599 -> 0.2599), so the question is whether a
fine-tune turns it into a real member. Microsoft's own recipe (finetuning-asr/lora_finetune.py) is written for
their non-HF package; this ports its hyperparameters onto the transformers model we already run:
LoRA r16 / alpha 32 / dropout 0.05, lr 1e-4, 10 % warmup, weight decay 0.01, clip 1.0, bf16, gradient
checkpointing, batch 1 x accumulation 4. Targets are the decoder's attention projections, as for Voxtral and
SeaLLMs, so the three bases stay comparable; the acoustic and semantic tokenizers are frozen.

The label is the model's own output format - `[{"Start":0,"End":<dur>,"Speaker":0,"Content":"<text>"}]` in
the assistant turn - so the fine-tune changes what it hears, not how it writes, and the existing parser still
applies. Only the assistant turn carries loss. The data is the same 12,000-clip v5mix sample as the other two.

Runs in /kaggle/working/venv_qwen (transformers 5.15; VibeVoice needs >= 5.3).
usage: vibevoice_lora_train_jv.py --run v5_r16 [--n_train 12000] [--eval_every 1500] [--eval_only ADAPTER]"""
import os
import argparse, importlib.util, json, math, os, random, re, sys, time, types, unicodedata
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pool", default="v4", choices=["v4", "v4_clean"])
ap.add_argument("--mix", default="v5mix", choices=["v5mix", "v6mix"])
ap.add_argument("--n_train", type=int, default=12000)
ap.add_argument("--min_s", type=float, default=1.0)
ap.add_argument("--max_s", type=float, default=30.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=1500)
ap.add_argument("--eval_only", default=None)
args = ap.parse_args()

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration  # noqa: E402

R = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
if importlib.util.find_spec("typer") is None:
    t = types.ModuleType("typer")
    t.Typer = lambda *a, **k: types.SimpleNamespace(command=lambda *a, **k: (lambda f: f))
    t.Option = t.Argument = lambda *a, **k: None
    sys.modules["typer"] = t
spec = importlib.util.spec_from_file_location("competition_score", R / "runtime_repo/score.py")
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
norm = lambda s: unicodedata.normalize("NFC", score.normalize_text(str(s))).strip()

S = R / "models/candidates/vibevoice-asr-hf"
RUNS = R / "training/vibevoice_lora" / args.run
RUNS.mkdir(parents=True, exist_ok=True)
OUT = R / "predictions/newmodels"
POOL = R / "data/processed/qwen_v2" / {"v4": "train_pool_v4.jsonl",
                                       "v4_clean": "train_pool_v4_clean.jsonl"}[args.pool]
MIXES = {
    "v5mix": {"jember": .19, "homostoria": .13, "hari": .05, "slr35": .19, "central": .08, "fleurs_jv": .07, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .08, "fleurs_id": .08},
    "v6mix": {"jember": .16, "homostoria": .11, "hari": .05, "slr35": .11, "central": .06, "fleurs_jv": .05, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .05, "fleurs_id": .05, "banyumasan": .10, "gresik": .05, "jombang": .05, "multidialect": .03},
}
WEIGHTS = MIXES[args.mix]
CONTENT = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"')
SPECIAL = re.compile(r"<\|[^|]*\|>")


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def parse_transcript(raw):
    """The Content fields of the JSON-ish output; the stripped text when there are none (as run_vibevoice.py)."""
    txt = SPECIAL.sub(" ", raw)
    parts = CONTENT.findall(txt)
    if parts:
        out = []
        for p in parts:
            try:
                out.append(json.loads('"' + p + '"'))
            except json.JSONDecodeError:
                out.append(p.replace('\\"', '"'))
        return " ".join(s.strip() for s in out if s.strip())
    return re.sub(r"\s+", " ", txt.replace("assistant", " ", 1)).strip()


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


processor = AutoProcessor.from_pretrained(str(S))
tok = processor.tokenizer
SR = processor.feature_extractor.sampling_rate


def load(path):
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav.astype(np.float32)


def build(path, target):
    wav = load(path)
    x = processor.apply_chat_template([[{"role": "user", "content": [{"type": "audio", "audio": wav}]}]],
                                      tokenize=True, add_generation_prompt=True, return_dict=True)
    dur = len(wav) / SR
    end = f"{dur:.2f}".rstrip("0").rstrip(".")
    ans = ('<|im_start|>assistant\n[{"Start":0,"End":' + end + ',"Speaker":0,"Content":'
           + json.dumps(target, ensure_ascii=False) + "}]<|im_end|>\n")
    tgt = tok(ans, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([x["input_ids"], tgt], dim=1)
    labels = ids.clone()
    labels[:, : x["input_ids"].shape[1]] = -100
    b = {k: v for k, v in x.items() if k not in ("input_ids", "attention_mask")}
    b = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.cuda()) for k, v in b.items()}
    b.update(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids).cuda(), labels=labels.cuda())
    return b


t0 = time.time()
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(str(S), dtype=torch.bfloat16, attn_implementation="eager",  # the acoustic tokenizer has no SDPA path
                                                             device_map={"": 0})
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
log(f"model loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB, SR {SR}")
if args.eval_only:
    pm = PeftModel.from_pretrained(model, args.eval_only)
else:
    pm = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=r"model\.language_model.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
    pm.print_trainable_parameters()
model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

dev = pd.read_csv(R / "data/processed/newmodels/dev.csv", keep_default_na=False)


@torch.no_grad()
def evaluate(tag, batch=4):
    pm.eval(); model.config.use_cache = True
    order = sorted(range(len(dev)), key=lambda i: -float(dev.duration[i]))
    hyp, t1 = {}, time.time()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        x = processor.apply_transcription_request(audio=[load(dev.path[i]) for i in idx], sampling_rate=SR)
        x = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.cuda()) for k, v in x.items()}
        out = pm.generate(**x, max_new_tokens=512, do_sample=False)
        for i, raw in zip(idx, processor.decode(out[:, x["input_ids"].shape[1]:])):
            hyp[i] = parse_transcript(str(raw))
    path = OUT / f"vibevoice_lora_{args.run}_{tag}_dev.csv"
    pd.DataFrame({"uid": dev.uid, "transcript": [hyp.get(i, "") for i in range(len(dev))]}).to_csv(path, index=False)
    e = n = 0
    for i in range(len(dev)):
        r, h = norm(dev.ref[i]).split(), norm(hyp.get(i, "")).split()
        e += lev(r, h); n += len(r)
    log(f"EVAL {tag}: dev WER {e/n:.4f} ({time.time() - t1:.0f}s) -> {path}")
    model.config.use_cache = False; pm.train()


if args.eval_only:
    evaluate(Path(args.eval_only).name)
    sys.exit(0)

pool = {}
for line in open(POOL, encoding="utf-8"):
    r = json.loads(line)
    if r["source"] in WEIGHTS and args.min_s <= float(r["duration"]) <= args.max_s and r["text_ref1"].strip():
        pool.setdefault(r["source"], []).append(r)
missing = [s for s in WEIGHTS if s not in pool]
assert not missing, f"mixture names sources the pool does not have: {missing}"
rng = random.Random(args.seed)
clips = []
for src, w in WEIGHTS.items():
    rows = pool[src][:]; rng.shuffle(rows)
    k = round(args.n_train * w)
    clips += (rows * math.ceil(k / len(rows)))[:k]
rng.shuffle(clips)
log(f"train clips {len(clips)}, {sum(float(c['duration']) for c in clips) / 3600:.1f} h")

lora_params = [p for n, p in pm.named_parameters() if "lora_" in n]
opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
total = math.ceil(len(clips) / args.accum)
warm = max(1, int(0.10 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
pm.train()
step, run_loss, t1 = 0, [], time.time()
for i, c in enumerate(clips):
    b = build(c["audio_filepath"], c["text_ref1"])
    out = pm(**b)
    (out.loss / args.accum).backward()
    run_loss.append(out.loss.item())
    if (i + 1) % args.accum == 0:
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            log(f"step {step}/{total} loss {sum(run_loss) / len(run_loss):.4f} "
                f"{(time.time() - t1) / (i + 1):.2f}s/sample mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            run_loss = []
        if step % args.eval_every == 0 or step == total:
            pm.save_pretrained(str(RUNS / f"step{step}"))
            evaluate(f"step{step}")
log("all done")
print("VIBEVOICE_TRAIN_DONE", flush=True)
