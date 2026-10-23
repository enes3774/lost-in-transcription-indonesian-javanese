"""Full fine-tune of SeamlessM4T-v2-large (speech -> text, 2.3B) on this track's pool, with dev eval.

Why: it is the only untested model on disk that declares Javanese natively (`jav` source and target), and zero shot
it is already 0.283 on dev - the best zero-shot number of any non-shipped model. Its decoder is an NLLB-style
seq2seq decoder, not an LLM, so its errors may complement the four LLM-decoder members the way the CTC voter's do.
Its speech encoder is the same w2v-BERT 2.0 our CTC member was fully fine-tuned from, so a full fine-tune of
encoder + decoder is the natural recipe (fp32 master weights, bf16 autocast - the omni lesson: pure-bf16 AdamW
freezes most weights). Same 12,000-clip v5mix sample, same label convention (text_ref1), as the other members.
Labels carry the target-language token `__jav__` and </s>, exactly what generate(tgt_lang="jav") forces.
Eval: greedy dev decode in-process every --eval_every optimizer steps, dev CSV to predictions/newmodels, bf16
checkpoint saved next to it. Env: /kaggle/working/verify_venv (transformers 4.57.6).
usage: seamless_ft_jv.py --run v5_full [--n_train 12000] [--batch 4] [--accum 2] [--lr 2e-5] [--eval_every 500]"""
import os
import argparse, importlib.util, json, math, os, random, sys, time, types, unicodedata
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pool", default="v4", choices=["v4", "v4_clean"])
ap.add_argument("--mix", default="v5mix", choices=["v5mix", "v6mix"])
ap.add_argument("--n_train", type=int, default=12000)
ap.add_argument("--min_s", type=float, default=1.0)
ap.add_argument("--max_s", type=float, default=30.0)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--tgt", default="jav")
args = ap.parse_args()

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from jiwer import wer  # noqa: E402
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText  # noqa: E402

R = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
if importlib.util.find_spec("typer") is None:
    t = types.ModuleType("typer")
    t.Typer = lambda *a, **k: types.SimpleNamespace(command=lambda *a, **k: (lambda f: f))
    t.Option = t.Argument = lambda *a, **k: None
    sys.modules["typer"] = t
spec = importlib.util.spec_from_file_location("competition_score", R / "runtime_repo/score.py")
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
norm = lambda s: unicodedata.normalize("NFC", score.normalize_text(str(s))).strip()

S = R / "models/candidates/seamless-m4t-v2-large"
RUNS = R / "training/seamless_ft" / args.run
RUNS.mkdir(parents=True, exist_ok=True)
OUT = R / "predictions/newmodels"
POOL = R / "data/processed/qwen_v2" / {"v4": "train_pool_v4.jsonl",
                                       "v4_clean": "train_pool_v4_clean.jsonl"}[args.pool]
MIXES = {
    "v5mix": {"jember": .19, "homostoria": .13, "hari": .05, "slr35": .19, "central": .08, "fleurs_jv": .07, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .08, "fleurs_id": .08},
    "v6mix": {"jember": .16, "homostoria": .11, "hari": .05, "slr35": .11, "central": .06, "fleurs_jv": .05, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .05, "fleurs_id": .05, "banyumasan": .10, "gresik": .05, "jombang": .05, "multidialect": .03},
}
WEIGHTS = MIXES[args.mix]
SR = 16000


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def load(path):
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav.astype(np.float32)


processor = AutoProcessor.from_pretrained(str(S))
tok = processor.tokenizer
tok.tgt_lang = args.tgt                      # labels = [__jav__, ..., </s>]; generate(tgt_lang) forces the same token
t0 = time.time()
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(str(S), dtype=torch.float32).cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable()
n_par = sum(p.numel() for p in model.parameters())
log(f"model loaded in {time.time() - t0:.0f}s, {n_par / 1e9:.2f}B params, all trainable, fp32 master + bf16 autocast")

dev = pd.read_csv(R / "data/processed/newmodels/dev.csv", keep_default_na=False)
dev_ref = [norm(r).casefold() for r in dev.ref]


def build(rows):
    audio = [load(r["audio_filepath"]) for r in rows]
    x = processor(audios=audio, sampling_rate=SR, return_tensors="pt", padding=True)
    lab = tok([r["text_ref1"] for r in rows], text_target=[r["text_ref1"] for r in rows], return_tensors="pt",
              padding=True, truncation=True, max_length=256)["labels"]
    lab[lab == tok.pad_token_id] = -100
    return {"input_features": x["input_features"].cuda(), "attention_mask": x["attention_mask"].cuda(),
            "labels": lab.cuda()}


def evaluate(tag):
    model.eval(); model.config.use_cache = True
    order = sorted(range(len(dev)), key=lambda i: -float(dev.duration[i]))
    hyp, t1 = {}, time.time()
    for b in range(0, len(order), 8):
        idx = order[b:b + 8]
        audio = [load(dev.path[i]) for i in idx]
        x = processor(audios=audio, sampling_rate=SR, return_tensors="pt", padding=True)
        x = {k: v.cuda() for k, v in x.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model.generate(**x, tgt_lang=args.tgt, num_beams=1, max_new_tokens=256)
        for i, t in zip(idx, processor.batch_decode(y, skip_special_tokens=True)):
            hyp[i] = " ".join(str(t).split())
    texts = [hyp[i] for i in range(len(dev))]
    w = wer(dev_ref, [norm(t).casefold() or "_" for t in texts])
    out = OUT / f"seamless_ft_{args.run}_{tag}_dev.csv"
    pd.DataFrame({"uid": dev.uid, "transcript": texts}).to_csv(out, index=False)
    log(f"EVAL {tag}: dev WER {w:.4f} ({time.time() - t1:.0f}s) -> {out}")
    model.train(); model.config.use_cache = False
    return w


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

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = math.ceil(len(clips) / (args.batch * args.accum))
warm = max(1, int(0.10 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
model.train()
step, run_loss, t1, best = 0, [], time.time(), (9.9, None)
batches = [clips[i:i + args.batch] for i in range(0, len(clips), args.batch)]
for bi, rows in enumerate(batches):
    b = build(rows)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b)
    (out.loss / args.accum).backward()
    run_loss.append(out.loss.item())
    if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            log(f"step {step}/{total} loss {sum(run_loss) / len(run_loss):.4f} "
                f"{(time.time() - t1) / ((bi + 1) * args.batch):.2f}s/sample mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            run_loss = []
        if step % args.eval_every == 0 or step == total:
            w = evaluate(f"step{step}")
            if w < best[0]:
                best = (w, step)
                d = RUNS / "best"
                model.to(torch.bfloat16).save_pretrained(str(d), safe_serialization=True)
                model.to(torch.float32)
                processor.save_pretrained(str(d))
                log(f"saved best (step {step}, {w:.4f}) -> {d}")
log(f"all done; best dev WER {best[0]:.4f} at step {best[1]}")
print("SEAMLESS_FT_DONE", flush=True)
