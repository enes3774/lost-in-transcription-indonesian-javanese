"""Manifests for the omniASR-LLM-7B fine-tune: one epoch = the v5mix sampling plan made explicit (~200 h),
held-out = the 372 dev clips (pool-only training, so dev stays honest)."""
import os
import json, random, collections, yaml
from pathlib import Path
import pandas as pd
R = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription")); O = R / "training/omni_jv"; O.mkdir(parents=True, exist_ok=True)
W = yaml.safe_load(open(R / "training/meralion_v1/meralion_v5mix.yaml"))["sampling"]["weights"]
print("weights", W, "sum", round(sum(W.values()), 3))
by = collections.defaultdict(list)
for l in open(R / "data/processed/qwen_v2/train_pool_v4.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["source"] in W and W[r["source"]] > 0 and 0.5 <= r["duration"] <= 40 and r["text_ref1"].strip(): by[r["source"]].append(r)
random.seed(5); H = 200.0; out = []
for s, w in W.items():
    rows = by.get(s, [])
    if not rows or w <= 0: print("  skip", s); continue
    tgt = w / sum(W.values()) * H * 3600; got = 0; n = 0
    while got < tgt:
        random.shuffle(rows)
        for r in rows:
            out.append({"audio_filepath": r["audio_filepath"], "offset": 0.0, "duration": r["duration"], "text": r["text_ref1"], "lang": "ind_Latn", "src": s}); got += r["duration"]; n += 1
            if got >= tgt: break
    print(f"  {s:16s} w {w:.2f}  pool {len(rows):6d} rows {sum(r['duration'] for r in rows)/3600:6.1f} h -> {n:6d} rows {got/3600:5.1f} h ({got/max(1,sum(r['duration'] for r in rows)):.1f} passes)")
random.shuffle(out)
with open(O / "train_v5mix.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train rows", len(out), "hours", round(sum(r["duration"] for r in out) / 3600, 1))
dev = pd.read_csv(R / "data/processed/newmodels/dev.csv", keep_default_na=False)
import librosa, soundfile as sf
S = R / "tmp/omni_wav"; S.mkdir(parents=True, exist_ok=True)
with open(O / "dev_held.jsonl", "w", encoding="utf-8") as f:
    for i, r in dev.iterrows():
        dst = S / (Path(r.path).stem + ".wav")
        if not dst.exists():
            w, _ = librosa.load(r.path, sr=16000, mono=True); sf.write(dst, w, 16000)
        f.write(json.dumps({"audio_filepath": str(dst), "offset": 0.0, "duration": float(r.duration), "text": r.ref, "lang": "ind_Latn", "rec": r.uid, "speaker": str(r.speaker)}, ensure_ascii=False) + "\n")
print("held", len(dev))
