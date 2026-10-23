"""train_pool_v4_clean.jsonl: pool v4 without the 30 rows whose labels are broken.

30 `banyumasan` rows carry leaked manifest fields - hex clip ids, `webm`, tabs - from a broken tab-separated
parse, one of them 1,104 words long on a 6.8 s clip (that row took MERaLiON from 46.6 GB to 231 GB in one step).
Rules only, no model in the loop, so no selection toward what our systems already hear. Every other row of the
four sources added for the v6 mix is checked with the same rules and kept."""
import os
import json, re
from pathlib import Path
R = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
Q = R / "data/processed/qwen_v2"
HEX = re.compile(r"\b[0-9a-f]{12,}\b"); JUNK = re.compile(r"webm|\.wav|\.mp3|\t|https?:")
NEW = {"banyumasan", "gresik", "jombang", "multidialect"}
keep, drop = [], []
for l in open(Q / "train_pool_v4.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = str(r["text_ref1"]); d = float(r["duration"]); nw = len(t.split())
    # only the four sources new to v6mix: cleaning the old ones too would add a second change to the v5mix/v6mix A/B
    bad = r["source"] in NEW and (HEX.search(t) or JUNK.search(t) or nw / max(d, 0.5) > 6 or (d > 3 and nw / d < 0.3))
    (drop if bad else keep).append(r)
with open(Q / "train_pool_v4_clean.jsonl", "w", encoding="utf-8") as h:
    for r in keep:
        h.write(json.dumps(r, ensure_ascii=False) + "\n")
by = {}
for r in drop:
    by[r["source"]] = by.get(r["source"], 0) + 1
print(f"kept {len(keep)}, dropped {len(drop)}: {by}")
