"""Move every pool row whose label still contains a digit out of train_pool_v2.jsonl (118 rows).

prepare_qwen_v2_data.py spells numbers out as words; the rows it could not convert keep a digit and are set aside in
digit_rows_excluded.jsonl. This was a one-off inline step on 2026-09-02; this file applies the same rule
(any digit in text_ref1). Note: the Qwen3-ASR stage-1 LoRA had already started and read the file with these
118 rows still in it; every later member reads the 216,263-row file."""
import os
import json
import re
from pathlib import Path

Q = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/data/processed/qwen_v2")
rows = [json.loads(l) for l in open(Q / "train_pool_v2.jsonl", encoding="utf-8")]
keep = [r for r in rows if not re.search(r"\d", r["text_ref1"])]
drop = [r for r in rows if re.search(r"\d", r["text_ref1"])]
for name, part in (("train_pool_v2.jsonl", keep), ("digit_rows_excluded.jsonl", drop)):
    with open(Q / name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"kept {len(keep)}, excluded {len(drop)}")
