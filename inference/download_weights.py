"""Download the trained models from the Hugging Face Hub into inference/models/.

usage:  python download_weights.py [--repo <hf_repo_id>] [--models-dir inference/models] [--only whisper,qwen]

After this, inference/models/ holds the tree main.py expects: whisper qwen meralion ctc omni vibevoice seamless owsm
lm (about 65 GB). The rescoring LLM is used as released and comes from its own repository (--with-llm, 18.5 GB).
Re-running is incremental: files already present are skipped.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "enesssssw/lost-in-transcription-indonesian-javanese-ensemble"
LLM_REPO = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
MEMBERS = ("whisper", "qwen", "meralion", "ctc", "omni", "vibevoice", "seamless", "owsm", "lm")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=DEFAULT_REPO)
ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"))
ap.add_argument("--only", default=None, help="comma-separated subset of member folders")
ap.add_argument("--with-llm", action="store_true", help=f"also download {LLM_REPO} into models/llm")
a = ap.parse_args()
want = a.only.split(",") if a.only else list(MEMBERS)
bad = [m for m in want if m not in MEMBERS]
if bad:
    sys.exit(f"unknown member(s) {bad}; choose from {MEMBERS}")
out = Path(a.models_dir)
out.mkdir(parents=True, exist_ok=True)
# the repo stores every member under models/<name>/ ; download straight into the target tree
snapshot_download(a.repo, repo_type="model", local_dir=str(out.parent), allow_patterns=[f"models/{m}/**" for m in want])
if a.with_llm:
    snapshot_download(LLM_REPO, repo_type="model", local_dir=str(out / "llm"))
missing = [m for m in want if not (out / m).is_dir()] + (["llm"] if a.with_llm and not (out / "llm").is_dir() else [])
if missing:
    sys.exit(f"download finished but these folders are missing: {missing}")
print("weights ready:", ", ".join(want + (["llm"] if a.with_llm else [])))
