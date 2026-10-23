#!/usr/bin/env python3
"""Download approved MDC datasets after web terms have been accepted.

Set MDC_API_TOKEN in the environment. Tokens are created in MDC profile settings.
"""
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

BASE = "https://mozilladatacollective.com/api"
ROOT = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/data/raw")
DATASETS = [
    ("dev", "cmsxiwqdo000fny07e5fx0yi5"),
    ("jember", "cmlgm5a94008kny07nz2intus"),
    ("homostoria", "cmiepnyu1001jo207hzslvb5m"),
    ("common_voice_indonesian_26", "cmqinqwef00xonr07z4vbfovw"),
    ("central_javanese", "cml5bn4k900aame07u0rwidcg"),
    ("hari_minggoean", "cmhlvswz200fio007c7szypw8"),
    ("common_voice_spontaneous_javanese_4", "cmqi294n3004mmf07wvybp3fn"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    token = os.environ.get("MDC_API_TOKEN")
    if not token:
        raise SystemExit("MDC_API_TOKEN is not set")
    wanted = set(sys.argv[1:])
    headers = {"Authorization": f"Bearer {token}"}
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, dataset_id in DATASETS:
        if wanted and name not in wanted and dataset_id not in wanted:
            continue
        outdir = ROOT / name
        done = outdir / ".complete"
        if done.exists():
            print(f"SKIP complete: {name}", flush=True)
            continue
        print(f"\n=== Requesting {name} ({dataset_id}) ===", flush=True)
        r = requests.post(f"{BASE}/datasets/{dataset_id}/download", headers=headers, timeout=60)
        if not r.ok:
            print(f"FAILED {name}: HTTP {r.status_code}: {r.text[:1000]}", flush=True)
            continue
        info = r.json()
        filename = info.get("filename") or f"{name}.tar.gz"
        archive = ROOT / filename
        subprocess.run([
            "curl", "-fL", "--retry", "6", "--retry-all-errors", "-C", "-",
            "-o", str(archive), info["downloadUrl"]
        ], check=True)
        expected = (info.get("checksum") or "").removeprefix("sha256:")
        if expected:
            actual = sha256(archive)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {expected}")
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {outdir}", flush=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(outdir, filter="data")
        done.write_text(f"dataset_id={dataset_id}\narchive={archive.name}\n")
        print(f"DONE {name}", flush=True)

if __name__ == "__main__":
    main()
