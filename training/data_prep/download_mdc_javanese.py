#!/usr/bin/env python3
"""Download the MDC Javanese dialect corpora that are missing from the training pool.

Terms must be accepted on the web first, per dataset (the API returns 403 otherwise;
this script prints the page URL for each one that still needs it).  Signed download
URLs are short-lived, so each archive is fetched immediately after it is requested.

    MDC_API_TOKEN=... python scripts/download_mdc_javanese.py            # all
    MDC_API_TOKEN=... python scripts/download_mdc_javanese.py banyumasan ngapak
"""
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

BASE = "https://mozilladatacollective.com/api"
WEB = "https://mozilladatacollective.com/datasets"
ROOT = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/data/raw")

# id known from the dataset page URL; None = resolved by name from the API listing.
DATASETS = [
    # speech
    ("banyumasan_tts",        "cmmamtaf104nemh07xa9e7sdx"),  # Banyumasan / ngapak, 9h40m, CC-BY-SA
    ("ngapak_tts",            "cmlgmf58l0096nx07tahttd6y"),  # Ngapak north coast, CC-BY-SA
    ("lumajang_tts",          "cml5bgysg00bhkr07g23kewke"),  # East Java, CC-BY-SA
    ("purworejo",             "cmtr89ry305csny07rrpxqe9d"),  # Purworejo speech corpus, CC-BY-NC-SA
    ("jombang_tts",           "cmpo53l5b000pku07cem453ac"),  # East Java, CC-BY-NC-SA
    ("gresik_tts",            "cmpwi0ac200zbo007hebmkzqy"),  # East Java, CC-BY-NC-SA
    ("multidialect_tts",      "cmtimj5tm0045mj07qrjo74gc"),  # CC-BY-NC-SA
    ("cv_spontaneous_jv",     "cmqi294n3004mmf07wvybp3fn"),  # Common Voice Spontaneous Speech 4.0 Javanese, CC0
    # text (dialect parallel corpora + magazine) -- for the synthetic-dialect idea and LM work
    ("panjebar_semangat",     "cmk719jvs02z1nt07tjswt52s"),
    ("ngapak_parallel",       "cmtr7p2v105bdny07rmc0xfmo"),
    ("gresik_parallel",       "cmtr7pgo605bhny07e9o9b6k8"),
    ("demakan_parallel",      "cmtufo0cx00gant07sgudmy3r"),
    ("multidialect_parallel", "cmtrd505k05reo107esjgl8s3"),
    ("lumajang_parallel",     "cmtbjsbum00vmnr0701e5imnc"),
    ("semarang_parallel",     "cmtbjkm6400vbnr075arsqkst"),
    ("jombang_parallel",      "cmtr7r73c05hoo107ene7xanq"),
    # scout report 2026-09-12: CC0 / open MDC items not yet held
    ("cv_sps_bundle_traindev", "cmfzu8u8wa555eq8onrk334h4"),
    ("cv_sps_bundle_test",     "cminc35no007no707hql26lzk"),
    ("cv_sps_betawi",          "cmqi2mr4z006omf076bl739t9"),
    ("jav_image_yogyakarta",   "cmtll3rp101zqnv07fr9tsj86"),
    ("cv_scripted_26_id",      "cmqinqwef00xonr07z4vbfovw"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def listing(headers):
    """id -> title for everything the account can see that mentions Javanese."""
    out = {}
    for q in ("javanese", "jawa", "ngapak"):
        try:
            r = requests.get(f"{BASE}/datasets", params={"q": q, "limit": 100},
                             headers=headers, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print(f"listing q={q} failed: {e}", flush=True)
            continue
        items = j if isinstance(j, list) else (j.get("data") or j.get("datasets") or j.get("items") or [])
        for d in items:
            if d.get("id"):
                out[d["id"]] = d.get("title") or d.get("name") or ""
    return out


def resolve(name, index):
    """Match a slug like 'gresik_tts' against the titles in the API listing."""
    words = [w for w in name.split("_") if w != "tts"]
    for did, title in index.items():
        t = title.lower()
        if all(w in t for w in words):
            return did
    return None


def main():
    token = os.environ.get("MDC_API_TOKEN")
    if not token:
        raise SystemExit("MDC_API_TOKEN is not set")
    wanted = set(sys.argv[1:])
    headers = {"Authorization": f"Bearer {token}"}
    ROOT.mkdir(parents=True, exist_ok=True)

    index = {}
    print(f"{len(index)} Javanese-matching datasets visible to this account\n", flush=True)
    for did, title in sorted(index.items(), key=lambda kv: kv[1]):
        print(f"  {did}  {title}")
    print(flush=True)

    needs_terms = []
    for name, dataset_id in DATASETS:
        if wanted and name not in wanted:
            continue
        outdir = ROOT / name
        if (outdir / ".complete").exists():
            print(f"SKIP complete: {name}", flush=True)
            continue
        dataset_id = dataset_id or resolve(name, index)
        if not dataset_id:
            print(f"UNRESOLVED {name}: not in the listing -- accept its terms on the web first", flush=True)
            needs_terms.append((name, None))
            continue

        print(f"\n=== Requesting {name} ({dataset_id}) ===", flush=True)
        r = requests.post(f"{BASE}/datasets/{dataset_id}/download", headers=headers, timeout=60)
        if r.status_code in (401, 403):
            print(f"NEEDS TERMS {name}: {WEB}/{dataset_id}", flush=True)
            needs_terms.append((name, dataset_id))
            continue
        if not r.ok:
            print(f"FAILED {name}: HTTP {r.status_code}: {r.text[:500]}", flush=True)
            continue
        info = r.json()
        archive = ROOT / (info.get("filename") or f"{name}.tar.gz")
        subprocess.run([
            "curl", "-fL", "--retry", "6", "--retry-all-errors", "-C", "-",
            "-o", str(archive), info["downloadUrl"]
        ], check=True)
        expected = (info.get("checksum") or "").removeprefix("sha256:")
        if expected:
            actual = sha256(archive)
            if actual.lower() != expected.lower():
                print(f"WARNING checksum mismatch for {name}: {actual[:12]} != {expected[:12]} -- keeping the archive if it extracts", flush=True)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {outdir}", flush=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(outdir, filter="data")
        (outdir / ".complete").write_text(f"dataset_id={dataset_id}\narchive={archive.name}\n")
        print(f"DONE {name}", flush=True)

    if needs_terms:
        print("\nAccept the terms for these on the web, then re-run "
              "(finished ones are skipped):", flush=True)
        for name, did in needs_terms:
            print(f"  {name}: {WEB}/{did}" if did else f"  {name}: search '{name}' on {WEB}")


if __name__ == "__main__":
    main()
