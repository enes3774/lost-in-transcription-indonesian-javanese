#!/usr/bin/env python3
"""Unified eval manifests for the MERaLiON-3 / Qwen3-Omni comparison.

Emits one CSV per eval set with a common schema so every runner and the scorer
can stay set-agnostic:

    uid, path, ref, lang, source, duration, speaker, convo_id

`lang` is the grouping we break results down by:
  dev    -> ind | javind   (competition dev's own field)
  robust -> id  | jv       (the pool tag carried in robust_val.jsonl)

robust_val.jsonl holds 700 rows but only 606 distinct audio files: 79 fleurs_jv
clips are repeated 2-3x, which is how every historical `robust WER` in the
project report was weighted. We keep all 700 rows so the numbers stay
comparable, give each row a unique uid, and let the runners transcribe each
distinct path once.
"""
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
OUT = ROOT / 'data/processed/newmodels'
OUT.mkdir(parents=True, exist_ok=True)


def uniquify(uids):
    """Append an occurrence index to repeated uids, preserving row order."""
    seen = Counter()
    out = []
    for u in uids:
        seen[u] += 1
        out.append(u if seen[u] == 1 else f'{u}#{seen[u]}')
    return out


def build_dev():
    df = pd.read_csv(ROOT / 'data/processed/dev/manifest.csv', keep_default_na=False)
    # The dev manifest carries no duration column; read it off the files once.
    durations = [sf.info(p).frames / sf.info(p).samplerate for p in df.path]
    return pd.DataFrame({
        'uid': df.audio_filename,
        'path': df.path,
        'ref': df.transcript,
        'lang': df.language,
        'source': 'dev',
        'duration': durations,
        'speaker': df.speaker,
        'convo_id': df.convo_id,
    })


def build_robust():
    rows = [json.loads(l) for l in open(ROOT / 'data/processed/qwen_v2/robust_val.jsonl')]
    return pd.DataFrame({
        'uid': uniquify([r['uid'] for r in rows]),
        'path': [r['audio_filepath'] for r in rows],
        'ref': [r['text_ref1'] for r in rows],
        'lang': [r['language'] for r in rows],
        'source': [r['source'] for r in rows],
        'duration': [r['duration'] for r in rows],
        'speaker': '',
        'convo_id': '',
    })


for name, build in [('dev', build_dev), ('robust', build_robust)]:
    df = build()
    assert df.uid.is_unique, f'{name}: duplicate uids'
    missing = [p for p in df.path.unique() if not Path(p).exists()]
    assert not missing, f'{name}: {len(missing)} missing audio files, e.g. {missing[:2]}'
    path = OUT / f'{name}.csv'
    df.to_csv(path, index=False)
    le30 = int((df.duration <= 30).sum())
    print(f'{name}: {len(df)} rows / {df.path.nunique()} distinct clips, '
          f'{df.duration.sum()/3600:.3f} h, <=30s {le30}, >30s {len(df)-le30}')
    print(f'   by lang  : {df.groupby("lang").size().to_dict()}')
    print(f'   by source: {df.groupby("source").size().to_dict()}')
    print(f'   -> {path}')
