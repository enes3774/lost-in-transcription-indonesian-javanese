"""Shared plumbing for the MERaLiON-3 / Qwen3-Omni candidate evaluation.

Every runner takes a unified eval set (see build_eval_sets.py), transcribes each
*distinct* audio file once, and writes `uid,transcript` for all rows -- so the
79 repeated fleurs_jv rows in robust_val cost nothing extra to evaluate but keep
their historical weighting when scored.
"""
import os
import re
import time
from pathlib import Path

import librosa
import pandas as pd

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
SETS = ROOT / 'data/processed/newmodels'
PRED = ROOT / 'predictions/newmodels'


SPEAKER_TAG = re.compile(r'<\s*speaker\s*\d*\s*>\s*:?', re.I)


def clean_text(text):
    """Drop diarization tags the audio LLMs prepend (MERaLiON emits `<Speaker1>:`).

    They are a response-format artifact, not a transcription choice, and the
    official normalizer keeps them as words -- one guaranteed error per clip.
    """
    return SPEAKER_TAG.sub(' ', text).strip()


def add_common_args(ap):
    ap.add_argument('eval_set')
    ap.add_argument('--out', required=True, help='output name under predictions/newmodels')
    ap.add_argument('--max-duration', type=float, default=None,
                    help='skip clips longer than this many seconds')
    ap.add_argument('--sample', type=int, default=None,
                    help='probe mode: evenly spaced N clips')
    ap.add_argument('--batch-size', type=int, default=4)
    return ap


def load_work(args):
    """Return (work, rows): distinct clips to transcribe, and the full row table."""
    rows = pd.read_csv(SETS / f'{args.eval_set}.csv', keep_default_na=False)
    kept = rows
    if args.max_duration is not None:
        kept = kept[kept.duration <= args.max_duration]
    work = kept.drop_duplicates(subset='path', keep='first').copy()
    if args.sample:
        step = max(1, len(work) // args.sample)
        work = work.iloc[::step].head(args.sample).copy()
    # Longest first: a batch's cost is set by its longest member, and sorting
    # keeps padding waste down while surfacing OOM on the very first batch.
    work = work.sort_values('duration', ascending=False).reset_index(drop=True)
    print(f'{args.eval_set}: {len(rows)} rows -> {len(work)} clips to transcribe '
          f'({work.duration.sum()/3600:.3f} h)', flush=True)
    return work, rows


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def batches(df, size):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]


class Writer:
    """Incremental writer: maps transcripts back onto every row sharing a path."""

    def __init__(self, rows, out_name):
        self.rows = rows
        self.path = PRED / f'{out_name}.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.by_path = {}
        self.raw_by_path = {}
        self.start = time.time()

    def add(self, path, text, raw=None):
        self.by_path[path] = text
        self.raw_by_path[path] = text if raw is None else raw

    def flush(self):
        df = self.rows[self.rows.path.isin(self.by_path)].copy()
        df['transcript'] = df.path.map(self.by_path)
        df['raw'] = df.path.map(self.raw_by_path)
        df[['uid', 'transcript', 'raw']].to_csv(self.path, index=False)
        return len(df)

    def progress(self, done, total):
        el = time.time() - self.start
        rate = done / el if el else 0
        eta = (total - done) / rate if rate else 0
        print(f'  {done}/{total} clips  {el/60:.1f} min elapsed  ETA {eta/60:.1f} min',
              flush=True)

    def finish(self):
        n = self.flush()
        print(f'DONE clips={len(self.by_path)} rows={n} '
              f'seconds={time.time()-self.start:.1f} output={self.path}', flush=True)
