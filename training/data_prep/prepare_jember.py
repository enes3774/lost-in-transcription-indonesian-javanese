#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
SOURCE = ROOT / 'data/raw/jember/Jember Javanese Spontaneous Speech Corpus'
TSV = SOURCE / 'Jember Javanese Spontaneous Speech Corpus - 1-200.tsv'
AUDIO = SOURCE / 'mp3 audio'
OUT = ROOT / 'data/processed/jember'
CLIPS = OUT / 'clips'
TMP = ROOT / 'tmp/jember_decode'
OUT.mkdir(parents=True, exist_ok=True)
CLIPS.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

def seconds(value):
    h, m, s = str(value).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

df = pd.read_csv(TSV, sep='\t')
df['start_seconds'] = df['start'].map(seconds)
df['end_seconds'] = df['end'].map(seconds)
df['duration_seconds'] = df['end_seconds'] - df['start_seconds']
rows = []
for session, group in df.groupby('Audio file name', sort=True):
    session = int(session)
    src = AUDIO / f'{session}.mp3'
    decoded = TMP / f'{session}.wav'
    subprocess.run([
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(src), '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(decoded)
    ], check=True)
    wave, sr = sf.read(decoded, dtype='float32', always_2d=False)
    assert sr == 16000
    for within_session, (source_row, row) in enumerate(group.iterrows()):
        filename = f'jember_{session:03d}_{within_session:04d}.wav'
        target = CLIPS / filename
        a = max(0, round(float(row.start_seconds) * sr))
        b = min(len(wave), round(float(row.end_seconds) * sr))
        if b <= a:
            print(f'SKIP non-positive interval row={source_row} session={session} {a}:{b}', flush=True)
            continue
        sf.write(target, wave[a:b], sr, subtype='PCM_16')
        rows.append({
            'audio_filename': filename,
            'path': str(target),
            'session': session,
            'source_row': int(source_row),
            'start_seconds': float(row.start_seconds),
            'end_seconds': float(row.end_seconds),
            'duration_seconds': (b-a)/sr,
            'transcript': str(row.text),
        })
    decoded.unlink()
    if session % 20 == 0:
        print(f'prepared session {session}/208', flush=True)
manifest = pd.DataFrame(rows)
manifest.to_csv(OUT / 'manifest.csv', index=False)
# Two deterministic, evenly spaced utterances from every session: 416 clips.
selected = []
for _, g in manifest.groupby('session', sort=True):
    positions = np.linspace(0, len(g)-1, num=min(2, len(g)), dtype=int)
    selected.append(g.iloc[positions])
subset = pd.concat(selected, ignore_index=True)
subset.to_csv(OUT / 'benchmark_subset.csv', index=False)
print(f'DONE clips={len(manifest)} hours={manifest.duration_seconds.sum()/3600:.3f}')
print(f'SUBSET clips={len(subset)} hours={subset.duration_seconds.sum()/3600:.3f} sessions={subset.session.nunique()}')
