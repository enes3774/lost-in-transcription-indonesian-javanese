#!/usr/bin/env python3
"""Ingest the MDC Javanese dialect corpora (Banyumasan, Ngapak, Bojonegoro) into the pool.

Same recipe as prepare_qwen_lit_data.py: mapping.tsv -> ffmpeg 16 kHz mono wav under
data/processed/qwen_train/audio/<source>/, official normaliser on the text, rows in the
train_pool_v2 schema.  Writes data/processed/qwen_v2/dialect_rows.jsonl and
train_pool_v3.jsonl (= v2 + dialect rows).
"""
import os
import importlib.util, json, os, subprocess, sys, unicodedata, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
RAW, AUDIO, OUT = ROOT / 'data/raw', ROOT / 'data/processed/qwen_train/audio', ROOT / 'data/processed/qwen_v2'
spec = importlib.util.spec_from_file_location('competition_score', ROOT / 'runtime_repo/score.py')
score_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(score_mod)
def norm(t): return unicodedata.normalize('NFC', score_mod.normalize_text(str(t))).strip()

def read_mapping_lines(lines):
    """mapping.tsv with occasionally line-wrapped sentences (copied from prepare_qwen_lit_data)."""
    records, pending = [], None
    for raw in lines:
        if raw.startswith('audio_filename\t'): continue
        if '\t' in raw:
            fn, text = raw.split('\t', 1)
            if fn.lower().endswith(('.webm', '.wav', '.mp3')):
                if pending: records.append(pending)
                pending = [fn, text.strip()]
            elif pending: pending[1] += ' ' + raw.strip()
        elif pending and raw.strip(): pending[1] = (pending[1] + ' ' + raw.strip()).strip()
    if pending: records.append(pending)
    return [(f, t) for f, t in records if f and t]

def jobs_banyumasan():
    base = RAW / 'banyumasan_tts/FULL TTS JAWA BANYUMASAN'
    for bi, d in enumerate(sorted(p for p in base.iterdir() if (p / 'mapping.tsv').exists())):
        for ri, (fn, text) in enumerate(read_mapping_lines((d / 'mapping.tsv').read_text('utf-8-sig').splitlines())):
            yield d / fn, f'banyumasan_{bi:03d}_{ri:04d}', text, f'banyumasan_{bi:03d}'

def jobs_bojonegoro():
    for bi, d in enumerate(sorted((RAW / 'bojonegoro_javanese/batches').glob('batch_*'))):
        for ri, (fn, text) in enumerate(read_mapping_lines((d / 'mapping.tsv').read_text('utf-8-sig').splitlines())):
            yield d / fn, f'bojonegoro_{bi:03d}_{ri:04d}', text, f'bojonegoro_{bi:03d}'

def jobs_ngapak():
    base = RAW / 'ngapak_tts/TTS-Javanese Ngapak Dialect'
    index = {}
    for dp, _, fns in os.walk(base / 'unzipped'):
        for f in fns: index.setdefault(f, Path(dp) / f)
    for bi, z in enumerate(sorted(base.glob('*.zip'))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith('mapping.tsv'):
                    lines = zf.read(n).decode('utf-8-sig', 'replace').splitlines()
                    for ri, (fn, text) in enumerate(read_mapping_lines(lines)):
                        src = index.get(Path(fn).name)
                        if src: yield src, f'ngapak_{bi:03d}_{ri:04d}', text, f'ngapak_{bi:03d}'

def convert(job):
    src, uid, text, group, source = job
    dst = AUDIO / source / f'{uid}.wav'
    try:
        if not dst.exists():
            subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src),
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(dst)], check=True)
        info = sf.info(dst); dur = info.frames / info.samplerate
    except Exception as e:
        return ('ERR', f'{src}: {e}')
    t = norm(text)
    if not t or dur < 0.5 or dur > 40.0: return ('DROP', uid)
    return ('OK', {'audio_filepath': str(dst), 'duration': round(dur, 4), 'text_ref1': t, 'text_ref2': t,
                   'source': source, 'uid': uid, 'group': group, 'language': 'jv'})

rows, errs = [], []
for source, gen in (('banyumasan', jobs_banyumasan), ('ngapak', jobs_ngapak), ('bojonegoro', jobs_bojonegoro)):
    (AUDIO / source).mkdir(parents=True, exist_ok=True)
    jobs = [(src, uid, text, group, source) for src, uid, text, group in gen() if src.exists()]
    n0 = len(rows)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (st, v) in enumerate(ex.map(convert, jobs), 1):
            if st == 'OK': rows.append(v)
            elif st == 'ERR': errs.append(v)
            if i % 1000 == 0: print(source, i, '/', len(jobs), flush=True)
    got = rows[n0:]
    print(f'{source}: {len(got)} rows, {sum(r["duration"] for r in got)/3600:.2f} h, {len(jobs)-len(got)} dropped/err', flush=True)

with open(OUT / 'dialect_rows.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
v2 = (OUT / 'train_pool_v2.jsonl').read_text(encoding='utf-8')
with open(OUT / 'train_pool_v3.jsonl', 'w', encoding='utf-8') as f:
    f.write(v2 if v2.endswith('\n') else v2 + '\n')
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
if errs: print('errors:', len(errs), errs[:3])
NG = set('nyong inyong kiye kepimen sekang disit rika bae nyung manjing kencot'.split())
for source in ('banyumasan', 'ngapak', 'bojonegoro'):
    ws = [w for r in rows if r['source'] == source for w in r['text_ref1'].casefold().split()]
    print(f'{source}: ngapak markers {100*sum(w in NG for w in ws)/max(1,len(ws)):.2f} per 100 words')
print('DIALECT_PREP_DONE', len(rows))
