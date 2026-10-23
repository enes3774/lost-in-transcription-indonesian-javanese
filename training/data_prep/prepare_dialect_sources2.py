#!/usr/bin/env python3
"""Ingest zip-of-zips MDC dialect corpora (each inner zip carries its own mapping.tsv + webm).

Sources: purworejo, jombang, gresik, multidialect, lumajang.  Same recipe as
prepare_dialect_sources.py; writes dialect_rows2.jsonl and train_pool_v4.jsonl (= v3 + rows2).
"""
import os
import importlib.util, json, subprocess, unicodedata, zipfile, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
RAW, AUDIO, OUT = ROOT / 'data/raw', ROOT / 'data/processed/qwen_train/audio', ROOT / 'data/processed/qwen_v2'
spec = importlib.util.spec_from_file_location('competition_score', ROOT / 'runtime_repo/score.py')
score_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(score_mod)
def norm(t): return unicodedata.normalize('NFC', score_mod.normalize_text(str(t))).strip()
sys.path.insert(0, str(ROOT / 'scripts'))

def read_mapping_lines(lines):
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

SOURCES = {'purworejo': 'purworejo', 'jombang': 'jombang_tts', 'gresik': 'gresik_tts',
           'multidialect': 'multidialect_tts', 'lumajang': 'lumajang_tts'}

def jobs(source, rawdir):
    zips = sorted(p for p in (RAW / rawdir).rglob('*.zip'))
    for bi, z in enumerate(zips):
        ex = z.parent / 'unzipped' / f'{bi:03d}'
        try:
            with zipfile.ZipFile(z) as zf:
                maps = [n for n in zf.namelist() if n.endswith('mapping.tsv')]
                if not maps: continue
                if not ex.exists():
                    ex.mkdir(parents=True); zf.extractall(ex)
                index = {}
                for p in ex.rglob('*'):
                    if p.is_file(): index.setdefault(p.name, p)
                for n in maps:
                    for ri, (fn, text) in enumerate(read_mapping_lines(zf.read(n).decode('utf-8-sig', 'replace').splitlines())):
                        src = index.get(Path(fn).name)
                        if src: yield src, f'{source}_{bi:03d}_{ri:04d}', text, f'{source}_{bi:03d}', source
        except zipfile.BadZipFile:
            print('BAD ZIP', z, flush=True)

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
for source, rawdir in SOURCES.items():
    (AUDIO / source).mkdir(parents=True, exist_ok=True)
    js = list(jobs(source, rawdir)); n0 = len(rows)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (st, v) in enumerate(ex.map(convert, js), 1):
            if st == 'OK': rows.append(v)
            elif st == 'ERR': errs.append(v)
            if i % 1000 == 0: print(source, i, '/', len(js), flush=True)
    got = rows[n0:]
    print(f'{source}: {len(got)} rows, {sum(r["duration"] for r in got)/3600:.2f} h, {len(js)-len(got)} dropped/err', flush=True)
with open(OUT / 'dialect_rows2.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
v3 = (OUT / 'train_pool_v3.jsonl').read_text(encoding='utf-8')
with open(OUT / 'train_pool_v4.jsonl', 'w', encoding='utf-8') as f:
    f.write(v3 if v3.endswith('\n') else v3 + '\n')
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
if errs: print('errors:', len(errs), errs[:3])
print('DIALECT_PREP2_DONE', len(rows))
