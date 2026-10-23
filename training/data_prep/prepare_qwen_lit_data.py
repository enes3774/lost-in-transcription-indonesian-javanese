#!/usr/bin/env python3
import importlib.util
import json
import os
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/processed/qwen_train'
AUDIO = OUT / 'audio'
TMP = ROOT / 'tmp/qwen_data'
OUT.mkdir(parents=True, exist_ok=True)
AUDIO.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('competition_score', ROOT/'runtime_repo/score.py')
score_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(score_mod)

def norm(text):
    return unicodedata.normalize('NFC', score_mod.normalize_text(str(text))).strip()

def hms(value):
    h,m,s = str(value).split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def row(path, duration, text, source, uid, group=None):
    text = norm(text)
    if not text or duration < 0.5 or duration > 40.0:
        return None
    return {'audio_filepath':str(path),'duration':round(float(duration),4),
            'text_ref1':text,'text_ref2':text,'source':source,'uid':uid,
            'group':str(group if group is not None else uid)}

all_rows=[]
# Jember: already segmented and validated against timestamp bounds.
jm=pd.read_csv(ROOT/'data/processed/jember/manifest.csv',keep_default_na=False)
for x in jm.itertuples():
    z=row(Path(x.path),x.duration_seconds,x.transcript,'jember',x.audio_filename,f'jember_{int(x.session):03d}')
    if z: all_rows.append(z)
print('jember',sum(x['source']=='jember' for x in all_rows),flush=True)

# Decode each long session once, then slice timestamped utterances.
def segment_timestamped(tsv, audio_dir, source, name_col, start_col, end_col, text_col, extension='.mp3'):
    df=pd.read_csv(tsv,sep='\t',keep_default_na=False)
    stem_map={p.stem.lower():p for p in audio_dir.glob(f'*{extension}')}
    target=AUDIO/source;target.mkdir(parents=True,exist_ok=True)
    made=[]
    for session,g in df.groupby(name_col,sort=False):
        key=str(session).lower(); src=stem_map.get(key)
        if src is None:
            # Be tolerant of harmless filename case differences.
            matches=[p for k,p in stem_map.items() if k==key]
            if not matches: raise FileNotFoundError(f'{source}: no audio for {session!r}')
            src=matches[0]
        decoded=TMP/f'{source}_{len(made)}_{abs(hash(key))}.wav'
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(src),
                        '-ar','16000','-ac','1','-c:a','pcm_s16le',str(decoded)],check=True)
        wave,sr=sf.read(decoded,dtype='float32'); assert sr==16000
        for local_i,(idx,x) in enumerate(g.iterrows()):
            a=max(0,round(hms(x[start_col])*sr));b=min(len(wave),round(hms(x[end_col])*sr))
            if b<=a: continue
            fn=f'{source}_{len(made):06d}.wav';dst=target/fn
            sf.write(dst,wave[a:b],sr,subtype='PCM_16')
            z=row(dst,(b-a)/sr,x[text_col],source,fn,f'{source}_{key}')
            if z: made.append(z)
            else: dst.unlink(missing_ok=True)
        decoded.unlink(missing_ok=True)
        if len(made) and len(made)%1000<50: print(source,'clips',len(made),flush=True)
    return made

homo=segment_timestamped(RAW/'homostoria/homostoria/transcript_homostoria.tsv',
                         RAW/'homostoria/homostoria','homostoria','Audio file','Start','End','Text')
all_rows += homo; print('homostoria',len(homo),flush=True)
hari_tsv=next((RAW/'hari_minggoean').rglob('*.tsv'))
hari=segment_timestamped(hari_tsv,hari_tsv.parent,'hari','Audio file','Start','End','Text')
all_rows += hari; print('hari',len(hari),flush=True)

# Central Javanese: paired WEBM/text; convert once to submission-compatible 16 kHz WAV.
def read_mapping(path):
    """Read occasionally line-wrapped mapping.tsv files without shifting columns."""
    records=[]; pending=None
    for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
        if raw_line.startswith('audio_filename\t'):
            continue
        if '\t' in raw_line:
            filename,text=raw_line.split('\t',1)
            if filename.lower().endswith(('.webm','.wav','.mp3')):
                if pending is not None:
                    records.append(pending)
                pending=[filename,text.strip()]
            elif pending is not None:
                pending[1]+=' '+raw_line.strip()
        elif pending is not None and raw_line.strip():
            pending[1]=(pending[1]+' '+raw_line.strip()).strip()
    if pending is not None:
        records.append(pending)
    return [(f,t) for f,t in records if f and t]

central_jobs=[]
for bi,batch in enumerate(sorted((RAW/'central_javanese/batches').glob('batch_*'))):
    for ri,(filename,text) in enumerate(read_mapping(batch/'mapping.tsv')):
        src=batch/filename
        if not src.exists():
            print('SKIP central missing audio',src,flush=True); continue
        dst=AUDIO/'central'/f'central_{bi:03d}_{ri:04d}.wav'
        central_jobs.append((src,dst,text))
(AUDIO/'central').mkdir(parents=True,exist_ok=True)
def convert(job):
    src,dst,text=job
    if not dst.exists():
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(src),
                        '-ar','16000','-ac','1','-c:a','pcm_s16le',str(dst)],check=True)
    info=sf.info(dst);return row(dst,info.frames/info.samplerate,text,'central',dst.name,dst.stem.rsplit('_',1)[0])
with ThreadPoolExecutor(max_workers=8) as ex:
    for i,z in enumerate(ex.map(convert,central_jobs),1):
        if z: all_rows.append(z)
        if i%500==0: print('central converted',i,'/',len(central_jobs),flush=True)
print('central',sum(x['source']=='central' for x in all_rows),flush=True)

# Common Voice: keep all validated rows in the pool. MP3 is decoded/resampled lazily by the collator.
cvroot=next((RAW/'common_voice_indonesian_26').glob('cv-corpus-*'))/'id'
cv=pd.read_csv(cvroot/'validated.tsv',sep='\t',keep_default_na=False)
durs=pd.read_csv(cvroot/'clip_durations.tsv',sep='\t').set_index('clip')['duration[ms]'].to_dict()
for x in cv.itertuples():
    dur=durs.get(x.path,0)/1000
    z=row(cvroot/'clips'/x.path,dur,x.sentence,'cv_id',x.path,x.client_id)
    if z: all_rows.append(z)
print('cv_id',sum(x['source']=='cv_id' for x in all_rows),flush=True)

# Verify paths and make group-disjoint internal validation. Competition dev remains test-only.
assert all(Path(x['audio_filepath']).exists() for x in all_rows)
import random
rng=random.Random(1337)
train_rows=[]; held_rows=[]
for source in sorted({x['source'] for x in all_rows}):
    src=[x for x in all_rows if x['source']==source]
    groups=sorted({x['group'] for x in src})
    n=max(1,round(0.10*len(groups)))
    held_groups=set(rng.sample(groups,n))
    train_rows += [x for x in src if x['group'] not in held_groups]
    held_rows += [x for x in src if x['group'] in held_groups]
# Keep autoregressive validation bounded while representing every source.
val_caps={'jember':100,'homostoria':100,'central':100,'cv_id':50,'hari':50}
internal_val=[]
for source,cap in val_caps.items():
    candidates=[x for x in held_rows if x['source']==source]
    internal_val += rng.sample(candidates,min(cap,len(candidates)))
for filename, rows_out in [('train_pool.jsonl',train_rows),('internal_val.jsonl',internal_val)]:
    with open(OUT/filename,'w',encoding='utf-8') as f:
        for x in rows_out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
# Development remains completely absent from the training and checkpoint-selection pools.
dev=pd.read_csv(ROOT/'data/processed/dev/manifest.csv',keep_default_na=False)
with open(OUT/'dev.jsonl','w',encoding='utf-8') as f:
    for x in dev.itertuples():
        info=sf.info(x.path)
        z=row(Path(x.path),info.frames/info.samplerate,x.transcript,'dev',x.audio_filename,f'convo_{x.convo_id}')
        if z:f.write(json.dumps(z,ensure_ascii=False)+'\n')
stats=pd.DataFrame(train_rows).groupby('source').agg(rows=('uid','size'),hours=('duration',lambda x:x.sum()/3600),avg_seconds=('duration','mean'))
stats.to_csv(OUT/'stats.csv')
print('TRAIN POOL');print(stats.to_string(),flush=True)
print('INTERNAL VAL',pd.Series([x['source'] for x in internal_val]).value_counts().to_dict(),flush=True)
print('TOTAL train',len(train_rows),'held groups rows',len(held_rows),'pool',OUT/'train_pool.jsonl',flush=True)
