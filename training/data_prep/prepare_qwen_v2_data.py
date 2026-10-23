#!/usr/bin/env python3
"""Build canonicalized Qwen v2 manifests, extracting SLR35 and FLEURS audio."""
import os
from pathlib import Path
import io,json,random,re,sys,unicodedata,hashlib
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from datasets import load_from_disk,Audio

ROOT=Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
OUT=ROOT/'data/processed/qwen_v2'
AUDIO=OUT/'audio';OUT.mkdir(parents=True,exist_ok=True);AUDIO.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'runtime_repo'))
from score import normalize_text

ONES_ID=['nol','satu','dua','tiga','empat','lima','enam','tujuh','delapan','sembilan','sepuluh','sebelas']
ONES_JV=['nol','siji','loro','telu','papat','lima','enem','pitu','wolu','sanga','sepuluh','sewelas']
def number_id(n):
    if n<12:return ONES_ID[n]
    if n<20:return number_id(n-10)+' belas'
    if n<100:return number_id(n//10)+' puluh'+((' '+number_id(n%10)) if n%10 else '')
    if n<200:return 'seratus'+((' '+number_id(n-100)) if n>100 else '')
    if n<1000:return number_id(n//100)+' ratus'+((' '+number_id(n%100)) if n%100 else '')
    if n<2000:return 'seribu'+((' '+number_id(n-1000)) if n>1000 else '')
    if n<1_000_000:return number_id(n//1000)+' ribu'+((' '+number_id(n%1000)) if n%1000 else '')
    if n<1_000_000_000:return number_id(n//1_000_000)+' juta'+((' '+number_id(n%1_000_000)) if n%1_000_000 else '')
    return str(n)
def number_jv(n):
    if n<12:return ONES_JV[n]
    if n<20:return number_jv(n-10)+' welas'
    if n<100:return number_jv(n//10)+' puluh'+((' '+number_jv(n%10)) if n%10 else '')
    if n<200:return 'satus'+((' '+number_jv(n-100)) if n>100 else '')
    if n<1000:return number_jv(n//100)+' atus'+((' '+number_jv(n%100)) if n%100 else '')
    if n<2000:return 'sewu'+((' '+number_jv(n-1000)) if n>1000 else '')
    if n<1_000_000:return number_jv(n//1000)+' ewu'+((' '+number_jv(n%1000)) if n%1000 else '')
    if n<1_000_000_000:return number_jv(n//1_000_000)+' yuta'+((' '+number_jv(n%1_000_000)) if n%1_000_000 else '')
    return str(n)

NON_ACUTE=str.maketrans({'è':'e','ê':'e','ě':'e','È':'E','Ê':'E','Ě':'E'})
DIGITS=re.compile(r'(?<![\w])\d+(?![\w])')
def canonical(text,language='id'):
    text=unicodedata.normalize('NFC',normalize_text(str(text))).translate(NON_ACUTE)
    conv=number_jv if language=='jv' else number_id
    text=DIGITS.sub(lambda m:conv(int(m.group())) if len(m.group())<=9 else m.group(),text)
    return re.sub(r'\s+',' ',text).strip()
def stable_bucket(s,n=10):return int(hashlib.sha1(str(s).encode()).hexdigest()[:8],16)%n
def record(path,duration,text,source,uid,group,language):
    text=canonical(text,language)
    if not text or not (0.5<=duration<=40):return None
    return {'audio_filepath':str(path),'duration':round(float(duration),4),'text_ref1':text,'text_ref2':'','source':source,'uid':uid,'group':group,'language':language}

# Recanonicalize the previous clean group-disjoint pool.
old=ROOT/'data/processed/qwen_train/train_pool.jsonl'
train=[]
for line in open(old,encoding='utf8'):
    x=json.loads(line);lang='id' if x['source']=='cv_id' else 'id'
    x['text_ref1']=canonical(x['text_ref1'],lang);x['language']=lang
    train.append(x)
print('existing',len(train),flush=True)

# SLR35: exact CC-BY-SA source mirrored as 38 parquet shards. Speaker-disjoint 10% holdout.
slr_root=ROOT/'data/raw/openslr35_javanese/hf_dataset/data'
slr_train=[];slr_hold=[]
for si,p in enumerate(sorted(slr_root.glob('*.parquet'))):
    table=pq.read_table(p,columns=['id','speaker_id','sentence','audio'])
    ids=table['id'].to_pylist();speakers=table['speaker_id'].to_pylist();texts=table['sentence'].to_pylist();audios=table['audio'].to_pylist()
    shard_dir=AUDIO/'slr35'/f'{si:02d}';shard_dir.mkdir(parents=True,exist_ok=True)
    for uid,spk,text,audio in zip(ids,speakers,texts,audios):
        blob=audio['bytes'];suffix=Path(audio.get('path') or '').suffix or '.flac';dst=shard_dir/f'{uid}{suffix}'
        if not dst.exists():dst.write_bytes(blob)
        try:
            info=sf.info(io.BytesIO(blob));dur=info.frames/info.samplerate
        except Exception:continue
        z=record(dst,dur,text,'slr35',f'slr35_{uid}',f'slr35_spk_{spk}','jv')
        if z:(slr_hold if stable_bucket(spk)==0 else slr_train).append(z)
    print('slr shard',si+1,'train',len(slr_train),'hold',len(slr_hold),flush=True)
train.extend(slr_train)

# FLEURS: train split for training; official validation/test remain external evaluation only.
fleurs_root=ROOT/'data/raw/fleurs_id_jv'
fleurs_vals={}
for cfg,lang,source in [('id_id','id','fleurs_id'),('jv_id','jv','fleurs_jv')]:
    ds=load_from_disk(str(fleurs_root/cfg))
    for split,dset in ds.items():
        dset=dset.cast_column('audio',Audio(decode=False));rows=[];dest=AUDIO/source/split;dest.mkdir(parents=True,exist_ok=True)
        for x in dset:
            a=x['audio'];blob=a.get('bytes');suffix=Path(a.get('path') or '').suffix or '.wav';dst=dest/f"{x['id']}{suffix}"
            if not dst.exists():dst.write_bytes(blob)
            dur=float(x['num_samples'])/16000
            z=record(dst,dur,x['transcription'],source,f"{source}_{split}_{x['id']}",f"{source}_{split}_{x['id']}",lang)
            if z:rows.append(z)
        if split=='train':train.extend(rows)
        else:fleurs_vals[f'{source}_{split}']=rows
        print(source,split,len(rows),flush=True)

# Existing internal validation, recanonicalized. Competition dev remains separate and untouched.
internal=[]
for line in open(ROOT/'data/processed/qwen_train/internal_val.jsonl',encoding='utf8'):
    x=json.loads(line);x['text_ref1']=canonical(x['text_ref1'],'id');x['language']='id';internal.append(x)
rng=random.Random(1337);rng.shuffle(slr_hold);slr_val=slr_hold[:1000]

def write_jsonl(name,rows):
    with open(OUT/name,'w',encoding='utf8') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(name,len(rows),sum(x['duration'] for x in rows)/3600,flush=True)
write_jsonl('train_pool_v2.jsonl',train)
write_jsonl('internal_val.jsonl',internal)
write_jsonl('slr35_speaker_val.jsonl',slr_val)
for name,rows in fleurs_vals.items():write_jsonl(name+'.jsonl',rows)
# Keep a canonicalized copy solely for local final evaluation, never training/checkpoint selection.
dev=[]
for line in open(ROOT/'data/processed/qwen_train/dev.jsonl',encoding='utf8'):
    x=json.loads(line);x['text_ref1']=canonical(x['text_ref1'],'id');dev.append(x)
write_jsonl('competition_dev_eval_only.jsonl',dev)

stats=pd.DataFrame(train).groupby('source').agg(rows=('uid','size'),hours=('duration',lambda x:x.sum()/3600),avg_seconds=('duration','mean'))
stats.to_csv(OUT/'stats.csv');print(stats.to_string(),flush=True)
(OUT/'CANONICALIZATION.md').write_text('''# Qwen v2 label canonicalization\n\n1. Official competition punctuation/annotation normalization.\n2. Unicode NFC.\n3. Preserve acute é.\n4. Map è, ê and ě to plain e.\n5. Expand numeric tokens to words: Indonesian for Indonesian/mixed sources, Javanese for pure-Javanese sources.\n6. Preserve original source datasets and provenance unchanged under data/raw/.\n''',encoding='utf8')
