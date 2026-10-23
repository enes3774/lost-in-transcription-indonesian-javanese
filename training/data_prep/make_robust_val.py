"""robust_val.jsonl (700 held-out pool rows used for checkpoint selection) and the dev language tags.

The Python part of the original launcher of the Qwen3-ASR stage-1 run (training/qwen_v2/launch_experiments.sh,
2026-09-02), unchanged: it tags each dev row with the official language field and draws a fixed sample of
held-out pool rows (internal 200, OpenSLR-35 speaker-held-out 200, FLEURS-jv 200, FLEURS-id 100, seed 404)."""
import os
from pathlib import Path
import json,random,re
p=Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription') + '/data/processed/qwen_v2')
def load(name): return [json.loads(x) for x in open(p/name,encoding='utf8')]
r=random.Random(404)
# Attach official dev metadata categories for ind/javind metric reporting.
dev=load('competition_dev_eval_only.jsonl')
import pandas as pd
meta=pd.read_csv(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription') + '/data/processed/dev/manifest.csv',keep_default_na=False)
labels=dict(zip(meta.audio_filename,meta.language))
for x in dev:x['eval_group']=labels.get(x['uid'],'unknown')
with open(p/'competition_dev_eval_only.jsonl','w',encoding='utf8') as f:
 for x in dev:f.write(json.dumps(x,ensure_ascii=False)+'\n')
parts=[]
for name,n in [('internal_val.jsonl',200),('slr35_speaker_val.jsonl',200),('fleurs_jv_validation.jsonl',200),('fleurs_id_validation.jsonl',100)]:
 x=load(name);r.shuffle(x);parts += x[:n]
r.shuffle(parts)
with open(p/'robust_val.jsonl','w',encoding='utf8') as f:
 for x in parts:f.write(json.dumps(x,ensure_ascii=False)+'\n')
train=load('train_pool_v2.jsonl')
from collections import Counter
c=Counter(x['source'] for x in train)
digit=sum(bool(re.search(r'\d',x['text_ref1'])) for x in train)
missing=sum(not Path(x['audio_filepath']).exists() for x in random.Random(7).sample(train,min(2000,len(train))))
print('TRAIN',len(train),dict(c),'digit_rows',digit,'sample_missing_audio',missing)
print('ROBUST_VAL',len(parts),Counter(x['source'] for x in parts))
assert not missing and len(parts)==700
