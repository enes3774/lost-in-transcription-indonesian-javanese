#!/usr/bin/env python3
import os
import json,math
from pathlib import Path
import pandas as pd
R=Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'));D=R/'data/processed/qwen_style_aggressive';D.mkdir(parents=True,exist_ok=True);pool=[json.loads(x) for x in open(R/'data/processed/qwen_v2/train_pool_v2.jsonl',encoding='utf8')];dev=[json.loads(x) for x in open(R/'data/processed/qwen_v2/competition_dev_eval_only.jsonl',encoding='utf8')];meta=pd.read_csv(R/'data/processed/dev/manifest.csv',keep_default_na=False);speaker=dict(zip(meta.audio_filename,meta.speaker))
def wr(p,rows):
 with open(p,'w',encoding='utf8') as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
rob=[json.loads(x) for x in open(R/'data/processed/qwen_v2/robust_val.jsonl',encoding='utf8')];wr(D/'robust.jsonl',rob)
allsp=[]
for x in dev:
 y=dict(x);y['eval_group']=f'speaker_{speaker[x["uid"]]}';allsp.append(y)
wr(D/'all_dev_by_speaker.jsonl',allsp)
folds=[]
for name,pred in [('conv_hold2',lambda x:x['group']=='convo_2'),('conv_hold5',lambda x:x['group']=='convo_5')]+[(f'speaker_hold{s}',lambda x,s=s:speaker[x['uid']]==s) for s in sorted(set(speaker.values()))]:
 held=[x for x in dev if pred(x)];style=[]
 for x in dev:
  if not pred(x):y=dict(x);y['source']='style_dev';style.append(y)
 steps=math.ceil(2*len(style)/(.8*4));epoch=steps*4;wr(D/f'{name}_train.jsonl',pool+style);wr(D/f'{name}_heldout.jsonl',held)
 cfg={'name':'qwen_style_aggressive_'+name,'seed':2080+len(folds),'paths':{'train_manifest':str(D/f'{name}_train.jsonl'),'train_epoch_manifest':str(D/f'unused_{name}.jsonl'),'val_unseen':str(D/f'{name}_heldout.jsonl'),'val_all':str(D/'robust.jsonl'),'exp_dir':str(R/f'training/qwen_style_aggressive/exp_{name}')},'model':{'path':str(R/'training/qwen_v2/models/lora_one_stage_step9000_hf5'),'attn_implementation':'sdpa','gradient_checkpointing':True,'num_decoder_layers':28},'freeze':{'train_audio_tower':False,'train_projector':False,'lora':{'enabled':True,'r':16,'alpha':32,'dropout':.05,'layer_min':14}},'data':{'text_key':'text_ref1','language':'None','batch_size':4,'min_duration':.5,'max_duration':40.,'num_workers':4,'val_batch_size':8,'val_max_seconds':40.,'max_new_tokens':256},'sampling':{'seed':2080+len(folds),'epoch_samples':epoch,'weights':{'style_dev':.8,'homostoria':.05,'hari':.04,'central':.04,'jember':.02,'slr35':.02,'fleurs_jv':.01,'cv_id':.01,'fleurs_id':.01}},'repetition_cap':{'enabled':False,'max_per_text':1000000,'seed':2080+len(folds),'group_key':'text_ref1'},'optim':{'lr_encoder':0.,'lr_projector':0.,'lr_lora':8e-6,'lr_decoder':0.,'betas':[.9,.98],'weight_decay':.01,'warmup_steps':min(20,max(5,steps//10)),'min_lr_ratio':.2},'trainer':{'max_steps':steps,'max_epochs':1,'gradient_clip_val':1.,'log_every_n_steps':10,'val_every_n_steps':math.ceil(steps/2),'val_all_every':2,'resume_if_exists':True,'keep_last_after_completion':False,'init_checkpoint':None}}
 (D/f'{name}.json').write_text(json.dumps(cfg,indent=2));folds.append((name,len(style),len(held),steps))
print(folds)
