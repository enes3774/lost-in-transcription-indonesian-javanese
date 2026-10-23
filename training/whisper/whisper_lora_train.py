#!/usr/bin/env python3
"""Whisper large-v3 LoRA adaptation for Lost In Transcription.
Competition dev is evaluation-only; robust validation selects checkpoints.
"""
import argparse,csv,gc,json,math,os,random,re,shutil,sys,time,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf,torch
from jiwer import wer,cer
from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from transformers import WhisperForConditionalGeneration,WhisperProcessor
ROOT=Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'));BASE=ROOT/'models/checkpoints/whisper-large-v3';DATA=ROOT/'data/processed/qwen_v2';POOL=DATA/'train_pool_v2.jsonl';ROBUST=DATA/'robust_val.jsonl';DEV=DATA/'competition_dev_eval_only.jsonl'
MIXES={
 'devmix':{'jember':.32,'homostoria':.25,'hari':.10,'central':.12,'slr35':.09,'fleurs_jv':.06,'cv_id':.03,'fleurs_id':.03},  # B's mixture (MERaLiON D / d_final)
 'qwen':{'jember':.30,'homostoria':.25,'slr35':.15,'fleurs_jv':.08,'fleurs_id':.04,'cv_id':.08,'central':.05,'hari':.05},
 'annotation':{'jember':.20,'homostoria':.30,'hari':.15,'central':.15,'slr35':.08,'fleurs_jv':.04,'cv_id':.05,'fleurs_id':.03},
}
def log(x):print(time.strftime('[%H:%M:%S]'),x,flush=True)
def readj(p):return [json.loads(x) for x in open(p,encoding='utf8')]
def audio(p):
 x,sr=sf.read(p,dtype='float32',always_2d=False)
 if x.ndim==2:x=x.mean(1)
 if sr!=16000:
  import librosa;x=librosa.resample(x,orig_sr=sr,target_sr=16000)
 return x
class DS(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):r=self.rows[i];return audio(r['audio_filepath']),r.get('text_ref1',''),r
class Collate:
 def __init__(self,p,labels):self.p=p;self.labels=labels
 def __call__(self,b):
  x=self.p.feature_extractor([z[0] for z in b],sampling_rate=16000,padding='max_length',max_length=480000,truncation=True,return_attention_mask=True,return_tensors='pt')
  o={'input_features':x.input_features,'attention_mask':x.attention_mask,'rows':[z[2] for z in b]}
  if self.labels:
   y=self.p.tokenizer([z[1] for z in b],padding=True,truncation=True,max_length=448,return_tensors='pt');lab=y.input_ids.masked_fill(y.attention_mask.ne(1),-100)
   if (lab[:,0]==self.p.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')).all():lab=lab[:,1:]
   o['labels']=lab
  return o
def scores(ref,hyp):
 sys.path.insert(0,str(ROOT/'runtime_repo'));from score import normalize_text
 r=[normalize_text(x) for x in ref];h=[normalize_text(x) for x in hyp];return {'wer':wer(r,h),'cer':cer(r,h)}
def sched(opt,step,total,warm=250):
 if step<warm:q=max(.001,step/warm)
 else:q=.08+.92*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
 for g in opt.param_groups:g['lr']=g['base_lr']*q
 return q
def validate(model,p,rows,name,step,exp,batch=8):
 rows=[r for r in rows if float(r['duration'])<=30.0];loader=DataLoader(DS(rows),batch_size=batch,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=Collate(p,False));ref=[];hyp=[];groups=[];model.eval();t=time.time()
 with torch.inference_mode():
  for b in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16):ids=model.generate(input_features=b['input_features'].cuda(non_blocking=True),attention_mask=b['attention_mask'].cuda(non_blocking=True),language='jw',task='transcribe',num_beams=1,do_sample=False,max_new_tokens=192,use_cache=True)
   hyp.extend(p.tokenizer.batch_decode(ids,skip_special_tokens=True));ref.extend([r['text_ref1'] for r in b['rows']]);groups.extend([r.get('eval_group',r.get('source','all')) for r in b['rows']])
 z={'step':step,'split':name,'overall':scores(ref,hyp),'groups':{}}
 for g in sorted(set(groups)):
  ii=[i for i,x in enumerate(groups) if x==g];z['groups'][g]=scores([ref[i] for i in ii],[hyp[i] for i in ii])
 log(f'[val:{name}] wer={z["overall"]["wer"]:.4f} cer={z["overall"]["cer"]:.4f} seconds={time.time()-t:.1f}')
 for g,v in z['groups'].items():log(f'[val:{name}:{g}] wer={v["wer"]:.4f}')
 with (exp/f'{name}_step{step}.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['audio_filename','transcript']);w.writeheader();w.writerows({'audio_filename':r.get('uid',Path(r['audio_filepath']).name),'transcript':h} for r,h in zip(rows,hyp))
 del loader;gc.collect();torch.cuda.empty_cache();model.train();model.config.use_cache=False;return z,hyp,rows
def full_dev_hybrid(short_hyp,short_rows,all_dev):
 q=pd.read_csv(ROOT/'predictions/dev_qwen_v2_lora_one_beam4_lp10.csv',keep_default_na=False);qd=dict(zip(q.audio_filename,q.transcript));wd={r['uid']:h for r,h in zip(short_rows,short_hyp)};ref=[r['text_ref1'] for r in all_dev];hyp=[wd[r['uid']] if r['duration']<=30 else qd[r['uid']] for r in all_dev];return scores(ref,hyp)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mix',choices=MIXES,required=True);ap.add_argument('--max-steps',type=int,default=4000);ap.add_argument('--val-every',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--accum',type=int,default=2);ap.add_argument('--seed',type=int,default=2040);ap.add_argument('--benchmark',action='store_true');ap.add_argument('--exp-name',default=None);ap.add_argument('--init-adapter',default=None);ap.add_argument('--lr-scale',type=float,default=1.0);a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 exp=ROOT/'training/whisper_v3_lora'/f'exp_{a.exp_name or a.mix}';exp.mkdir(parents=True,exist_ok=True);train=[r for r in readj(POOL) if .5<=float(r['duration'])<=30];robust=readj(ROBUST);dev=readj(DEV);counts=Counter(r['source'] for r in train);weights=torch.tensor([MIXES[a.mix][r['source']]/counts[r['source']] for r in train],dtype=torch.double);log(f'mix={a.mix} train={len(train)} filtered_gt30={216263-len(train)} weights={MIXES[a.mix]}')
 p=WhisperProcessor.from_pretrained(str(BASE),local_files_only=True);p.tokenizer.set_prefix_tokens(language='jw',task='transcribe',predict_timestamps=False)
 base=WhisperForConditionalGeneration.from_pretrained(str(BASE),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa');base.config.use_cache=False;base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 if a.init_adapter:model=PeftModel.from_pretrained(base,a.init_adapter,is_trainable=True).cuda();log(f'initialized adapter from {a.init_adapter}')
 else:
  cfg=LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,bias='none',target_modules=['q_proj','v_proj']);model=get_peft_model(base,cfg).cuda()
 model.print_trainable_parameters()
 enc=[];dec=[]
 for n,x in model.named_parameters():
  if x.requires_grad:(enc if '.encoder.' in n else dec).append(x)
 opt=torch.optim.AdamW([{'params':enc,'lr':1e-5*a.lr_scale,'base_lr':1e-5*a.lr_scale,'weight_decay':.01},{'params':dec,'lr':2e-5*a.lr_scale,'base_lr':2e-5*a.lr_scale,'weight_decay':.01}],betas=(.9,.98),eps=1e-8)
 start_step=0;best=99.;last_path=exp/'last.pt'
 if last_path.exists() and not a.benchmark:
  ck=torch.load(last_path,map_location='cpu',weights_only=False);set_peft_model_state_dict(model,ck['adapter']);opt.load_state_dict(ck['optimizer']);start_step=int(ck['step']);best=float(ck['best']);del ck;gc.collect();log(f'resumed step={start_step} best={best:.4f}')
 sampler=WeightedRandomSampler(weights,num_samples=(a.max_steps-start_step)*a.accum*a.batch_size,replacement=True,generator=torch.Generator().manual_seed(a.seed+start_step));loader=DataLoader(DS(train),batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=Collate(p,True));it=iter(loader);model.train();opt.zero_grad(set_to_none=True);start=time.time();run=0.
 for step in range(start_step+1,a.max_steps+1):
  ratio=sched(opt,step,a.max_steps);last=0.
  for micro in range(a.accum):
   b=next(it);features=b['input_features'].cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_features=features,attention_mask=b['attention_mask'].cuda(non_blocking=True),labels=b['labels'].cuda(non_blocking=True)).loss/a.accum
   loss.backward();last+=loss.item()*a.accum
   if step==start_step+1 and micro==a.accum-1:
    eg=sum(float(x.grad.float().norm()) for x in enc if x.grad is not None);dg=sum(float(x.grad.float().norm()) for x in dec if x.grad is not None);log(f'gradient_check encoder={eg:.4f} decoder={dg:.4f}')
  torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.);opt.step();opt.zero_grad(set_to_none=True);run+=last
  if step%25==0:log(f'step {step:4d}/{a.max_steps} loss={run/25:.4f} lr_enc={opt.param_groups[0]["lr"]:.2e} lr_dec={opt.param_groups[1]["lr"]:.2e} sec_step={(time.time()-start)/max(1,step-start_step):.3f} vram={torch.cuda.max_memory_allocated()/1e9:.1f}G');run=0.
  if not a.benchmark and (step%a.val_every==0 or step==a.max_steps):
   rv,_,_=validate(model,p,robust,'robust',step,exp);dv,dh,dr=validate(model,p,dev,'dev_short',step,exp);hybrid=full_dev_hybrid(dh,dr,dev);metrics={'robust':rv,'dev_short':dv,'dev_hybrid':hybrid};score=rv['overall']['wer'];new=score<best
   if new:
    best=score;tmp=exp/'best_adapter_tmp';shutil.rmtree(tmp,ignore_errors=True);model.save_pretrained(tmp);shutil.rmtree(exp/'best_adapter',ignore_errors=True);tmp.replace(exp/'best_adapter');(exp/'best.json').write_text(json.dumps({'step':step,'robust_wer':score,'dev_short_wer':dv['overall']['wer'],'dev_hybrid_wer':hybrid['wer']},indent=2))
   state={'adapter':get_peft_model_state_dict(model),'optimizer':opt.state_dict(),'step':step,'best':best};torch.save(state,exp/'last.pt');open(exp/'metrics.jsonl','a').write(json.dumps(metrics)+'\n');log(f'[ckpt] step={step} robust={score:.4f} dev_short={dv["overall"]["wer"]:.4f} dev_hybrid={hybrid["wer"]:.4f} best={best:.4f} new_best={new}')
 if (exp/'last.pt').exists():(exp/'last.pt').unlink();log('removed optimizer checkpoint after clean completion')
 log('training complete')
if __name__=='__main__':main()
