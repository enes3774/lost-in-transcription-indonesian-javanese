#!/usr/bin/env python3
"""Discriminative full fine-tuning of Whisper large-v3; all parameters train."""
import argparse,gc,json,math,os,random,re,shutil,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from transformers import WhisperForConditionalGeneration,WhisperProcessor
sys.path.insert(0,os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription') + '/training/whisper_v3_lora')
from whisper_lora_train import ROOT,BASE,DATA,POOL,ROBUST,DEV,MIXES,DS,Collate,readj,validate,full_dev_hybrid,log
MIXES["v5mix"]={"jember": .19, "homostoria": .13, "hari": .05, "slr35": .19, "central": .08, "fleurs_jv": .07, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .08, "fleurs_id": .08}  # the v5 mixture
MIXES["v6mix"]={"jember": .16, "homostoria": .11, "hari": .05, "slr35": .11, "central": .06, "fleurs_jv": .05, "ngapak": .05, "purworejo": .03, "lumajang": .03, "bojonegoro": .02, "cv_id": .05, "fleurs_id": .05, "banyumasan": .10, "gresik": .05, "jombang": .05, "multidialect": .03}  # v5mix + Banyumasan, Gresik, Jombang and Multidialect (pool train_pool_v4.jsonl)
MIXES["jvmix"]={"slr35":.25,"central":.18,"fleurs_jv":.15,"jember":.20,"homostoria":.12,"hari":.04,"cv_id":.03,"fleurs_id":.03}  # Javanese-weighted mixture (12.21), same weights as meralion_jv_mix

class DSAug(DS):
 """Speed 0.9-1.1 (p=.5) and Gaussian noise at 10-30 dB SNR (p=.3), the MERaLiON d_aug recipe; labels untouched."""
 def __getitem__(self,i):
  w,t,r=super().__getitem__(i);limit=30*16000
  if random.random()<.5:
   f=random.uniform(.9,1.1)
   if len(w)/f>limit:f=len(w)/limit*1.01
   n=int(len(w)/f);w=np.interp(np.linspace(0,len(w)-1,n),np.arange(len(w)),w).astype(np.float32)
  if random.random()<.3:
   snr=random.uniform(10,30);power=float(np.mean(w**2))+1e-8;w=w+np.random.randn(len(w)).astype(np.float32)*np.sqrt(power/(10**(snr/10)))
  return np.clip(w,-1.,1.),t,r

def make_opt(model):
 groups=defaultdict(list);lrs={}
 for n,p in model.named_parameters():
  if n.startswith('proj_out.') or n.startswith('model.decoder.embed_tokens.'):
   key='proj_out';lr=8e-6
  elif n.startswith('model.encoder.conv'):
   key='encoder_conv';lr=3e-7
  else:
   m=re.search(r'model\.encoder\.layers\.(\d+)\.',n)
   d=re.search(r'model\.decoder\.layers\.(\d+)\.',n)
   if m:
    i=int(m.group(1));key=f'enc_{i:02d}';lr=2e-6*(.97**(31-i))
   elif d:
    i=int(d.group(1));key=f'dec_{i:02d}';lr=4e-6*(.97**(31-i))
   elif n.startswith('model.encoder.'):
    key='encoder_other';lr=7e-7
   else:
    key='decoder_other';lr=1.5e-6
  groups[(key,lr)].append(p);lrs[key]=lr
 opt=torch.optim.AdamW([{'params':v,'lr':lr,'base_lr':lr,'weight_decay':.01} for (k,lr),v in groups.items()],betas=(.9,.98),eps=1e-8,foreach=False)
 log('LR range '+str(min(lrs.values()))+'..'+str(max(lrs.values()))+' groups='+str(len(groups)))
 return opt
def schedule(opt,step,total,warm=250):
 if step<warm:q=max(.001,step/warm)
 else:q=.08+.92*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
 for g in opt.param_groups:g['lr']=g['base_lr']*q
 return q
def save_best(model,exp):
 tmp=exp/'best_model_tmp';shutil.rmtree(tmp,ignore_errors=True);state={k:v.detach().to(device='cpu',dtype=torch.bfloat16).contiguous() for k,v in model.state_dict().items()};model.save_pretrained(tmp,state_dict=state,safe_serialization=True,max_shard_size='4GB');del state;shutil.rmtree(exp/'best_model',ignore_errors=True);tmp.replace(exp/'best_model');gc.collect()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mix',choices=MIXES,default='qwen');ap.add_argument('--max-steps',type=int,default=3000);ap.add_argument('--val-every',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--accum',type=int,default=4);ap.add_argument('--seed',type=int,default=2050);ap.add_argument('--benchmark',action='store_true');ap.add_argument('--augment',action='store_true');ap.add_argument('--pool',default=None);ap.add_argument('--init',default=None);ap.add_argument('--exp-name',default=None);a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 exp=ROOT/'training/whisper_v3_lora'/f'exp_full_{a.exp_name or a.mix}';exp.mkdir(parents=True,exist_ok=True);train=[r for r in readj(a.pool or POOL) if .5<=float(r['duration'])<=30];robust=readj(ROBUST);dev=readj(DEV);cnt=Counter(r['source'] for r in train);weights=torch.tensor([MIXES[a.mix].get(r['source'],0.0)/cnt[r['source']] for r in train],dtype=torch.double);log(f'FULL FT mix={a.mix} train={len(train)} filtered_gt30={216263-len(train)}')
 p=WhisperProcessor.from_pretrained(str(BASE),local_files_only=True);p.tokenizer.set_prefix_tokens(language='jw',task='transcribe',predict_timestamps=False);model=WhisperForConditionalGeneration.from_pretrained(str(a.init or BASE),local_files_only=True,attn_implementation='sdpa',dtype=torch.float32).cuda();log(f'initialized from {a.init or BASE}');model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});log(f'trainable={sum(x.numel() for x in model.parameters() if x.requires_grad)}')
 opt=make_opt(model);sampler=WeightedRandomSampler(weights,num_samples=a.max_steps*a.accum*a.batch_size,replacement=True,generator=torch.Generator().manual_seed(a.seed));loader=DataLoader((DSAug if a.augment else DS)(train),batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=Collate(p,True));it=iter(loader);opt.zero_grad(set_to_none=True);model.train();best=99.;start=time.time();run=0.
 for step in range(1,a.max_steps+1):
  ratio=schedule(opt,step,a.max_steps);sl=0.
  for micro in range(a.accum):
   b=next(it)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_features=b['input_features'].cuda(non_blocking=True),attention_mask=b['attention_mask'].cuda(non_blocking=True),labels=b['labels'].cuda(non_blocking=True)).loss/a.accum
   loss.backward();sl+=loss.item()*a.accum
  if step==1:
   eg=sum(float(x.grad.float().norm()) for n,x in model.named_parameters() if x.grad is not None and '.encoder.' in n);dg=sum(float(x.grad.float().norm()) for n,x in model.named_parameters() if x.grad is not None and '.decoder.' in n);log(f'gradient_check encoder={eg:.3f} decoder={dg:.3f}')
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True);run+=sl
  if step%25==0:log(f'step {step:4d}/{a.max_steps} loss={run/25:.4f} ratio={ratio:.3f} sec_step={(time.time()-start)/step:.3f} vram={torch.cuda.max_memory_allocated()/1e9:.1f}G');run=0.
  if os.environ.get('OFFLINE_VAL')=='1' and (step%a.val_every==0 or step==a.max_steps):
   save_best(model,exp);(exp/'best_model').replace(exp/f'step{step}');log(f'[saved] step={step} -> {exp}/step{step}');continue
  if not a.benchmark and (step%a.val_every==0 or step==a.max_steps):
   rv,_,_=validate(model,p,robust,'robust',step,exp,batch=4);dv,dh,dr=validate(model,p,dev,'dev_short',step,exp,batch=4);hy=full_dev_hybrid(dh,dr,dev);score=rv['overall']['wer'];new=score<best
   if new:best=score;save_best(model,exp);(exp/'best.json').write_text(json.dumps({'step':step,'robust_wer':score,'dev_short_wer':dv['overall']['wer'],'dev_hybrid_wer':hy['wer']},indent=2))
   # One bounded resumable checkpoint; overwritten at each validation.
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'best':best},exp/'last.pt');open(exp/'metrics.jsonl','a').write(json.dumps({'robust':rv,'dev_short':dv,'dev_hybrid':hy})+'\n');log(f'[ckpt] step={step} robust={score:.4f} dev_short={dv["overall"]["wer"]:.4f} hybrid={hy["wer"]:.4f} best={best:.4f} new={new}')
 if (exp/'last.pt').exists():(exp/'last.pt').unlink();log('removed optimizer checkpoint after clean completion')
 log('full training complete')
if __name__=='__main__':main()
