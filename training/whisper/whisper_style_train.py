#!/usr/bin/env python3
"""Dev-style stage for the Whisper large-v3 full FT (stage 1 = exp_full_qwen/best_model).

Same protocol as the Qwen style stage: 80% competition dev / 20% replay from the
ordinary pool, ~2 passes over dev, all parameters train at a scaled-down copy of
the stage-1 layer-wise LRs. The held-out baseline is decoded at step 0 through
the same validate() call as the adapted checkpoints, so both arms share batching
and the comparison is not contaminated by padding nondeterminism.
"""
import os
import argparse,json,math,random,sys,time
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from transformers import WhisperForConditionalGeneration,WhisperProcessor
sys.path.insert(0,os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription') + '/training/whisper_v3_lora')
from whisper_lora_train import ROOT,BASE,ROBUST,MIXES,DS,Collate,readj,validate,log
from whisper_full_train import make_opt,schedule,save_best
STAGE1=ROOT/'training/whisper_v3_lora/exp_full_qwen/best_model'

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--train-manifest',required=True)
 ap.add_argument('--heldout',default=None,help='held-out dev manifest; omit for the final full-dev run')
 ap.add_argument('--exp-name',required=True)
 ap.add_argument('--init',default=str(STAGE1))
 ap.add_argument('--passes',type=float,default=2.0)
 ap.add_argument('--style-weight',type=float,default=.8)
 ap.add_argument('--lr-scale',type=float,default=.5)
 ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--accum',type=int,default=4)
 ap.add_argument('--vals',type=int,default=2,help='validate this many times, evenly spaced (last = final step)')
 ap.add_argument('--seed',type=int,default=2050)
 a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 exp=ROOT/'training/whisper_v3_lora'/f'exp_style_{a.exp_name}';exp.mkdir(parents=True,exist_ok=True)
 train=[r for r in readj(a.train_manifest) if .5<=float(r['duration'])<=30]
 cnt=Counter(r['source'] for r in train);n_dev=cnt['style_dev'];assert n_dev>0,'no style_dev rows'
 mix={'style_dev':a.style_weight};rest=1-a.style_weight;tot=sum(MIXES['qwen'].values())
 mix.update({k:rest*v/tot for k,v in MIXES['qwen'].items()})
 per_step=a.accum*a.batch_size;steps=math.ceil(n_dev*a.passes/a.style_weight/per_step)
 val_at=sorted({max(1,round(steps*i/a.vals)) for i in range(1,a.vals+1)}|{steps});warm=max(3,steps//10)
 weights=torch.tensor([mix[r['source']]/cnt[r['source']] for r in train],dtype=torch.double)
 log(f'STYLE FT train={len(train)} style_dev={n_dev} passes={a.passes} steps={steps} val_at={val_at} warm={warm} lr_scale={a.lr_scale}')
 log('mix='+json.dumps({k:round(v,3) for k,v in mix.items()}))
 p=WhisperProcessor.from_pretrained(str(BASE),local_files_only=True);p.tokenizer.set_prefix_tokens(language='jw',task='transcribe',predict_timestamps=False)
 model=WhisperForConditionalGeneration.from_pretrained(a.init,local_files_only=True,attn_implementation='sdpa',dtype=torch.float32).cuda();log(f'initialized from {a.init}')
 model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 held=readj(a.heldout) if a.heldout else None
 if held:
  z,_,_=validate(model,p,held,'heldout',0,exp,batch=4);log(f'[baseline] heldout={z["overall"]["wer"]:.4f}')
 opt=make_opt(model)
 for g in opt.param_groups:g['base_lr']*=a.lr_scale
 sampler=WeightedRandomSampler(weights,num_samples=steps*per_step,replacement=True,generator=torch.Generator().manual_seed(a.seed))
 loader=DataLoader(DS(train),batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=Collate(p,True));it=iter(loader)
 opt.zero_grad(set_to_none=True);model.train();best=99.;start=time.time();run=0.
 for step in range(1,steps+1):
  ratio=schedule(opt,step,steps,warm=warm);sl=0.
  for _ in range(a.accum):
   b=next(it)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_features=b['input_features'].cuda(non_blocking=True),attention_mask=b['attention_mask'].cuda(non_blocking=True),labels=b['labels'].cuda(non_blocking=True)).loss/a.accum
   loss.backward();sl+=loss.item()*a.accum
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True);run+=sl
  if step%10==0:log(f'step {step:4d}/{steps} loss={run/10:.4f} ratio={ratio:.3f} sec_step={(time.time()-start)/step:.3f} vram={torch.cuda.max_memory_allocated()/1e9:.1f}G');run=0.
  if step in val_at:
   if held:
    dv,_,_=validate(model,p,held,'heldout',step,exp,batch=4);score=dv['overall']['wer'];new=score<best
    if new:best=score;save_best(model,exp);(exp/'best.json').write_text(json.dumps({'step':step,'heldout_wer':score},indent=2))
    log(f'[ckpt] step={step} passes={step*per_step*a.style_weight/n_dev:.2f} heldout={score:.4f} best={best:.4f} new={new}')
   elif step==steps:
    save_best(model,exp);(exp/'best.json').write_text(json.dumps({'step':step,'passes':a.passes},indent=2));log(f'[ckpt] saved final step={step}')
 rv,_,_=validate(model,p,readj(ROBUST),'robust',steps,exp,batch=4);log(f'[forgetting-guard] robust={rv["overall"]["wer"]:.4f} (stage-1 was 0.2136)')
 log('style training complete')
if __name__=='__main__':main()
