"""Score saved whisper_full_train checkpoints exactly as the trainer would have, in a fresh process.

Same validate() on the robust set and on dev_short, same full_dev_hybrid(), same batch size, so the numbers
line up with exp_full_v5mix_aug/metrics.jsonl (robust 0.2252 / dev_short 0.1867 / hybrid 0.1869 at 3000).
Usage: eval_whisper_ckpt.py <exp dir> [step ...]   (waits for each checkpoint to appear)"""
import os
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/training/whisper_v3_lora")
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import whisper_lora_train as W

exp = Path(sys.argv[1])
steps = [int(x) for x in sys.argv[2:]] or [1000, 2000, 3000]
p = WhisperProcessor.from_pretrained(str(W.BASE), local_files_only=True)
p.tokenizer.set_prefix_tokens(language="jw", task="transcribe", predict_timestamps=False)
robust, dev = W.readj(W.ROBUST), W.readj(W.DEV)
for st in steps:
    ck = exp / f"step{st}"
    while not (ck / "config.json").exists():
        time.sleep(60)
    time.sleep(30)
    model = WhisperForConditionalGeneration.from_pretrained(str(ck), local_files_only=True, attn_implementation="sdpa",
                                                            dtype=torch.float32).cuda()
    rv, _, _ = W.validate(model, p, robust, "robust", st, exp, batch=4)
    dv, dh, dr = W.validate(model, p, dev, "dev_short", st, exp, batch=4)
    hy = W.full_dev_hybrid(dh, dr, dev)
    open(exp / "metrics_offline.jsonl", "a").write(json.dumps({"robust": rv, "dev_short": dv, "dev_hybrid": hy}) + "\n")
    print(f"[ckpt] step={st} robust={rv['overall']['wer']:.4f} dev_short={dv['overall']['wer']:.4f} "
          f"hybrid={hy['wer']:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
print("EVAL_CKPT_DONE")
