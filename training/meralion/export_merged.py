#!/usr/bin/env python3
"""Export a fine-tuned MERaLiON-3-3B-ASR as one self-contained model folder.

    python training/meralion_v1/export_merged.py --config <yaml> --ckpt <exp>/best.ckpt --out <dir>

Loads the base model, applies the trainable state (LoRA + adapter + encoder),
merges the LoRA into the decoder weights, and saves a plain bf16 checkpoint
with the processor and the three remote-code files, so the submission loads it
with AutoModelForSpeechSeq2Seq(..., trust_remote_code=True) and nothing else.
Then reloads the folder and decodes a few dev clips to prove the round trip.
"""
import os
import argparse
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
sys.path.insert(0, str(ROOT / 'training/meralion_v1'))
sys.path.insert(0, str(ROOT / 'scripts'))
import meralion_train as mt  # noqa: E402
from newmodels_common import clean_text, load_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    cfg.model.gradient_checkpointing = False
    device = torch.device('cuda:0')
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg.model.path, trust_remote_code=True, local_files_only=True)
    model, _ = mt.build_model(cfg, device)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    res = model.load_state_dict(ck['trainable'], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    mt.log(f'[export] loaded {len(ck["trainable"])} trainable tensors from {a.ckpt} (step {ck.get("step")})')

    merged = model.merge_and_unload()          # LoRA folded into the Gemma-2 linears
    merged.eval()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    proc.save_pretrained(out)
    src = Path(cfg.model.path)
    for f in ['configuration_meralion3.py', 'modeling_meralion3.py', 'processing_meralion3.py',
              'preprocessor_config.json', 'processor_config.json', 'generation_config.json']:
        if (src / f).exists() and not (out / f).exists():
            shutil.copy(src / f, out / f)
    total = sum(p.stat().st_size for p in out.glob('*.safetensors')) / 1e9
    mt.log(f'[export] wrote {out}  ({total:.2f} GB of safetensors)')

    # Round trip: reload from the folder exactly as main.py will, decode 4 dev clips.
    del model, merged
    torch.cuda.empty_cache()
    m2 = AutoModelForSpeechSeq2Seq.from_pretrained(out, trust_remote_code=True, local_files_only=True,
                                                   dtype=torch.bfloat16, attn_implementation='eager').to(device).eval()
    p2 = AutoProcessor.from_pretrained(out, trust_remote_code=True, local_files_only=True)
    col = mt.Collator(p2, 'text_ref1', 40.0, train=False)
    import pandas as pd
    dev = pd.read_csv(ROOT / 'data/processed/newmodels/dev.csv', keep_default_na=False).head(4)
    batch = mt.to_device(col([{'audio_filepath': p} for p in dev.path]), device, m2.dtype)
    eos = [p2.tokenizer.eos_token_id, p2.tokenizer.convert_tokens_to_ids(mt.END_OF_TURN)]
    with torch.no_grad():
        o = m2.generate(**batch, max_new_tokens=256, do_sample=False, num_beams=1,
                        no_repeat_ngram_size=6, eos_token_id=eos, pad_token_id=p2.tokenizer.pad_token_id)
    L = batch['input_ids'].shape[1]
    seq = o[:, L:] if o.shape[1] > L else o
    for ref, hyp in zip(dev.ref, p2.batch_decode(seq, skip_special_tokens=True)):
        mt.log(f'    ref: {ref[:90]}')
        mt.log(f'    hyp: {clean_text(hyp)[:90]}')
    mt.log('[export] round trip OK')


if __name__ == '__main__':
    main()
