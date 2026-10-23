#!/usr/bin/env python3
"""Final style stage: ALL 372 dev clips in training, encoder+decoder recipe.

Hyperparameters validated out-of-fold on speaker- and session-disjoint splits:
LoRA r16 layers 14-27 @ 8e-6, audio tower @ 1e-6, projector @ 5e-6.
Pass count (2.0) comes from the sweep on speaker_hold1 / conv_hold2, which showed
no degradation up to 3 passes; there is no held-out dev left here, so it is fixed
from those folds rather than selected in this run.
"""
import json
import os
from pathlib import Path
R = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
A = R/'data/processed/qwen_style_aggressive'
E = R/'training/qwen_style_aggressive'
PASSES = 2.0

dst = A/'full_dev_train.jsonl'
n_dev = n_pool = 0
with open(dst, 'w', encoding='utf-8') as out:
    for line in open(A/'conv_hold2_train.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r.get('source') == 'style_dev': n_dev += 1
        else: n_pool += 1
        out.write(json.dumps(r, ensure_ascii=False)+'\n')
    for line in open(A/'conv_hold2_heldout.jsonl', encoding='utf-8'):
        r = json.loads(line); r['source'] = 'style_dev'; n_dev += 1
        out.write(json.dumps(r, ensure_ascii=False)+'\n')
print(f'{dst.name}: style_dev={n_dev}  pool={n_pool}')
assert n_dev == 372, n_dev

c = json.loads((A/'conv_hold2.json').read_text())
c['name'] = 'qwen_style_final_full_dev'
c['paths']['train_manifest'] = str(dst)
c['paths']['val_unseen'] = str(A/'all_dev_by_speaker.jsonl')  # IN-SAMPLE, monitor only
c['paths']['val_all'] = str(A/'robust.jsonl')                 # forgetting guard
c['paths']['exp_dir'] = str(E/'exp_final_full_dev')
c['freeze']['train_audio_tower'] = True
c['freeze']['train_projector'] = True
c['optim']['lr_encoder'] = 1e-6
c['optim']['lr_projector'] = 5e-6
c['optim']['lr_lora'] = 8e-6
w = c['sampling']['weights']['style_dev']
epoch_samples = int(round(372 * PASSES / w / 4) * 4)
c['sampling']['epoch_samples'] = epoch_samples
c['trainer']['max_steps'] = epoch_samples // c['data']['batch_size']
c['trainer']['max_epochs'] = 1
c['trainer']['val_every_n_steps'] = c['trainer']['max_steps']
c['trainer']['val_all_every'] = 1
(A/'full_dev_final.json').write_text(json.dumps(c, indent=2))
print(f'passes={PASSES} epoch_samples={epoch_samples} max_steps={c["trainer"]["max_steps"]} '
      f'style_dev/epoch={round(epoch_samples*w)} replay={round(epoch_samples*(1-w))}')
