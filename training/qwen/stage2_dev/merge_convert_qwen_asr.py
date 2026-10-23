"""Merge the dev-adapted style LoRA into the Qwen v2 base and write it in the
qwen_asr layout the competition container loads (transformers 4.57).

Merge happens in venv_qwen (transformers 5 + peft) on the hf5 directory; the
key mapping hf5 -> qwen_asr is derived from the existing pair of directories
(same model, two layouts) and verified tensor-by-tensor before use.
"""
import os
import json, re, shutil, struct, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

R = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
HF5 = R / 'training/qwen_v2/models/lora_one_stage_step9000_hf5'
QA = R / 'training/qwen_v2/models/lora_one_stage_step9000_qwen_asr'
CK = R / 'training/qwen_style_aggressive/exp_final_full_dev/best.ckpt'
OUT = R / 'submissions/rover3/models/qwen'

def to_qa(k):
    k = k.replace('model.language_model.', 'thinker.model.')
    k = k.replace('model.multi_modal_projector.linear_1', 'thinker.audio_tower.proj1')
    k = k.replace('model.multi_modal_projector.linear_2', 'thinker.audio_tower.proj2')
    k = k.replace('model.audio_tower.', 'thinker.audio_tower.')
    return k

# 1. verify the mapping on the existing pair
hf = load_file(str(HF5 / 'model.safetensors')); qa = load_file(str(QA / 'model.safetensors'))
mapped = {to_qa(k): v for k, v in hf.items()}
missing = [k for k in qa if k not in mapped]; extra = [k for k in mapped if k not in qa]
bad = [k for k in mapped if k in qa and not (mapped[k].shape == qa[k].shape and torch.equal(mapped[k], qa[k]))]
print(f'mapping check: qa tensors {len(qa)}, mapped {len(mapped)}, missing-in-mapped {missing}, extra {extra[:3]}, unequal {bad[:3]}')
assert not bad and not extra, 'mapping does not reproduce the existing conversion'
assert missing == ['thinker.lm_head.weight'], missing
assert torch.equal(qa['thinker.lm_head.weight'], hf['model.language_model.embed_tokens.weight']), 'lm_head is not tied to embeddings'
del qa

# 2. merge the style LoRA into the hf5 model
from transformers import Qwen3ASRForConditionalGeneration
from peft import LoraConfig, get_peft_model
model = Qwen3ASRForConditionalGeneration.from_pretrained(str(HF5), dtype=torch.bfloat16)
ids = '|'.join(str(i) for i in range(14, 28))
pattern = (rf"model\.language_model\.layers\.(?:{ids})\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', target_modules=pattern))
ck = torch.load(str(CK), map_location='cpu', weights_only=False)
res = model.load_state_dict(ck['trainable'], strict=False)
assert not res.unexpected_keys, res.unexpected_keys[:5]
n_lora = sum(1 for k in ck['trainable'] if 'lora_' in k)
print(f'loaded style state: {len(ck["trainable"])} tensors ({n_lora} LoRA); step {ck.get("step")}')
merged = model.merge_and_unload()
sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in merged.state_dict().items() if k != 'lm_head.weight'}  # tied alias of embed_tokens
changed = sum(1 for k in sd if k in hf and not torch.equal(sd[k], hf[k]))
print(f'merged state: {len(sd)} tensors, {changed} differ from the base (expect audio tower + projector + layers>=14 attn/mlp)')

# 3. write in qwen_asr layout
out_sd = {to_qa(k): v for k, v in sd.items()}
out_sd['thinker.lm_head.weight'] = sd['model.language_model.embed_tokens.weight'].clone()  # own storage for safetensors
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for f in QA.iterdir():
    if f.name != 'model.safetensors': shutil.copy(f, OUT / f.name)
save_file(out_sd, str(OUT / 'model.safetensors'), metadata={'format': 'pt'})
print('wrote', OUT, sum(p.stat().st_size for p in OUT.iterdir()) / 1e9, 'GB')
