"""The stage-1 Qwen model in the original qwen_asr layout, which merge_convert_qwen_asr.py reads as a reference: it
checks its hf5 -> qwen_asr key mapping against this folder tensor by tensor and copies the config and tokenizer files
from it. During the competition this folder came from an earlier conversion whose script was not kept; this file
writes the same thing (same key mapping, lm_head tied to the embeddings, files of the original Qwen/Qwen3-ASR-1.7B).
usage: python make_qa_reference.py <hf5 folder> <Qwen/Qwen3-ASR-1.7B folder> <out folder>"""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

hf5, orig, out = map(Path, sys.argv[1:4])


def to_qa(k):
    k = k.replace('model.language_model.', 'thinker.model.')
    k = k.replace('model.multi_modal_projector.linear_1', 'thinker.audio_tower.proj1')
    k = k.replace('model.multi_modal_projector.linear_2', 'thinker.audio_tower.proj2')
    k = k.replace('model.audio_tower.', 'thinker.audio_tower.')
    return k


sd = load_file(str(hf5 / 'model.safetensors'))
qa = {to_qa(k): v for k, v in sd.items() if k != 'lm_head.weight'}
qa['thinker.lm_head.weight'] = sd['model.language_model.embed_tokens.weight'].clone()
out.mkdir(parents=True, exist_ok=True)
for f in orig.iterdir():
    if f.is_file() and not f.name.endswith('.safetensors') and f.name != 'model.safetensors.index.json':
        shutil.copy(f, out / f.name)
save_file(qa, str(out / 'model.safetensors'), metadata={'format': 'pt'})
print('wrote', out, len(qa), 'tensors')
