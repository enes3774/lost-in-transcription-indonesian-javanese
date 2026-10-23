"""best.pt of w2vbert_train.py -> the inference folder (bf16 safetensors + CTC tokenizer + feature extractor).
Originally an inline step on 2026-09-10; this is that code with the paths as arguments.
usage: python export.py <exp_full/best.pt> <out dir>"""
import os
import sys
from pathlib import Path

import torch
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
ck_path, out = Path(sys.argv[1]), Path(sys.argv[2])
EXP = ck_path.parent
tok = Wav2Vec2CTCTokenizer(str(EXP / 'tokenizer/vocab.json'), unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
fe = SeamlessM4TFeatureExtractor.from_pretrained(str(ROOT / 'models/checkpoints/w2v-bert-2.0'), local_files_only=True)
m = Wav2Vec2BertForCTC.from_pretrained(str(ROOT / 'models/checkpoints/w2v-bert-2.0'), local_files_only=True,
    vocab_size=len(tok), pad_token_id=tok.pad_token_id, ctc_loss_reduction='mean', ctc_zero_infinity=True,
    add_adapter=True, layerdrop=0.0, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=0.0)
ck = torch.load(ck_path, map_location='cpu', weights_only=False)
m.load_state_dict(ck['model'], strict=True)
print('loaded step', ck['step'], 'dev_wer', round(ck['dev_wer'], 4), 'robust_wer', round(ck['robust_wer'], 4))
out.mkdir(parents=True, exist_ok=True)
m.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out); fe.save_pretrained(out)
print('exported to', out)
