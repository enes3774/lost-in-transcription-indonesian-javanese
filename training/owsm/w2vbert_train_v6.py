#!/usr/bin/env python3
"""CTC fine-tuning of w2v-BERT 2.0 on the competition pool.

Why this model: §12.15 showed that where a spelling distinction is acoustically present but
sub-lexical (`gak`/`nggak`, `wis`/`wes`, `tahu`/`tau`) a prior-free CTC model discriminates far
better than any of our seq2seq systems, because it has no decoder language model to snap the rare
form onto the frequent one -- but our only CTC model is an XLSR-53 fine-tune whose coverage of
those slots is 25-35 %, which caps the gain at 0.0006. w2v-BERT 2.0 is the same idea with 4.5M
hours of pretraining over 143 languages (vs XLSR-53's 56k), so the bet is that coverage rises
while discrimination holds.

Deliberately mirrors training/ctc_v2/ctc_train_llrd.py -- same pool, same source weights, same
label normalisation -- so the two CTC models are directly comparable, and the predictions land in
the same CSV shape that analysis/pair_discrimination.py consumes. No distillation teacher here:
the CTC head is new, so there is nothing to distil from.
"""
import argparse, csv, gc, json, math, os, random, re, sys, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np, soundfile as sf, torch
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer

ROOT = Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
BASE = ROOT / 'models/checkpoints/w2v-bert-2.0'
POOL = ROOT / 'data/processed/qwen_v2/train_pool_v4_clean.jsonl'
ROBUST = ROOT / 'data/processed/qwen_v2/robust_val.jsonl'
DEV = ROOT / 'data/processed/qwen_v2/competition_dev_eval_only.jsonl'
EXP = ROOT / 'training/ctc_w2vbert/exp_v6mix'
SOURCE_WEIGHTS = {'jember': .16, 'homostoria': .11, 'hari': .05, 'slr35': .11, 'central': .06, 'fleurs_jv': .05, 'ngapak': .05, 'purworejo': .03, 'lumajang': .03, 'bojonegoro': .02, 'cv_id': .05, 'fleurs_id': .05, 'banyumasan': .10, 'gresik': .05, 'jombang': .05, 'multidialect': .03}
ALLOWED = set('abcdefghijklmnopqrstuvwxyzé ')
DROP = set("'-–—….")


def log(s):
    print(time.strftime('[%H:%M:%S]'), s, flush=True)


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding='utf8')]


def label_text(x):
    """Identical to ctc_v2's, so the two CTC models see the same targets."""
    x = unicodedata.normalize('NFC', str(x)).lower().translate(str.maketrans({'è': 'e', 'ê': 'e', 'ě': 'e'}))
    out = []
    for c in x:
        if c in ALLOWED:
            out.append(c)
        elif c in DROP:
            pass
        else:
            d = ''.join(z for z in unicodedata.normalize('NFKD', c) if not unicodedata.combining(z))
            if d and all(z in ALLOWED for z in d):
                out.append(d)
            else:
                out.append(' ')
    return re.sub(r' +', ' ', ''.join(out)).strip()


def load_audio(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim == 2:
        x = x.mean(1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {'audio': load_audio(r['audio_filepath']), 'text': label_text(r.get('text_ref1', '')), 'row': r}


class Collate:
    def __init__(self, fe, tok, labels=True):
        self.fe = fe
        self.tok = tok
        self.labels = labels

    def __call__(self, b):
        x = self.fe([z['audio'] for z in b], sampling_rate=16000, padding=True, return_attention_mask=True,
                    return_tensors='pt')
        out = {'input_features': x.input_features, 'attention_mask': x.attention_mask, 'rows': [z['row'] for z in b]}
        if self.labels:
            y = self.tok([z['text'] for z in b], padding=True, return_tensors='pt')
            out['labels'] = y.input_ids.masked_fill(y.attention_mask.ne(1), -100)
        return out


def build_tokenizer(pool, out_dir):
    """Character vocabulary from the pool's normalised targets."""
    chars = sorted({c for r in pool for c in label_text(r.get('text_ref1', '')) if c != ' '})
    vocab = {'[PAD]': 0, '[UNK]': 1, '|': 2}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False), encoding='utf8')
    tok = Wav2Vec2CTCTokenizer(str(out_dir / 'vocab.json'), unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    log(f'vocab {len(vocab)} symbols: {"".join(chars)}')
    return tok


def make_optimizer(model, encoder_lr, head_lr, weight_decay=.005):
    groups = defaultdict(list)
    lrs = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('lm_head'):
            key, lr = 'lm_head', head_lr
        elif '.adapter.' in n or n.startswith('wav2vec2_bert.adapter'):
            key, lr = 'adapter', head_lr * .5
        else:
            m = re.search(r'encoder\.layers\.(\d+)\.', n)
            if m:
                i = int(m.group(1))
                key, lr = f'enc_{i:02d}', encoder_lr * (0.95 ** (23 - i))
            else:
                key, lr = 'front', encoder_lr * (0.95 ** 24)
        groups[(key, lr)].append(p)
        lrs[key] = lr
    opt = torch.optim.AdamW([{'params': v, 'lr': lr, 'base_lr': lr, 'weight_decay': weight_decay}
                             for (k, lr), v in groups.items()], betas=(.9, .98), eps=1e-8)
    log(f'LR range {min(lrs.values()):.2e}..{max(lrs.values()):.2e} over {len(groups)} groups')
    return opt


def set_schedule(opt, step, max_steps, warmup, min_ratio=.05):
    if step < warmup:
        ratio = max(1e-3, (step + 1) / warmup)
    else:
        prog = min(1., (step - warmup) / max(1, max_steps - warmup))
        ratio = min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * ratio
    return ratio


def validate(model, fe, tok, rows, name, step, batch_size=8):
    loader = DataLoader(Rows(rows), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=Collate(fe, tok, False))
    hyps, refs, groups = [], [], []
    model.eval()
    started = time.time()
    with torch.inference_mode():
        for b in loader:
            f = b['input_features'].cuda(non_blocking=True)
            am = b['attention_mask'].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = model(input_features=f, attention_mask=am).logits.float()
            hyps.extend(tok.batch_decode(y.argmax(-1).cpu().numpy()))
            refs.extend([label_text(r.get('text_ref1', '')) for r in b['rows']])
            groups.extend([r.get('eval_group', r.get('source', 'all')) for r in b['rows']])
    ok = [i for i in range(len(refs)) if refs[i].strip()]
    R = [refs[i] for i in ok]
    H = [hyps[i] or ' ' for i in ok]
    m = {'wer': float(jiwer_wer(R, H)), 'cer': float(jiwer_cer(R, H))}
    log(f'[val:{name}] wer={m["wer"]:.4f} cer={m["cer"]:.4f} n={len(R)} seconds={time.time()-started:.1f}')
    per = {}
    for g in sorted(set(groups)):
        ids = [i for i in ok if groups[i] == g]
        if ids:
            per[g] = float(jiwer_wer([refs[i] for i in ids], [hyps[i] or ' ' for i in ids]))
            log(f'[val:{name}:{g}] wer={per[g]:.4f}')
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / f'{name}_step{step}.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['audio_filename', 'transcript'])
        w.writeheader()
        for r, h in zip(rows, hyps):
            w.writerow({'audio_filename': r.get('uid', Path(r['audio_filepath']).name), 'transcript': h})
    del loader
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    return {'step': step, 'split': name, **m, 'groups': per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-steps', type=int, default=6000)
    ap.add_argument('--val-every', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--encoder-lr', type=float, default=5e-5)
    ap.add_argument('--head-lr', type=float, default=2e-4)
    ap.add_argument('--max-duration', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=2033)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    EXP.mkdir(parents=True, exist_ok=True)

    pool = [r for r in read_jsonl(POOL)
            if .5 <= float(r.get('duration', 0)) <= a.max_duration and label_text(r.get('text_ref1', ''))]
    robust = read_jsonl(ROBUST)
    dev = read_jsonl(DEV)
    dev = [r for r in dev if float(r.get('duration', 0)) <= 40]
    counts = Counter(r['source'] for r in pool)
    weights = torch.tensor([SOURCE_WEIGHTS.get(r['source'], 0.0) / counts[r['source']] for r in pool], dtype=torch.double)
    log(f'data train={len(pool)} robust={len(robust)} dev={len(dev)} sources={dict(counts)}')

    tok = build_tokenizer(pool, EXP / 'tokenizer')
    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(BASE), local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(
        str(BASE), local_files_only=True, vocab_size=len(tok), pad_token_id=tok.pad_token_id,
        ctc_loss_reduction='mean', ctc_zero_infinity=True, add_adapter=True, layerdrop=0.0,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=0.05,
    ).cuda()
    model.gradient_checkpointing_enable()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'model {sum(p.numel() for p in model.parameters())/1e6:.0f} M params, {n_tr/1e6:.0f} M trainable, '
        f'adapter={model.config.add_adapter}, vocab={model.config.vocab_size}')

    opt = make_optimizer(model, a.encoder_lr, a.head_lr)
    sampler = WeightedRandomSampler(weights, num_samples=a.max_steps * a.accum * a.batch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(Rows(pool), batch_size=a.batch_size, sampler=sampler, num_workers=6, pin_memory=True,
                        collate_fn=Collate(fe, tok, True), persistent_workers=True)
    it = iter(loader)
    model.train(); opt.zero_grad(set_to_none=True)
    started = time.time(); running = 0.; best = 99.
    for step in range(1, a.max_steps + 1):
        set_schedule(opt, step, a.max_steps, 500)
        sl = 0.
        for _ in range(a.accum):
            b = next(it)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_features=b['input_features'].cuda(non_blocking=True),
                            attention_mask=b['attention_mask'].cuda(non_blocking=True),
                            labels=b['labels'].cuda(non_blocking=True))
                loss = out.loss / a.accum
            loss.backward()
            sl += loss.item() * a.accum
            del out, loss, b
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        running += sl
        if step % 25 == 0:
            log(f'step {step:5d}/{a.max_steps} loss={running/25:.4f} lr={opt.param_groups[0]["lr"]:.2e} '
                f'sec_step={(time.time()-started)/step:.3f} vram={torch.cuda.max_memory_allocated()/1e9:.1f}G')
            running = 0.
        if step % a.val_every == 0 or step == a.max_steps:
            dm = validate(model, fe, tok, dev, 'dev', step)
            rm = validate(model, fe, tok, robust, 'robust', step)
            open(EXP / 'metrics.jsonl', 'a').write(json.dumps({'dev': dm, 'robust': rm}) + '\n')
            if dm['wer'] < best:
                best = dm['wer']
                torch.save({'model': model.state_dict(), 'step': step, 'dev_wer': dm['wer'], 'robust_wer': rm['wer']},
                           EXP / 'best.pt')
                (EXP / 'best.json').write_text(json.dumps({'step': step, 'dev_wer': dm['wer'], 'robust_wer': rm['wer']}, indent=2))
            log(f'[ckpt] step={step} dev={dm["wer"]:.4f} robust={rm["wer"]:.4f} best={best:.4f}')
    log('training complete')


if __name__ == '__main__':
    main()

