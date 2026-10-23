# -*- coding: utf-8 -*-
"""omniASR-LLM fine-tune, done the way the MMS Step-A runs proved works.

Differences from train_omni2.py:
  * fp32 MASTER WEIGHTS for every trainable tensor. The model stays bf16 for the forward pass,
    but AdamW runs on fp32 copies. In pure bf16 a 1e-5 step is rounded away for ~90 % of the
    weights (measured), so the old runs only ever moved the near-zero ones.
  * mix spec like mmsA ("dev_train:8,zac:1,tetel:1"), no Puebla unless asked for.
  * in-loop DECODED evaluation on a held-out speaker fold with the official normalizer:
    raw WER, spelling-blind WER (L1), hyp/ref length. Selection is on decoded WER, never on
    teacher-forced loss.
  * saves only the trainable tensors (bf16) for best/last.
"""
import os, sys, re, json, math, time, random, argparse, collections, unicodedata
import numpy as np, soundfile as sf, torch
from torch.utils.data import Dataset, DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--card', default='omniASR_LLM_7B')
ap.add_argument('--mix', required=True, help='path:weight,path:weight  (weight = repeat factor, may be <1)')
ap.add_argument('--held', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--steps', type=int, default=600)
ap.add_argument('--warmup', type=int, default=60)
ap.add_argument('--accum', type=int, default=6)
ap.add_argument('--max_samples', type=int, default=1_280_000)   # 80 s of audio per micro-batch
ap.add_argument('--max_bs', type=int, default=12)
ap.add_argument('--workers', type=int, default=3)
ap.add_argument('--enc_top', type=int, default=12, help='train only the top-N encoder layers (-1 = all)')
ap.add_argument('--eval_every', type=int, default=50)
ap.add_argument('--eval_bs', type=int, default=4)
ap.add_argument('--log_every', type=int, default=10)
ap.add_argument('--init', default='')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_sec', type=float, default=40.0)
ap.add_argument('--smoke', type=int, default=0)
ap.add_argument('--no_eval', type=int, default=0, help='final all-data runs: never decode in-process, just save')
ap.add_argument('--save_every', type=int, default=100)
ap.add_argument('--aug', type=float, default=0.0, help='probability of waveform augmentation per training clip')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device('cuda')
T0 = time.time()
def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)

from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import jiwer
sys.path.insert(0, '/kaggle/working/Enes_nah/mmsA')
from norm import normalize_text          # byte-exact port of the official scorer's normalizer

LANGMAP = {'azz': 'azz_Latn', 'nhi': 'nhi_Latn', 'nhw': 'nhw_Latn', 'nlv': 'nlv_Latn', 'ncx': 'ncx_Latn',
           'spa': 'spa_Latn', '': 'nhi_Latn'}
def omni_lang(l):
    return l if '_' in l else LANGMAP.get(l, 'nhi_Latn')

# ----------------------------------------------------------------------------- model
model = load_model(args.card, device=dev, dtype=torch.bfloat16)
tok = load_tokenizer(args.card)
enc = tok.create_encoder()
PAD = model.target_vocab_info.pad_idx

lyr = set()
for n, _ in model.named_parameters():
    m = re.match(r'encoder\.layers\.(\d+)\.', n)
    if m: lyr.add(int(m.group(1)))
NL = max(lyr) + 1
keep = set(range(NL)) if args.enc_top < 0 else set(range(max(0, NL - args.enc_top), NL))
for n, p in model.named_parameters():
    top = n.split('.')[0]
    if top in ('encoder', 'encoder_frontend', 'masker'):
        m = re.match(r'encoder\.layers\.(\d+)\.', n)
        if not (m and int(m.group(1)) in keep):
            p.requires_grad_(False)
if args.init and os.path.exists(args.init):
    sd = torch.load(args.init, map_location='cpu')
    sd = sd.get('model', sd)
    r = model.load_state_dict(sd, strict=False)
    log('init from', args.init, 'unexpected', len(r.unexpected_keys))

named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
ntr = sum(p.numel() for _, p in named); nall = sum(p.numel() for p in model.parameters())
log(f"{args.card}: {nall/1e9:.2f}B params, trainable {ntr/1e9:.2f}B "
    f"(encoder layers {min(keep)}..{max(keep)} of {NL} + everything outside the encoder)")

# fp32 master weights -------------------------------------------------------------
masters = [p.detach().float().clone().requires_grad_(True) for _, p in named]
for m_ in masters: m_.grad = torch.zeros_like(m_)
decay = [m_ for m_ in masters if m_.dim() > 1]; nodecay = [m_ for m_ in masters if m_.dim() <= 1]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': nodecay, 'weight_decay': 0.0}],
                        lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
def lr_at(s):
    if s < args.warmup: return args.lr * (s + 1) / max(1, args.warmup)
    p = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
def pull_grads():
    for (_, p), m_ in zip(named, masters):
        if p.grad is not None:
            m_.grad.add_(p.grad.float()); p.grad = None
def push_weights():
    for (_, p), m_ in zip(named, masters):
        p.data.copy_(m_.data)          # fp32 -> bf16 cast happens here

# ----------------------------------------------------------------------------- data
def clean_target(t):
    """The omni tokenizer has NO uppercase letters (every capital encodes to <unk>), so targets are lowercased here
    and capitals are restored after decoding. Also drops annotation glyphs the tokenizer cannot encode."""
    t = t.replace("’", "'").replace("‘", "'").lower()
    t = re.sub(r"[¨´\[\]{}#_]", "", t)
    return " ".join(t.split())

def load_rows(p):
    rows = [json.loads(l) for l in open(p, encoding='utf-8')]
    for r in rows: r['text_orig'] = r['text']; r['text'] = clean_target(r['text'])
    return rows
rows = []
for spec in args.mix.split(','):
    path, w = spec.rsplit(':', 1); w = float(w); rr = load_rows(path)
    rr = [r for r in rr if 0.4 <= r['duration'] <= args.max_sec and r['text'].strip()]
    k, frac = int(w), w - int(w)
    part = rr * k + (random.sample(rr, int(len(rr) * frac)) if frac > 0 else [])
    log(f"  mix {os.path.basename(path)} x{w:g}: {len(rr)} rows {sum(r['duration'] for r in rr)/3600:.2f} h "
        f"-> {len(part)} rows {sum(r['duration'] for r in part)/3600:.2f} h")
    rows += part
held = load_rows(args.held)
if args.smoke: rows = rows[:400]; held = held[:12]
log(f"train {len(rows)} rows {sum(r['duration'] for r in rows)/3600:.2f} h | held {len(held)} clips "
    f"{sum(r['duration'] for r in held)/60:.1f} min  langs {dict(collections.Counter(r['lang'] for r in held))}")

def read_audio(path, offset=0.0, duration=None):
    info = sf.info(path); sr = info.samplerate
    start = int(offset * sr); frames = int(duration * sr) if duration else -1
    a, _ = sf.read(path, start=start, frames=frames, dtype='float32')
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a.astype('float32')

def augment(a):
    """Cheap waveform augmentation: speed perturbation, gain, additive coloured noise, short time masks."""
    r = np.random
    if r.rand() < 0.7:                                   # speed 0.88..1.12 (changes tempo and pitch together)
        f = r.uniform(0.88, 1.12)
        n = max(1600, int(len(a) / f))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype('float32')
    if r.rand() < 0.5:                                   # gain +-6 dB
        a = a * (10 ** (r.uniform(-6, 6) / 20))
    if r.rand() < 0.5:                                   # noise at 12..30 dB SNR, white or low-passed
        p = float(np.mean(a ** 2)) + 1e-9
        nz = r.randn(len(a)).astype('float32')
        if r.rand() < 0.5:
            k = r.randint(4, 40); nz = np.convolve(nz, np.ones(k, dtype='float32') / k, mode='same')
        nz *= np.sqrt(p / (10 ** (r.uniform(12, 30) / 10)) / (float(np.mean(nz ** 2)) + 1e-9))
        a = a + nz
    if r.rand() < 0.5:                                   # 1-3 time masks of 20..120 ms
        for _ in range(r.randint(1, 4)):
            w = r.randint(320, 1920); s0 = r.randint(0, max(1, len(a) - w)); a[s0:s0 + w] = 0.0
    return np.clip(a, -1.0, 1.0).astype('float32')

class BatchDS(Dataset):
    def __init__(self, rows):
        self.rows = rows; self.rebuild()
    def rebuild(self):
        # bucket by duration with a little jitter so batches differ between epochs
        rs = sorted(self.rows, key=lambda r: r['duration'] * random.uniform(0.9, 1.1))
        self.b = []; cur = []; mx = 0
        for r in rs:
            n = int(r['duration'] * 16000)
            if cur and (max(mx, n) * (len(cur) + 1) > args.max_samples or len(cur) >= args.max_bs):
                self.b.append(cur); cur = []; mx = 0
            cur.append(r); mx = max(mx, n)
        if cur: self.b.append(cur)
        random.shuffle(self.b)
    def __len__(self): return len(self.b)
    def __getitem__(self, i):
        items = []
        for r in self.b[i]:
            try: a = read_audio(r['audio_filepath'], r.get('offset', 0.0), r['duration'])
            except Exception: continue
            if a.size < 1600: continue
            if args.aug > 0 and np.random.rand() < args.aug: a = augment(a)
            ids = enc(r['text'])
            if ids.numel() < 2 or ids.numel() > 512: continue
            items.append((a, ids, omni_lang(r.get('lang', ''))))
        if not items: return None
        L = max(len(a) for a, _, _ in items); T = max(i.numel() for _, i, _ in items)
        src = torch.zeros(len(items), L); tgt = torch.full((len(items), T), PAD, dtype=torch.long)
        sl, tl = [], []
        for k, (a, ii, _) in enumerate(items):
            src[k, :len(a)] = torch.from_numpy(a); sl.append(len(a))
            tgt[k, :ii.numel()] = ii; tl.append(int(ii.numel()))
        return src, sl, tgt, tl, [l for _, _, l in items]

def to_batch(x):
    src, sl, tgt, tl, lang = x
    return Seq2SeqBatch(source_seqs=src.to(dev, torch.bfloat16, non_blocking=True), source_seq_lens=sl,
                        target_seqs=tgt.to(dev, non_blocking=True), target_seq_lens=tl, example={"lang": lang})

tds = BatchDS(rows)
log(f"{len(tds)} micro-batches/epoch = {len(tds)/args.accum:.0f} optimizer steps/epoch")
def make_dl():
    return DataLoader(tds, batch_size=None, shuffle=True, num_workers=args.workers, collate_fn=lambda x: x,
                      prefetch_factor=4, persistent_workers=False, pin_memory=False,
                      worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32))  # distinct aug stream per worker/epoch

# ----------------------------------------------------------------------------- eval
def ortho_blind(word):
    w = unicodedata.normalize("NFD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    w = w.replace(":", "").replace("tz", "ts")
    w = w.replace("ch", "\x01").replace("qu", "k").replace("q", "k")
    w = re.sub(r"c([ei])", r"s\1", w); w = w.replace("c", "k").replace("\x01", "ch")
    w = w.replace("z", "s").replace("hu", "u").replace("uh", "u").replace("w", "u")
    w = w.replace("ll", "y").replace("h", "j").replace("v", "b")
    return re.sub(r"(.)\1+", r"\1", w)
blind = lambda t: t.lower()

def split_long(w, maxs=34.0, sr=16000):
    if len(w) <= maxs * sr: return [w]
    lo, hi = int(len(w) * .35), int(len(w) * .65); fr = 512
    e = np.array([np.abs(w[i:i + fr]).mean() for i in range(lo, hi, fr)])
    c = lo + int(np.argmin(e)) * fr
    return split_long(w[:c], maxs, sr) + split_long(w[c:], maxs, sr)

pipe = ASRInferencePipeline(None, model=model, tokenizer=tok)
HELD_AUDIO = None
@torch.no_grad()
def evaluate(step):
    global HELD_AUDIO
    t = time.time()
    if HELD_AUDIO is None:
        HELD_AUDIO = [read_audio(r['audio_filepath'], r.get('offset', 0.0), r['duration']) for r in held]
    model.eval()      # NOTE: no empty_cache() here - returning memory to the WSL driver and re-allocating it is what
                      # produced the sporadic, context-poisoning 'CUDA error: invalid argument' at evaluation time
    pieces = []
    for i, w in enumerate(HELD_AUDIO):
        for j, p in enumerate(split_long(w)): pieces.append((i, j, p))
    pieces.sort(key=lambda x: len(x[2]))
    # Never decode a batch made only of ultra-short clips: on this box a (4 x <2.5 s) conv input
    # raises "CUDA error: invalid argument" once training has filled the allocator, and the error
    # can poison the context. Fold the short pieces into the first batch that reaches >= 4 s
    # (padding is masked by seq_lens, so results are unchanged).
    batches = []; cur = []
    for pc in pieces:
        cur.append(pc)
        if len(cur) >= args.eval_bs and len(cur[-1][2]) >= 4 * 16000:
            batches.append(cur); cur = []
        elif len(cur) >= 3 * args.eval_bs:
            batches.append(cur); cur = []
    if cur: batches.append(cur)
    res = {}
    for sub in batches:
        try:
            out = pipe.transcribe([{"waveform": p, "sample_rate": 16000} for _, _, p in sub], lang=["ind_Latn"] * len(sub), batch_size=len(sub))
        except Exception as e:
            # seen on this WSL box for batches of ultra-short (<2.5 s) clips: retry them one at a time
            log("  decode ERR (retrying singly)", repr(e)[:80]); out = []
            for _, _, p1 in sub:
                try: out += pipe.transcribe([{"waveform": p1, "sample_rate": 16000}], lang=["ind_Latn"], batch_size=1)
                except Exception: out.append("")
        for (i, j, _), o in zip(sub, out): res.setdefault(i, {})[j] = o
    hyps = [" ".join(res.get(i, {}).get(j, "") for j in sorted(res.get(i, {}))).strip() for i in range(len(held))]
    R = [normalize_text(r.get('text_orig', r['text'])).strip() for r in held]
    H = [normalize_text(h).strip() for h in hyps]
    out = {}
    for lang in sorted(set(r['lang'] for r in held)) + ['ALL']:
        sel = [i for i, r in enumerate(held) if lang == 'ALL' or r['lang'] == lang]
        rr = [R[i] for i in sel]; hh = [H[i] if H[i] else "_" for i in sel]
        nr = sum(len(x.split()) for x in rr); nh = sum(len(H[i].split()) for i in sel)
        out[lang] = dict(wer=jiwer.wer(rr, hh), l1=jiwer.wer([blind(x) for x in rr], [blind(x) for x in hh]),
                         cer=jiwer.cer(rr, hh), len=nh / max(nr, 1), n=len(sel))
    with open(f"{args.out}/hyps_step{step}.jsonl", 'w', encoding='utf-8') as f:
        for r, h in zip(held, hyps):
            f.write(json.dumps({"rec": r.get('rec'), "lang": r['lang'], "speaker": r.get('speaker'),
                                "dur": r['duration'], "ref": r.get('text_orig', r['text']), "hyp": h}, ensure_ascii=False) + "\n")
    model.train()
    out['_sec'] = time.time() - t
    return out

def save(name):
    sd = {n: p.detach().to(torch.bfloat16).cpu() for n, p in named}
    torch.save({"model": sd, "args": vars(args)}, f"{args.out}/{name}.pt")

# ----------------------------------------------------------------------------- loop
history = []; best = (9e9, -1)
def do_eval(step):
    global best
    r = evaluate(step); a = r['ALL']
    parts = " | ".join(f"{k} raw {100*v['wer']:.2f} L1 {100*v['l1']:.2f} cer {100*v['cer']:.2f} len {v['len']:.3f}"
                       for k, v in r.items() if not k.startswith('_') and k != 'ALL')
    if a['wer'] < best[0]:
        best = (a['wer'], step)
        if step > 0: save('best')
    log(f"EVAL step {step} | {parts} | best {100*best[0]:.2f}@{best[1]} | {r['_sec']:.0f}s")
    history.append(dict(step=step, **{k: v for k, v in r.items() if not k.startswith('_')}))
    json.dump(dict(history=history, best_wer=best[0], best_step=best[1], args=vars(args)),
              open(f"{args.out}/results.json", 'w'), indent=1)

if not args.no_eval: do_eval(0)
model.train(); step = 0; acc = 0; run = collections.deque(maxlen=60); ooms = 0; t_tr = time.time()
log("start training")
while step < args.steps:
    tds.rebuild()
    for x in make_dl():
        if x is None: continue
        try:
            b = to_batch(x); nb = b.source_seqs.size(0)
            loss = model(b) / nb / args.accum
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            ooms += 1; log(f"OOM #{ooms} (batch {x[0].shape}) - skipped")
            for _, p in named: p.grad = None
            for m_ in masters: m_.grad.zero_()
            torch.cuda.empty_cache(); acc = 0
            if ooms > 40: log("FATAL too many OOMs"); sys.exit(3)
            continue
        if not torch.isfinite(loss):
            log("WARN non-finite loss - micro-batch skipped")
            for _, p in named: p.grad = None
            continue
        run.append(loss.item() * args.accum); pull_grads(); acc += 1
        if acc < args.accum: continue
        for g in opt.param_groups: g['lr'] = lr_at(step)
        gn = torch.nn.utils.clip_grad_norm_(masters, 1.0)
        opt.step(); push_weights()
        for m_ in masters: m_.grad.zero_()
        acc = 0; step += 1
        if step % args.log_every == 0:
            log(f"step {step}/{args.steps} loss {np.mean(run):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} "
                f"{(time.time()-t_tr)/step:.1f}s/step mem {torch.cuda.max_memory_allocated()/1e9:.1f}G")
        if args.no_eval:
            if step % args.save_every == 0 or step == args.steps:
                save(f'step{step}'); log(f'saved step{step}.pt')
        elif step % args.eval_every == 0 or step == args.steps:
            save('last'); do_eval(step)     # save first: a poisoned CUDA context must not cost us the weights
        if step >= args.steps: break
log(f"DONE best {100*best[0]:.2f}@{best[1]}")
