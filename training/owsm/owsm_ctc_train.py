#!/usr/bin/env python3
"""Fine-tuning of OWSM-CTC v4 1B (espnet/owsm_ctc_v4_1B) on the competition pool, through the model's own forward:
its CTC + intermediate-CTC loss, its BPE tokenizer and its <lang>/<task> prefix. Unlike XEUS it starts from a working
recogniser. Same pool, v6 source mix and sampler as training/ctc_w2vbert/w2vbert_train_v6.py; labels = the pool's
text_ref1 as written (cased); every clip is padded to the model's fixed 30 s input; language token <jav> for all.
A checkpoint is saved BEFORE every validation; dev csv (audio_filename,transcript) every --val-every steps.
The run sizes itself to end by --end-utc unless --max-steps is given.  --smoke 1 --device cpu = one step + 2 decodes."""
import os
import argparse, csv, datetime, gc, glob, importlib.util, json, math, os, random, re, sys, time, types
from collections import Counter
from pathlib import Path
sys.path.insert(0, os.environ.get("ESPNET_DIR", "/kaggle/working/espnet_side"))
import numpy as np, torch
from jiwer import process_words
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
ROOT = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
spec = importlib.util.spec_from_file_location("w2v", ROOT / "training/ctc_w2vbert/w2vbert_train_v6.py"); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
t_ = types.ModuleType("typer"); t_.Typer = lambda *a, **k: None; t_.Option = t_.Argument = lambda *a, **k: None; t_.run = lambda *a, **k: None
sys.modules.setdefault("typer", t_)
spec = importlib.util.spec_from_file_location("competition_score", ROOT / "runtime_repo/score.py"); score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
NORM = lambda s: " ".join(score.normalize_text(str(s)).split()).casefold()
EXP = ROOT / "training/owsm_ctc/exp_v6mix"
CK = EXP                                            # /kaggle/working has only 18 GB free: 4 GB checkpoints stay on the big share
_DEV_AUDIO = {}
SNAP = glob.glob(str(ROOT / "models/hf_home/hub/models--espnet--owsm_ctc_v4_1B/snapshots/*"))[0]
CFG = glob.glob(SNAP + "/exp/*/config.yaml")[0]; PTH = glob.glob(SNAP + "/exp/*/*.pth")[0]
SR = 16000; WIN = 30 * SR
log = W.log


def build(device):
    os.chdir(SNAP)                                  # the config refers to its stats file and bpe model by relative path
    from espnet2.tasks.s2t_ctc import S2TTask
    from espnet2.text.build_tokenizer import build_tokenizer
    from espnet2.text.token_id_converter import TokenIDConverter
    model, targs = S2TTask.build_model_from_file(CFG, PTH, device)
    for _, m in model.named_modules():
        if hasattr(m, "use_flash_attn"): m.use_flash_attn = False
    tok = build_tokenizer(token_type="bpe", bpemodel=targs.bpemodel); conv = TokenIDConverter(token_list=model.token_list)
    return model, tok, conv


class Rows(Dataset):
    def __init__(self, rows, tok, conv): self.rows, self.tok, self.conv = rows, tok, conv
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        try: a = W.load_audio(r["audio_filepath"])[:WIN]
        except Exception: a = np.zeros(SR, np.float32)
        ids = self.conv.tokens2ids(self.tok.text2tokens(" ".join(str(r.get("text_ref1", "")).split())))[:180]
        return a, ids


def collate(b):
    wav = torch.zeros(len(b), WIN, dtype=torch.float32)                       # the model's fixed 30 s input
    for i, (a, _) in enumerate(b): wav[i, :len(a)] = torch.from_numpy(a)
    tl = torch.tensor([len(ids) for _, ids in b], dtype=torch.long)
    text = torch.full((len(b), int(tl.max())), -1, dtype=torch.long)
    for i, (_, ids) in enumerate(b): text[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return wav, text, tl


def inputs(model, wav, lang_id, task_id, device):
    B = wav.shape[0]
    return dict(speech=wav.to(device), speech_lengths=torch.full((B,), wav.shape[1], dtype=torch.long, device=device),
                text_prev=torch.full((B, 1), model.na, dtype=torch.long, device=device), text_prev_lengths=torch.ones(B, dtype=torch.long, device=device),
                prefix=torch.tensor([[lang_id, task_id]], dtype=torch.long, device=device).repeat(B, 1), prefix_lengths=torch.full((B,), 2, dtype=torch.long, device=device))


def decode(model, tok, conv, wavs, lang_id, task_id, device, amp):
    """Greedy CTC over 30 s windows (padded), special tokens dropped. wavs: list of 1-D float arrays (<= 30 s each)."""
    x = torch.zeros(len(wavs), WIN, dtype=torch.float32)
    for i, w in enumerate(wavs): x[i, :len(w)] = torch.from_numpy(w)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
        enc, _ = model.encode(**inputs(model, x, lang_id, task_id, device))
        if isinstance(enc, tuple): enc = enc[0]
        ids = model.ctc.argmax(enc.float())
    out = []
    for row in ids:
        t = [int(v) for v in torch.unique_consecutive(row).cpu().tolist() if int(v) not in (model.blank_id, model.sos, model.eos)]
        toks = [z for z in conv.ids2tokens(t) if not (z.startswith("<") and z.endswith(">"))]
        out.append(" ".join(tok.tokens2text(toks).split()))
    return out


def validate(model, tok, conv, rows, step, lang_id, task_id, device, amp, batch=8, limit=None):
    model.eval(); t0 = time.time(); hyps = []
    pend, owner = [], []
    for i, r in enumerate(rows[:limit] if limit else rows):
        if i not in _DEV_AUDIO: _DEV_AUDIO[i] = W.load_audio(r["audio_filepath"])      # decoded once, reused by every validation
        wav = _DEV_AUDIO[i]
        for k in range(0, len(wav), WIN):
            w = wav[k:k + WIN]
            if len(w) >= SR // 2 or k == 0: pend.append(w); owner.append(i)
    texts = []
    for b in range(0, len(pend), batch): texts += decode(model, tok, conv, pend[b:b + batch], lang_id, task_id, device, amp)
    n_rows = len(rows[:limit] if limit else rows); parts = [[] for _ in range(n_rows)]
    for o, t in zip(owner, texts):
        if t: parts[o].append(t)
    hyps = [" ".join(p) for p in parts]
    e = n = 0
    for r, h in zip(rows, hyps):
        ref = NORM(r.get("text_ref1", ""))
        if not ref: continue
        o = process_words(ref, NORM(h) or "_"); e += o.substitutions + o.deletions + o.insertions; n += len(ref.split())
    w = e / max(n, 1)
    log(f"[val:dev] wer={w:.4f} n={n_rows} seconds={time.time()-t0:.1f} step={step}   e.g. {hyps[0][:110]}")
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / f"dev_step{step}.csv").open("w", encoding="utf8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["audio_filename", "transcript"]); wr.writeheader()
        for r, h in zip(rows, hyps): wr.writerow({"audio_filename": r.get("uid", Path(r["audio_filepath"]).name), "transcript": h})
    gc.collect(); torch.cuda.empty_cache(); model.train()
    return w


def make_optimizer(model, lr, decay):
    n_layers = len(model.encoder.encoders); groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        m = re.match(r"encoder\.encoders\.(\d+)\.", n)
        if m: key, l = f"enc{int(m.group(1)):02d}", lr * decay ** (n_layers - 1 - int(m.group(1)))
        elif n.startswith(("encoder.embed", "frontend", "normalize")): key, l = "front", lr * decay ** n_layers
        else: key, l = "top", lr                     # ctc heads, conditioning layers, prompt encoder, token embeddings
        groups.setdefault((key, l), []).append(p)
    opt = torch.optim.AdamW([{"params": v, "lr": l, "base_lr": l, "weight_decay": .01} for (k, l), v in groups.items()], betas=(.9, .98), eps=1e-8)
    ls = [l for (_, l) in groups]; log(f"{n_layers} encoder layers; LR range {min(ls):.2e}..{max(ls):.2e} over {len(groups)} groups; top-group tensors {len(groups[('top', lr)])}")
    return opt


def set_schedule(opt, step, max_steps, warmup, min_ratio=.05):
    if step < warmup: ratio = max(1e-3, (step + 1) / warmup)
    else:
        prog = min(1., (step - warmup) / max(1, max_steps - warmup)); ratio = min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups: g["lr"] = g["base_lr"] * ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-utc", default="21:00"); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--val-every", type=int, default=750)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--lr", type=float, default=1.5e-5); ap.add_argument("--layer-decay", type=float, default=0.97)
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--max-duration", type=float, default=30.0); ap.add_argument("--lang", default="jav")
    ap.add_argument("--amp", type=int, default=1); ap.add_argument("--seed", type=int, default=2033); ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=""); ap.add_argument("--start-step", type=int, default=0); ap.add_argument("--no-val", type=int, default=0)
    ap.add_argument("--val0", type=int, default=1, help="validate the untouched model first (zero-shot reference)"); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); EXP.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True)
    dev_ = a.device; amp = bool(a.amp) and dev_ == "cuda"
    model, tok, conv = build(dev_)
    lang_id, task_id = conv.token2id[f"<{a.lang}>"], conv.token2id["<asr>"]
    log(f"model {sum(p.numel() for p in model.parameters())/1e6:.0f} M params; children {[n for n, _ in model.named_children()]}; lang <{a.lang}>={lang_id} asr={task_id} na={model.na} blank={model.blank_id}; "
        f"interctc_weight={model.interctc_weight} ctc_asr_only={model.ctc_asr_only}; amp={amp}")
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); log(f"resumed from {a.resume} (step {ck.get('step')})"); del ck
    pool = [r for r in W.read_jsonl(W.POOL) if .5 <= float(r.get("duration", 0)) <= a.max_duration and str(r.get("text_ref1", "")).strip()]
    dev = [r for r in W.read_jsonl(W.DEV) if float(r.get("duration", 0)) <= 40]
    counts = Counter(r["source"] for r in pool)
    weights = torch.tensor([W.SOURCE_WEIGHTS.get(r["source"], 0.0) / counts[r["source"]] for r in pool], dtype=torch.double)
    log(f"data train={len(pool)} dev={len(dev)}")
    opt = make_optimizer(model, a.lr, a.layer_decay)
    bs = 2 if a.smoke else a.batch_size
    sampler = WeightedRandomSampler(weights, num_samples=40000 * bs, replacement=True, generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(Rows(pool, tok, conv), batch_size=bs, sampler=sampler, num_workers=0 if a.smoke else 3, collate_fn=collate, pin_memory=False,
                        persistent_workers=not a.smoke)
    it = iter(loader)
    if a.smoke:
        model.train(); wav, text, tl = next(it); t0 = time.time()
        loss, stats, _ = model(**inputs(model, wav, lang_id, task_id, dev_), text=text.to(dev_), text_lengths=tl.to(dev_), text_ctc=text.clone().to(dev_), text_ctc_lengths=tl.to(dev_))
        loss.backward(); log(f"SMOKE: loss {loss.item():.4f} finite {bool(torch.isfinite(loss))} in {time.time()-t0:.1f}s; stats keys {list(stats)[:6]}; target lens {tl.tolist()}")
        validate(model, tok, conv, dev, 0, lang_id, task_id, dev_, amp, batch=2, limit=2); log("SMOKE_DONE"); return
    best = 99.
    if a.val0 and not a.start_step:
        best = validate(model, tok, conv, dev, 0, lang_id, task_id, dev_, amp); log(f"[ckpt] step=0 dev={best:.4f} (zero-shot, <{a.lang}>)")
    hh, mm = map(int, a.end_utc.split(":")); end = datetime.datetime.utcnow().replace(hour=hh, minute=mm, second=0, microsecond=0)
    max_steps = a.max_steps or 4000
    model.train(); opt.zero_grad(set_to_none=True); running = 0.; bad = 0; step = a.start_step; s0 = step; t_run = time.time()
    while step < max_steps:
        step += 1
        set_schedule(opt, step, max_steps, a.warmup)
        wav, text, tl = next(it)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, stats, _ = model(**inputs(model, wav, lang_id, task_id, dev_), text=text.to(dev_), text_lengths=tl.to(dev_), text_ctc=text.clone().to(dev_), text_ctc_lengths=tl.to(dev_))
        except RuntimeError as ex:
            if amp and step - s0 <= 3:
                log(f"bf16 autocast failed at step {step} ({repr(ex)[:160]}) -> falling back to fp32"); amp = False; opt.zero_grad(set_to_none=True); step -= 1; continue
            raise
        if not torch.isfinite(loss):
            bad += 1; opt.zero_grad(set_to_none=True)
            if amp and bad >= 3 and step - s0 <= 20: log("non-finite losses under bf16 -> fp32"); amp = False
            continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        running += loss.item()
        if step % 25 == 0:
            log(f"step {step:5d}/{max_steps} loss={running/25:.4f} lr={max(g['lr'] for g in opt.param_groups):.2e} sec_step={(time.time()-t_run)/max(step-s0,1):.3f} "
                f"vram={torch.cuda.max_memory_allocated()/1e9:.1f}G bad={bad} amp={amp}"); running = 0.
        if step - s0 == 100 and not a.max_steps:
            sps = (time.time() - t_run) / 100; left = (end - datetime.datetime.utcnow()).total_seconds()
            n_ev = max(1, int(left / sps) // a.val_every + 1); max_steps = max(step + 200, step + int((left - n_ev * 150 - 60) / sps)); log(f"sized: {sps:.3f} s/step, {left:.0f} s left -> max_steps {max_steps}")
        if step % a.val_every == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "step": step}, CK / "last.pt")       # before validating: a crash there must not cost the run
            if a.no_val:
                os.replace(CK / "last.pt", CK / f"step{step}.pt"); log(f"[ckpt] saved step{step}.pt (validation runs in a separate process)"); continue
            w = validate(model, tok, conv, dev, step, lang_id, task_id, dev_, amp)
            if w < best:
                best = w; os.replace(CK / "last.pt", CK / "best.pt"); (EXP / "best.json").write_text(json.dumps({"step": step, "dev_wer": w}))
            log(f"[ckpt] step={step} dev={w:.4f} best={best:.4f}")
    log("training complete")


if __name__ == "__main__":
    main()
