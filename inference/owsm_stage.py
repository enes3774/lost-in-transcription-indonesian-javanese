"""OWSM-CTC v4 1B (espnet/owsm_ctc_v4_1B, fine-tuned on this track's pool) as a candidate source, run by main.py as a
SEPARATE PROCESS alongside the other stages, like the Seamless stage. The runner has no espnet: main.py starts this
script with PYTHONPATH pointing at the vendored pure-python bundle (vendor_espnet/); only the model's own building
blocks are imported (mel frontend, global MVN, E-Branchformer CTC encoder, prompt transformer, CTC) - no espnet
task/trainer code, no yaml.

Model dir: model.pt = {confs, token_list, state (bf16)}, feats_stats.npz (global MVN), bpe.model (sentencepiece).
Decode: 30 s windows padded to the model's fixed 30 s input, prefix <lang><asr>, greedy CTC, special tokens dropped.
Results are written after every batch. No exception is caught: a failure exits non-zero and main.py raises on it.
Usage: owsm_stage.py <model dir> <pieces json: [[path, duration], ...]> <out json> [batch] [lang=jav]"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "vendor_espnet").is_dir() and str(HERE / "vendor_espnet") not in sys.path:
    sys.path.insert(0, str(HERE / "vendor_espnet"))
SR, WINDOW_S = 16000, 30.0


def build(model_dir, torch):
    from espnet2.asr.frontend.default import DefaultFrontend
    from espnet2.layers.global_mvn import GlobalMVN
    from espnet2.asr.encoder.e_branchformer_ctc_encoder import EBranchformerCTCEncoder
    from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
    from espnet2.asr.ctc import CTC
    from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
    import sentencepiece as spm
    pkg = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    c = pkg["confs"]; tl = pkg["token_list"]; V = len(tl)
    net = torch.nn.Module()
    net.frontend = DefaultFrontend(**c["frontend_conf"])
    net.normalize = GlobalMVN(stats_file=os.path.join(model_dir, "feats_stats.npz"))
    ec = dict(c["encoder_conf"]); ec["use_flash_attn"] = False
    net.encoder = EBranchformerCTCEncoder(input_size=net.frontend.output_size(), **ec)
    pc = dict(c["promptencoder_conf"]); pc["use_flash_attn"] = False
    net.prompt_encoder = TransformerEncoder(input_size=pc["output_size"], input_layer=None, **pc)
    D = net.prompt_encoder.output_size(); E = net.encoder.output_size()
    net.embed = torch.nn.Embedding(V, D); net.pos_enc = PositionalEncoding(D, 0.0)
    net.embed_proj = torch.nn.Linear(D, E) if D != E else torch.nn.Identity()
    net.prompt_proj = torch.nn.Linear(D, E) if D != E else torch.nn.Identity()
    if getattr(net.encoder, "interctc_use_conditioning", False):
        net.encoder.conditioning_layer = torch.nn.Linear(V, E)
    net.ctc = CTC(odim=V, encoder_output_size=E, **c["ctc_conf"])
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in pkg["state"].items()}
    res = net.load_state_dict(state, strict=False)
    bad = [k for k in res.missing_keys if not k.endswith(".pe")] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"state dict mismatch: {bad[:10]} (+{max(0, len(bad) - 10)})")
    for m in net.modules():
        if hasattr(m, "use_flash_attn"):
            m.use_flash_attn = False
    net = net.cuda().eval()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "bpe.model"))
    ids = dict(blank=tl.index("<blank>"), sos=tl.index("<sos>"), eos=tl.index("<eos>"), na=tl.index("<na>"), asr=tl.index("<asr>"))
    return net, tl, sp, ids


def encode(net, wav, lens, prefix, na, torch):
    from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
    B = wav.shape[0]
    tp = torch.full((B, 1), na, dtype=torch.long, device=wav.device); tpl = torch.ones(B, dtype=torch.long, device=wav.device)
    with torch.autocast("cuda", enabled=False):
        feats, fl = net.frontend(wav, lens); feats, fl = net.normalize(feats, fl)
    memory, ml, _ = net.prompt_encoder(net.pos_enc(net.embed(tp)), tpl)
    mmask = (~make_pad_mask(ml)[:, None, :]).to(memory.device)
    pre = torch.tensor(prefix, dtype=torch.long, device=wav.device).repeat(B, 1)
    enc, el, _ = net.encoder(feats, fl, ctc=net.ctc, prefix_embeds=net.embed_proj(net.embed(pre)), memory=net.prompt_proj(memory), memory_mask=mmask)
    if isinstance(enc, tuple):
        enc = enc[0]
    return net.ctc.argmax(enc.float()), el


def to_text(row, n, tl, sp, ids):
    import torch
    t = [int(v) for v in torch.unique_consecutive(row[:n]).tolist() if int(v) not in (ids["blank"], ids["sos"], ids["eos"])]
    toks = [tl[i] for i in t if not (tl[i].startswith("<") and tl[i].endswith(">"))]
    return " ".join(sp.decode_pieces(toks).split())


def main():
    import librosa, numpy as np, torch
    model_dir, pieces_json, out_json = sys.argv[1:4]
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    lang = sys.argv[5] if len(sys.argv) > 5 else "jav"
    pieces = json.load(open(pieces_json, encoding="utf-8"))
    t0 = time.time()
    net, tl, sp, ids = build(model_dir, torch)
    prefix = [tl.index(f"<{lang}>"), ids["asr"]]
    print(f"owsm: model loaded in {time.time()-t0:.0f}s; {len(pieces)} pieces; torch {torch.__version__}; lang <{lang}>", flush=True)
    win = int(WINDOW_S * SR); work = []
    # audio decoding is the stage's real cost (single-threaded it was ~90 % of the wall time), so it runs on a small pool
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        waves = list(pool.map(lambda pd: librosa.load(pd[0], sr=SR, mono=True)[0].astype(np.float32), pieces))
    for i, wav in enumerate(waves):
        chunks = [wav] if len(wav) <= win else [wav[k:k + win] for k in range(0, len(wav), win)]
        for j, c in enumerate(chunks):
            work.append((i, j, c))
    del waves
    work.sort(key=lambda w: -len(w[2]))
    print(f"owsm: audio loaded in {time.time()-t0:.0f}s; {len(work)} windows", flush=True)
    got, t1 = {}, time.time()
    for b in range(0, len(work), batch):
        part = work[b:b + batch]
        wav = torch.zeros(len(part), win, dtype=torch.float32)             # the model's fixed 30 s input
        for r, (_, _, c) in enumerate(part):
            wav[r, :len(c)] = torch.from_numpy(c)
        lens = torch.full((len(part),), win, dtype=torch.long)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            rows, el = encode(net, wav.cuda(), lens.cuda(), prefix, ids["na"], torch)
        rows, el = rows.cpu(), el.cpu()
        for r, (i, j, _) in enumerate(part):
            got.setdefault(str(i), {})[str(j)] = to_text(rows[r], int(el[r]), tl, sp, ids)
        json.dump(got, open(out_json + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(out_json + ".tmp", out_json)
        if (b // batch) % 25 == 0:
            print(f"owsm: {min(b + batch, len(work))}/{len(work)} windows, {time.time()-t1:.0f}s", flush=True)
    final = {k: " ".join(v[j] for j in sorted(v, key=int)).strip() for k, v in got.items()}
    if len(final) != len(pieces):
        raise RuntimeError(f"owsm output lacks {len(pieces) - len(final)} of {len(pieces)} pieces")
    json.dump(final, open(out_json + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(out_json + ".tmp", out_json)
    print(f"owsm: done, {len(final)} pieces, decode {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s, "
          f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    print("OWSM_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
