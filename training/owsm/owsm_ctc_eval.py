"""Standalone (out-of-process) dev validation of an OWSM-CTC trainer checkpoint.
usage: CUDA_VISIBLE_DEVICES=<gpu> python owsm_ctc_eval.py <checkpoint> [lang=jav] ; writes EXP/dev_step{N}.csv, logs [val:dev] wer=..."""
import os
import importlib.util, json, sys, torch
spec = importlib.util.spec_from_file_location("ot", os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/training/owsm_ctc/owsm_ctc_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
lang = sys.argv[2] if len(sys.argv) > 2 else "jav"
model, tok, conv = T.build("cuda")
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); step = int(ck.get("step", 0)); model.load_state_dict(ck["model"]); del ck
dev = [r for r in T.W.read_jsonl(T.W.DEV) if float(r.get("duration", 0)) <= 40]
w = T.validate(model, tok, conv, dev, step, conv.token2id["<" + lang + ">"], conv.token2id["<asr>"], "cuda", True)
bj = T.EXP / "best.json"; best = json.loads(bj.read_text())["dev_wer"] if bj.exists() else 99.
if w < best: bj.write_text(json.dumps({"step": step, "dev_wer": w, "ckpt": sys.argv[1]}))
T.log("[ckpt] step=%d dev=%.4f best=%.4f (standalone eval of %s)" % (step, w, min(w, best), sys.argv[1]))
