"""Slim OWSM-CTC package dir for owsm_stage.py: model.pt (confs + token list + bf16 state), feats_stats.npz, bpe.model.
usage: owsm_export.py <out dir> [trainer checkpoint]   (no checkpoint = the raw pretrained weights)"""
import os
import glob, shutil, sys
from pathlib import Path
import torch, yaml
ROOT = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
SNAP = glob.glob(str(ROOT / "models/hf_home/hub/models--espnet--owsm_ctc_v4_1B/snapshots/*"))[0]
CFG = glob.glob(SNAP + "/exp/*/config.yaml")[0]; PTH = glob.glob(SNAP + "/exp/*/*.pth")[0]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
c = yaml.safe_load(open(CFG))
if len(sys.argv) > 2:
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False); state = ck["model"]; src = f"{sys.argv[2]} step {ck.get('step')}"
else:
    state = torch.load(PTH, map_location="cpu", weights_only=False); src = "raw pretrained"
state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items() if not k.startswith("specaug")}
confs = {k: c[k] for k in ("frontend_conf", "normalize_conf", "encoder_conf", "promptencoder_conf", "ctc_conf", "model_conf")}
torch.save({"confs": confs, "token_list": list(c["token_list"]), "state": state, "source": src}, out / "model.pt")
shutil.copy(SNAP + "/" + c["normalize_conf"]["stats_file"], out / "feats_stats.npz"); shutil.copy(SNAP + "/" + c["bpemodel"], out / "bpe.model")
print("wrote", out, f"model.pt {(out / 'model.pt').stat().st_size/1e9:.2f} GB, {len(state)} tensors;", src)
