"""Base omniASR-LLM checkpoint + the fine-tune's trainable tensors (train_omni_jv.py best.pt/stepN.pt) -> the
port's converted bf16 file. Usage: omni_overlay.py <base .pt> <finetune .pt> <out .pt>"""
import sys, time, torch
sys.path.insert(0, "/kaggle/working")
from omni_asr import load_omni_asr, save_converted
base, ft, out = sys.argv[1:4]
t = time.time()
model = load_omni_asr(base, device="cpu", dtype=torch.bfloat16)
obj = torch.load(ft, map_location="cpu", weights_only=False)
sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
assert not res.unexpected_keys, res.unexpected_keys[:5]
own = set(model.state_dict().keys())
print(f"overlaid {len(sd)} tensors ({sum(v.numel() for v in sd.values())/1e9:.2f} B params); missing (frozen, kept from base): {len(res.missing_keys)}; base keys {len(own)}", flush=True)
save_converted(model, out)
import os; print(f"wrote {out} {os.path.getsize(out)/1e9:.2f} GB in {time.time()-t:.0f}s")
