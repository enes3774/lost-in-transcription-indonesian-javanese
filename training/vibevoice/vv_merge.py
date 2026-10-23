"""Fold the v5mix LoRA (step 3000) into VibeVoice-ASR's weights and save a plain HF model dir for shipping.
Sanity: decode 3 dev clips before and after merging; they must match."""
import os
import json, sys, time
from pathlib import Path
import librosa, pandas as pd, torch
from peft import PeftModel
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
R = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription"))
S = R / "models/candidates/vibevoice-asr-hf"
A = R / "training/vibevoice_lora/v5_r16/step3000"
OUT = R / "models/checkpoints/vibevoice-asr-v5ft"
proc = AutoProcessor.from_pretrained(str(S))
SR = proc.feature_extractor.sampling_rate
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(str(S), dtype=torch.bfloat16, attn_implementation="eager",
                                                             device_map={"": 0})
pm = PeftModel.from_pretrained(model, str(A)).eval()
dev = pd.read_csv(R / "data/processed/newmodels/dev.csv", keep_default_na=False)
wavs = [librosa.load(dev.path[i], sr=SR, mono=True)[0] for i in range(3)]


def dec(m):
    x = proc.apply_transcription_request(audio=wavs, sampling_rate=SR)
    x = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.cuda()) for k, v in x.items()}
    with torch.inference_mode():
        y = m.generate(**x, max_new_tokens=256, do_sample=False)
    return [str(t) for t in proc.decode(y[:, x["input_ids"].shape[1]:])]


before = dec(pm)
merged = pm.merge_and_unload()
after = dec(merged)
print("identical before/after merge:", before == after, flush=True)
t0 = time.time()
merged.save_pretrained(str(OUT), safe_serialization=True, max_shard_size="4GB")
proc.save_pretrained(str(OUT))
print(f"saved {OUT} in {time.time()-t0:.0f}s:", sorted(p.name for p in OUT.iterdir()), flush=True)
print("VV_MERGE_DONE", flush=True)
