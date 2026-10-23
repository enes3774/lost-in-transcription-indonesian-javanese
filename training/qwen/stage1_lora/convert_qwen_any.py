#!/usr/bin/env python3
"""Freeze any qwen_train.py checkpoint into a self-contained HF model dir.

    /kaggle/working/venv_qwen/bin/python sapc/convert_qwen_any.py \
        --base /kaggle/working/models/Qwen3-ASR-1.7B-hf \
        --ckpt /kaggle/working/exp_fold0_qwen/best.ckpt \
        --out  /kaggle/working/models/qwen_fold0_hf

convert_qwen.py assumes an unwrapped full fine-tune and asserts on unexpected
keys. That is true of exp_qwenfull but not of a LoRA run, whose state dict is
PEFT-wrapped: every key carries a `base_model.model.` prefix and half of them
are lora_A/lora_B. Loading those into a bare model fails on 392 unexpected keys.

qwen_train.py's LoRA is partial in a way that matters here: adapters go on the
LANGUAGE model only, matched by a fully-qualified regex, while the audio tower
and the projector are full fine-tuned. So merging needs the same LoraConfig
reconstructed -- the checkpoint carries it under config.freeze.lora -- and then
merge_and_unload() to fold the adapters into the base weights.

Reads the training config out of the checkpoint rather than taking it on the
command line, so the merge cannot silently disagree with how the model was
actually trained.
"""
import argparse
import json
import os
import shutil

_C = os.environ.setdefault("SAPC_CACHE_ROOT", "/kaggle/working/cache")
os.environ.setdefault("HF_HOME", f"{_C}/hf")
for _v in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_v] = f"{_C}/tmp"

import torch  # noqa: E402
from transformers import (Qwen3ASRForConditionalGeneration,  # noqa: E402
                          Qwen3ASRProcessor)

# Must match qwen_train.py exactly or the adapters land on different modules.
LORA_PATTERN = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/kaggle/working/models/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["trainable"]
    cfg = ck.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    lora = (((cfg.get("freeze") or {}).get("lora")) or {})
    wrapped = any(k.startswith("base_model.model.") for k in state)
    print(f"ckpt step={ck.get('step')} epoch={ck.get('epoch')} "
          f"metrics={ck.get('metrics')}", flush=True)
    print(f"{len(state)} tensors, lora_enabled={bool(lora.get('enabled'))}, "
          f"peft_wrapped={wrapped}", flush=True)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa")

    if wrapped or lora.get("enabled"):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=int(lora.get("r", 32)),
            lora_alpha=int(lora.get("alpha", 64)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none",
            target_modules=LORA_PATTERN,
        ))
        res = model.load_state_dict(state, strict=False)
        if res.unexpected_keys:
            raise SystemExit(f"unexpected keys after PEFT wrap: "
                             f"{res.unexpected_keys[:5]}")
        print(f"loaded; {len(res.missing_keys)} missing (frozen base weights "
              "come from --base, so this is expected)", flush=True)
        model = model.merge_and_unload()
        print("adapters merged into the base weights", flush=True)
    else:
        res = model.load_state_dict(state, strict=False)
        if res.unexpected_keys:
            raise SystemExit(f"unexpected keys: {res.unexpected_keys[:5]}")
        print(f"loaded full fine-tune; missing: {res.missing_keys}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    Qwen3ASRProcessor.from_pretrained(args.base).save_pretrained(args.out)
    for f in os.listdir(args.base):
        if (f.endswith((".json", ".txt", ".model"))
                and not os.path.exists(os.path.join(args.out, f))):
            shutil.copyfile(os.path.join(args.base, f),
                            os.path.join(args.out, f))
    tot = sum(os.path.getsize(os.path.join(args.out, f))
              for f in os.listdir(args.out))
    print(f"\nwrote {args.out}  {tot/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
