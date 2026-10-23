#!/bin/bash
# Create the training environments under <repo>/.venvs (the run.sh scripts look there by default).
# Needs: Linux x86_64, NVIDIA driver for CUDA 12.8 and 13.0, python3.10 (3.10.12 used), python3-venv, ffmpeg.
#   sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv ffmpeg git curl
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); E=$REPO/environment; V=$REPO/.venvs
mkdir -p $V
# train: data preparation, Whisper, MERaLiON, w2v-BERT CTC, SeamlessM4T, OWSM, exports (torch 2.8 cu128, transformers 4.57.6)
python3.10 -m venv $V/train
$V/train/bin/pip install -r $E/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128
# ESPnet for OWSM, on the side (no dependency resolution, so the train environment's torch stays untouched)
$V/train/bin/pip install --no-deps --target $V/espnet_side -r $E/requirements-espnet-side.txt
# qwen: Qwen3-ASR and VibeVoice (torch 2.13, transformers 5.15.1, peft 0.20)
python3.10 -m venv $V/qwen
$V/qwen/bin/pip install -r $E/requirements-qwen.txt --extra-index-url https://download.pytorch.org/whl/cu130
# omniasr: omniASR-LLM fine-tune (fairseq2 0.6 + omnilingual-asr 0.2.0, torch 2.8 cu128)
python3.10 -m venv $V/omniasr
$V/omniasr/bin/pip install -r $E/requirements-omniasr.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
echo "environments ready in $V"
