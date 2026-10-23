#!/bin/bash
# Train everything from the raw data to a complete inference/models/ folder, on one GPU, in order:
#   data preparation (pools + trigram) -> the eight models -> the rescoring LLM (used as released).
#   bash train_all.sh [gpu]          (LIT_ROOT = working folder for data, base models and runs; default ./work)
# Prerequisites: environment/setup_training_envs.sh, MDC_API_TOKEN exported, roughly 300 GB of free disk (about 70 GB of data, 60 GB of base models, the rest checkpoints; an estimate).
# Every step stops the whole script on error (set -e); each training/<model>/run.sh can also be run on its own.
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); GPU=${1:-0}
export LIT_ROOT=${LIT_ROOT:-$REPO/work}
bash $REPO/training/data_prep/run.sh
bash $REPO/training/download_base_models.sh
for m in whisper qwen meralion ctc omni vibevoice seamless owsm; do
  echo "=== $m $(date -u +%FT%TZ)"
  bash $REPO/training/$m/run.sh $GPU
done
$REPO/.venvs/train/bin/hf download GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct --local-dir $REPO/inference/models/llm
echo "TRAIN_ALL_DONE: inference/models/ is complete"
