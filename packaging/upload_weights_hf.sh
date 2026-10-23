#!/bin/bash
# Publish the trained models to one Hugging Face model repo, in the models/<member>/ layout download_weights.py expects.
#   bash upload_weights_hf.sh <staging_dir> <hf_repo_id> <solution_code.zip>
# <staging_dir> is created with real folders holding per-file symlinks to the trained models (nothing is copied).
# Resumable: hf upload-large-folder keeps its state in <staging_dir>/.cache/huggingface and skips finished files.
# Needs HF_TOKEN (write) in the environment.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); REPO_DIR=$(cd "$HERE/.." && pwd)
STAGE=$1; REPO=$2; CODE_ZIP=$3
M=${MODELS:-$REPO_DIR/inference/models}
mkdir -p "$STAGE/models"
for m in whisper qwen meralion ctc omni vibevoice seamless owsm lm; do
  src=$(readlink -f "$M/$m"); mkdir -p "$STAGE/models/$m"
  for f in "$src"/*; do [ -f "$f" ] && ln -sfn "$(readlink -f "$f")" "$STAGE/models/$m/$(basename "$f")"; done
done
cp "$HERE/HF_MODEL_CARD.md" "$STAGE/README.md"
ln -sfn "$(readlink -f "$CODE_ZIP")" "$STAGE/solution_code.zip"
hf repo create "$REPO" --repo-type model --exist-ok
hf upload-large-folder "$REPO" "$STAGE" --repo-type model --num-workers 8
