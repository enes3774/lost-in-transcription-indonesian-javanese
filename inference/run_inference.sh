#!/bin/bash
# Run the final pipeline on one data directory and write the submission CSV.
#
#   bash run_inference.sh <work_dir> <out_csv> [gpu]
#
# <work_dir>/data must hold clips/ and test_metadata.csv (prepare_data.py builds it from a folder of audio).
# Needs one GPU with >= 60 GB (the competition runner was an A100 80 GB), the competition runtime environment
# (README.md, Setup) and the folders in inference/models/ (download_weights.py). The full test set took
# about 110 of the 120 minutes allowed on the A100. The run is strict: if any stage fails or misses a clip, no
# output is written and the exit code is 1.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "$1" && pwd); OUT=$2; GPU=${3:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export CODE_EXECUTION_ROOT="$WORK" LIT_OUT_CSV="$OUT"
mkdir -p "$(dirname "$OUT")"
cd "$HERE"
python -u main.py
