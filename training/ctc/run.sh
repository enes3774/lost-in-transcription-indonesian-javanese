#!/usr/bin/env bash
# w2v-BERT 2.0 + CTC head (voter): full fine-tune on the v2 pool, 6,000 steps. One GPU, ~1 h 45 min.
# Base: facebook/w2v-bert-2.0 in $R/models/checkpoints/w2v-bert-2.0. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
C=$R/training/ctc_w2vbert; mkdir -p $C; cp $HERE/w2vbert_train.py $C/
cd $R
# encoder LR 5e-5 with x0.95 per-layer decay, head 2e-4, batch 8 x 2, clips 0.5-20 s; best.pt = lowest dev WER (step 6000)
$P $C/w2vbert_train.py --max-steps 6000 --val-every 1500 --batch-size 8 --accum 2 > $R/logs/w2vbert_ctc.log 2>&1
$P $HERE/export.py $C/exp_full/best.pt $OUT/ctc
echo CTC_DONE
