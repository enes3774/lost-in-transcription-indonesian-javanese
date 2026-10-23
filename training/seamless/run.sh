#!/usr/bin/env bash
# SeamlessM4T-v2 large (candidate source): full fine-tune, fp32 master weights + bf16 autocast, 36,000 clips drawn with
# the v6 mix, lr 2e-5, batch 4 x 2, 4,500 steps; the best dev checkpoint (step 3000 in our run) is saved to best/.
# Base: facebook/seamless-m4t-v2-large in $R/models/candidates/seamless-m4t-v2-large. ~2.5 h. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
T=$R/training/seamless_ft; mkdir -p $T; cp $HERE/seamless_ft_jv.py $T/
$P $T/seamless_ft_jv.py --run v6_36k --mix v6mix --pool v4_clean --n_train 36000 --eval_every 1500 > $R/logs/seamless_ft_v6_36k.log 2>&1
rm -rf $OUT/seamless; cp -r $T/v6_36k/best $OUT/seamless
echo SEAMLESS_DONE
