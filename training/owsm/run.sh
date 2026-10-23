#!/usr/bin/env bash
# OWSM-CTC v4 1B (candidate source): full fine-tune on the v6 mix (weighted sampler over train_pool_v4_clean.jsonl),
# batch 8, lr 1.5e-5 with 0.97 layer decay, CTC + 0.8 intermediate CTC, 3,800 steps. Base: espnet/owsm_ctc_v4_1B in
# $R/models/hf_home (download_base_models.sh). ~50 min. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
export ESPNET_DIR=${ESPNET_DIR:-$REPO/.venvs/espnet_side}  # ESPnet 202511 (environment/requirements-espnet-side.txt); put on sys.path by the trainer
O=$R/training/owsm_ctc; mkdir -p $O $R/training/ctc_w2vbert
cp $HERE/owsm_ctc_train.py $HERE/owsm_ctc_eval.py $O/; cp $HERE/w2vbert_train_v6.py $R/training/ctc_w2vbert/   # pool, dev set and mix weights
export PYTHONUNBUFFERED=1 HF_HOME=$R/models/hf_home
cd $R
# Validation runs out of process (in-process validation crashed this machine): every step*.pt is scored by owsm_ctc_eval.py.
# Our run was interrupted after the step-750 checkpoint and resumed from it (--resume ... --start-step 750), which
# restarts the optimizer state and the seeded sampler; running straight through is the same recipe otherwise.
$P $O/owsm_ctc_train.py --max-steps 3800 --val0 0 --no-val 1 > $R/logs/owsm_ctc_v6mix.log 2>&1
for f in $(ls $O/exp_v6mix/step*.pt | sort -V); do $P $O/owsm_ctc_eval.py $f >> $R/logs/owsm_ctc_v6mix.log 2>&1; done
# step 3800 (the last) had the best dev WER. Export: bf16 state dict + configs + token list, feats_stats.npz, bpe.model.
$P $HERE/owsm_export.py $OUT/owsm $O/exp_v6mix/step3800.pt
echo OWSM_DONE
