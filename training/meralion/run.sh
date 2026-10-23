#!/usr/bin/env bash
# MERaLiON-3-3B-ASR (voter): LoRA + encoder fine-tune on the v5 mix, then a dev-adaptation LoRA, then export.
# Base: MERaLiON/MERaLiON-3-3B-ASR (revision 9f226d73) in $R/models/candidates/meralion-3-3b-asr. ~3 h. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
T=$R/training/meralion_v1; mkdir -p $T $R/training/qwen_v2 $R/scripts
cp $HERE/meralion_train.py $HERE/export_merged.py $HERE/*.yaml $T/
cp $HERE/lit_metrics.py $R/training/qwen_v2/; cp $HERE/newmodels_common.py $R/scripts/
fix_paths $T/*.yaml
cd $R

# 1. v5 mix: LoRA r32 on all 26 decoder layers (3e-5), Whisper encoder unfrozen (1e-5), adapter 1e-5; 36,000 weighted
#    samples, batch 4, speed + noise + SpecAugment; robust_val selects, dev is only watched.
$P $T/meralion_train.py --config $T/meralion_v5mix.yaml > $R/logs/meralion_v5mix.log 2>&1
LAST=$(ls $T/exp_v5mix/step*.ckpt | sort -V | tail -1)          # the epoch ends at the last saved step (8982 in our run)

# 2. dev adaptation: LoRA on decoder layers 13-25 only (8e-6), dev weight .45 + pool replay, 413 steps, initialised
#    from step 1's checkpoint (style_alldev_v5mix.yaml is style_alldev_dec_final.yaml re-pointed at it).
sed -e 's#^name: .*#name: style_alldev_v5mix#' -e 's#exp_style_alldev_dec_final#exp_style_alldev_v5mix#'     -e "s#init_checkpoint: .*#init_checkpoint: $LAST#" $T/style_alldev_dec_final.yaml > $T/style_alldev_v5mix.yaml
$P $T/meralion_train.py --config $T/style_alldev_v5mix.yaml > $R/logs/meralion_devpass.log 2>&1
MCK=$(ls $T/exp_style_alldev_v5mix/step*.ckpt | sort -V | tail -1)

# 3. export (bf16 safetensors + processor + remote code). export_merged.py builds the base model with empty LoRA and
#    loads this one checkpoint, i.e. the base model + the dev-pass LoRA of layers 13-25 merged. That is exactly the
#    shipped model; see training/README.md.
$P $T/export_merged.py --config $T/style_alldev_v5mix.yaml --ckpt $MCK --out $OUT/meralion
echo MERALION_DONE
