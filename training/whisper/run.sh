#!/usr/bin/env bash
# Whisper large-v3 (anchor): full fine-tune on the v6 mix, then one pass over the dev set. One GPU, ~2 h + 10 min.
# Base: openai/whisper-large-v3 in $R/models/checkpoints/whisper-large-v3. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
PW=${PW:-$P}                                                # step A ran in an env with the same torch/transformers (environment/requirements-whisper-stepA.txt)
W=$R/training/whisper_v3_lora; mkdir -p $W; cp $HERE/*.py $W/
cd $W

# A. Full fine-tune, all parameters, layer-wise LRs 3e-7..8e-6, batch 2 x 4, 3,000 steps, speed (0.9-1.1) and noise
#    (10-30 dB) augmentation. OFFLINE_VAL=1: checkpoints at 1000/2000/3000 are saved without in-process validation
#    and scored by eval_whisper_ckpt.py in its own process (same validate() as the trainer). Step 3000 was best on
#    every measure and is the one used.
OFFLINE_VAL=1 $PW whisper_full_train.py --mix v6mix --pool $R/data/processed/qwen_v2/train_pool_v4.jsonl   --exp-name v6mix_aug --augment --seed 2050 --max-steps 3000 --val-every 1000 > $R/logs/whisper_v6mix_aug.log 2>&1
$PW eval_whisper_ckpt.py $W/exp_full_v6mix_aug 1000 2000 3000 > $R/logs/whisper_v6mix_eval.log 2>&1

# B. One pass over all dev clips (dev weight .8, pool replay .2 with the "qwen" mix; LRs x0.5; 53 steps of 8 clips).
$P whisper_style_train.py --train-manifest $R/data/processed/qwen_style_aggressive/all_dev_train.jsonl   --exp-name v6mix_full_dev_p1 --init $W/exp_full_v6mix_aug/step3000 --passes 1 > $R/logs/whisper_v6mix_devpass.log 2>&1

# C. Inference folder = weights/configs of B + tokenizer and processor files of the base model.
O=$OUT/whisper; rm -rf $O; mkdir -p $O
for f in preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json special_tokens_map.json normalizer.json; do
  cp $R/models/checkpoints/whisper-large-v3/$f $O/; done
cp $W/exp_style_v6mix_full_dev_p1/best_model/* $O/
echo WHISPER_DONE
