#!/usr/bin/env bash
# VibeVoice-ASR (candidate source): LoRA r16 on the language model's attention, 12,000 clips drawn with the v5 mix,
# 3,000 steps; merged into the weights in bf16. Base: microsoft/VibeVoice-ASR-HF in $R/models/candidates/vibevoice-asr-hf.
# ~2 h. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
PQ=${PQ:-$REPO/.venvs/qwen/bin/python}                    # transformers 5.15.1 + peft 0.20 (environment/requirements-qwen.txt)
V=$R/training/vibevoice_lora; mkdir -p $V; cp $HERE/vibevoice_lora_train_jv.py $V/
cd $V
# lr 1e-4, batch 1 x 4, 10 % warmup then linear decay; pool train_pool_v4.jsonl, mix v5mix (the defaults); step 3000 used
$PQ vibevoice_lora_train_jv.py --run v5_r16 --n_train 12000 --eval_every 1500 > $R/logs/vibevoice_v5_r16.log 2>&1
$PQ $HERE/vv_merge.py > $R/logs/vv_merge.log 2>&1        # -> $R/models/checkpoints/vibevoice-asr-v5ft
M=$R/models/checkpoints/vibevoice-asr-v5ft
for f in processor_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do cp $R/models/candidates/vibevoice-asr-hf/$f $M/; done   # save_pretrained drops the audio settings
rm -rf $OUT/vibevoice; cp -r $M $OUT/vibevoice
echo VIBEVOICE_DONE
