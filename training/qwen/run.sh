#!/usr/bin/env bash
# Qwen3-ASR 1.7B (voter; anchor above 30 s): LoRA on the v2 pool, then adaptation on all dev clips, merged and written
# in the qwen_asr layout the competition image loads. Bases: Qwen/Qwen3-ASR-1.7B-hf (transformers layout) in $QB and
# Qwen/Qwen3-ASR-1.7B (original layout, for the shipped config/tokenizer files) in $QO. ~2 h 20 min. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
PQ=${PQ:-$REPO/.venvs/qwen/bin/python}                    # transformers 5.15.1 + peft 0.20 (environment/requirements-qwen.txt)
QB=${QB:-$R/models/checkpoints/Qwen3-ASR-1.7B-hf}
QO=${QO:-$R/models/checkpoints/qwen3-asr-1.7b}
export HF_HOME=${HF_HOME:-$R/cache/hf} SAPC_CACHE_ROOT=$R/cache/qwen
D=$R/training/qwen_v2; S=$R/training/qwen_style_aggressive; A=$R/data/processed/qwen_style_aggressive
mkdir -p $D $S; cp $HERE/stage1_lora/* $D/; cp $HERE/stage2_dev/*.py $S/
fix_paths $D/qwen_v2_lora_one_stage.yaml; sed -i "s#/kaggle/working/models/Qwen3-ASR-1.7B-hf#$QB#g" $D/qwen_v2_lora_one_stage.yaml

# 1. LoRA r32/a64 on every LM projection (3e-5), projector 1e-5, audio tower frozen; 9,000 steps x batch 4 over the v2
#    pool (mix in the yaml); best of steps 3000/6000/9000 on robust_val (9000). Then merge to a plain HF folder.
$PQ $D/qwen_train.py --config $D/qwen_v2_lora_one_stage.yaml > $R/logs/qwen_v2_lora_one_stage.log 2>&1
$PQ $D/convert_qwen_any.py --base $QB --ckpt $D/exp_lora_one_stage/best.ckpt --out $D/models/lora_one_stage_step9000_hf5
$PQ $HERE/make_qa_reference.py $D/models/lora_one_stage_step9000_hf5 $QO $D/models/lora_one_stage_step9000_qwen_asr

# 2. dev adaptation: new LoRA r16 on LM layers 14-27 (8e-6), audio tower 1e-6, projector 5e-6; all 372 dev clips
#    at weight .8 (~2 passes) + pool replay, 232 steps. Fold manifests come from prepare.py (run in data_prep/run.sh).
$PQ $S/build_final.py                                     # -> $A/full_dev_train.jsonl + full_dev_final.json
$PQ $S/qwen_style_train.py --config $A/full_dev_final.json > $R/logs/final_full_dev.log 2>&1

# 3. merge the dev LoRA, write bf16 in the qwen_asr layout (merge_convert writes to $R/submissions/rover3/models/qwen).
cd $R && $PQ $S/merge_convert_qwen_asr.py
rm -rf $OUT/qwen; cp -r $R/submissions/rover3/models/qwen $OUT/qwen
echo QWEN_DONE
