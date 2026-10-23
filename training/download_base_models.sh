#!/bin/bash
# Download every base model the training scripts start from, into the folders they expect under $LIT_ROOT.
# About 60 GB. Needs network access and `hf` (huggingface_hub CLI, in the train environment).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=${LIT_ROOT:-$REPO/work}; M=$R/models
HF=${HF:-$REPO/.venvs/train/bin/hf}
$HF download openai/whisper-large-v3                  --local-dir $M/checkpoints/whisper-large-v3
$HF download facebook/w2v-bert-2.0                    --local-dir $M/checkpoints/w2v-bert-2.0
$HF download Qwen/Qwen3-ASR-1.7B                      --local-dir $M/checkpoints/qwen3-asr-1.7b
$HF download Qwen/Qwen3-ASR-1.7B-hf                   --local-dir $M/checkpoints/Qwen3-ASR-1.7B-hf
$HF download MERaLiON/MERaLiON-3-3B-ASR --revision 9f226d738dee62c60a2f43b0ab7b9a2585a132c3 --local-dir $M/candidates/meralion-3-3b-asr
$HF download microsoft/VibeVoice-ASR-HF               --local-dir $M/candidates/vibevoice-asr-hf
$HF download facebook/seamless-m4t-v2-large           --local-dir $M/candidates/seamless-m4t-v2-large
HF_HOME=$M/hf_home $HF download espnet/owsm_ctc_v4_1B          # read from the HF cache layout under $M/hf_home
mkdir -p $M/omniasr
curl -L -o $M/omniasr/omniASR-LLM-3B.pt        https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt
curl -L -o $M/omniasr/omniASR_tokenizer.model  https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model
echo "base models ready under $M"
