#!/usr/bin/env bash
# omniASR-LLM-3B (candidate source): fine-tune of the top 12 encoder layers + LLM decoder on 200 h drawn with the v5
# mix weights, 1,500 steps, fp32 master weights; then the plain-PyTorch bf16 export read by inference/omni_asr.py.
# Base: fairseq2 card omniASR_LLM_3B (https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt) and omniASR_tokenizer.model
# in $K (download_base_models.sh); the installed fairseq2 card is pointed at them below. ~2.5 h. Usage: bash run.sh [gpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R              # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}                    # environment/requirements-train.txt
OUT=${OUT:-$REPO/inference/models}                         # where the finished member lands for inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $R/logs $OUT
fix_paths() { sed -i -e "s#/home/jovyan/work/Enes/lost_in_transcription#$R#g" "$@"; }   # configs carry our absolute paths
PO=${PO:-$REPO/.venvs/omniasr/bin/python}                 # fairseq2 + omnilingual_asr (environment/requirements-omniasr.txt)
K=${K:-$R/models/omniasr}                                   # omniASR-LLM-3B.pt, omniASR_tokenizer.model (download_base_models.sh)
O=$R/training/omni_jv; mkdir -p $O; cp $HERE/train_omni_jv.py $HERE/omni_jv_data.py $HERE/norm.py $O/   # norm.py = the official normaliser
# point the installed omniASR model card at the local base files (once; the original card is kept as .bak)
CARD=$($PO -c "import omnilingual_asr, pathlib; print(pathlib.Path(omnilingual_asr.__file__).parent / 'cards/models/rc_models_v1.yaml')")
[ -f $CARD.bak ] || cp $CARD $CARD.bak
sed -e "s#file:///home/jovyan/work/Enes/nahuatl/ckpt#file://$K#g" $HERE/fairseq2_cards/rc_models_v1.yaml > $CARD

# 1. training list: the v5 mix weights (read from training/meralion_v1/meralion_v5mix.yaml) made explicit as ~200 h of
#    rows (seed 5), plus the 372 dev clips as the held-out set (watched only).
$P $O/omni_jv_data.py
# 2. fine-tune: lr 1e-5, warmup 100, cosine to 5 %, accum 6, waveform augmentation p .3; last.pt (step 1500) is used.
cd $O && $PO $O/train_omni_jv.py --card omniASR_LLM_3B --mix $O/train_v5mix.jsonl:1 --held $O/dev_held.jsonl   --out $O/exp3b_v5mix --steps 1500 --warmup 100 --accum 6 --eval_every 300 --workers 2 --aug 0.3 --seed 5 > $R/logs/omni_jv_3b_v5mix.log 2>&1
# 3. export: base + the 306 trained tensors -> one bf16 file for the plain-PyTorch port (CPU is enough).
mkdir -p $OUT/omni
CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO/inference $P $HERE/omni_overlay.py $K/omniASR-LLM-3B.pt $O/exp3b_v5mix/last.pt $OUT/omni/model.pt
cp $K/omniASR_tokenizer.model $OUT/omni/tokenizer.model
echo OMNI_DONE
