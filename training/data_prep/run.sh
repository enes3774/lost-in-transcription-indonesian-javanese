#!/usr/bin/env bash
# Data preparation: raw downloads -> the training pools every member reads. CPU only (ffmpeg needed).
# Paths: every script works under the project root $R (see training/README.md). Run from anywhere.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${LIT_ROOT:-$REPO/work}; export LIT_ROOT=$R            # working root: data/, models/, training runs, logs/
P=${P:-$REPO/.venvs/train/bin/python}
Q=$R/data/processed/qwen_v2
mkdir -p $R/runtime_repo $R/scripts $R/analysis $R/logs
cp $REPO/training/common/score.py $R/runtime_repo/score.py        # the organizers' normaliser; most scripts import it from here
cp $HERE/*.py $R/scripts/                                         # the scripts expect to live in $R/scripts
cp $REPO/training/lm/style_postprocess.py $R/analysis/

# 1. Downloads. Accept each dataset's terms on mozilladatacollective.com first, then export MDC_API_TOKEN.
$P $R/scripts/download_mdc.py              # dev, jember, homostoria, common_voice_indonesian_26, central_javanese, hari_minggoean
$P $R/scripts/download_mdc_javanese.py     # banyumasan, ngapak, bojonegoro, purworejo, lumajang, jombang, gresik, multidialect TTS
$P $R/scripts/fetch_open_sources.py        # OpenSLR-35 Javanese, FLEURS id_id / jv_id

# 2. Dev manifest (data/processed/dev/manifest.csv)
$P $R/scripts/prepare_dev_manifest.py

# 3. Pool v2 (Jember, Homostoria, Hari, Central, CV-26 id, OpenSLR-35, FLEURS jv/id), canonical labels:
#    official normaliser, NFC, é kept, è/ê/ě -> e, digits as Indonesian (or Javanese) number words
$P $R/scripts/prepare_jember.py
$P $R/scripts/prepare_qwen_lit_data.py
$P $R/scripts/prepare_qwen_v2_data.py      # -> train_pool_v2.jsonl (216,381 rows), held-out sets, competition_dev_eval_only.jsonl
$P $R/scripts/remove_digit_rows.py         # -> 216,263 rows (118 rows that kept a digit set aside)
$P $R/scripts/make_robust_val.py           # -> robust_val.jsonl (700 held-out pool rows), dev language tags
$P $R/scripts/build_eval_sets.py           # -> data/processed/newmodels/dev.csv, robust.csv (evaluation lists)

# 4. Jember spelling convention, then the dialect TTS corpora
$P $R/scripts/jemberconv_rewrite.py        # -> train_pool_v2_jemberconv.jsonl
$P $R/scripts/prepare_dialect_sources.py   # -> dialect_rows.jsonl (ngapak, banyumasan, bojonegoro)
cat $Q/train_pool_v2_jemberconv.jsonl $Q/dialect_rows.jsonl > $Q/train_pool_v3.jsonl     # 229,997 rows
$P $R/scripts/prepare_dialect_sources2.py  # -> dialect_rows2.jsonl + train_pool_v4.jsonl (242,885 rows; v5 mix pool)
$P $R/scripts/clean_pool.py                # -> train_pool_v4_clean.jsonl (30 broken banyumasan labels dropped; v6 mix pool)

# 5. Dev-set manifests for the dev adaptation passes (Qwen stage 2, Whisper and MERaLiON one-pass)
mkdir -p $R/training/qwen_style_aggressive
cp $REPO/training/qwen/stage2_dev/prepare.py $REPO/training/qwen/stage2_dev/build_final.py $R/training/qwen_style_aggressive/
$P $R/training/qwen_style_aggressive/prepare.py      # speaker / conversation folds of dev + pool, robust.jsonl, all_dev_by_speaker.jsonl
$P $R/scripts/build_all_dev_train.py                 # -> qwen_style_aggressive/all_dev_train.jsonl

# 6. Text resources of the inference package (CPU, a few minutes)
$P $REPO/training/lm/build_trigram.py $Q/train_pool_v4.jsonl $REPO/inference/models/lm/trigram.pkl   # rescoring trigram
$P $REPO/training/lm/build_lowercase_vocab.py $REPO/inference/lowercase_vocab.txt                      # same 4,324 words as shipped
echo DATA_PREP_DONE
