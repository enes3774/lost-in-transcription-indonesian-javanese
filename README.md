# Solution - Lost in Transcription: Indonesian-Javanese

Username: enesmahmut3774, romansolovyevzf

2nd place in the Indonesian-Javanese track of [Lost in Transcription](https://competitions.mozilladatacollective.com/)
(Mozilla Data Collective, 2026). Final submission `rover18`, public leaderboard WER **0.2026**. Trained weights:
[huggingface.co/enesssssw/lost-in-transcription-indonesian-javanese-ensemble](https://huggingface.co/enesssssw/lost-in-transcription-indonesian-javanese-ensemble).

## Summary

Eight open speech-recognition models are fine-tuned on Indonesian and Javanese speech, decode every clip independently,
and are combined by a word-level vote followed by a language-model rescoring stage.

- **Preprocessing.** Training audio is converted to 16 kHz mono. Every transcript goes through the organizers'
  normaliser, Unicode NFC, `é` kept, `è/ê/ě` folded to `e`, and digits written as number words, so all models learn one
  spelling. Small conversational and dialect corpora are up-weighted by sampling mixtures (the "v5" and "v6" mixes,
  [`training/README.md`](training/README.md)). At inference, clips longer than 30 s are cut at pauses into 12-28 s
  pieces, and the piece transcripts are joined afterwards.
- **Models.** Whisper large-v3, Qwen3-ASR 1.7B, MERaLiON-3 3B ASR and w2v-BERT 2.0 + CTC (the four voters);
  omniASR-LLM 3B, VibeVoice-ASR, SeamlessM4T-v2 large and OWSM-CTC v4 1B (four extra candidate sources); Sahabat-AI
  Gemma-2 9B (rescoring language model, used as released). Whisper, Qwen3-ASR and MERaLiON get a final adaptation pass
  on the competition dev set.
- **Ensemble.** Yes. Whisper anchors clips up to 30 s, Qwen3-ASR longer ones; an anchor word changes only when two
  voters agree on a different word. The other four models only add alternatives: letting them vote made results worse.
- **Output modifications.** All eight texts are aligned into a confusion network; a word trigram keeps the 24 best paths;
  the LLM scores them and the path with the highest 2·votes + 2·words + 0.25·LM-score wins. Spelling and casing rules
  learned from the training transcripts and a repetition guard (a word repeated 8 times or a phrase 6 times keeps only
  its last copy) finish the text.

```
test clip ──► cut at pauses (clips > 30 s become 12-28 s pieces)
               ├─► Whisper large-v3 (anchor ≤ 30 s) ─┐
               ├─► Qwen3-ASR 1.7B (anchor > 30 s) ───┤
               ├─► MERaLiON-3 3B ────────────────────┼─► word-level vote (an anchor word changes only when 2 voters agree)
               ├─► w2v-BERT 2.0 CTC ─────────────────┘         │
               ├─► omniASR-LLM 3B ──────┐                      ▼
               ├─► VibeVoice-ASR ───────┼──────► confusion network over all eight texts, trigram keeps 24 paths,
               ├─► SeamlessM4T-v2 ──────┤        Sahabat-AI Gemma-2 9B scores them
               └─► OWSM-CTC v4 1B ──────┘                      │
                                                               ▼
                                  spelling / casing rules, repetition guard ─► transcript
```

Every stage is strict: if one fails or misses a piece, the run stops with an error instead of writing a degraded result.

# Setup

Tested on Ubuntu 22.04 with an NVIDIA driver for CUDA 13.0. Inference needs steps 1-2 only; training needs all steps.

1. Install the prerequisites:
   - Python 3.12 for inference (installed by `uv` below) and Python 3.10 (3.10.12 used) for training
   - an NVIDIA driver that supports CUDA 12.8 and 13.0
   - system tools (`environment/apt.txt` lists those of the inference image; ffmpeg converts the training audio):
     ```bash
     sudo apt-get update
     sudo apt-get install -y git curl unzip ffmpeg python3.10 python3.10-venv $(cat environment/apt.txt)
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
2. Create the inference environment. It is the organizers' runtime image with its locked package versions
   (`environment/pyproject.toml`, `environment/uv.lock`):
   ```bash
   cd environment && uv sync --frozen && cd ..       # environment/.venv: Python 3.12, torch 2.11 cu130, transformers 4.57
   source environment/.venv/bin/activate
   ```
   or with Docker: `docker build --target runtime -t lit-runtime environment/`. Two libraries the image lacks ship
   inside `inference/` (`vendor_tf.zip`, `vendor_espnet_owsm.zip`); `main.py` unpacks them itself.
3. Create the training environments (`.venvs/train`, `.venvs/qwen`, `.venvs/omniasr`, `.venvs/espnet_side`), pinned
   by `environment/requirements-*.txt`:
   ```bash
   bash environment/setup_training_envs.sh
   ```
4. Data. Create an account on mozilladatacollective.com, accept the terms of each MDC dataset listed under
   [Run training](#run-training), create an API key in the profile settings and `export MDC_API_TOKEN=...`.
   `training/data_prep/run.sh` downloads all data (including OpenSLR-35 and FLEURS from their own hosts);
   `training/download_base_models.sh` downloads the base models.

Expected file structure (`LIT_ROOT`, default `./work`, holds data, base models and training runs):

```
.
├── inference/                 code of the final submission + helpers
│   └── models/                from download_weights.py or from training: whisper/ qwen/ meralion/ ctc/ omni/
│                              vibevoice/ seamless/ owsm/ lm/trigram.pkl llm/
├── training/                  data preparation, one folder per model with run.sh, training/README.md
├── environment/               runtime image definition, training requirements, setup_training_envs.sh
├── train_all.sh               main entry point for training
└── work/                      $LIT_ROOT, created by training
    ├── data/raw/<source>/                           downloads
    ├── data/processed/qwen_v2/train_pool_*.jsonl    training pools (16 kHz audio under data/processed/*/audio)
    ├── models/checkpoints/ candidates/ hf_home/ omniasr/   base models
    └── training/<run folders>/  logs/
```

# Hardware

The solution was run on one workstation: Ubuntu 22.04 (Docker on WSL2), three NVIDIA RTX PRO 6000 Blackwell 96 GB and
one RTX A6000 48 GB, 8 usable CPU cores, 251 GB RAM. Every training job used a single GPU (peak memory 17-55 GB).
Inference was scored on the competition runner: one NVIDIA A100 80 GB, 24 CPU cores, 120-minute limit, no network.

Training time (one GPU each, from our run logs): Whisper 2 h 05 min, Qwen3-ASR 2 h 05 min, MERaLiON 2 h 55 min,
w2v-BERT CTC 1 h 45 min, omniASR 2 h 30 min, VibeVoice 2 h 10 min, SeamlessM4T-v2 2 h 30 min, OWSM 50 min; about 17
GPU-hours in sequence. Data preparation is CPU-only and was not timed.

Inference time: about 110 minutes for the full test set on the A100 (MERaLiON with VibeVoice and SeamlessM4T beside it
29.5 min, Qwen3-ASR 19.0 min, Whisper 24.1 min, CTC 3.6 min, omniASR with OWSM beside it 8.9 min, rescoring about
10 min; unpacking the packaged archive took 12.9 min on the runner).

# Run training

```bash
bash train_all.sh 0          # GPU 0; LIT_ROOT=./work by default
```

`train_all.sh` runs, in order: `training/data_prep/run.sh` (downloads, pools, the rescoring trigram),
`training/download_base_models.sh`, `training/<model>/run.sh` for the eight models, and finally the download of the
rescoring LLM. Each step can also be run on its own; the eight model runs are independent and can run in parallel on
separate GPUs (`bash training/<model>/run.sh <gpu>`). Per-model recipes, checkpoint choice and export are in
[`training/README.md`](training/README.md).

**Data sources** (all on Mozilla Data Collective except OpenSLR-35 and FLEURS, which are openly licensed):

| source | language / style | hours | licence | used by |
| --- | --- | ---: | --- | --- |
| OpenSLR-35 Javanese | Javanese, read | 259.5 | CC BY-SA 4.0 | all models |
| Common Voice 26 Indonesian | Indonesian, read | 33.5 | CC0 | all models |
| FLEURS Javanese / Indonesian | read | 11.0 / 9.0 | CC BY 4.0 | all models |
| Jember Javanese Spontaneous Speech | conversation | 9.2 | MDC dataset terms | all models |
| Homostoria | Indonesian, conversation | 8.2 | MDC dataset terms | all models |
| Hari Minggoean | Indonesian, conversation | 10.0 | MDC dataset terms | all models |
| Central Javanese | Central Javanese | 7.1 | MDC dataset terms | all models |
| TTS Javanese, Ngapak dialect | Ngapak | 10.2 | CC BY-SA 4.0 | v5 / v6 mixes |
| Bojonegoro Javanese | East Javanese | 8.3 | CC BY-SA 4.0 | v5 / v6 mixes |
| Purworejo dialect speech | Purworejo | 8.1 | MDC dataset terms | v5 / v6 mixes |
| Lumajang dialect TTS | Lumajang | 6.0 | MDC dataset terms | v5 / v6 mixes |
| Javanese TTS of Banyumasan Dialect | Banyumasan | 9.5 | CC BY-SA 4.0 | v6 mix |
| Gresik / Jombang dialect TTS | Gresik / Jombang | 5.1 / 5.1 | MDC dataset terms | v6 mix |
| Multidialect TTS | several | 2.3 | MDC dataset terms | v6 mix |
| Competition dev set (372 clips) | conversation | 2.6 | competition data | validation; adaptation pass of Whisper, Qwen3-ASR, MERaLiON |

- **Weights.** About 46 GB for the eight models and the trigram, plus 18.5 GB for the LLM. Each `run.sh` writes its
  model to `inference/models/<model>/` by default (`OUT=` overrides); intermediate checkpoints stay under
  `$LIT_ROOT/training/`.
- **Trained weights without retraining.** `python inference/download_weights.py --with-llm` downloads the models of the
  final submission from the public Hugging Face repository above into `inference/models/`.
- **Network access** is needed for the downloads only (data, base models, LLM). The training runs set
  `HF_HUB_OFFLINE=1`.

# Run inference

```bash
python inference/download_weights.py --with-llm               # once: about 65 GB into inference/models/
bash inference/run_inference.sh <run_dir> <run_dir>/submission/submission.csv 0
```

- **Input.** `<run_dir>/data/test_metadata.csv` and `<run_dir>/data/clips/`, the layout of the competition test data
  (columns `audio_filename`, `file_duration_seconds`, `language`). There is no separate preprocessing step: cutting and
  resampling happen inside `main.py`, clip by clip. For a folder of new audio files,
  `python inference/prepare_data.py <audio_folder> <run_dir>` writes that layout.
- **Output.** The CSV given as second argument (`audio_filename,transcript`); without it,
  `<run_dir>/submission/submission.csv`.
- **Files written on the way.** In the system temp folder (`TMPDIR`): `pieces_*` (16 kHz WAV pieces of clips longer
  than 30 s), and `vibevoice_*`, `seamless_*`, `owsm_*` (each stage's input list, output and log). A successful run
  deletes them at the end; after a failure they stay, so the stage logs can be read. On first run `main.py` unpacks
  `vendor_tf/` and `vendor_espnet_owsm/` next to itself, so `inference/` must be writable.
- **Pitfalls.**
  - The run is strict: a failing stage or a missing piece ends it with exit code 1 and no output file. The reason is
    in the console log or in the stage logs left under `TMPDIR`.
  - One GPU with at least 60 GB of memory is needed; the third argument selects it (`CUDA_VISIBLE_DEVICES`).
  - Run it inside the runtime environment of Setup step 2 (Python 3.12, transformers 4.57); the VibeVoice and OWSM
    stages bring their own libraries.
  - The full test set takes about 110 minutes on an A100 80 GB.

## Repository layout

```
README.md                 this file
train_all.sh              training entry point
inference/
  main.py                 entry point: cutting, the four voters, the vote, the candidate stages, rescoring, checks
  omni_asr.py             plain-PyTorch port of omniASR-LLM (no fairseq2 at inference)
  vibevoice_stage.py seamless_stage.py owsm_stage.py   candidate sources, each in its own process
  rescore.py              confusion network + trigram beam + LLM scoring
  postprocess.py          spelling, casing and the repetition guard; lowercase_vocab.txt is its word list
  split_audio.py          pause-based cutting of long clips
  vendor_tf.zip vendor_espnet_owsm.zip   libraries the runtime image lacks (unpacked by main.py)
  NOTICE.md               licence notices shipped with the weights
  run_inference.sh prepare_data.py download_weights.py
training/                 data_prep/, lm/, one folder per model with run.sh, download_base_models.sh, README.md
environment/              runtime image definition, training requirements, setup_training_envs.sh
packaging/                Hugging Face model card and upload script
```

## Submission history

| package | public WER | what changed |
| --- | ---: | --- |
| rover5 | 0.236 | v5-mix Whisper and MERaLiON, strict spelling stage |
| rover6 | 0.230 | + context rescoring (trigram + Sahabat-AI LLM) |
| rover10 | 0.2192 | + omniASR-LLM 3B as a candidate source |
| rover13 | 0.2076 | + VibeVoice-ASR and SeamlessM4T-v2 as candidate sources |
| rover18 | **0.2026** | + repetition guard, Seamless on 36k clips, cutting at 30 s, OWSM-CTC as a candidate source |

## Licences

Models: Whisper large-v3 (Apache 2.0), Qwen3-ASR (Apache 2.0), MERaLiON-3 (MERaLiON Public Licence; notice in
`inference/NOTICE.md`), w2v-BERT 2.0 (MIT), omniASR (Apache 2.0), VibeVoice-ASR (MIT), SeamlessM4T-v2 (CC BY-NC 4.0),
OWSM-CTC v4 (CC BY 4.0), Sahabat-AI Gemma-2 (Gemma Terms of Use). Data: see the table under Run training.
