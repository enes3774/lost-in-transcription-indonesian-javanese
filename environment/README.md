# Environments

| name | used for | Python | key packages | spec file |
| --- | --- | --- | --- | --- |
| **competition runtime** | running `inference/main.py`; the platform ran the submission in this image | 3.12 | torch 2.11 cu130, vLLM 0.23.0, transformers 4.57, qwen-asr, soundfile, librosa, sentencepiece | `Dockerfile`, `pyproject.toml`, `uv.lock`, `apt.txt`, `.python-version` (verbatim from the organizers' runtime repository) |
| **train** (`.venvs/train`) | MERaLiON, w2v-BERT CTC, SeamlessM4T, OWSM, Whisper dev pass, data preparation, exports | 3.10.12 | torch 2.8.0 cu128, transformers 4.57.6, peft 0.17.1, jiwer 4.0.0 | `requirements-train.txt` |
| **whisper step A** | the environment the Whisper full fine-tune ran in; same torch and transformers as **train**, which `run.sh` uses | 3.10.12 | torch 2.8.0 cu128, transformers 4.57.6 | `requirements-whisper-stepA.txt` |
| **qwen** (`.venvs/qwen`) | Qwen3-ASR (both stages and the merge), VibeVoice | 3.10.12 | torch 2.13.0, transformers 5.15.1, peft 0.20.0 | `requirements-qwen.txt` |
| **omniasr** (`.venvs/omniasr`) | omniASR-LLM fine-tune | 3.10.12 | torch 2.8.0, fairseq2 0.6, omnilingual-asr | `requirements-omniasr.txt` |

`bash environment/setup_training_envs.sh` creates all of them under `.venvs/`. The requirement lists are `pip freeze`
of the training machine (taken 2026-10-22; the runs were 2026-09-02 to 2026-10-01). OWSM also needs ESPnet 202511,
installed without dependency resolution into `.venvs/espnet_side` (`requirements-espnet-side.txt`); the trainer puts
that folder on `sys.path` (`ESPNET_DIR`).

## Inference environment

The files of the organizers' runtime image are here unchanged (`Dockerfile`, `entrypoint.sh`, `.dockerignore`,
`pyproject.toml`, `uv.lock`, `apt.txt`, `.python-version`). Either create the same Python environment directly:

```bash
sudo apt-get install -y $(cat environment/apt.txt)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd environment && uv sync --frozen        # environment/.venv: Python 3.12, torch 2.11 cu130, transformers 4.57
```

or build the image: `docker build --target runtime -t lit-runtime environment/`.

Two libraries the runtime does not have ship inside `inference/` and need no installation; `main.py` unpacks them
next to itself at start-up:

- `vendor_tf.zip`: transformers 5.15.1 and huggingface_hub, used only by the VibeVoice stage (its own process).
- `vendor_espnet_owsm.zip`: the parts of ESPnet (plus torch_complex, humanfriendly, typeguard) the OWSM stage needs.

## Hardware

Training ran on one workstation (Ubuntu 22.04 in Docker on WSL2): three RTX PRO 6000 Blackwell 96 GB cards and one
RTX A6000 48 GB, 8 usable CPU cores; every job used a single GPU. The final pipeline ran on the competition runner:
one A100 80 GB, 24 CPU cores, 120-minute limit, no network.
