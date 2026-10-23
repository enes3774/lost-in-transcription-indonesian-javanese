# Training: how every model in the final pipeline is produced

Each folder holds the training code of one member and a `run.sh` that runs it end to end and writes the finished
folder into `inference/models/<member>`. Run `data_prep/run.sh` once first; after that the members are independent
and can run in parallel on separate GPUs (`bash <member>/run.sh <gpu index>`).

Every script reads its working root from `LIT_ROOT` (default in the `run.sh` scripts: `./work` in this repository;
data, base models, training runs and logs go there). The `run.sh` scripts copy their code into the folder layout
under `$LIT_ROOT/training/` that the scripts were developed in, rewrite the absolute paths inside the YAML configs
to `$LIT_ROOT`, and use the environments created by `environment/setup_training_envs.sh`. Variables that can be
overridden: `LIT_ROOT`, `OUT` (finished models, default `inference/models`), `P` / `PQ` / `PO` (python of the
`train` / `qwen` / `omniasr` environment), `ESPNET_DIR`, `QB` / `QO` (Qwen bases), `K` (omniASR base files).

Order: `bash training/data_prep/run.sh`, `bash training/download_base_models.sh`, then any `training/<model>/run.sh
<gpu>`; `train_all.sh` in the repository root runs all of it.

| base model | downloaded to (`download_base_models.sh`) |
| --- | --- |
| `openai/whisper-large-v3` | `$LIT_ROOT/models/checkpoints/whisper-large-v3` |
| `facebook/w2v-bert-2.0` | `$LIT_ROOT/models/checkpoints/w2v-bert-2.0` |
| `Qwen/Qwen3-ASR-1.7B-hf`, `Qwen/Qwen3-ASR-1.7B` | `$LIT_ROOT/models/checkpoints/Qwen3-ASR-1.7B-hf`, `.../qwen3-asr-1.7b` |
| `MERaLiON/MERaLiON-3-3B-ASR` (revision 9f226d73) | `$LIT_ROOT/models/candidates/meralion-3-3b-asr` |
| `microsoft/VibeVoice-ASR-HF` | `$LIT_ROOT/models/candidates/vibevoice-asr-hf` |
| `facebook/seamless-m4t-v2-large` | `$LIT_ROOT/models/candidates/seamless-m4t-v2-large` |
| `espnet/owsm_ctc_v4_1B` | Hugging Face cache in `$LIT_ROOT/models/hf_home` |
| omniASR-LLM-3B + tokenizer (dl.fbaipublicfiles.com/mms) | `$LIT_ROOT/models/omniasr` |

## 1. Data (`data_prep/`)

`data_prep/run.sh` downloads every source, builds the pools and the two text resources of the inference package.
Accept each dataset's terms on the Mozilla Data Collective website and export `MDC_API_TOKEN` before running it.

| pool (in `$R/data/processed/qwen_v2/`) | built by | content | rows |
| --- | --- | --- | ---: |
| `train_pool_v2.jsonl` | `prepare_jember.py`, `prepare_qwen_lit_data.py`, `prepare_qwen_v2_data.py`, `remove_digit_rows.py` | Jember, Homostoria, Hari Minggoean, Central Javanese, Common Voice 26 id, OpenSLR-35, FLEURS jv/id | 216,263 |
| `train_pool_v2_jemberconv.jsonl` | `jemberconv_rewrite.py` | v2 with five Jember spellings set to the dev convention | 216,263 |
| `train_pool_v3.jsonl` | `prepare_dialect_sources.py` + `cat` | + Ngapak, Banyumasan, Bojonegoro TTS | 229,997 |
| `train_pool_v4.jsonl` | `prepare_dialect_sources2.py` | + Purworejo, Lumajang, Jombang, Gresik, Multidialect TTS (pool of the v5 mix) | 242,885 |
| `train_pool_v4_clean.jsonl` | `clean_pool.py` | v4 minus 30 Banyumasan rows with broken labels (pool of the v6 mix) | 242,855 |

Labels everywhere: the organizers' normaliser, Unicode NFC, `é` kept, `è/ê/ě` folded to `e`, digits written as
number words (`prepare_qwen_v2_data.py::canonical`). Held-out sets: `robust_val.jsonl` (700 pool rows the models
never train on) selects checkpoints; the competition dev set is watched, and used for training only in the
dev-adaptation passes listed below.

**Mixtures.** A mixture is the share of each training batch drawn from each source. The same two are defined in every
trainer that uses them (`MIXES["v5mix"]` / `MIXES["v6mix"]`, `SOURCE_WEIGHTS`, `sampling.weights`):

| source | v5 mix | v6 mix |
| --- | ---: | ---: |
| Jember | .19 | .16 |
| Homostoria | .13 | .11 |
| Hari Minggoean | .05 | .05 |
| OpenSLR-35 Javanese | .19 | .11 |
| Central Javanese | .08 | .06 |
| FLEURS Javanese | .07 | .05 |
| Common Voice 26 Indonesian | .08 | .05 |
| FLEURS Indonesian | .08 | .05 |
| Ngapak TTS | .05 | .05 |
| Purworejo | .03 | .03 |
| Lumajang TTS | .03 | .03 |
| Bojonegoro | .02 | .02 |
| Banyumasan TTS | | .10 |
| Gresik TTS | | .05 |
| Jombang TTS | | .05 |
| Multidialect TTS | | .03 |

`lm/build_trigram.py` builds the rescoring trigram (`inference/models/lm/trigram.pkl`, Kneser-Ney, D = 0.75) from
`train_pool_v4.jsonl`; `lm/build_lowercase_vocab.py` rebuilds `inference/lowercase_vocab.txt` from the v2 pool.

## 2. Members

| member | role in the pipeline | recipe | data |
| --- | --- | --- | --- |
| Whisper large-v3 | anchor, clips ≤ 30 s | full fine-tune, 3,000 steps, speed + noise augmentation; then one pass over all dev clips | v6 mix on `train_pool_v4.jsonl`; dev |
| Qwen3-ASR 1.7B | voter; anchor > 30 s | LoRA r32, 9,000 steps; then LoRA r16 on layers 14-27 + audio tower on all dev clips (~2 passes) | v2 pool (mix in the yaml); dev |
| MERaLiON-3 3B | voter | LoRA r32 + unfrozen encoder, 36,000 samples; then a dev LoRA on decoder layers 13-25, 413 steps | v5 mix on `train_pool_v4.jsonl`; dev |
| w2v-BERT 2.0 CTC | voter | full fine-tune with a new CTC head, 6,000 steps | v2 pool |
| omniASR-LLM 3B | candidate source | top 12 encoder layers + LLM decoder, fp32 master weights, 1,500 steps | 200 h drawn with the v5 mix |
| VibeVoice-ASR | candidate source | LoRA r16 on attention, 3,000 steps, merged in bf16 | 12,000 clips, v5 mix |
| SeamlessM4T-v2 large | candidate source | full fine-tune, fp32 master + bf16, best of steps 1500/3000/4500 on dev | 36,000 clips, v6 mix on `train_pool_v4_clean.jsonl` |
| OWSM-CTC v4 1B | candidate source | full fine-tune, CTC + 0.8 intermediate CTC, 3,800 steps | v6 mix on `train_pool_v4_clean.jsonl` |
| Sahabat-AI Gemma-2 9B | rescoring LM | none; `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct` is used as released | |

Details worth knowing before re-running:

- **Checkpoint choice.** Whisper: step 3000 (scored out of process by `eval_whisper_ckpt.py`); Qwen stage 1: best of
  3000/6000/9000 on `robust_val` (9000); CTC: lowest dev WER (6000, the last); omniASR: `last.pt` (step 1500);
  VibeVoice: step 3000; Seamless: best on dev (3000); OWSM: 3800 (the last, also the best on dev). The dev passes of
  Whisper, Qwen and MERaLiON have no held-out data and keep their final step.
- **MERaLiON export.** `export_merged.py` builds the base model with empty LoRA adapters and loads the one checkpoint
  it is given. Given the dev-pass checkpoint, which stores only the LoRA of decoder layers 13-25, the exported model
  is the base model plus that LoRA merged. This is how the model in the final pipeline was exported; the dev-pass
  LoRA starts from the stage-1 values of those layers, so stage 1 shapes it through its initialisation.
- **OWSM.** In-process validation crashed our machine, so `run.sh` trains with `--no-val 1` and scores every
  `step*.pt` afterwards. Our run was resumed once from the step-750 checkpoint (`--resume ... --start-step 750`),
  which restarts the optimizer state and the seeded sampler; straight through is the same recipe otherwise.
- **Qwen stage 1** started while the 118 digit-bearing rows were still in `train_pool_v2.jsonl`; every other run
  reads the 216,263-row file.
- **Seeds** are fixed in every trainer; GPU kernels are not bit-deterministic, so expect small differences.
