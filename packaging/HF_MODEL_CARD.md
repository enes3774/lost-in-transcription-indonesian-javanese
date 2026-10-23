---
language:
  - id
  - jv
tags:
  - automatic-speech-recognition
  - ensemble
  - javanese
  - indonesian
  - lost-in-transcription
license: other
license_name: mixed-per-member
---

# Lost in Transcription: Indonesian-Javanese — 2nd place ensemble weights

The eight fine-tuned speech models and the rescoring trigram of the 2nd-place solution to the Mozilla Data Collective
[Lost in Transcription](https://competitions.mozilladatacollective.com/) Indonesian-Javanese track (final submission
`rover18`, public WER 0.2026).

The inference code, training scripts and guides are on GitHub,
[github.com/enes3774/lost-in-transcription-indonesian-javanese](https://github.com/enes3774/lost-in-transcription-indonesian-javanese),
and in `solution_code.zip` in this repository.
`python inference/download_weights.py --with-llm` pulls this tree into `inference/models/` (plus the rescoring LLM
from its own repository), and `bash inference/run_inference.sh <work_dir> <out.csv>` runs the pipeline.

| folder | role | base model | base licence | what is inside |
| --- | --- | --- | --- | --- |
| `models/whisper` | anchor (clips ≤ 30 s) | `openai/whisper-large-v3` | Apache-2.0 | full fine-tune + one pass over the competition dev set, bf16 |
| `models/qwen` | voter; anchor > 30 s | `Qwen/Qwen3-ASR-1.7B` | Apache-2.0 | LoRA (all LM layers) + dev LoRA (layers 14-27) + audio tower, merged, bf16, qwen_asr layout |
| `models/meralion` | voter | `MERaLiON/MERaLiON-3-3B-ASR` | MERaLiON Public Licence | base + dev-adaptation LoRA of decoder layers 13-25, merged, bf16; notice in `solution_code.zip:inference/NOTICE.md` |
| `models/ctc` | voter | `facebook/w2v-bert-2.0` | MIT | full fine-tune with a CTC head (Indonesian/Javanese characters), bf16 |
| `models/omni` | candidate source | omniASR-LLM-3B (`facebookresearch/omnilingual-asr`) | Apache-2.0 | top 12 encoder layers + LLM decoder fine-tuned; one bf16 file for the plain-PyTorch port in `omni_asr.py` |
| `models/vibevoice` | candidate source | `microsoft/VibeVoice-ASR-HF` | MIT | LoRA r16 merged, bf16 |
| `models/seamless` | candidate source | `facebook/seamless-m4t-v2-large` | CC-BY-NC-4.0 | full fine-tune (speech-to-text part), bf16 |
| `models/owsm` | candidate source | `espnet/owsm_ctc_v4_1B` | CC-BY-4.0 | full fine-tune, bf16 state dict + configs (`model.pt`), `bpe.model`, `feats_stats.npz` |
| `models/lm` | rescoring | — | see training data | word trigram (Kneser-Ney counts) over the training transcripts |

The rescoring LLM, `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct` (Gemma Terms of Use), is used unchanged and is
not re-uploaded here.

The models are meant to be used together: four voters (Whisper, Qwen3-ASR, MERaLiON, CTC) are merged by a word-level
vote, the other four add alternatives, and a confusion-network rescoring stage with the trigram and the LLM picks the
final words.

**Training data of the recipe** (`training/README.md` in `solution_code.zip`): OpenSLR-35 Javanese (CC BY-SA 4.0), FLEURS id/jv (CC BY 4.0), and from the Mozilla Data
Collective: Common Voice 26 Indonesian, Jember Javanese Spontaneous Speech, Homostoria, Hari Minggoean, Central
Javanese, and the Javanese dialect TTS corpora (Ngapak, Banyumasan, Bojonegoro, Purworejo, Lumajang, Gresik, Jombang,
Multidialect); the competition dev set for the adaptation passes of Whisper, Qwen3-ASR and MERaLiON.

Each fine-tuned member inherits the licence of its base model (column above); SeamlessM4T-v2 is non-commercial.
