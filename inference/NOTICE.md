# Notices

This submission contains a fine-tuned derivative of **MERaLiON-3-3B-ASR**
(https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR), developed by I2R, A*STAR, Singapore,
and released under the MERaLiON Public Licence
(https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-3-Public-Licence.pdf).

## Modifications

The released weights were fine-tuned (LoRA on the Gemma-2 decoder, merged into the weights;
the speech adapter and Whisper encoder further trained) on Indonesian and Javanese speech
corpora for the Lost in Transcription competition. The model architecture, processor and
remote-code files (`configuration_meralion3.py`, `modeling_meralion3.py`,
`processing_meralion3.py`) are unchanged from the release.

## Component licences carried by the base model

- MERaLiON code and adapter: MIT (Copyright 2025/2026 Agency for Science, Technology and Research).
- Speech encoder: OpenAI Whisper large-v3, MIT.
- Text decoder: Google Gemma 2. **Gemma is provided under and subject to the Gemma Terms of Use
  found at ai.google.dev/gemma/terms.** Use is subject to the Gemma Prohibited Use Policy
  (ai.google.dev/gemma/prohibited_use_policy). Any further distribution of this model or its
  derivatives must pass on these terms and this notice.

## Acknowledgment

This work uses MERaLiON, developed by the MERaLiON team at A*STAR and funded by the National
Research Foundation, Singapore, under its National Large Language Models Funding Initiative.
The software is provided "as is", without warranty of any kind.

## Other models in this submission

- **Qwen3-ASR-1.7B** (https://huggingface.co/Qwen/Qwen3-ASR-1.7B), Alibaba Cloud, Apache License 2.0 —
  fine-tuned (LoRA, merged) on Indonesian and Javanese speech corpora for this competition.
- **Whisper large-v3** (https://github.com/openai/whisper), OpenAI, MIT License — fully fine-tuned on the
  same corpora for this competition.

## Context rescoring language model (models/llm)

`GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct` (Sahabat-AI, GoTo / Indosat / AI Singapore), a Gemma-2-9B
continued-pretrained on Indonesian, Javanese and Sundanese text. Distributed under the Gemma Terms of Use
(https://ai.google.dev/gemma/terms); the model files are packaged unmodified. It is used only to score
candidate transcripts of one clip at a time (rescore.py). `models/lm/trigram.pkl` holds word trigram counts
estimated on our own training transcripts.


## Fifth candidate source (models/omni)

`facebook/omniASR-LLM-3B` (Meta Omnilingual ASR), packaged as a plain-PyTorch
re-implementation (`omni_asr.py`, no fairseq2) with the weights converted to bfloat16 unchanged; fine-tuned on our training corpus (top encoder layers and decoder, 1,500 steps) and used with
the Indonesian language embedding. Licence: Apache-2.0 (https://huggingface.co/facebook/omniASR-LLM-3B).

## Eighth candidate source (models/owsm)

`espnet/owsm_ctc_v4_1B` (OWSM-CTC v4, CMU WAVLab / ESPnet; E-Branchformer CTC speech model trained on public data),
fine-tuned on our training corpus and stored in bfloat16. Licence of the released weights: CC BY 4.0 (Creative Commons Attribution 4.0 International)
(https://huggingface.co/espnet/owsm_ctc_v4_1B). `vendor_espnet_owsm.zip` carries the pure-python parts of ESPnet
(Apache License 2.0, https://github.com/espnet/espnet), humanfriendly (MIT), torch_complex (Apache License 2.0) and
typeguard (MIT) that the stage imports; nothing else in the package uses them.
