#!/usr/bin/env python3
"""Lost In Transcription submission: four-model ROVER with context rescoring over five systems.

The three fine-tuned-and-adapted systems of the accepted submission, unchanged, plus a
fourth voice and three text stages. Systems:
  * Whisper large-v3 full-FT      -- anchor on clips <= 30 s (its window)
  * Qwen3-ASR-1.7B v2 LoRA        -- anchor on longer clips; voter elsewhere
  * MERaLiON-3-3B-ASR             -- voter (30 s windows joined for long clips)

ROVER: the anchor's words survive unless both voters agree on a different word
at the same alignment slot (replace), agree that the anchor's word is absent
(delete), or agree on the same inserted word (insert). Fusion runs on
transcripts passed through the competition's own normalizer, so the vote sees
what the scorer sees.

Every clip is handled on its own: which model anchors it depends only on the
clip's own duration, the 30 s windows come from the clip's own audio, and the
vote uses the three transcripts of that clip alone. Batching shares compute
only. Models are loaded one at a time and released. Logs carry no counts,
filenames or predictions.
"""
import csv
import gc
import os
import re
import json
import shutil
import subprocess
import zipfile
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error", "TF_CPP_MIN_LOG_LEVEL": "3",
})

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import postprocess, strict_spelling
from rescore import rescore_all
from omni_asr import load_converted, normalize_waveform, split_long
from split_audio import split as split_pauses, SPLIT_TRIGGER_S

WORK = Path(os.environ.get("CODE_EXECUTION_ROOT", "/code_execution"))
SRC = Path(__file__).resolve().parent
MODELS = SRC / "models"
METADATA = WORK / "data" / "test_metadata.csv"
CLIPS = WORK / "data" / "clips"
OUTPUT = Path(os.environ.get("LIT_OUT_CSV", str(WORK / "submission" / "submission.csv")))

SR = 16000
# No stage is cut short: an overrun or an error returns the submission slot, a silently degraded run does not.
RESCORE_DEADLINE_MIN = 10 ** 6
OMNI_DEADLINE_MIN = 10 ** 6
VIBEVOICE_DEADLINE_MIN = 10 ** 6
VIBEVOICE_BATCH = 24
SEAMLESS_BATCH = 16
OWSM_BATCH, OWSM_LANG = 4, "jav"     # ~6-11 GB; the stage overlaps omniASR only
OMNI_LANG = 596                   # ind_Latn in omniASR's language table
OMNI_BATCH, OMNI_MAX_BATCH_S, OMNI_MAX_CHUNK_S, OMNI_MAX_NEW = 24, 480.0, 34.0, 448
WINDOW_SEC = 30.0            # Whisper's window; also MERaLiON's training window
# Whisper full-FT: the beam setting of the reported ensemble (beam 4 is within
# 0.0002 of beam 6 and a third faster).
WHISPER_BEAMS, WHISPER_BATCH, WHISPER_MAX_NEW = 4, 12, 192
# Qwen v2: exactly the accepted submission's decoding.
QWEN_BEAMS, QWEN_LP, QWEN_BATCH, QWEN_MAX_NEW = 4, 1.0, 16, 256
# MERaLiON-3-3B-ASR: official defaults, greedy. Beams are worth 0.017 standalone but
# only 0.0006 inside this vote, and the 2 h budget cannot afford them.
MER_BEAMS, MER_LP, MER_REP, MER_NRNG, MER_BATCH, MER_MAX_NEW = 1, 1.0, 1.05, 6, 8, 256
MER_PROMPT = ("Instruction: Please transcribe this speech. \n"
              "Follow the text instruction based on the following audio: <SpeechHere>")
# w2v-BERT 2.0 CTC: a fourth voice with no Whisper encoder and no autoregressive decoder, so its
# errors decorrelate from the other three. Greedy; a CTC posterior has no beam to search.
CTC_BATCH = 16
SPEAKER_TAG = re.compile(r"<\s*speaker\s*\d*\s*>\s*:?", re.I)
ARTIFACT = re.compile(r"\b(nospeech|err)\b", re.I)

NON_ACUTE_E = str.maketrans({"è": "e", "ê": "e", "ě": "e", "È": "E", "Ê": "E", "Ě": "E"})
DIGITS = re.compile(r"(?<![\w])\d+(?![\w])")
ONES = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"]


def log(message):
    print("[LIT] " + message, flush=True)


# ---------------------------------------------------------------------------
# text: the accepted submission's canonicalization, and the official normalizer
# ---------------------------------------------------------------------------

def number_words(n):
    if n < 12: return ONES[n]
    if n < 20: return number_words(n - 10) + " belas"
    if n < 100: return number_words(n // 10) + " puluh" + ((" " + number_words(n % 10)) if n % 10 else "")
    if n < 200: return "seratus" + ((" " + number_words(n - 100)) if n > 100 else "")
    if n < 1000: return number_words(n // 100) + " ratus" + ((" " + number_words(n % 100)) if n % 100 else "")
    if n < 2000: return "seribu" + ((" " + number_words(n - 1000)) if n > 1000 else "")
    if n < 1_000_000: return number_words(n // 1000) + " ribu" + ((" " + number_words(n % 1000)) if n % 1000 else "")
    return str(n)


def canonicalize(text):
    text = str(text).translate(NON_ACUTE_E)
    return DIGITS.sub(lambda m: number_words(int(m.group())) if len(m.group()) <= 6 else m.group(), text)


def _lowercase_sentence_initial(match):
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    """Verbatim copy of runtime_repo/score.py::normalize_text."""
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")
    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text


def clean_meralion(text):
    text = SPEAKER_TAG.sub(" ", text)
    text = ARTIFACT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# ROVER (analysis/rover_ablate_dev.py, unchanged)
# ---------------------------------------------------------------------------

def _key(x):
    return None if x is None else x.casefold()


def _align(a, b):
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i; bt[i][0] = "D"
    for j in range(1, m + 1):
        d[0][j] = j; bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = [(d[i - 1][j - 1] + (_key(a[i - 1]) != _key(b[j - 1])), "M"),
                 (d[i - 1][j] + 1, "D"), (d[i][j - 1] + 1, "I")]
            d[i][j], bt[i][j] = min(v, key=lambda x: (x[0], {"M": 0, "D": 1, "I": 2}[x[1]]))
    out = []; i = n; j = m
    while i or j:
        o = bt[i][j]
        if o == "M":
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif o == "D":
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    return out[::-1]


def _amap(a, b):
    p = {}; g = defaultdict(list); i = 0
    for x, y in _align(a, b):
        if x is None:
            g[i].append(y)
        else:
            p[i] = y; i += 1
    return p, g


def fuse(anchor, b, c):
    a = anchor.split(); b = b.split(); c = c.split()
    bm, bg = _amap(a, b); cm, cg = _amap(a, c); out = []
    for i in range(len(a) + 1):
        for x, y in _align(bg.get(i, []), cg.get(i, [])):
            if x is not None and y is not None and _key(x) == _key(y):
                out.append(x)
        if i < len(a):
            x, y = bm.get(i), cm.get(i)
            if x is None and y is None:
                continue
            if x is not None and y is not None and _key(x) == _key(y) and _key(x) != _key(a[i]):
                out.append(x)
            else:
                out.append(a[i])
    return " ".join(out)


def fuse_n(anchor, voters, k=2):
    """ROVER over any number of voters: a slot changes when at least `k` voters propose the
    same alternative. With two voters and k=2 this is exactly `fuse`."""
    a = anchor.split()
    maps = [_amap(a, v.split()) for v in voters]
    out = []
    for i in range(len(a) + 1):
        ins = defaultdict(int)
        for _, gm in maps:
            for w in dict.fromkeys(_key(x) for x in gm.get(i, [])):
                ins[w] += 1
        for w, c in ins.items():
            if c >= k:
                for _, gm in maps:
                    cand = [x for x in gm.get(i, []) if _key(x) == w]
                    if cand:
                        out.append(cand[0])
                        break
        if i < len(a):
            props = [pm.get(i) for pm, _ in maps]
            if sum(1 for p in props if p is None) >= k:
                continue
            alts = defaultdict(int)
            for p in props:
                if p is not None and _key(p) != _key(a[i]):
                    alts[_key(p)] += 1
            best = max(alts.items(), key=lambda x: x[1]) if alts else None
            if best and best[1] >= k:
                out.append(next(p for p in props if p is not None and _key(p) == best[0]))
            else:
                out.append(a[i])
    return " ".join(out)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def load_audio(path):
    import librosa
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav.astype(np.float32)


def windows(wav):
    n = int(WINDOW_SEC * SR)
    return [wav] if len(wav) <= n else [wav[i:i + n] for i in range(0, len(wav), n)]


def release(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# the three systems
# ---------------------------------------------------------------------------

def run_qwen(paths):
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        str(MODELS / "qwen"), dtype=torch.bfloat16, device_map="cuda",
        max_inference_batch_size=QWEN_BATCH, max_new_tokens=QWEN_MAX_NEW,
        local_files_only=True, attn_implementation="sdpa")
    g = model.model.thinker.generation_config
    g.do_sample = False; g.num_beams = QWEN_BEAMS; g.length_penalty = QWEN_LP
    g.early_stopping = False; g.no_repeat_ngram_size = 0; g.repetition_penalty = 1.0
    g.pad_token_id = 151645; g.temperature = None
    with torch.inference_mode():
        results = model.transcribe(audio=paths, language=None)
    out = [canonicalize(r.text.strip()) for r in results]
    release(model)
    return out


def run_whisper(paths, durations):
    """Clips <= 30 s only; longer clips get None (Qwen anchors those)."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    d = MODELS / "whisper"
    proc = WhisperProcessor.from_pretrained(d, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        d, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    out = [None] * len(paths)
    idx = [i for i, s in enumerate(durations) if s <= WINDOW_SEC]
    for k in range(0, len(idx), WHISPER_BATCH):
        sel = idx[k:k + WHISPER_BATCH]
        wav = [load_audio(paths[i]) for i in sel]
        x = proc.feature_extractor(wav, sampling_rate=SR, padding="max_length", max_length=480000,
                                   truncation=True, return_attention_mask=True, return_tensors="pt")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model.generate(input_features=x.input_features.cuda(), attention_mask=x.attention_mask.cuda(),
                               language="jw", task="transcribe", num_beams=WHISPER_BEAMS, do_sample=False,
                               length_penalty=1.0, max_new_tokens=WHISPER_MAX_NEW, use_cache=True)
        for i, t in zip(sel, proc.batch_decode(y, skip_special_tokens=True)):
            out[i] = canonicalize(t.strip())
    release(model, proc)
    return out


def run_meralion(paths):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    d = MODELS / "meralion"
    proc = AutoProcessor.from_pretrained(d, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        d, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    tok = proc.tokenizer
    tok.padding_side = "left"
    prompt = tok.apply_chat_template([{"role": "user", "content": MER_PROMPT}], tokenize=False, add_generation_prompt=True)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]

    def _reorder_cache(past, beam_idx):     # legacy hook the remote code calls under beam search
        if hasattr(past, "reorder_cache"):
            past.reorder_cache(beam_idx); return past
        return tuple(tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer) for layer in past)
    model.text_decoder._reorder_cache = _reorder_cache

    pending = []                              # (clip index, window index, wave)
    for i, p in enumerate(paths):
        for j, w in enumerate(windows(load_audio(p))):
            pending.append((i, j, w))
    pieces = defaultdict(dict)
    for k in range(0, len(pending), MER_BATCH):
        chunk = pending[k:k + MER_BATCH]
        inputs = proc(text=[prompt] * len(chunk), audios=[w for _, _, w in chunk], return_tensors="pt", padding=True)
        moved = {kk: (v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()) for kk, v in inputs.items()}
        with torch.inference_mode():
            y = model.generate(**moved, max_new_tokens=MER_MAX_NEW, do_sample=False, num_beams=MER_BEAMS,
                               repetition_penalty=MER_REP,
                               no_repeat_ngram_size=MER_NRNG, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        L = moved["input_ids"].shape[1]
        seq = y[:, L:] if y.shape[1] > L else y
        for (i, j, _), t in zip(chunk, proc.batch_decode(seq, skip_special_tokens=True)):
            pieces[i][j] = clean_meralion(t)
    out = [canonicalize(" ".join(pieces[i][j] for j in sorted(pieces[i])).strip()) for i in range(len(paths))]
    release(model, proc)
    return out


def run_ctc(paths):
    """w2v-BERT 2.0 CTC, greedy. Long clips are windowed like MERaLiON and rejoined."""
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer
    d = MODELS / "ctc"
    fe = SeamlessM4TFeatureExtractor.from_pretrained(d, local_files_only=True)
    tok = Wav2Vec2CTCTokenizer.from_pretrained(d, local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(
        d, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    # This stage is bound by audio decoding, not by the GPU (batch 8/16/24 are within 3 %),
    # so the clips are decoded on a small thread pool while the model waits.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        waves = list(pool.map(load_audio, paths))
    pending = []
    for i, wav in enumerate(waves):
        for j, w in enumerate(windows(wav)):
            if len(w) < SR:                # a sub-second tail window alone in a batch gives the front end 0 frames
                w = np.pad(w, (0, SR - len(w)))
            pending.append((i, j, w))
    del waves
    pieces = defaultdict(dict)
    for k in range(0, len(pending), CTC_BATCH):
        chunk = pending[k:k + CTC_BATCH]
        x = fe([w for _, _, w in chunk], sampling_rate=SR, padding=True,
               return_attention_mask=True, return_tensors="pt")
        with torch.inference_mode():
            y = model(input_features=x.input_features.cuda().to(torch.bfloat16),
                      attention_mask=x.attention_mask.cuda()).logits.float()
        for (i, j, _), t in zip(chunk, tok.batch_decode(y.argmax(-1).cpu().numpy())):
            pieces[i][j] = t.strip()
    out = [canonicalize(" ".join(pieces[i][j] for j in sorted(pieces[i])).strip()) for i in range(len(paths))]
    release(model, fe)
    return out


# ---------------------------------------------------------------------------

_TAIL_PUNCT = re.compile(r"[^\w]+$")


def collapse_repeats(text, word_run=8, phrase_run=6, max_phrase=4):
    """Runaway-repetition guard on a finished transcript. One word repeated `word_run` or more times in a row, or
    a phrase of 2..`max_phrase` words repeated `phrase_run` or more times, is a decoder loop, not speech: only the
    last copy is kept (a trailing '...' sits on the last copy). The rescoring stage prefers such a loop - each
    extra word earns the per-word bonus and costs the language model almost nothing - so a loop in any one
    candidate source reaches the output. On the 372 dev clips it shortens 1 output (-64 errors) and makes
    none worse."""
    ws = text.split()
    for p in range(1, max_phrase + 1):
        need = word_run if p == 1 else phrase_run
        k = [_TAIL_PUNCT.sub("", w.casefold()) for w in ws]
        out, i = [], 0
        while i < len(ws):
            n = 1
            while i + (n + 1) * p <= len(ws) and k[i + n * p:i + (n + 1) * p] == k[i:i + p]:
                n += 1
            if n >= need and all(k[i:i + p]) and (p == 1 or len(set(k[i:i + p])) > 1):
                out += ws[i + (n - 1) * p:i + n * p]
                i += n * p
            else:
                out.append(ws[i])
                i += 1
        ws = out
    return " ".join(ws)


def fuse_clip(q, w, m, c, o):  # o is kept in the signature; it reaches the candidates, not the vote
    """One piece. Every system's text is first put in the references' spelling (strict_spelling), so a
    spelling difference neither splits the vote nor survives it; then ROVER, then the text stages.
    omni does not vote - it only widens the rescoring stage's candidates, as in rover8, because letting it
    vote cost 0.0046 on the dev dry run. `w` is None only if Whisper produced nothing."""
    qn, mn, cn = (strict_spelling(normalize_text(x)) for x in (q, m, c))
    if w is None:
        anchor, voters = qn, [mn, cn]
    else:
        anchor, voters = strict_spelling(normalize_text(w)), [qn, mn, cn]
    fused = fuse_n(anchor, voters, k=2)
    return strict_spelling(postprocess(fused, mn, [anchor] + voters)), [anchor] + voters


def run_omni(paths, deadline):
    """omniASR-LLM through the plain-torch port: greedy transcription of every clip (long clips split at quiet
    points), Indonesian language embedding. Returns one text per clip, "" for clips not reached before the
    deadline. Its output never enters the vote; it only adds candidates to the rescoring stage."""
    import sentencepiece as spm
    gc.collect()
    torch.cuda.empty_cache()
    model = load_converted(str(MODELS / "omni" / "model.pt"), device="cuda", dtype=torch.bfloat16)
    sp = spm.SentencePieceProcessor(model_file=str(MODELS / "omni" / "tokenizer.model"))
    pieces = []
    for i, p in enumerate(paths):
        for j, w in enumerate(split_long(load_audio(p), SR, OMNI_MAX_CHUNK_S)):
            pieces.append((i, j, normalize_waveform(torch.from_numpy(w))))
    pieces.sort(key=lambda x: x[2].numel())
    batches, cur = [], []
    for pc in pieces:
        n = pc[2].numel() / SR
        if cur and (len(cur) >= OMNI_BATCH or n * (len(cur) + 1) > OMNI_MAX_BATCH_S):
            batches.append(cur); cur = []
        cur.append(pc)
    if cur:
        batches.append(cur)
    texts = {}
    for b in batches:
        if time.monotonic() > deadline:
            break
        ids = model.generate([w for _, _, w in b], lang_ids=[OMNI_LANG] * len(b), max_new_tokens=OMNI_MAX_NEW)
        for (i, j, _), s in zip(b, ids):
            texts[(i, j)] = sp.decode(s) if s else ""
    out = []
    for i in range(len(paths)):
        parts = [texts.get((i, j)) for j in range(sum(1 for k in texts if k[0] == i))]
        out.append(" ".join(x for x in parts if x).strip() if parts and all(x is not None for x in parts) else "")
    release(model, sp)
    return out


def main():
    started = time.monotonic()
    log("startup" + (" (smoke test)" if os.environ.get("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1" else ""))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    for d in ("qwen", "whisper", "meralion", "ctc", "llm", "lm", "omni", "owsm"):
        if not (MODELS / d).is_dir():
            raise FileNotFoundError("packaged model is missing")
    if not METADATA.is_file() or not CLIPS.is_dir():
        raise FileNotFoundError("competition input is missing")
    metadata = pd.read_csv(METADATA, keep_default_na=False)
    required = {"audio_filename", "file_duration_seconds", "language"}
    if not required.issubset(metadata.columns):
        raise ValueError("invalid metadata schema")
    if metadata["audio_filename"].duplicated().any():
        raise ValueError("duplicate audio identifiers")
    paths = [str(CLIPS / name) for name in metadata["audio_filename"]]
    if not all(Path(p).is_file() for p in paths):
        raise FileNotFoundError("manifest audio is missing")
    durations = [float(x) for x in metadata["file_duration_seconds"]]

    # Long clips are cut at pauses and decoded piece by piece; their texts are joined afterwards, in order.
    # Nothing else changes: a clip at or under the limit becomes a single piece and follows the old path.
    import tempfile, soundfile as sf
    piece_dir = tempfile.mkdtemp(prefix="pieces_")
    owner, ppaths, pdur = [], [], []
    for i, (p, d) in enumerate(zip(paths, durations)):
        if d <= SPLIT_TRIGGER_S:
            owner.append(i); ppaths.append(p); pdur.append(d)
            continue
        for j, seg in enumerate(split_pauses(load_audio(p))):
            sp = os.path.join(piece_dir, f"{i:06d}_{j:03d}.wav")
            sf.write(sp, seg, SR)
            owner.append(i); ppaths.append(sp); pdur.append(len(seg) / SR)

    # VibeVoice runs in its own process, with the vendored transformers 5 on ITS path only, sharing the GPU with
    # the stages below. Its log goes to a file: the runner keeps only the first 500 lines of ours.
    vendor = SRC / "vendor_tf"
    if not vendor.is_dir():
        with zipfile.ZipFile(SRC / "vendor_tf.zip") as z:
            z.extractall(SRC)
    vv_dir = tempfile.mkdtemp(prefix="vibevoice_")
    vv_pieces, vv_out = os.path.join(vv_dir, "pieces.json"), os.path.join(vv_dir, "out.json")
    with open(vv_pieces, "w", encoding="utf-8") as h:
        json.dump([[p, d] for p, d in zip(ppaths, pdur)], h)
    vv_env = dict(os.environ, PYTHONPATH=str(vendor), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    vv_log = open(os.path.join(vv_dir, "vibevoice.log"), "w")
    vv_proc = subprocess.Popen([sys.executable, str(SRC / "vibevoice_stage.py"), str(MODELS / "vibevoice"),
                                vv_pieces, vv_out, str(VIBEVOICE_BATCH)], env=vv_env, stdout=vv_log, stderr=subprocess.STDOUT)
    log(f"vibevoice process started (pid {vv_proc.pid})")
    sea_dir = tempfile.mkdtemp(prefix="seamless_")
    sea_pieces, sea_out = os.path.join(sea_dir, "pieces.json"), os.path.join(sea_dir, "out.json")
    with open(sea_pieces, "w", encoding="utf-8") as h:
        json.dump([[p, d] for p, d in zip(ppaths, pdur)], h)
    sea_env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    sea_log = open(os.path.join(sea_dir, "seamless.log"), "w")
    sea_proc = subprocess.Popen([sys.executable, str(SRC / "seamless_stage.py"), str(MODELS / "seamless"),
                                 sea_pieces, sea_out, str(SEAMLESS_BATCH)], env=sea_env, stdout=sea_log, stderr=subprocess.STDOUT)
    log(f"seamless process started (pid {sea_proc.pid})")
    log("system 1/4")
    t = time.monotonic(); mer = run_meralion(ppaths); log(f"system 1/4 done in {time.monotonic() - t:.0f}s")
    log("system 2/4")
    t = time.monotonic(); qwen = run_qwen(ppaths); log(f"system 2/4 done in {time.monotonic() - t:.0f}s")
    log("system 3/4")
    t = time.monotonic(); whisper = run_whisper(ppaths, pdur); log(f"system 3/4 done in {time.monotonic() - t:.0f}s")
    log("system 4/4")
    t = time.monotonic(); ctc = run_ctc(ppaths); log(f"system 4/4 done in {time.monotonic() - t:.0f}s")
    # OWSM-CTC (an eighth candidate source) runs in its own process with the vendored ESPnet bundle on ITS path only.
    # It is started only after the VibeVoice and Seamless processes have ended, so it never joins their overlap with
    # MERaLiON (the pipeline's memory peak); from here it shares the GPU with omniASR alone. The exit codes of those
    # two processes are checked where they always were, below.
    vv_proc.wait()
    sea_proc.wait()
    vendor_o = SRC / "vendor_espnet_owsm"
    if not vendor_o.is_dir():
        with zipfile.ZipFile(SRC / "vendor_espnet_owsm.zip") as z:
            z.extractall(SRC)
    ow_dir = tempfile.mkdtemp(prefix="owsm_")
    ow_pieces, ow_out = os.path.join(ow_dir, "pieces.json"), os.path.join(ow_dir, "out.json")
    with open(ow_pieces, "w", encoding="utf-8") as h:
        json.dump([[p, d] for p, d in zip(ppaths, pdur)], h)
    ow_env = dict(os.environ, PYTHONPATH=str(vendor_o), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ow_log = open(os.path.join(ow_dir, "owsm.log"), "w")
    ow_proc = subprocess.Popen([sys.executable, str(SRC / "owsm_stage.py"), str(MODELS / "owsm"), ow_pieces, ow_out,
                                str(OWSM_BATCH), OWSM_LANG], env=ow_env, stdout=ow_log, stderr=subprocess.STDOUT)
    log(f"owsm process started (pid {ow_proc.pid})")
    log("system 5/6")
    t = time.monotonic()
    omni = run_omni(ppaths, started + OMNI_DEADLINE_MIN * 60)
    log(f"system 5/6 done in {time.monotonic() - t:.0f}s")

    # Anchor on Whisper for clips inside its window and on Qwen above it; the remaining
    # systems vote, and a slot changes only when at least two of them agree. The text stage
    # then applies the style gate and restores casing (see postprocess.py). Every step uses
    # only this clip's own transcripts.
    vv_proc.wait()
    vv_log.close()
    if vv_proc.returncode != 0:
        raise RuntimeError(f"vibevoice process failed with exit code {vv_proc.returncode}")
    with open(vv_out, encoding="utf-8") as h:
        vv_got = json.load(h)
    missing = [i for i in range(len(ppaths)) if str(i) not in vv_got]
    if missing:
        raise RuntimeError(f"vibevoice output lacks {len(missing)} of {len(ppaths)} pieces")
    vibevoice = [" ".join(vv_got[str(i)][j] for j in sorted(vv_got[str(i)], key=int)).strip() for i in range(len(ppaths))]
    log(f"vibevoice joined: {sum(1 for t in vibevoice if t.strip())}/{len(ppaths)} pieces with text")
    shutil.rmtree(vv_dir, ignore_errors=True)
    sea_proc.wait()
    sea_log.close()
    if sea_proc.returncode != 0:
        raise RuntimeError(f"seamless process failed with exit code {sea_proc.returncode}")
    with open(sea_out, encoding="utf-8") as h:
        sea_got = json.load(h)
    missing = [i for i in range(len(ppaths)) if str(i) not in sea_got]
    if missing:
        raise RuntimeError(f"seamless output lacks {len(missing)} of {len(ppaths)} pieces")
    seamless = [sea_got[str(i)].strip() for i in range(len(ppaths))]
    log(f"seamless joined: {sum(1 for t in seamless if t.strip())}/{len(ppaths)} pieces with text")
    shutil.rmtree(sea_dir, ignore_errors=True)
    ow_proc.wait()
    ow_log.close()
    if ow_proc.returncode != 0:
        raise RuntimeError(f"owsm process failed with exit code {ow_proc.returncode}")
    with open(ow_out, encoding="utf-8") as h:
        ow_got = json.load(h)
    missing = [i for i in range(len(ppaths)) if str(i) not in ow_got]
    if missing:
        raise RuntimeError(f"owsm output lacks {len(missing)} of {len(ppaths)} pieces")
    owsm = [canonicalize(str(ow_got[str(i)]).strip()) for i in range(len(ppaths))]
    log(f"owsm joined: {sum(1 for t in owsm if t.strip())}/{len(ppaths)} pieces with text")
    shutil.rmtree(ow_dir, ignore_errors=True)

    predictions, systems = [], []
    for q, w, m, c, o, v, s, ow in zip(qwen, whisper, mer, ctc, omni, vibevoice, seamless, owsm):
        fused, sys_texts = fuse_clip(q, w, m, c, o)
        predictions.append(fused)
        extra = [strict_spelling(normalize_text(x)) for x in (o, v, s, ow) if x.strip()]
        systems.append(sys_texts + extra)
    # Context rescoring (rescore.py): for each clip on its own, choose among the transcripts its
    # systems proposed, using a language model; nothing crosses between clips. Stops at the deadline.
    log("system 6/6")
    t = time.monotonic()
    predictions = rescore_all(predictions, systems, MODELS / "llm", MODELS / "lm" / "trigram.pkl",
                              deadline=started + RESCORE_DEADLINE_MIN * 60, log=log)
    log(f"system 6/6 done in {time.monotonic() - t:.0f}s")
    if len(predictions) != len(ppaths):
        raise RuntimeError("unexpected prediction count")
    guarded = [collapse_repeats(text) for text in predictions]
    log(f"repetition guard: {sum(1 for a, b in zip(predictions, guarded) if a != b)}/{len(guarded)} pieces shortened")
    predictions = guarded
    # join the pieces of each clip, in order
    joined = [[] for _ in paths]
    for i, text in zip(owner, predictions):
        if text.strip():
            joined[i].append(text.strip())
    predictions = [" ".join(parts) for parts in joined]
    shutil.rmtree(piece_dir, ignore_errors=True)
    if len(predictions) != len(metadata):
        raise RuntimeError("unexpected prediction count")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["audio_filename", "transcript"])
        w.writeheader()
        for name, text in zip(metadata["audio_filename"], predictions):
            w.writerow({"audio_filename": name, "transcript": text})
    tmp.replace(OUTPUT)
    with OUTPUT.open("r", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    if [r["audio_filename"] for r in rows] != metadata["audio_filename"].tolist():
        raise RuntimeError("output identifiers do not match manifest")
    log(f"completed in {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("fatal error")
        traceback.print_exc()
        sys.exit(1)
