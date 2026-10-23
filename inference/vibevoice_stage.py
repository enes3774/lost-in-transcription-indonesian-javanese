"""VibeVoice-ASR as a sixth candidate source, run by main.py as a SEPARATE PROCESS alongside the other stages.

Separate because VibeVoice needs transformers >= 5.3 while the runtime pins 4.57.6: main.py starts this script with
PYTHONPATH pointing at the vendored transformers 5 (vendor_tf/), so nothing else in the package sees it. It shares
the GPU with whatever stage main.py is running - measured on the same 96 clips, two HF generate() stages on one card
finish in 61 % of their sequential time and slow each other by at most 17 %.

Decode: 30 s windows, batch 8, greedy, no prompt - the fine-tune's own evaluation setting. Every window's text is
parsed out of the model's JSON-like output, then a fixed guard drops a window that is clearly a runaway
(over 4 words per second, or one word 5+ times in a row): +0.029 on the weighted dev score without the guard, -0.003 with it.
Results are written after every batch, so a deadline kill in main.py keeps every window finished so far.
No exception is caught here: a failure exits non-zero and main.py raises on it.
Usage: vibevoice_stage.py <model dir> <pieces json: [[path, duration], ...]> <out json> [batch]"""
import json, re, sys, time

CONTENT = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"')
SPECIAL = re.compile(r"<\|[^|]*\|>")
PREAMBLE = re.compile(r"^\s*(here\s+is\s+(the\s+)?(verbatim\s+)?transcript(ion)?|transcript(ion)?|"
                      r"berikut\s+(adalah\s+)?transkrip(si)?(nya)?|transkrip(si)?)\s*[:\-]\s*", re.IGNORECASE)
WINDOW_S, MAX_WPS, MAX_RUN = 30.0, 4.0, 5


def parse(raw):
    txt = SPECIAL.sub(" ", raw)
    parts = CONTENT.findall(txt)
    if parts:
        out = []
        for p in parts:
            try:
                out.append(json.loads('"' + p + '"'))
            except json.JSONDecodeError:
                out.append(p.replace('\\"', '"'))
        t = " ".join(s.strip() for s in out if s.strip())
    elif '"Content":' in txt:                      # generation cut off before the closing quote
        t = txt.split('"Content":', 1)[1].lstrip(' "')
        t = re.sub(r'"?\s*\}?\s*\]?\s*$', "", t)
    else:
        t = re.sub(r"\s+", " ", txt.replace("assistant", " ", 1)).strip()
    t = PREAMBLE.sub("", t.strip()).strip()
    if len(t) > 1 and t[0] in '"«“' and t[-1] in '"»”':
        t = t[1:-1].strip()
    return " ".join(t.split())


def runaway(text, seconds):
    ws = text.split()
    run = best = 1
    for a, b in zip(ws, ws[1:]):
        run = run + 1 if a.casefold() == b.casefold() else 1
        best = max(best, run)
    return len(ws) / max(seconds, 1.0) > MAX_WPS or best >= MAX_RUN


def main():
    import librosa, torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    model_dir, pieces_json, out_json = sys.argv[1:4]
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    pieces = json.load(open(pieces_json, encoding="utf-8"))
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    sr = proc.feature_extractor.sampling_rate
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager", device_map={"": 0}).eval()
    print(f"vibevoice: model loaded in {time.time()-t0:.0f}s; {len(pieces)} pieces; transformers "
          f"{__import__('transformers').__version__}", flush=True)
    win = int(WINDOW_S * sr)
    work = []                                       # (piece index, window index, waveform, seconds)
    for i, (path, dur) in enumerate(pieces):
        wav = librosa.load(path, sr=sr, mono=True)[0]
        chunks = [wav] if len(wav) <= win else [wav[k:k + win] for k in range(0, len(wav), win)]
        for j, c in enumerate(chunks):
            work.append((i, j, c, len(c) / sr))
    work.sort(key=lambda w: -len(w[2]))
    print(f"vibevoice: audio loaded in {time.time()-t0:.0f}s; {len(work)} windows", flush=True)
    got, dropped, t1 = {}, 0, time.time()
    for b in range(0, len(work), batch):
        part = work[b:b + batch]
        x = proc.apply_transcription_request(audio=[c for _, _, c, _ in part], sampling_rate=sr)
        x = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.cuda()) for k, v in x.items()}
        with torch.inference_mode():
            y = model.generate(**x, max_new_tokens=512, do_sample=False)
        for (i, j, _, secs), raw in zip(part, proc.decode(y[:, x["input_ids"].shape[1]:])):
            text = parse(str(raw))
            if runaway(text, secs):
                text = ""; dropped += 1
            got.setdefault(str(i), {})[str(j)] = text
        json.dump(got, open(out_json + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        import os
        os.replace(out_json + ".tmp", out_json)
        if (b // batch) % 25 == 0:
            print(f"vibevoice: {min(b + batch, len(work))}/{len(work)} windows, {time.time()-t1:.0f}s", flush=True)
    print(f"vibevoice: done, {len(got)} pieces, {dropped} windows dropped by the guard, decode {time.time()-t1:.0f}s, "
          f"total {time.time()-t0:.0f}s", flush=True)
    print("VIBEVOICE_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
