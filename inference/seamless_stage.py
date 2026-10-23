"""SeamlessM4T-v2 (speech -> text, fine-tuned on this track's pool) as a SEVENTH candidate source, run by main.py as
a separate process alongside the other stages, like the VibeVoice stage. It uses the runtime's own transformers
(4.57.6 supports SeamlessM4Tv2ForSpeechToText); nothing is vendored.

Decode: whole pieces (the pause splitter keeps them under 40 s), batch 16, greedy, target language `jav` - the
fine-tune's own evaluation setting (dev 0.2232). Then the same fixed runaway guard as VibeVoice's (over 4 words per
second, or one word 5+ times in a row -> the piece contributes no candidate; 3 of 372 dev clips). Results are
written after every batch. No exception is caught: a failure exits non-zero and main.py raises on it.
Usage: seamless_stage.py <model dir> <pieces json: [[path, duration], ...]> <out json> [batch]"""
import json, os, sys, time

MAX_WPS, MAX_RUN, TGT = 4.0, 5, "jav"


def runaway(text, seconds):
    ws = text.split()
    run = best = 1
    for a, b in zip(ws, ws[1:]):
        run = run + 1 if a.casefold() == b.casefold() else 1
        best = max(best, run)
    return len(ws) / max(seconds, 1.0) > MAX_WPS or best >= MAX_RUN


def main():
    import librosa, torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    model_dir, pieces_json, out_json = sys.argv[1:4]
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    pieces = json.load(open(pieces_json, encoding="utf-8"))
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    sr = proc.feature_extractor.sampling_rate
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    print(f"seamless: model loaded in {time.time()-t0:.0f}s; {len(pieces)} pieces; transformers "
          f"{__import__('transformers').__version__}", flush=True)
    work = []                                       # (piece index, waveform, seconds)
    for i, (path, dur) in enumerate(pieces):
        wav = librosa.load(path, sr=sr, mono=True)[0]
        work.append((i, wav, len(wav) / sr))
    work.sort(key=lambda w: -len(w[1]))
    print(f"seamless: audio loaded in {time.time()-t0:.0f}s", flush=True)
    got, dropped, t1 = {}, 0, time.time()
    for b in range(0, len(work), batch):
        part = work[b:b + batch]
        x = proc(audios=[w for _, w, _ in part], sampling_rate=sr, return_tensors="pt", padding=True)
        x = {k: v.cuda() for k, v in x.items()}
        with torch.inference_mode():
            y = model.generate(**x, tgt_lang=TGT, num_beams=1, max_new_tokens=256)
        for (i, _, secs), raw in zip(part, proc.batch_decode(y, skip_special_tokens=True)):
            text = " ".join(str(raw).split())
            if runaway(text, secs):
                text = ""; dropped += 1
            got[str(i)] = text
        json.dump(got, open(out_json + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(out_json + ".tmp", out_json)
        if (b // batch) % 25 == 0:
            print(f"seamless: {min(b + batch, len(work))}/{len(work)} pieces, {time.time()-t1:.0f}s", flush=True)
    print(f"seamless: done, {len(got)} pieces, {dropped} dropped by the guard, decode {time.time()-t1:.0f}s, "
          f"total {time.time()-t0:.0f}s", flush=True)
    print("SEAMLESS_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
