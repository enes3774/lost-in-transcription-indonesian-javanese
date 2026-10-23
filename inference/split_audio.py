"""Cut long audio at pauses, with signal processing only: no model, no alignment.

20 ms frame energy smoothed over 0.3 s; from the current start, look only inside a window of MIN_PIECE_S to
MAX_PIECE_S and cut at the quietest frame there; repeat while the tail is still too long, and fold a tail
under 3 s into the piece before it. Clips at or under MAX_PIECE_S are never touched, so nothing changes for
ordinary clips."""
import numpy as np

SR = 16000
FRAME_S, SMOOTH_S = 0.02, 0.3
MIN_PIECE_S, MAX_PIECE_S = 12.0, 28.0
# Only clips longer than this are cut at all.
SPLIT_TRIGGER_S = 30.0


def cut_points(wav, sr=SR, min_s=MIN_PIECE_S, max_s=MAX_PIECE_S):
    """Sample indices where each piece starts; [0] when the clip is short enough."""
    f = int(FRAME_S * sr)
    n = len(wav) // f
    if n == 0 or len(wav) / sr <= max_s:
        return [0]
    e = np.abs(wav[:n * f].reshape(n, f)).mean(1)
    k = max(int(SMOOTH_S / FRAME_S), 1)
    pad = np.pad(e, (k // 2, k - k // 2 - 1), mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")
    starts, s = [0], 0
    while (len(wav) - s) / sr > max_s:
        fs = s // f
        lo, hi = fs + int(min_s / FRAME_S), min(fs + int(max_s / FRAME_S), n - 1)
        if hi <= lo:
            break
        cut = (lo + int(np.argmin(sm[lo:hi]))) * f
        if cut <= s:
            break
        starts.append(cut)
        s = cut
    if len(starts) > 1 and (len(wav) - starts[-1]) / sr < 3.0:
        starts.pop()
    return starts


def split(wav, sr=SR, min_s=MIN_PIECE_S, max_s=MAX_PIECE_S):
    """The clip as a list of pieces, in order; one element when it is short enough."""
    st = cut_points(wav, sr, min_s, max_s) + [len(wav)]
    return [wav[st[i]:st[i + 1]] for i in range(len(st) - 1)]
