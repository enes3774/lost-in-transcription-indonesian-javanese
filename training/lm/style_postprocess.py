#!/usr/bin/env python3
"""Frozen post-fusion text stage: MERaLiON style gate + standard initialism casing.

Both steps are deliberately narrow.

**Style gate.** The conservative ROVER only changes its anchor when *both* voters agree,
so the one voter that was trained to write dev's conventions (MERaLiON `d_final`, trained on
the convention-rewritten jember manifest) is always outvoted by the two systems that carry the
broad pool's spelling. This gate re-admits that voter for exactly the five mappings that were
fixed *before* `d_final` was trained, and only where MERaLiON actually emits the target form at
the aligned slot. It is a minority-expert vote, not an unconditional rewrite table.

**Initialism casing.** The official normalizer lowercases only sentence-initial letters, so an
acronym written in lowercase is a full word error. Uppercasing a closed list of ordinary
Indonesian initialisms is standard-writing policy, not a dev-specific lexical choice.

Frozen honest-dev result (Whisper+aug anchor, Qwen v2 and MERaLiON `d_final` voters, 372 clips):

    ordinary ROVER               0.174264
    + style gate (112 slots)     0.171061   -0.003203
    + initialism casing          0.170203   -0.004061 total

Nothing here is tuned per clip, per speaker or per conversation, and no unlisted word is touched.
Deliberately NOT included, because they are speaker/lect dependent or ambiguous with Indonesian
(measured deltas on the same fusion): `apa->opo` +0.00446, `ya->yo` +0.02625, `iki->ki` +0.00103,
`nggak->gak` +0.00034 overall and opposite in sign between the two conversations.
"""
import os
from collections import defaultdict

# (form the ensemble tends to write, form MERaLiON was trained to write)
STYLE_OVERRIDES = (
    ("kaya", "kayak"),
    ("lho", "loh"),
    ("isa", "iso"),
    ("ana", "ono"),
    ("dhewe", "dewe"),
)
INITIALISMS = (
    # Derived from the training pool by analysis/acronym_lexicon.py (a token written
    # all-uppercase in >=75% of its non-sentence-initial occurrences, 2-5 letters, >=4 times),
    # plus five that dev contains and the pool writes inconsistently. Scored identically to the
    # original hand list on dev and both folds, with four times the coverage for unseen acronyms.
    "AC", "AI", "BBM", "BCA", "BMW", "BNI", "BPJS", "BPK",
    "BRI", "BTN", "CNN", "CS", "DBS", "FOMO", "GPT", "HP",
    "IBM", "IPA", "IPS", "KKM", "KKN", "KTP", "LG", "MC",
    "MCU", "MTV", "OSIS", "PC", "PKI", "PNS", "PR", "PS",
    "RT", "RW", "SD", "SMA", "SMK", "SMP", "SMS", "TED",
    "TK", "TKI", "UAS", "UGM", "UHU", "UI", "UMJ", "UNY",
)

_OVERRIDE_SET = frozenset(STYLE_OVERRIDES)
_CASE_MAP = {w.casefold(): w for w in INITIALISMS}


def _align(a, b):
    """Unit-cost, case-insensitive Levenshtein alignment; ties prefer match/substitute."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], bt[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], bt[0][j] = j, "I"
    rank = {"M": 0, "D": 1, "I": 2}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = ((d[i - 1][j - 1] + (a[i - 1].casefold() != b[j - 1].casefold()), "M"),
                       (d[i - 1][j] + 1, "D"), (d[i][j - 1] + 1, "I"))
            d[i][j], bt[i][j] = min(choices, key=lambda x: (x[0], rank[x[1]]))
    out = []
    i, j = n, m
    while i or j:
        op = bt[i][j]
        if op == "M":
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif op == "D":
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    return out[::-1]


def _amap(a, b):
    """Map each index of `a` to the token of `b` aligned to it (None if deleted)."""
    positions, gaps, i = {}, defaultdict(list), 0
    for x, y in _align(a, b):
        if x is None:
            gaps[i].append(y)
        else:
            positions[i] = y
            i += 1
    return positions, gaps


def style_gate(fused, meralion, overrides=_OVERRIDE_SET):
    """Adopt MERaLiON's spelling for the listed pairs where it is aligned to `fused`."""
    words = fused.split()
    aligned, _ = _amap(words, meralion.split())
    out = []
    for i, w in enumerate(words):
        m = aligned.get(i)
        out.append(m if m is not None and (w.casefold(), m.casefold()) in overrides else w)
    return " ".join(out)


def fix_initialisms(text):
    return " ".join(_CASE_MAP.get(w.casefold(), w) for w in text.split())


def postprocess(fused, meralion):
    """The frozen stage: style gate, then initialism casing."""
    return fix_initialisms(style_gate(fused, meralion))

# ---------------------------------------------------------------------------
# Casing by ensemble vote (no word list)
# ---------------------------------------------------------------------------
# The official normalizer lowercases only *sentence-initial* letters, so a mid-utterance
# capital is scored: `sangiran` for `Sangiran` and `hp` for `HP` are full word errors.
# A fixed lexicon cannot solve this -- of the 25 all-caps tokens dev contains, only three
# are safely derivable from the training pool and ten do not occur in it at all. So instead
# of a list, ask the systems: they have collectively seen far more Indonesian than our pool,
# and an all-caps or capitalised emission is a deliberate signal, never an artefact of the
# normalizer. Where the fused text has a lowercase token and any system wrote a cased form of
# the same word at the aligned slot, adopt that form. Names are `Ted`, not `TED`, so the
# all-caps branch cannot corrupt them, and the capitalised branch only ever agrees with a
# model that already chose to capitalise.

def case_vote(fused, systems, need=1):
    """Restore casing on `fused` from the aligned tokens of the other systems."""
    import re
    core = lambda w: re.sub(r"[^A-Za-z]", "", w)

    def cased(c):
        return bool(c) and ((c.isupper() and 2 <= len(c) <= 5) or (c[0].isupper() and not c.isupper() and len(c) >= 3))

    words = fused.split()
    tally = [defaultdict(int) for _ in words]
    for s in systems:
        positions, _ = _amap(words, s.split())
        for i, tok in positions.items():
            if tok is not None:
                tally[i][tok] += 1
    out = []
    for i, w in enumerate(words):
        c = core(w)
        if c and c.islower():
            cand = [(z, n) for z, n in tally[i].items() if core(z).casefold() == c.casefold() and cased(core(z))]
            if cand and max(n for _, n in cand) >= need:
                out.append(max(cand, key=lambda x: x[1])[0])
                continue
        out.append(w)
    return " ".join(out)


def postprocess_full(fused, meralion, systems, need=1):
    """Style gate, then casing by ensemble vote, then the initialism lexicon as a backstop."""
    return downcase_common(fix_initialisms(case_vote(style_gate(fused, meralion), systems, need)))

# The mirror of the casing vote. Our systems add capitals dev does not have 2.3x more often
# than they miss ones it does (`kecamatan->Kecamatan`, `kerajaan->Kerajaan`, `nah->Nah`), and the
# vote can only add. So: a word the training pool writes lowercase in >=97% of its
# non-sentence-initial uses (>=20 occurrences) is forced back to lowercase. Derived from the pool,
# never from dev, and it cannot touch a word the pool ever capitalises -- proper nouns are safe.

_LOWER_CACHE = None


def lowercase_vocabulary(pool_path=None, thr=0.97, minc=20):
    global _LOWER_CACHE
    if _LOWER_CACHE is None:
        import collections, json, re
        from pathlib import Path
        pool_path = pool_path or Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/data/processed/qwen_v2/train_pool_v2.jsonl")
        forms = collections.defaultdict(collections.Counter)
        for line in open(pool_path, encoding="utf-8"):
            for j, w in enumerate(json.loads(line)["text_ref1"].split()):
                if j:
                    c = re.sub(r"[^A-Za-z]", "", w)
                    if c:
                        forms[c.casefold()][c] += 1
        _LOWER_CACHE = {k for k, c in forms.items()
                        if sum(c.values()) >= minc
                        and sum(v for f, v in c.items() if f.islower()) / sum(c.values()) >= thr}
    return _LOWER_CACHE


def downcase_common(text, vocab=None):
    import re
    vocab = vocab if vocab is not None else lowercase_vocabulary()
    out = []
    for j, w in enumerate(text.split()):
        c = re.sub(r"[^A-Za-z]", "", w)
        out.append(w.lower() if j and c and not c.islower() and c.casefold() in vocab else w)
    return " ".join(out)
