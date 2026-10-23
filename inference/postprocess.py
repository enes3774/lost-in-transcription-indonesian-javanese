"""Post-fusion text stages. Self-contained: no training data is read at run time.

Three stages, in order, each derived from the training pool or from the systems themselves
rather than from the development set:

1. **Style gate.** ROVER changes its anchor only when the voters agree, which systematically
   silences the one system trained to write the competition's conventions whenever the others
   share the broad pool's spelling. For five pre-declared pairs -- the same five used to build
   that system's training target -- MERaLiON's spelling is adopted where it is aligned to the
   fused text. A minority-expert vote, not an unconditional rewrite table.
2. **Casing by ensemble vote.** The official normalizer lowercases only *sentence-initial*
   letters, so a mid-utterance capital is scored: `sangiran` for `Sangiran` and `hp` for `HP`
   are full word errors. A fixed lexicon cannot cover this (of the initialisms in a held-out
   set, most do not occur in the training pool at all), so the systems are asked instead: where
   the fused text has a lowercase token and any system wrote a cased form of the same word at
   the aligned slot, that form is adopted. Names are `Ted`, not `TED`, so the all-caps branch
   cannot corrupt them.
3. **Casing lexicon and downcase.** A closed list of initialisms derived from the pool as a
   backstop, then the mirror rule: a word the pool writes lowercase in at least 97 % of its
   non-sentence-initial uses is forced back to lowercase. The systems add capitals the
   references do not have about twice as often as they miss ones they do, and the vote can only
   add, so this stage removes the spurious ones. It cannot touch a word the pool ever
   capitalises, so proper nouns are safe.
"""
from collections import defaultdict
from pathlib import Path
import re

# (form the ensemble tends to write, form MERaLiON was trained to write)
STYLE_OVERRIDES = frozenset((
    ("kaya", "kayak"),
    ("lho", "loh"),
    ("isa", "iso"),
    ("ana", "ono"),
    ("dhewe", "dewe"),
))

# Tokens written all-uppercase in >=75% of their non-sentence-initial occurrences in the
# training pool (>=4 occurrences, 2-5 letters), plus five the pool writes inconsistently.
INITIALISMS = (
    "AC", "AI", "BBM", "BCA", "BMW", "BNI", "BPJS", "BPK",
    "BRI", "BTN", "CNN", "CS", "DBS", "FOMO", "GPT", "HP",
    "IBM", "IPA", "IPS", "KKM", "KKN", "KTP", "LG", "MC",
    "MCU", "MTV", "OSIS", "PC", "PKI", "PNS", "PR", "PS",
    "RT", "RW", "SD", "SMA", "SMK", "SMP", "SMS", "TED",
    "TK", "TKI", "UAS", "UGM", "UHU", "UI", "UMJ", "UNY",
)
_CASE_MAP = {w.casefold(): w for w in INITIALISMS}
_CORE = lambda w: re.sub(r"[^A-Za-z]", "", w)

_LOWER = None


def lowercase_vocabulary(path=None):
    """Words the training pool writes lowercase; frozen to a file at packaging time."""
    global _LOWER
    if _LOWER is None:
        p = Path(path) if path else Path(__file__).resolve().parent / "lowercase_vocab.txt"
        _LOWER = frozenset(p.read_text(encoding="utf-8").split()) if p.exists() else frozenset()
    return _LOWER


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


def _positions(a, b):
    """Token of `b` aligned to each index of `a` (None where `b` deletes it)."""
    out, i = {}, 0
    for x, y in _align(a, b):
        if x is None:
            continue
        out[i] = y
        i += 1
    return out


def style_gate(fused, meralion):
    words = fused.split()
    aligned = _positions(words, meralion.split())
    out = []
    for i, w in enumerate(words):
        m = aligned.get(i)
        out.append(m if m is not None and (w.casefold(), m.casefold()) in STYLE_OVERRIDES else w)
    return " ".join(out)


def case_vote(fused, systems, need=1):
    def cased(c):
        return bool(c) and ((c.isupper() and 2 <= len(c) <= 5)
                            or (c[0].isupper() and not c.isupper() and len(c) >= 3))

    words = fused.split()
    tally = [defaultdict(int) for _ in words]
    for s in systems:
        for i, tok in _positions(words, s.split()).items():
            if tok is not None:
                tally[i][tok] += 1
    out = []
    for i, w in enumerate(words):
        c = _CORE(w)
        if c and c.islower():
            cand = [(z, n) for z, n in tally[i].items()
                    if _CORE(z).casefold() == c.casefold() and cased(_CORE(z))]
            if cand and max(n for _, n in cand) >= need:
                out.append(max(cand, key=lambda x: x[1])[0])
                continue
        out.append(w)
    return " ".join(out)


def fix_initialisms(text):
    return " ".join(_CASE_MAP.get(w.casefold(), w) for w in text.split())


def downcase_common(text):
    vocab = lowercase_vocabulary()
    out = []
    for j, w in enumerate(text.split()):
        c = _CORE(w)
        out.append(w.lower() if j and c and not c.islower() and c.casefold() in vocab else w)
    return " ".join(out)


# Strict spelling: forms the development references write one way almost without exception. Each pair
# was kept only if rewriting it wins on both the dev bench and the held-out fold bench by >= 3 to 1.
STRICT_SPELLING = {
    "isa": "iso", "kaya": "kayak", "pengin": "pengen", "nduwe": "duwe", "mesthi": "mesti",
    "pake": "pakai", "sik": "sek", "rada": "rodo", "temanku": "temenku", "taun": "tahun",
    "njok": "njuk", "adikku": "adekku", "kanca-kanca": "konco-konco",
}
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$")


def strict_spelling(text):
    out = []
    for tok in text.split():
        pre, core, post = _EDGE.match(tok).groups()
        new = STRICT_SPELLING.get(core.casefold())
        if new is None:
            out.append(tok)
        else:
            if core[:1].isupper():
                new = new[:1].upper() + new[1:]
            out.append(pre + new + post)
    return " ".join(out)


def postprocess(fused, meralion, systems):
    """Style gate, casing vote, initialism lexicon, then the pool's lowercase rule."""
    return downcase_common(fix_initialisms(case_vote(style_gate(fused, meralion), systems)))

