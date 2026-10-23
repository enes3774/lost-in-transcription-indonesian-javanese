"""Build the word trigram used by the rescoring stage (inference/models/lm/trigram.pkl).

Interpolated Kneser-Ney counts (discount D = 0.75) over every training transcript of the final pool, after the
competition normaliser and casefolding. rescore.py reads exactly these keys. CPU only, a few minutes.

usage:  python build_trigram.py <train_pool.jsonl> <out.pkl>
        (the pool is data/processed/qwen_v2/train_pool_v4.jsonl, see training/data_prep/README in training/README.md)
"""
import collections
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from score import normalize_text   # the organizers' normaliser (runtime_repo/score.py)


def toks(t):
    return normalize_text(t).casefold().split()


pool, out = Path(sys.argv[1]), Path(sys.argv[2])
texts = collections.Counter()
n = 0
for line in open(pool, encoding="utf-8"):
    w = toks(json.loads(line)["text_ref1"])
    if w:
        texts[" ".join(w)] += 1
        n += len(w)
print(f"  {pool.name:28s} {n / 1e6:6.2f} M words", flush=True)

D = 0.75
c3 = collections.Counter(); c2 = collections.Counter(); c1 = collections.Counter()
for s, k in texts.items():
    w = ["<s>", "<s>"] + s.split() + ["</s>"]
    for i in range(2, len(w)):
        c3[(w[i - 2], w[i - 1], w[i])] += k
for (a, b, c), k in c3.items():
    c2[(b, c)] += 1
for (b, c), k in c2.items():
    c1[c] += 1
h3 = collections.Counter(); t3 = collections.Counter()
for (a, b, c), k in c3.items():
    h3[(a, b)] += k; t3[(a, b)] += 1
h2 = collections.Counter(); t2 = collections.Counter()
for (b, c), k in c2.items():
    h2[b] += k; t2[b] += 1
N1 = sum(c1.values()); V = len(c1) + 1
print("vocab", len(c1), "trigrams", len(c3), flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(dict(c3=dict(c3), c2=dict(c2), c1=dict(c1), h3=dict(h3), t3=dict(t3), h2=dict(h2), t2=dict(t2),
                 N1=N1, V=V, D=D), open(out, "wb"), protocol=4)
print("dumped trigram to", out, flush=True)
