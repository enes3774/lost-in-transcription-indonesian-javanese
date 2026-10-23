"""Context rescoring stage (rover6).

After the vote, every clip is turned into a confusion network: each word of the fused transcript is a slot,
and each system's aligned text proposes for that slot the same word, a substitute, a deletion, or the word
followed by an insertion. A beam over the slots (guided by a word trigram estimated on our own training
transcripts) keeps the NBEST most plausible full transcripts; a causal language model then scores each one,
and the transcript with the best combination of vote support, anchor agreement, length and LM score is
kept. Every candidate is built from this clip's own transcripts only; the LM sees one candidate at a time.
Weights were fixed on the development benches before this file was packaged.
"""
import gc
import math
import pickle
import time
from collections import Counter

import torch

PREFIX = "Transkrip percakapan lisan (bahasa Jawa dan bahasa Indonesia):\n"
NBEST = 24
BATCH = 32
# lam: per-system vote for the chosen alternative; mu: bonus when the fused word is kept;
# wip: per-word bonus (deleting a word is not free); llm: weight of the LM log-probability.
WEIGHTS = dict(lam=2.0, mu=0.0, wip=2.0, llm=0.25)


def _key(w):
    return w.casefold()


# ---------------------------------------------------------------------------
# Kneser-Ney word trigram (counts pickled from the training transcripts)
# ---------------------------------------------------------------------------

class Trigram:
    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.c3, self.c2, self.c1 = d["c3"], d["c2"], d["c1"]
        self.h3, self.t3, self.h2, self.t2 = d["h3"], d["t3"], d["h2"], d["t2"]
        self.N1, self.V, self.D = d["N1"], d["V"], d["D"]
        self.unk = self.D * len(self.c1) / self.V / self.N1 / 10

    def p1(self, c):
        if c not in self.c1:
            return self.unk
        return (max(self.c1[c] - self.D, 0) + self.D * len(self.c1) / self.V) / self.N1

    def p2(self, b, c):
        h = self.h2.get(b, 0)
        if h == 0:
            return self.p1(c)
        return (max(self.c2.get((b, c), 0) - self.D, 0) + self.D * self.t2[b] * self.p1(c)) / h

    def p3(self, a, b, c):
        h = self.h3.get((a, b), 0)
        if h == 0:
            return self.p2(b, c)
        return (max(self.c3.get((a, b, c), 0) - self.D, 0) + self.D * self.t3[(a, b)] * self.p2(b, c)) / h

    def logp(self, a, b, c):
        return math.log(self.p3(a, b, c))


# ---------------------------------------------------------------------------
# alignment and confusion network
# ---------------------------------------------------------------------------

def _chunks(ref, hyp):
    """Levenshtein alignment of two key lists, as runs of (type, r0, r1, h0, h1)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + (ri != hyp[j - 1]), dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append("equal"); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append("substitute"); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append("delete"); i -= 1
        else:
            ops.append("insert"); j -= 1
    ops.reverse()
    out = []
    i = j = 0
    for op in ops:
        di, dj = (op != "insert"), (op != "delete")
        if out and out[-1][0] == op:
            out[-1][2] += di; out[-1][4] += dj
        else:
            out.append([op, i, i + di, j, j + dj])
        i += di; j += dj
    return out


def slots(fused, systems):
    """Per slot: Counter over alternatives (keys) and the surface form each key was first seen with."""
    F = fused.split()
    FK = [_key(w) for w in F]
    alt = [Counter() for _ in F]
    form = [{} for _ in F]
    for s in systems:
        S = s.split()
        if not F or not S:
            continue
        SK = [_key(w) for w in S]
        cur = [None] * len(F)
        for op, r0, r1, h0, h1 in _chunks(FK, SK):
            rs = list(range(r0, r1)); hs = S[h0:h1]
            if op == "equal":
                for a in rs:
                    cur[a] = F[a]
            elif op == "substitute":
                for j, a in enumerate(rs):
                    cur[a] = hs[j] if j < len(hs) else ""
                if len(hs) > len(rs):
                    cur[rs[-1]] = " ".join(hs[len(rs) - 1:])
            elif op == "delete":
                for a in rs:
                    cur[a] = ""
            elif op == "insert" and r0 > 0:
                k = r0 - 1
                cur[k] = (cur[k] if cur[k] is not None else F[k]) + " " + " ".join(hs)
        for k in range(len(F)):
            c = cur[k] if cur[k] is not None else F[k]
            ck = _key(c)
            alt[k][ck] += 1
            form[k].setdefault(ck, c)
    for k in range(len(F)):
        form[k][FK[k]] = F[k]
    return F, FK, alt, form


def nbest(fused, systems, lm, n=NBEST):
    """Beam over the slots. Returns candidates as dicts: keys (for the LM), words (surface), votes, anch, nw."""
    F, FK, alt, form = slots(fused, systems)
    if not F:
        return []
    if all(len(a) == 1 for a in alt):
        return []                      # every system agrees with the vote: nothing to choose
    hyps = [(0, 0, [], [], 0.0)]       # votes, anchor matches, keys, words, trigram log-prob
    for k in range(len(F)):
        d = {}
        for v, a, keys, words, l in hyps:
            for ck, votes in alt[k].items():
                ks = ck.split() if ck else []
                ws = form[k][ck].split() if ck else []
                keys2 = keys + ks; l2 = l
                ctx = ["<s>", "<s>"] + keys2
                for j in range(len(keys), len(keys2)):
                    l2 += lm.logp(ctx[j], ctx[j + 1], ctx[j + 2])
                key = " ".join(keys2); v2 = v + votes; a2 = a + (ck == FK[k])
                if key not in d or d[key][0] < v2:
                    d[key] = (v2, a2, keys2, words + ws, l2)
        hyps = sorted(d.values(), key=lambda x: -(4 * x[0] + 2 * x[1] + 4 * len(x[2]) + x[4]))[:n]
    out = [dict(keys=" ".join(k), text=" ".join(w), votes=v, anch=a, nw=len(k)) for v, a, k, w, l in hyps]
    if all(o["keys"] != " ".join(FK) for o in out):
        out.append(dict(keys=" ".join(FK), text=fused, votes=sum(alt[k][FK[k]] for k in range(len(F))),
                        anch=len(F), nw=len(F)))
    return out


# ---------------------------------------------------------------------------
# LM scoring and the choice
# ---------------------------------------------------------------------------

def _score(model, tok, texts, pre, eos, deadline=None):
    """Sum of token log-probs of each text after the prefix. Batches are padded to a length bucket (multiple
    of 16) so tensor shapes repeat and the CUDA allocator reuses its blocks; without that, thousands of
    distinct shapes fragment the pool and the run slows down by an order of magnitude."""
    encs = [pre + tok(t, add_special_tokens=False).input_ids + [eos] for t in texts]
    order = sorted(range(len(encs)), key=lambda j: len(encs[j]))
    res = [None] * len(encs)
    p0 = len(pre) - 1
    for b in range(0, len(order), BATCH):
        if deadline is not None and time.monotonic() > deadline:
            break                      # out of time: the remaining clips keep the vote's transcript
        idx = order[b:b + BATCH]
        L = -(-max(len(encs[j]) for j in idx) // 16) * 16
        ids = torch.full((BATCH, L), eos, dtype=torch.long)
        mask = torch.zeros((BATCH, L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, :len(encs[j])] = torch.tensor(encs[j]); mask[r, :len(encs[j])] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, p0:-1]
            tgt = ids[:, p0 + 1:]
            lps = -torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape).cpu()
        for r, j in enumerate(idx):
            res[j] = float(lps[r, :len(encs[j]) - 1 - p0].sum())
        del logits, lps, ids, mask
    return res


def rescore_all(fused_list, systems_list, llm_dir, lm_path, weights=WEIGHTS, deadline=None, log=None):
    """fused_list[i]: the vote's transcript for clip i; systems_list[i]: that clip's system transcripts
    (already normalized and strict-spelled). Returns the rescored transcripts, same order. `deadline` is a
    time.monotonic() value after which no more scoring is done; clips not fully scored keep the vote's text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.monotonic()
    lm = Trigram(lm_path)
    cands = [nbest(f, s, lm) for f, s in zip(fused_list, systems_list)]
    del lm
    gc.collect()
    if log: log(f"rescoring: candidates built in {time.monotonic() - t0:.0f}s")
    todo = [(i, c) for i, cs in enumerate(cands) for c in cs]
    out = list(fused_list)
    if not todo or (deadline is not None and deadline - time.monotonic() < 300):
        return out                     # nothing to choose, or too little time left to load the LM
    t0 = time.monotonic()
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(str(llm_dir))
    model = AutoModelForCausalLM.from_pretrained(str(llm_dir), dtype=torch.bfloat16, device_map="cuda").eval()
    pre = tok(PREFIX, add_special_tokens=True).input_ids
    eos = tok.eos_token_id
    if log: log(f"rescoring: language model loaded in {time.monotonic() - t0:.0f}s")
    t0 = time.monotonic()
    scores = _score(model, tok, [c["keys"] for _, c in todo], pre, eos, deadline)
    if log: log(f"rescoring: scored in {time.monotonic() - t0:.0f}s")
    for (_, c), s in zip(todo, scores):
        c["llm"] = s
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    w = weights
    for i, cs in enumerate(cands):
        if cs and all(c["llm"] is not None for c in cs):
            best = max(cs, key=lambda c: w["lam"] * c["votes"] + w["mu"] * c["anch"] + w["wip"] * c["nw"] + w["llm"] * c["llm"])
            out[i] = best["text"]
    return out
