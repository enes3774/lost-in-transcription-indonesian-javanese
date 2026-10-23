# train_pool_v2_jemberconv.jsonl: the v2 pool with five Jember spellings rewritten to the form the dev transcripts use.
# Originally an inline step on 2026-09-05; this is that code, unchanged. Re-running it on train_pool_v2.jsonl reproduces
# the file byte for byte (1,566 jember rows: kaya->kayak 696, ana->ono 531, isa->iso 421, dhewe->dewe 339, lho->loh 138).
import os
import json, re, collections
R=os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription')
# jember-only rewrites toward dev's majority convention (dev share of the kept form in parentheses)
RULES={'kaya':'kayak','lho':'loh','dhewe':'dewe','isa':'iso','ana':'ono'}   # .94 .72 .73 .95 .62
pat=re.compile(r"(?<![\w'-])(" + '|'.join(map(re.escape, RULES)) + r")(?![\w'-])", re.I)
def rw(t):
    def sub(m):
        w=m.group(1); r=RULES[w.lower()]; return r.capitalize() if w[0].isupper() else r
    return pat.sub(sub, t)
n=collections.Counter(); out=[]
for line in open(f'{R}/data/processed/qwen_v2/train_pool_v2.jsonl', encoding='utf-8'):
    r=json.loads(line)
    if r['source']=='jember':
        for k in ('text_ref1','text_ref2'):
            if r.get(k):
                new=rw(r[k]); n[k]+=sum(1 for _ in pat.finditer(r[k])); r[k]=new
    out.append(r)
with open(f'{R}/data/processed/qwen_v2/train_pool_v2_jemberconv.jsonl','w', encoding='utf-8') as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False)+'\n')
print('rewrote jember tokens:', dict(n), '-> train_pool_v2_jemberconv.jsonl', len(out), 'rows')
