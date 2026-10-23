"""data/processed/qwen_style_aggressive/all_dev_train.jsonl: the v2 pool + all 372 dev clips (source=style_dev).

Training manifest of the one-pass dev adaptation of Whisper and MERaLiON. Needs the outputs of
training/qwen/stage2_dev/prepare.py (speaker_hold1_train.jsonl, all_dev_by_speaker.jsonl). Originally an inline step
on 2026-09-05; this is that code, unchanged except that the second (unused) manifest variant is left out."""
import os
import json, collections
R=os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'); S=R+'/data/processed/qwen_style_aggressive'
dev=[json.loads(l) for l in open(f'{S}/all_dev_by_speaker.jsonl')]
for r in dev: r['source']='style_dev'
for src,dst in [('speaker_hold1_train.jsonl','all_dev_train.jsonl')]:
    pool=[json.loads(l) for l in open(f'{S}/{src}')]
    pool=[r for r in pool if r.get('source')!='style_dev']
    rows=pool+dev
    with open(f'{S}/{dst}','w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    c=collections.Counter(r['source'] for r in rows)
    print(dst, len(rows), {k:c[k] for k in ('style_dev','jember','homostoria') if k in c})
