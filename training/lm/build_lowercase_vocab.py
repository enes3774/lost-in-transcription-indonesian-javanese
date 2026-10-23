"""lowercase_vocab.txt of the inference package: words the pool writes lowercase in >= 97 % of >= 20 uses.

postprocess.py uses it to undo sentence-initial capitalisation that the references do not have. Built from
train_pool_v2.jsonl (the default inside style_postprocess.lowercase_vocabulary); 4,324 words.
usage: python build_lowercase_vocab.py <out.txt>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_postprocess as sp

vocab = sorted(sp.lowercase_vocabulary())
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text("\n".join(vocab), encoding="utf-8")
print(len(vocab), "words ->", sys.argv[1])
