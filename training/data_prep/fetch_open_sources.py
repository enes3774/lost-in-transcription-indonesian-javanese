"""Download the two openly licensed sources that are not on MDC, in the layout prepare_qwen_v2_data.py reads.

  OpenSLR-35 Javanese (CC BY-SA 4.0, https://www.openslr.org/35/) from the Parquet mirror ctaguchi/SLR35_javanese
      -> data/raw/openslr35_javanese/hf_dataset/data/train-000NN-of-00038.parquet
  FLEURS (CC BY 4.0) google/fleurs, configurations id_id and jv_id, revision 70bb2e84
      -> data/raw/fleurs_id_jv/{id_id,jv_id}   (datasets.save_to_disk)
This step was done by hand during the competition; this script fetches the same files."""
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

RAW = Path(os.environ.get("LIT_ROOT", "/home/jovyan/work/Enes/lost_in_transcription") + "/data/raw")
snapshot_download("ctaguchi/SLR35_javanese", repo_type="dataset", local_dir=str(RAW / "openslr35_javanese/hf_dataset"))
for cfg in ("id_id", "jv_id"):
    ds = load_dataset("google/fleurs", cfg, revision="70bb2e84b976b7e960aa89f1c648e09c59f894dd", trust_remote_code=True,
                      cache_dir=str(RAW / "fleurs_id_jv/hf_cache"))
    ds.save_to_disk(str(RAW / "fleurs_id_jv" / cfg))
print("open sources ready under", RAW)
