import os
from pathlib import Path
import pandas as pd
ROOT=Path(os.environ.get('LIT_ROOT', '/home/jovyan/work/Enes/lost_in_transcription'))
src=ROOT/'data/raw/dev/indonesian_dev'
df=pd.read_csv(src/'metadata.tsv',sep='\t',keep_default_na=False)
df['path']=df.audio_filename.map(lambda x:str(src/'clips'/x))
assert len(df)==372 and df.path.map(lambda x:Path(x).exists()).all()
out=ROOT/'data/processed/dev';out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/'manifest.csv',index=False)
print(df.language.value_counts().to_string());print(f'DONE {len(df)} -> {out/"manifest.csv"}')
