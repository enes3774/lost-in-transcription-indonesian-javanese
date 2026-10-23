"""Turn a folder of audio files into the data directory main.py reads (the competition's layout).

usage:  python prepare_data.py <audio_folder> <work_dir>

Writes <work_dir>/data/clips/<file> (audio copied as-is) and <work_dir>/data/test_metadata.csv with the columns
audio_filename, file_duration_seconds, language. Duration only steers batching and the cutting of long
clips; the language column is required by the schema check but not used.
Needs soundfile (librosa as fallback for mp3/m4a)."""
import csv
import shutil
import sys
from pathlib import Path

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}


def duration(path):
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        import librosa
        return float(librosa.get_duration(path=str(path)))


src, work = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO)
if not files:
    sys.exit(f"no audio files in {src}")
(work / "data" / "clips").mkdir(parents=True, exist_ok=True)
with open(work / "data" / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["audio_filename", "file_duration_seconds", "language"])
    for p in files:
        shutil.copy(p, work / "data" / "clips" / p.name)
        w.writerow([p.name, f"{duration(p):.2f}", "jv"])
print(f"{len(files)} clips -> {work / 'data'}")
