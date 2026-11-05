"""Download the official UCI Student Dropout and Academic Success dataset."""
from pathlib import Path
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "student_data.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        return CSV_PATH
    response = requests.get(UCI_ZIP_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        candidates = [n for n in archive.namelist()
                      if n.lower().endswith(".csv") and "data" in Path(n).name.lower()]
        if not candidates:
            candidates = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError("No CSV found in the UCI archive.")
        chosen = sorted(candidates, key=lambda n: (Path(n).name.lower() != "data.csv", len(n)))[0]
        CSV_PATH.write_bytes(archive.read(chosen))
    return CSV_PATH

if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset ready: {path}")
