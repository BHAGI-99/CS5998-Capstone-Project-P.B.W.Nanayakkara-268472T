"""Download the Smart-Sleeve dataset into data/raw/.

Source: https://github.com/xghgithub/Smart-Sleeve-Dataset (Apache-2.0)
Paper : Xu et al., "Smart-Sleeve: A Wearable Textile Pressure Sensor Array
        for Human Activity Recognition", Sensors 2022, 22(5), 1702.
        https://doi.org/10.3390/s22051702

Requires git. Run from the repo root:  python src/download_data.py
"""
import subprocess
from pathlib import Path

URL = "https://github.com/xghgithub/Smart-Sleeve-Dataset.git"
DEST = Path("data/raw/Smart-Sleeve-Dataset")

if DEST.exists():
    print(f"Already downloaded: {DEST}")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", URL, str(DEST)], check=True)
    print(f"Done. Data folder: {DEST / 'Smart-Sleeve Dataset'}")
