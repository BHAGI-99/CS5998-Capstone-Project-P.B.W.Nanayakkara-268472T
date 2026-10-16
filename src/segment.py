"""Slice every round's continuous data.csv into labelled activity segments.

Verified dataset facts (checked on all 14 subjects x 10 rounds):
  * data.csv   : no header, 203 columns = [counter, timestamp_ms, 200 pressure
                 values (20x10 grid), empty trailing column]. ~52 Hz.
  * labels.csv : HAS a header: label, frame_idx_start, frame_idx_end.
                 label is 1..18 and appears in shuffled order within a round.
  * 2,520 segments total, exactly 140 per activity. Pressure range 0-4095.
  * ~100 of the 140 files contain ONE truncated row with NaNs (dropped packet); none
    lie inside a labelled segment (asserted below).
  * Segment length (frames): min 31, median 124, 95th pct 208, max 358.

Output (data/processed/):
  X.npy            (N, SEQ_LEN, 200) uint16, zero-padded at the end
  y.npy            (N,) int, activity 0..17
  subject.npy      (N,) int, subject 1..14   (needed for leave-one-subject-out)
  round.npy        (N,) int
  lengths.npy      (N,) int, true (unpadded) length of every segment

Run from the repo root:  python src/segment.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/Smart-Sleeve-Dataset/Smart-Sleeve Dataset")
OUT = Path("data/processed")
SEQ_LEN = 208          # 95th percentile of segment lengths; longer ones are truncated
N_SUBJECTS, N_ROUNDS = 14, 10


def load_round(subject: int, rnd: int):
    folder = RAW / f"subject_{subject}" / f"round {rnd}"
    data = pd.read_csv(folder / "data.csv", header=None)
    pressure = data.iloc[:, 2:202].to_numpy(dtype=np.float32)  # drop 2 timestamps + empty last col
    labels = pd.read_csv(folder / "labels.csv")
    return pressure, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, subj, rnd_ids, lens = [], [], [], [], []
    for s in range(1, N_SUBJECTS + 1):
        for r in range(1, N_ROUNDS + 1):
            pressure, labels = load_round(s, r)
            for _, row in labels.iterrows():
                a, b = int(row.frame_idx_start), int(row.frame_idx_end)
                seg = pressure[a:b + 1]                       # inclusive of end frame
                # Each round has one truncated (NaN) row between activities; it must never fall inside a segment.
                assert not np.isnan(seg).any(), f"NaN inside segment: subject {s} round {r} rows {a}-{b}"
                seg = seg.astype(np.uint16)
                n = min(len(seg), SEQ_LEN)
                padded = np.zeros((SEQ_LEN, 200), dtype=np.uint16)
                padded[:n] = seg[:n]
                X.append(padded); y.append(int(row.label) - 1)
                subj.append(s); rnd_ids.append(r); lens.append(len(seg))
    X = np.stack(X)
    np.save(OUT / "X.npy", X)
    np.save(OUT / "y.npy", np.array(y))
    np.save(OUT / "subject.npy", np.array(subj))
    np.save(OUT / "round.npy", np.array(rnd_ids))
    np.save(OUT / "lengths.npy", np.array(lens))
    print("X", X.shape, X.dtype, "| classes:", len(set(y)), "| subjects:", len(set(subj)))
    print("truncated segments:", int((np.array(lens) > SEQ_LEN).sum()))


if __name__ == "__main__":
    main()
