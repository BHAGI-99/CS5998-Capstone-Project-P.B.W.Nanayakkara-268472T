"""Plot total pressure over time for a few segments, to eyeball that the
label start/end frames line up with real activity. Run after segment.py."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("data/processed")
X, y = np.load(P / "X.npy"), np.load(P / "y.npy")
lens = np.load(P / "lengths.npy")

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
for ax, cls in zip(axes.ravel(), [0, 3, 6, 9, 12, 15]):
    idx = np.where(y == cls)[0][:3]
    for i in idx:
        n = min(lens[i], X.shape[1])
        ax.plot(X[i, :n].astype(float).sum(axis=1), alpha=.8)
    ax.set_title(f"Activity {cls + 1}")
    ax.set_xlabel("frame")
axes[0, 0].set_ylabel("total pressure")
plt.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("results/figures/segment_examples.png", dpi=120)
print("saved results/figures/segment_examples.png")
