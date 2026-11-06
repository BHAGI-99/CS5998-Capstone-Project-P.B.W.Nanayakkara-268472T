"""Plausibility check: does the CNN confuse the activities whose pressure patterns look alike?

Needs no anatomical ground truth (the dataset does not say which end of the sleeve is the wrist).
Uses the saved leave-one-subject-out predictions (results/cnn_loso_preds.npy) and the data.
Run:  python -m src.plausibility
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from src.common import ACTIVITIES, load_data

X, y, subj, L = load_data()
cands = [Path("results/cnn_loso_preds.npy"), Path("results/figures/cnn_loso_preds.npy")]
pred_file = next((p for p in cands if p.exists()), None)
assert pred_file is not None, "Run the CNN LOSO cell first (it saves results/cnn_loso_preds.npy)"
pred = np.load(pred_file); assert len(pred) == len(y), "predictions and data must have the same number of trials"
Path("results/figures").mkdir(parents=True, exist_ok=True)

# mean pressure map (20x10) of every activity, from the true frames only
M = np.stack([np.mean([X[i, :L[i]].mean(0) for i in np.where(y == c)[0]], axis=0) for c in range(18)])   # (18,200)
cos = (M / np.linalg.norm(M, axis=1, keepdims=True)) @ (M / np.linalg.norm(M, axis=1, keepdims=True)).T

cm = confusion_matrix(y, pred, labels=range(18)).astype(float); cm /= cm.sum(1, keepdims=True)           # row-normalised
conf = (cm + cm.T) / 2                                                                                   # symmetric confusion rate
iu = np.triu_indices(18, 1)
rho, p = spearmanr(cos[iu], conf[iu])
print(f"LOSO accuracy from the saved predictions: {(pred == y).mean():.4f}")
print(f"Spearman correlation between map similarity and confusion rate over the 153 activity pairs: rho={rho:.2f} (p={p:.2g})")

order = np.argsort(-conf[iu])[:8]
print("\nMost confused activity pairs (symmetric confusion rate, map cosine similarity):")
for k in order:
    a, b = iu[0][k], iu[1][k]; print(f"  {a+1:2d} {ACTIVITIES[a]:30s} <-> {b+1:2d} {ACTIVITIES[b]:30s}  confusion={conf[a,b]:.2f}  similarity={cos[a,b]:.2f}")

pairs = [(14, 15, "write hunched vs straight"), (9, 10, "lean forward vs back at work"), (4, 5, "side vs back against wall"),
         (16, 18, "hands on armrests vs arms on legs"), (1, 11, "stand vs sleep on the table (unrelated)")]
print("\nHand-picked semantic pairs:")
for a, b, why in pairs:
    print(f"  {a:2d}<->{b:2d} {why:38s} confusion={conf[a-1,b-1]:.2f}  similarity={cos[a-1,b-1]:.2f}")
print(f"\nFor scale: mean confusion over all pairs = {conf[iu].mean():.3f}; mean similarity = {cos[iu].mean():.2f}")

fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(3, 12)
for c in range(18):
    ax = fig.add_subplot(gs[c // 6, (c % 6) * 2: (c % 6) * 2 + 2]); ax.imshow(M[c].reshape(20, 10), cmap="viridis")
    ax.set_title(f"{c+1}. {ACTIVITIES[c]}", fontsize=7); ax.axis("off")
plt.suptitle("Mean pressure pattern per activity (20x10 sensor grid, names from the dataset readme)")
plt.tight_layout(); plt.savefig("results/figures/plausibility_activity_maps.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
im = ax[0].imshow(cos, cmap="viridis"); ax[0].set(title="Cosine similarity of mean pressure maps", xlabel="activity", ylabel="activity")
ax[0].set_xticks(range(0, 18, 2)); ax[0].set_xticklabels(range(1, 19, 2)); ax[0].set_yticks(range(0, 18, 2)); ax[0].set_yticklabels(range(1, 19, 2)); plt.colorbar(im, ax=ax[0])
ax[1].scatter(cos[iu], conf[iu], s=14); ax[1].set(title=f"Similar patterns are confused more (rho={rho:.2f})", xlabel="map similarity", ylabel="confusion rate")
plt.tight_layout(); plt.savefig("results/figures/plausibility_similarity_vs_confusion.png", dpi=120); plt.close()
