"""Stability check: do independently trained CNNs highlight the same sensors (and the same moments)?

Trains N_MODELS CNNs that differ ONLY in the random seed (same data split), explains the same hold-out
trials with Integrated Gradients (which sensors) and CAM (which moments) for the TRUE class, and measures agreement.
Run:  python -m src.stability     (about 10 minutes on a Colab GPU; not yet timed)
"""
import os
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.common import load_data, make_split, train_model, integrated_gradients, sensor_importance, cam_curve

SEEDS = [int(s) for s in os.environ.get("XAI_SEEDS", "1,2,3,4,5").split(",")]
N_TRIALS, EPOCHS = int(os.environ.get("XAI_NTRIALS", 200)), int(os.environ.get("XAI_EPOCHS", 60))
X, y, subj, L = load_data()
tr_in, va, te = make_split(y, seed=42)                      # the split is fixed; only the training seed changes
ts = te[:N_TRIALS]
Path("results").mkdir(exist_ok=True)

imp, cam = [], []                                           # per model: (N,200) sensor importance, list of (n,) CAM curves
for s in SEEDS:
    m = train_model(X, y, tr_in, va, seed=s, epochs=EPOCHS)
    acc = (m.predict(X[ts], verbose=0).argmax(1) == y[ts]).mean()
    imp.append(np.stack([sensor_importance(integrated_gradients(m, X[i], int(y[i])), int(L[i])) for i in ts]))
    cam.append([cam_curve(m, X[i], int(y[i]), int(L[i])) for i in ts])
    print(f"model seed {s}: accuracy on the {len(ts)} explained trials = {acc:.3f}")

pressure = np.stack([X[i, :L[i]].mean(0) for i in ts])      # trivial baseline: raw mean pressure per sensor
def rank_corr(a, b): return float(np.nanmean([spearmanr(a[j], b[j])[0] for j in range(len(a))]))
def topk_overlap(a, b, k=20):                               # top 10% of 200 sensors
    return float(np.mean([len(set(np.argsort(-a[j])[:k]) & set(np.argsort(-b[j])[:k])) / len(set(np.argsort(-a[j])[:k]) | set(np.argsort(-b[j])[:k])) for j in range(len(a))]))

pairs = list(combinations(range(len(SEEDS)), 2))
same = np.mean([rank_corr(imp[a], imp[b]) for a, b in pairs])                   # same trial, different models
top = np.mean([topk_overlap(imp[a], imp[b]) for a, b in pairs])
# control: same model pair, but a DIFFERENT trial of the same activity (is the map trial-specific or just class-level?)
order = np.argsort(y[ts], kind="stable"); shifted = np.empty_like(order)
for c in np.unique(y[ts]):
    idx = order[y[ts][order] == c]; shifted[idx] = np.roll(idx, 1)
diff_trial = np.mean([rank_corr(imp[a], imp[b][shifted]) for a, b in pairs])
to_pressure = np.mean([rank_corr(imp[a], pressure) for a in range(len(SEEDS))])
cam_corr = np.mean([np.nanmean([np.corrcoef(cam[a][j], cam[b][j])[0, 1] for j in range(len(ts))]) for a, b in pairs])

print(f"\nSTABILITY over {len(SEEDS)} models, {len(ts)} trials ({len(pairs)} model pairs)")
print(f"  IG sensor ranking, same trial, different models (Spearman)  : {same:.2f}")
print(f"  IG top-10% sensors overlap (Jaccard)                        : {top:.2f}")
print(f"  control: different trial of the same activity (Spearman)   : {diff_trial:.2f}")
print(f"  IG vs raw mean pressure, within one model (Spearman)        : {to_pressure:.2f}")
print(f"  CAM time curves, same trial, different models (Pearson)     : {cam_corr:.2f}")
pd.DataFrame([{"models": len(SEEDS), "trials": len(ts), "IG_spearman_same_trial": same, "IG_top10_jaccard": top,
               "IG_spearman_other_trial_same_class": diff_trial, "IG_vs_pressure_spearman": to_pressure,
               "CAM_pearson_same_trial": cam_corr}]).round(3).to_csv("results/xai_stability.csv", index=False)
