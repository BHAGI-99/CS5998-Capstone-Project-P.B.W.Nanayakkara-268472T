"""Train the final 1D-CNN on a fixed split (seed 42) and save it for the demo.
Run:  python -m src.train_final        (about a minute on a Colab GPU)
Saves: models/cnn_final.keras and models/test_indices.npy
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from src.common import load_data, make_split, train_model

EPOCHS = int(__import__("os").environ.get("XAI_EPOCHS", 60))
X, y, subj, L = load_data()
tr_in, va, te = make_split(y, seed=42)
model = train_model(X, y, tr_in, va, seed=42, epochs=EPOCHS)
acc = accuracy_score(y[te], model.predict(X[te], verbose=0).argmax(1))
print(f"final CNN test accuracy on the fixed hold-out set (n={len(te)}): {acc:.4f}")
Path("models").mkdir(exist_ok=True)
model.save("models/cnn_final.keras"); np.save("models/test_indices.npy", te)
print("saved models/cnn_final.keras and models/test_indices.npy")
