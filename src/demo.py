"""Interactive demo: pick an activity and a hold-out trial -> prediction, confidence and a sensor explanation.
In Colab (after `python -m src.train_final`):
    from src.demo import launch; launch()
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.common import ACTIVITIES, load_data, integrated_gradients, sensor_importance

_cache = {}
def _load():
    if not _cache:
        import tensorflow as tf
        _cache["model"] = tf.keras.models.load_model("models/cnn_final.keras")
        _cache["data"] = load_data(); _cache["te"] = np.load("models/test_indices.npy")
    return _cache

def explain_trial(idx, show=True, save_to=None):
    """idx: index of a trial in the data (use trials from models/test_indices.npy for an honest demo)."""
    c = _load(); model = c["model"]; X, y, subj, L = c["data"]; n = int(L[idx])
    prob = model.predict(X[idx][None], verbose=0)[0]; top = np.argsort(-prob)[:3]; pred = int(top[0])
    imp = sensor_importance(integrated_gradients(model, X[idx], pred), n)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    ax[0].imshow(X[idx, :n].mean(0).reshape(20, 10), cmap="viridis"); ax[0].set_title("Sleeve pressure (average)"); ax[0].axis("off")
    ax[1].imshow(imp.reshape(20, 10), cmap="magma"); ax[1].set_title("Which sensors drove the decision\n(Integrated Gradients)"); ax[1].axis("off")
    ax[2].barh([f"{k+1}. {ACTIVITIES[k]}" for k in top][::-1], prob[top][::-1], color=["#bbb", "#bbb", "#2E7D32" if pred == y[idx] else "#B3261E"])
    ax[2].set(xlim=(0, 1), xlabel="model confidence", title=f"Prediction: {ACTIVITIES[pred]}\nTrue activity: {ACTIVITIES[y[idx]]}  (subject {subj[idx]})")
    plt.tight_layout()
    if save_to: plt.savefig(save_to, dpi=120)
    if show: plt.show()
    else: plt.close()
    return pred, float(prob[pred])

def launch():
    import ipywidgets as w
    from IPython.display import display
    c = _load(); X, y, subj, L = c["data"]; te = c["te"]
    act = w.Dropdown(options=[(f"{k+1}. {ACTIVITIES[k]}", k) for k in range(18)], description="Activity:")
    num = w.IntSlider(min=1, max=1, value=1, description="Trial:"); out = w.Output()
    def refresh(*_):
        ids = [i for i in te if y[i] == act.value]; num.max = max(1, len(ids)); num.value = min(num.value, num.max)
        with out:
            out.clear_output(wait=True); explain_trial(ids[num.value - 1])
    act.observe(refresh, "value"); num.observe(refresh, "value"); display(act, num, out); refresh()
