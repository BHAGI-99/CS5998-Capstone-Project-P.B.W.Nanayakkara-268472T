"""Shared helpers: data loading, the 1D-CNN, training, and explanation methods.

Run any script from the repo root with:  python -m src.<name>
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path

import numpy as np

DATA = Path("data/processed")
ACTIVITIES = [
    "Stand", "Fold arms", "Think", "Side against the wall", "Back against the wall",
    "Arm on the baffle", "Hug a doll", "Carry a box", "Lean forward at work",
    "Lean back at work", "Sleep on the table", "Hold cheeks", "Play mobile phone",
    "Write with a hunchback", "Write with a straight back",
    "Sit with hands on the armrests", "Sit leaning to the right", "Sit with arms on the legs",
]  # names from the dataset readme, in label order 1..18


def set_seed(seed):
    import tensorflow as tf
    np.random.seed(seed); tf.random.set_seed(seed); tf.keras.utils.set_random_seed(seed)


def load_data():
    """Returns X (N,208,200) scaled to 0-1, y (0..17), subject (1..14), L (true frames per trial).
    Set XAI_STEP=k (dev option) to use every k-th trial for quick tests."""
    step = int(os.environ.get("XAI_STEP", 1))
    X = np.load(DATA / "X.npy").astype(np.float32) / 4095.0      # sensor range is 0-4095
    y = np.load(DATA / "y.npy"); subj = np.load(DATA / "subject.npy")
    L = np.minimum(np.load(DATA / "lengths.npy"), 208)
    return X[::step], y[::step], subj[::step], L[::step]


def build_cnn(seq_len=208, n_feat=200, n_classes=18):
    import tensorflow as tf
    from tensorflow.keras import layers
    inp = tf.keras.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(64, 5, padding="same")(inp);  x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(128, 5, padding="same")(x);   x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, padding="same", name="last_conv")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)        # needed for CAM
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def make_split(y, seed=42):
    """Stratified 72% train / 8% validation / 20% test. The validation part comes from the training part only."""
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    tr_in, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
    return tr_in, va, te


def train_model(X, y, tr_in, va, seed, epochs=60):
    import tensorflow as tf
    set_seed(seed)
    m = build_cnn()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    m.fit(X[tr_in], y[tr_in], validation_data=(X[va], y[va]), epochs=epochs, batch_size=32, callbacks=[es], verbose=0)
    return m


def integrated_gradients(model, x, cls, steps=50):
    """Integrated Gradients with an all-zero (no pressure) baseline. x: (208,200). Returns (208,200)."""
    import tensorflow as tf
    x = tf.constant(x[None]); b = tf.zeros_like(x)
    a = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), (-1, 1, 1)); inter = b + a * (x - b)
    with tf.GradientTape() as tape:
        tape.watch(inter); p = model(inter, training=False)[:, cls]
    g = tape.gradient(p, inter); g = (g[:-1] + g[1:]) / 2.0
    return ((x - b)[0] * tf.reduce_mean(g, axis=0)).numpy()


def sensor_importance(attr, n):
    """Average absolute attribution over the trial's true frames -> (200,) one value per sensor."""
    return np.abs(attr[:n]).mean(0)


_fmap_cache = {}
def cam_curve(model, x, cls, n):
    """Class Activation Map over time (Wang et al., 2017) for one trial -> (n,)."""
    import tensorflow as tf
    from tensorflow.keras import layers
    if id(model) not in _fmap_cache:
        gap = [l for l in model.layers if isinstance(l, layers.GlobalAveragePooling1D)][0]
        _fmap_cache[id(model)] = (tf.keras.Model(model.input, gap.input), model.layers[-1].get_weights()[0])
    fmap_model, W = _fmap_cache[id(model)]
    f = fmap_model.predict(x[None], verbose=0)[0]
    return f[:n] @ W[:, cls]
