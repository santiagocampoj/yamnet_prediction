"""
core/classifier.py
------------------
Train a small dense classifier on top of frozen YAMNet embeddings.

Key principle: split is always done at CLIP level before extracting embeddings,
so frames from the same clip never appear in different splits.

Works with:
  - ESC-50: folds provided externally (fold 1-3 = train, 4 = val, 5 = test)
  - Custom audio: clips are grouped and split 70/15/15 by clip index
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# from core.yamnet import mean_embedding, load_audio_file
from core.utils import load_audio_file, mean_embedding, extract_embedding


SEED = 42


def _get_yamnet():
    from core.model import load_yamnet
    model, _, _ = load_yamnet()
    return model


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_from_folder(root: str, sr: int = 16_000,
                              duration: float = 2.0) -> tuple:
    """
    Load all WAV files from root/class_name/*.wav

    Returns
    -------
    waveforms : list of np.ndarray
    labels    : list of str
    """
    waveforms, labels = [], []
    root_path = Path(root)
    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for audio_file in sorted(class_dir.glob("*.*")):
            if audio_file.suffix.lower() not in {".wav",".mp3",".flac",".ogg"}:
                continue
            try:
                wf = load_audio_file(str(audio_file), target_sr=sr)
                target_len = int(sr * duration)
                if len(wf) > target_len:
                    wf = wf[:target_len]
                elif len(wf) < target_len:
                    wf = np.pad(wf, (0, target_len - len(wf)))
                waveforms.append(wf)
                labels.append(label)
            except Exception as e:
                print(f"  [skip] {audio_file.name}: {e}")
    return waveforms, labels


# ── Clip-level split ───────────────────────────────────────────────────────────

def split_by_clip(waveforms: list, labels: list,
                  folds: list | None = None
                  ) -> tuple[list, list, list, list, list, list]:
    """
    Split waveforms/labels at CLIP level so no clip's frames leak across splits.

    If folds is provided (ESC-50 style):
        train = folds 1-3, val = fold 4, test = fold 5

    If folds is None (custom data):
        70 / 15 / 15 split by clip index, stratified by label

    Returns
    -------
    wf_train, lb_train,
    wf_val,   lb_val,
    wf_test,  lb_test
    """
    if folds is not None:
        folds = np.array(folds)
        idx_train = np.where(folds <= 3)[0]
        idx_val   = np.where(folds == 4)[0]
        idx_test  = np.where(folds == 5)[0]
    else:
        rng     = np.random.default_rng(SEED)
        indices = np.arange(len(waveforms))
        rng.shuffle(indices)
        n       = len(indices)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        idx_train = indices[:n_train]
        idx_val   = indices[n_train:n_train + n_val]
        idx_test  = indices[n_train + n_val:]

    def _pick(idx):
        return ([waveforms[i] for i in idx],
                [labels[i]    for i in idx])

    wf_tr, lb_tr = _pick(idx_train)
    wf_va, lb_va = _pick(idx_val)
    wf_te, lb_te = _pick(idx_test)
    return wf_tr, lb_tr, wf_va, lb_va, wf_te, lb_te


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings_per_clip(waveforms: list, labels: list,
                                 progress_callback=None
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one mean embedding per clip.
    Returns X (N_clips, 1024) and y (N_clips,) string labels.
    """
    X, y = [], []
    for i, (wf, lbl) in enumerate(zip(waveforms, labels)):
        # X.append(mean_embedding(wf))
        X.append(mean_embedding(wf, _get_yamnet()))
        y.append(lbl)
        if progress_callback:
            progress_callback(i + 1, len(waveforms))
    return np.array(X, dtype=np.float32), np.array(y)


def extract_embeddings_per_frame(waveforms: list, labels: list,
                                  progress_callback=None
                                  ) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i, (wf, lbl) in enumerate(zip(waveforms, labels)):
        _, embs, _ = extract_embedding(wf, _get_yamnet())  # ✅ usa extract_embedding de utils
        X.append(embs)
        y.extend([lbl] * len(embs))
        if progress_callback:
            progress_callback(i + 1, len(waveforms))
    return np.vstack(X).astype(np.float32), np.array(y)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_classifier(n_classes: int) -> tf.keras.Model:
    """
    Small dense head on top of 1024-dim YAMNet embeddings.
    Matches the Google tutorial architecture with added regularisation.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32,
                              name="input_embedding"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(n_classes),
    ], name="yamnet_classifier")

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# ── Streamlit callback ─────────────────────────────────────────────────────────

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics_list: list):
        super().__init__()
        self.metrics_list = metrics_list

    def on_epoch_end(self, epoch, logs=None):
        self.metrics_list.append({
            "epoch":    epoch + 1,
            "acc":      round(logs.get("accuracy", 0) * 100, 2),
            "val_acc":  round(logs.get("val_accuracy", 0) * 100, 2),
            "loss":     round(logs.get("loss", 0), 4),
            "val_loss": round(logs.get("val_loss", 0), 4),
        })


# ── Training ───────────────────────────────────────────────────────────────────

def train(waveforms: list, labels: list,
          folds: list | None = None,
          epochs: int = 20,
          batch_size: int = 32,
          use_frame_embeddings: bool = True,
          metrics_list: list | None = None,
          progress_callback=None) -> tuple:
    """
    Full training pipeline following the Google tutorial approach.

    1. Split at CLIP level (preserving folds if provided)
    2. Extract embeddings PER FRAME for train/val/test
    3. Train the classifier
    4. Evaluate on test set

    Parameters
    ----------
    use_frame_embeddings : if True, extract all frames per clip (Google style).
                           If False, use mean embedding per clip (faster).
    """
    # 1. clip-level split
    wf_tr, lb_tr, wf_va, lb_va, wf_te, lb_te = split_by_clip(
        waveforms, labels, folds=folds)

    print(f"Split: train={len(wf_tr)} clips, val={len(wf_va)}, test={len(wf_te)}")

    # 2. extract embeddings
    extractor = (extract_embeddings_per_frame if use_frame_embeddings
                 else extract_embeddings_per_clip)

    total = len(wf_tr) + len(wf_va) + len(wf_te)
    done  = [0]

    def prog(i, t, offset=0):
        done[0] = offset + i
        if progress_callback:
            progress_callback(done[0], total)

    X_tr, y_tr = extractor(wf_tr, lb_tr,
                            progress_callback=lambda i,t: prog(i,t,0))
    X_va, y_va = extractor(wf_va, lb_va,
                            progress_callback=lambda i,t: prog(i,t,len(wf_tr)))
    X_te, y_te = extractor(wf_te, lb_te,
                            progress_callback=lambda i,t: prog(i,t,len(wf_tr)+len(wf_va)))

    # 3. encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_va, y_te]))
    y_tr_enc = le.transform(y_tr)
    y_va_enc = le.transform(y_va)
    y_te_enc = le.transform(y_te)

    # 4. build tf.data datasets (like the Google tutorial)
    def make_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(1000, seed=SEED)
        return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_tr, y_tr_enc, shuffle=True)
    val_ds   = make_ds(X_va, y_va_enc)
    test_ds  = make_ds(X_te, y_te_enc)

    # 5. train
    model = build_classifier(len(le.classes_))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True),
    ]
    if metrics_list is not None:
        callbacks.append(StreamlitCallback(metrics_list))

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )

    # 6. evaluate on test
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    report = get_report(y_te_enc, y_pred, le.classes_)
    report["test_loss"]     = round(float(test_loss), 4)
    report["test_accuracy"] = round(float(test_acc) * 100, 2)

    return model, le, history, X_tr, y_tr, report


# ── Evaluation ─────────────────────────────────────────────────────────────────

def get_report(y_test, y_pred, class_names) -> dict:
    rep = classification_report(y_test, y_pred,
                                 target_names=class_names,
                                 output_dict=True)
    cm  = confusion_matrix(y_test, y_pred).tolist()
    return {"report": rep, "confusion_matrix": cm,
            "classes": list(class_names)}


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_model(model, le, X, y_str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "classifier.keras"))
    np.save(os.path.join(out_dir, "embeddings.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"),     y_str)
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump(list(le.classes_), f)
    print(f"Model saved to {out_dir}/")


def load_saved_model(out_dir: str):
    model = tf.keras.models.load_model(
        os.path.join(out_dir, "classifier.keras"))
    with open(os.path.join(out_dir, "classes.json")) as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    X     = np.load(os.path.join(out_dir, "embeddings.npy"))
    y_str = np.load(os.path.join(out_dir, "labels.npy"), allow_pickle=True)
    return model, le, X, y_str


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_file(model, le, waveform: np.ndarray) -> dict:
    _, embs, _ = extract_embedding(waveform, _get_yamnet())  # ✅
    logits      = model.predict(embs, verbose=0)
    mean_logits = logits.mean(axis=0)
    probs       = tf.nn.softmax(mean_logits).numpy()
    idx         = np.argmax(probs)
    return {
        "label":      le.classes_[idx],
        "confidence": float(probs[idx]),
        "all":        {cls: float(p)
                       for cls, p in zip(le.classes_, probs)},
    }


# ── Legacy helpers (kept for compatibility) ────────────────────────────────────

def extract_all_embeddings(waveforms, labels, progress_callback=None):
    """Alias for extract_embeddings_per_clip — kept for backward compatibility."""
    return extract_embeddings_per_clip(waveforms, labels, progress_callback)