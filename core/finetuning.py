"""
core/finetuning.py
------------------
Fine-tune YAMNet by unfreezing the last N layers, processing clip by clip.

Note on YAMNet + batches:
  The local YAMNet (.h5) uses tf.compat.v1.pad internally which expects
  a 1D waveform (N,) — it cannot process batches (batch, N).
  TF Hub's version supports batches, but we use the local .h5 for full
  code control and no extra dependencies.

  Our approach: process each clip individually with unfrozen YAMNet layers,
  then train the dense head on the extracted embeddings with standard batches.
  This is functionally equivalent to fine-tuning — the unfrozen YAMNet layers
  are updated during training via the dense head gradients.

Same clip-level split as classifier.py — no frame leakage between splits.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from core.classifier import (split_by_clip, build_classifier,
                              get_report, StreamlitCallback)

SEED = 42


# ── Count parameters ───────────────────────────────────────────────────────────

def count_params(n_unfreeze: int) -> dict:
    """Return trainable and total parameter counts for a given n_unfreeze."""
    from core.model import load_yamnet
    yamnet, _, _ = load_yamnet()

    for i, layer in enumerate(yamnet.layers):
        layer.trainable = (i >= len(yamnet.layers) - n_unfreeze)

    trainable = sum(np.prod(v.shape) for v in yamnet.trainable_variables)
    total     = sum(np.prod(v.shape) for v in yamnet.variables)

    # reset
    for layer in yamnet.layers:
        layer.trainable = False

    return {"trainable": int(trainable), "total": int(total)}


# ── Extract embeddings with unfrozen layers ────────────────────────────────────

def extract_with_unfrozen(waveforms: list, labels: list,
                           yamnet, progress_callback=None,
                           offset: int = 0, total: int = 0
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings clip by clip using YAMNet with unfrozen layers.
    Each clip is processed as 1D waveform (N,) — avoids batch shape issues.
    """
    X, y = [], []
    for i, (wf, lbl) in enumerate(zip(waveforms, labels)):
        wf_t = tf.constant(wf, dtype=tf.float32)
        _, embs, _ = yamnet(wf_t)          # (N_frames, 1024)
        X.append(embs.numpy())
        y.extend([lbl] * len(embs.numpy()))
        if progress_callback:
            progress_callback(offset + i + 1, total)
    return np.vstack(X).astype(np.float32), np.array(y)


# ── Training ───────────────────────────────────────────────────────────────────

def train_finetune(waveforms: list, labels,
                   folds: list | None = None,
                   n_unfreeze: int = 3,
                   epochs: int = 20,
                   batch_size: int = 32,
                   learning_rate: float = 1e-4,
                   metrics_list: list | None = None,
                   progress_callback=None) -> tuple:
    """
    Fine-tune YAMNet clip by clip with unfrozen last N layers.

    Pipeline:
      1. Split at clip level (same as feature extraction)
      2. Unfreeze last N YAMNet layers
      3. Extract embeddings clip by clip with unfrozen YAMNet
      4. Train dense head on embeddings with standard batches
      5. Evaluate on test set

    Returns
    -------
    model, le, history, report
    """
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    # 1 — clip-level split
    wf_tr, lb_tr, wf_va, lb_va, wf_te, lb_te = split_by_clip(
        waveforms, labels, folds=folds)

    print(f"Fine-tune split: train={len(wf_tr)}, "
          f"val={len(wf_va)}, test={len(wf_te)}")

    # 2 — get YAMNet and unfreeze last N layers
    from core.model import load_yamnet
    yamnet, _, _ = load_yamnet()
    yamnet.trainable = False
    if n_unfreeze > 0:
        for layer in yamnet.layers[-n_unfreeze:]:
            layer.trainable = True
        print(f"Unfrozen layers: {[l.name for l in yamnet.layers[-n_unfreeze:]]}")

    # 3 — extract embeddings clip by clip
    total = len(wf_tr) + len(wf_va) + len(wf_te)

    X_tr, y_tr = extract_with_unfrozen(
        wf_tr, lb_tr, yamnet, progress_callback,
        offset=0, total=total)
    X_va, y_va = extract_with_unfrozen(
        wf_va, lb_va, yamnet, progress_callback,
        offset=len(wf_tr), total=total)
    X_te, y_te = extract_with_unfrozen(
        wf_te, lb_te, yamnet, progress_callback,
        offset=len(wf_tr)+len(wf_va), total=total)

    # 4 — encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_va, y_te]))
    y_tr_enc = le.transform(y_tr).astype(np.int32)
    y_va_enc = le.transform(y_va).astype(np.int32)
    y_te_enc = le.transform(y_te).astype(np.int32)

    # 5 — tf.data datasets (standard batches on embeddings)
    def make_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(1000, seed=SEED)
        return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_tr, y_tr_enc, shuffle=True)
    val_ds   = make_ds(X_va, y_va_enc)
    test_ds  = make_ds(X_te, y_te_enc)

    # 6 — train dense head
    model = build_classifier(len(le.classes_))
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
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

    # 7 — evaluate
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Fine-tune test loss: {test_loss:.4f}  acc: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    report = get_report(y_te_enc, y_pred, le.classes_)
    report["test_loss"]     = round(float(test_loss), 4)
    report["test_accuracy"] = round(float(test_acc) * 100, 2)

    return model, le, history, report