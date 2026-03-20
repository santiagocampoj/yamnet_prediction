"""
datasets/esc50.py
-----------------
Download, parse and prepare the ESC-50 dataset.

ESC-50 has 5 cross-validation folds where clips from the same
original source are always in the same fold.

  train = folds 1, 2, 3
  val   = fold 4
  test  = fold 5
"""

import os, shutil, zipfile, urllib.request
import pandas as pd
import numpy as np

ESC50_URL   = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP   = "./datasets/data/esc50.zip"
ESC50_ROOT  = "./datasets/data/ESC-50-master"
ESC50_CSV   = "./datasets/data/ESC-50-master/meta/esc50.csv"
ESC50_AUDIO = "./datasets/data/ESC-50-master/audio/"


# ── Download ───────────────────────────────────────────────────────────────────

def is_downloaded() -> bool:
    return os.path.isfile(ESC50_CSV)


def download(progress_callback=None) -> bool:
    os.makedirs("./datasets/data", exist_ok=True)

    def reporthook(count, block_size, total_size):
        if progress_callback and total_size > 0:
            progress_callback(min(count * block_size / total_size, 1.0))

    try:
        urllib.request.urlretrieve(ESC50_URL, ESC50_ZIP,
                                    reporthook=reporthook)
        with zipfile.ZipFile(ESC50_ZIP, "r") as z:
            z.extractall("./datasets/data")
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


# ── Metadata ───────────────────────────────────────────────────────────────────

def get_metadata() -> pd.DataFrame:
    return pd.read_csv(ESC50_CSV)


def available_classes() -> list[str]:
    if not is_downloaded():
        return []
    return sorted(get_metadata()["category"].unique().tolist())


# ── Prepare folder ─────────────────────────────────────────────────────────────

def prepare_train_folder(selected_classes: list[str],
                          out_dir: str = "./data/train",
                          progress_callback=None) -> dict:
    df       = get_metadata()
    filtered = df[df["category"].isin(selected_classes)].copy()
    filtered["full_path"] = filtered["filename"].apply(
        lambda f: os.path.join(ESC50_AUDIO, f))

    stats = {}
    total = len(filtered)
    done  = 0

    for cls in selected_classes:
        cls_dir  = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        cls_rows = filtered[filtered["category"] == cls]
        count    = 0
        for _, row in cls_rows.iterrows():
            src = row["full_path"]
            dst = os.path.join(cls_dir, row["filename"])
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                count += 1
            done += 1
            if progress_callback:
                progress_callback(done / total)
        stats[cls] = count

    return stats


# ── Load with folds ────────────────────────────────────────────────────────────

def load_dataset_with_folds(selected_classes: list[str],
                             sr: int = 16_000,
                             duration: float = 5.0,
                             progress_callback=None
                             ) -> tuple[list, list, list]:
    """
    Load ESC-50 clips preserving fold structure.

    Returns
    -------
    waveforms : list of np.ndarray
    labels    : list of str
    folds     : list of int  (1-5)
    """
    from core.utils import load_audio_file

    df       = get_metadata()
    filtered = df[df["category"].isin(selected_classes)].copy()
    filtered["full_path"] = filtered["filename"].apply(
        lambda f: os.path.join(ESC50_AUDIO, f))

    waveforms, labels, folds = [], [], []
    total = len(filtered)

    for i, (_, row) in enumerate(filtered.iterrows()):
        try:
            wf = load_audio_file(row["full_path"], target_sr=sr)
            target_len = int(sr * duration)
            if len(wf) > target_len:
                wf = wf[:target_len]
            elif len(wf) < target_len:
                wf = np.pad(wf, (0, target_len - len(wf)))
            waveforms.append(wf)
            labels.append(row["category"])
            folds.append(int(row["fold"]))
        except Exception as e:
            print(f"  [skip] {row['filename']}: {e}")

        if progress_callback:
            progress_callback(i + 1, total)

    return waveforms, labels, folds