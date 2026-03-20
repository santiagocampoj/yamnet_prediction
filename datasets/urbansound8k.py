"""
datasets/urbansound8k.py
------------------------
Download, parse and prepare the UrbanSound8K dataset.

10 classes, 8732 clips, 10 folds.
Classes: air_conditioner, car_horn, children_playing, dog_bark,
         drilling, engine_idling, gun_shot, jackhammer, siren, street_music

  train = folds 1-8  → remapped to folds 1-3
  val   = fold 9     → remapped to fold 4
  test  = fold 10    → remapped to fold 5
"""

import os, shutil, tarfile, urllib.request
import pandas as pd
import numpy as np

US8K_URL   = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
US8K_ROOT  = "./datasets/data/UrbanSound8K"
US8K_CSV   = "./datasets/data/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO = "./datasets/data/UrbanSound8K/audio/"

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]


# ── Download ───────────────────────────────────────────────────────────────────

def is_downloaded() -> bool:
    return os.path.isfile(US8K_CSV)


def download(progress_callback=None) -> bool:
    os.makedirs("./datasets/data", exist_ok=True)
    tar_path = "./datasets/data/urbansound8k.tar.gz"

    def reporthook(count, block_size, total_size):
        if progress_callback and total_size > 0:
            progress_callback(min(count * block_size / total_size, 1.0))

    try:
        urllib.request.urlretrieve(US8K_URL, tar_path,
                                    reporthook=reporthook)
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall("./datasets/data")
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


# ── Metadata ───────────────────────────────────────────────────────────────────

def get_metadata() -> pd.DataFrame:
    return pd.read_csv(US8K_CSV)


def available_classes() -> list[str]:
    if not is_downloaded():
        return []
    return sorted(get_metadata()["class"].unique().tolist())


# ── Prepare folder ─────────────────────────────────────────────────────────────

def prepare_train_folder(selected_classes: list[str],
                          out_dir: str = "./data/train",
                          progress_callback=None) -> dict:
    df       = get_metadata()
    filtered = df[df["class"].isin(selected_classes)].copy()

    stats = {}
    total = len(filtered)
    done  = 0

    for cls in selected_classes:
        cls_dir  = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        cls_rows = filtered[filtered["class"] == cls]
        count    = 0
        for _, row in cls_rows.iterrows():
            src = os.path.join(US8K_AUDIO,
                               f"fold{row['fold']}", row["slice_file_name"])
            dst = os.path.join(cls_dir, row["slice_file_name"])
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
                             duration: float = 4.0,
                             progress_callback=None
                             ) -> tuple[list, list, list]:
    """
    Load UrbanSound8K clips preserving fold structure.
    Folds remapped to 1-5 scale to match split_by_clip expectations.

    Returns
    -------
    waveforms : list of np.ndarray
    labels    : list of str
    folds     : list of int  (1-5)
    """
    from core.utils import load_audio_file

    df       = get_metadata()
    filtered = df[df["class"].isin(selected_classes)].copy()

    waveforms, labels, folds = [], [], []
    total = len(filtered)

    for i, (_, row) in enumerate(filtered.iterrows()):
        src = os.path.join(US8K_AUDIO,
                           f"fold{row['fold']}", row["slice_file_name"])
        try:
            wf = load_audio_file(src, target_sr=sr)
            target_len = int(sr * duration)
            if len(wf) > target_len:
                wf = wf[:target_len]
            elif len(wf) < target_len:
                wf = np.pad(wf, (0, target_len - len(wf)))
            waveforms.append(wf)
            labels.append(row["class"])
            # remap 10 folds → 5 scale
            fold = int(row["fold"])
            if fold <= 8:
                fold = min(fold, 3)   # train
            elif fold == 9:
                fold = 4              # val
            else:
                fold = 5              # test
            folds.append(fold)
        except Exception as e:
            print(f"  [skip] {row['slice_file_name']}: {e}")

        if progress_callback:
            progress_callback(i + 1, total)

    return waveforms, labels, folds