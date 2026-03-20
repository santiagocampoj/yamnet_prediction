"""
datasets/dcase2020t1.py
-----------------------
DCASE 2020 Task 1 — Acoustic Scene Classification.
TAU Urban Acoustic Scenes 2020 Mobile, Development dataset.

Two versions available:
  - "3class" : indoor / outdoor / transportation (~40h, 3 classes)
               Zenodo: https://zenodo.org/records/3670185
  - "10class": 10 individual scenes (~64h, 10 classes)
               Zenodo: https://zenodo.org/records/3670167

10 acoustic scenes:
  airport, bus, metro, metro_station, park, public_square,
  shopping_mall, street_pedestrian, street_traffic, tram
"""

import os, zipfile, urllib.request
import pandas as pd
import numpy as np

# ── Version configs ────────────────────────────────────────────────────────────

VERSIONS = {
    "3class": {
        "label":       "DCASE 2020 Task1 - 3 Classes (indoor / outdoor / transportation)",
        "zenodo_id":   "3670185",
        "prefix":      "TAU-urban-acoustic-scenes-2020-3class-development",
        "n_audio_zips": 3,
        "size_gb":     40,
    },
    "10class": {
        "label":       "DCASE 2020 Task1 - 10 Classes",
        "zenodo_id":   "3670167",
        "prefix":      "TAU-urban-acoustic-scenes-2020-mobile-development",
        "n_audio_zips": 16,
        "size_gb":     64,
    },
}

DEFAULT_VERSION = "3class"
BASE_DIR        = "./datasets/data"


def _paths(version: str = DEFAULT_VERSION) -> dict:
    cfg    = VERSIONS[version]
    prefix = cfg["prefix"]
    root   = os.path.join(BASE_DIR, prefix)
    return {
        "root":  root,
        "csv":   os.path.join(root, "meta.csv"),
        "audio": os.path.join(root, "audio"),
    }


# ── Download ───────────────────────────────────────────────────────────────────

def is_downloaded(version: str = DEFAULT_VERSION) -> bool:
    return os.path.isfile(_paths(version)["csv"])


def download(progress_callback=None,
             version: str = DEFAULT_VERSION) -> bool:
    """
    Download metadata + audio zip files from Zenodo.
    Audio is split into multiple zip files downloaded sequentially.
    """
    cfg    = VERSIONS[version]
    prefix = cfg["prefix"]
    zid    = cfg["zenodo_id"]
    n_zips = cfg["n_audio_zips"]
    base   = f"https://zenodo.org/record/{zid}/files"

    os.makedirs(BASE_DIR, exist_ok=True)

    # build file list: meta zip + audio zips
    files = [
        (f"{base}/{prefix}.meta.zip",
         os.path.join(BASE_DIR, f"dcase2020_{version}_meta.zip")),
    ]
    for i in range(1, n_zips + 1):
        files.append((
            f"{base}/{prefix}.audio.{i}.zip",
            os.path.join(BASE_DIR, f"dcase2020_{version}_audio{i}.zip"),
        ))

    total = len(files)

    def reporthook(count, block_size, total_size, idx=0):
        if progress_callback and total_size > 0:
            file_p   = min(count * block_size / total_size, 1.0)
            overall  = (idx + file_p) / total
            progress_callback(overall)

    try:
        for i, (url, path) in enumerate(files):
            if not os.path.isfile(path):
                print(f"  Downloading {os.path.basename(url)}…")
                urllib.request.urlretrieve(
                    url, path,
                    lambda c, b, t, idx=i: reporthook(c, b, t, idx))
            print(f"  Extracting {os.path.basename(path)}…")
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(BASE_DIR)

        return is_downloaded(version)

    except Exception as e:
        print(f"Download error: {e}")
        return False


# ── Metadata ───────────────────────────────────────────────────────────────────

def get_metadata(version: str = DEFAULT_VERSION) -> pd.DataFrame:
    """
    Parse meta.csv and add pseudo-folds 1-5 for compatibility
    with split_by_clip (DCASE has no official fold structure).
    """
    csv_path = _paths(version)["csv"]
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    # normalise column names across versions
    # Priority: exact matches first, then partial — avoid renaming multiple
    # columns (e.g. both "scene_label" and "source_label") to "category"
    rename = {}
    scene_col = next(
        (c for c in df.columns if c.lower() in ("scene_label", "scene", "tags")),
        next((c for c in df.columns if "scene" in c.lower()), None)
    )
    if scene_col:
        rename[scene_col] = "category"
    file_col = next(
        (c for c in df.columns if "file" in c.lower() and c != scene_col),
        None
    )
    if file_col and "category" not in file_col.lower():
        rename[file_col] = "filename"
    df = df.rename(columns=rename)

    # pseudo-folds 1-5 by file index
    df = df.reset_index(drop=True)
    df["fold"] = (df.index % 5) + 1

    return df


def available_classes(version: str = DEFAULT_VERSION) -> list[str]:
    if not is_downloaded(version):
        return []
    try:
        return sorted(get_metadata(version)["category"].unique().tolist())
    except Exception:
        return []


def available_versions() -> dict[str, str]:
    """Return {key: label} for UI selector."""
    return {k: v["label"] for k, v in VERSIONS.items()}


# ── Load with folds ────────────────────────────────────────────────────────────

def load_dataset_with_folds(selected_classes: list[str],
                             sr: int = 16_000,
                             duration: float = 10.0,
                             progress_callback=None,
                             version: str = DEFAULT_VERSION,
                             ) -> tuple[list, list, list]:
    """
    Load DCASE 2020 Task 1 clips with pseudo-fold structure.

    Returns
    -------
    waveforms : list of np.ndarray
    labels    : list of str
    folds     : list of int  (1-5 pseudo-folds)
    """
    from core.utils import load_audio_file

    paths    = _paths(version)
    df       = get_metadata(version)
    filtered = df[df["category"].isin(selected_classes)].copy()

    waveforms, labels, folds = [], [], []
    total = len(filtered)

    for i, (_, row) in enumerate(filtered.iterrows()):
        fname = row["filename"]
        # filename in meta.csv may be relative (e.g. "audio/airport-…wav")
        path  = os.path.join(paths["root"], fname) \
                if not os.path.isabs(fname) else fname

        try:
            wf = load_audio_file(path, target_sr=sr)
            target_len = int(sr * duration)
            wf = wf[:target_len] if len(wf) > target_len \
                 else np.pad(wf, (0, target_len - len(wf)))
            waveforms.append(wf)
            labels.append(row["category"])
            folds.append(int(row["fold"]))
        except Exception as e:
            print(f"  [skip] {fname}: {e}")

        if progress_callback:
            progress_callback(i + 1, total)

    return waveforms, labels, folds