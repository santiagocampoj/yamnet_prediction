"""
datasets/registry.py
--------------------
Unified registry for all supported audio datasets.

To add a new dataset:
  1. Create datasets/newdataset.py
  2. Register it in _build_registry() below.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DatasetInfo:
    name:             str
    key:              str
    description:      str
    classes_fn:       Callable
    download_fn:      Callable
    is_downloaded_fn: Callable
    load_fn:          Callable
    default_duration: float = 2.0
    color:            str   = "#534AB7"
    # optional: version selector support
    versions_fn:      Callable | None = None   # () → {key: label}
    default_version:  str | None      = None


# ── Registry ───────────────────────────────────────────────────────────────────

def _build_registry() -> dict[str, DatasetInfo]:
    registry = {}

    # ESC-50
    try:
        from datasets.esc50 import (is_downloaded, download,
                                     available_classes,
                                     load_dataset_with_folds)
        registry["esc50"] = DatasetInfo(
            name="ESC-50", key="esc50",
            description="50 environmental sound classes · 2000 clips · 5s",
            classes_fn=available_classes,
            download_fn=download,
            is_downloaded_fn=is_downloaded,
            load_fn=load_dataset_with_folds,
            default_duration=5.0, color="#1D9E75",
        )
    except ImportError:
        pass

    # UrbanSound8K
    try:
        from datasets.urbansound8k import (is_downloaded, download,
                                            available_classes,
                                            load_dataset_with_folds)
        registry["urbansound8k"] = DatasetInfo(
            name="UrbanSound8K", key="urbansound8k",
            description="10 urban sound classes · 8732 clips · ≤4s",
            classes_fn=available_classes,
            download_fn=download,
            is_downloaded_fn=is_downloaded,
            load_fn=load_dataset_with_folds,
            default_duration=4.0, color="#185FA5",
        )
    except ImportError:
        pass

    # DCASE 2020 Task 1
    try:
        from datasets.dcase2020t1 import (is_downloaded, download,
                                           available_classes,
                                           load_dataset_with_folds,
                                           available_versions,
                                           DEFAULT_VERSION)
        registry["dcase2020t1"] = DatasetInfo(
            name="DCASE Dataset", key="dcase2020t1",
            description="Acoustic scene classification · 3 or 10 classes · 10s clips",
            classes_fn=available_classes,
            download_fn=download,
            is_downloaded_fn=is_downloaded,
            load_fn=load_dataset_with_folds,
            default_duration=10.0, color="#D85A30",
            versions_fn=available_versions,
            default_version=DEFAULT_VERSION,
        )
    except ImportError:
        pass

    return registry


DATASETS = _build_registry()


# ── Public API ─────────────────────────────────────────────────────────────────

def available_datasets() -> list[DatasetInfo]:
    return list(DATASETS.values())


def get_dataset(key: str) -> DatasetInfo | None:
    return DATASETS.get(key)


def load_multi_dataset(selections: dict[str, list[str]],
                        versions: dict[str, str] | None = None,
                        progress_callback=None,
                        ) -> tuple[list, list, list]:
    """
    Load and merge clips from multiple datasets.

    Parameters
    ----------
    selections : {dataset_key: [class_name, ...]}
    versions   : {dataset_key: version_key}  optional, for versioned datasets
    """
    all_wf, all_lb, all_folds = [], [], []
    versions = versions or {}

    for ds_key, classes in selections.items():
        if not classes:
            continue
        ds = DATASETS.get(ds_key)
        if ds is None:
            print(f"  [skip] unknown dataset: {ds_key}")
            continue

        # resolve version
        version = versions.get(ds_key, ds.default_version)

        # check downloaded (version-aware if supported)
        try:
            downloaded = (ds.is_downloaded_fn(version)
                          if version else ds.is_downloaded_fn())
        except TypeError:
            downloaded = ds.is_downloaded_fn()

        if not downloaded:
            print(f"  [skip] {ds.name} not downloaded")
            continue

        def _prog(i, t, key=ds_key):
            if progress_callback:
                progress_callback(key, i, t)

        # call load_fn with version if supported
        try:
            wf, lb, fo = ds.load_fn(
                classes, sr=16_000,
                duration=ds.default_duration,
                progress_callback=_prog,
                version=version,
            ) if version else ds.load_fn(
                classes, sr=16_000,
                duration=ds.default_duration,
                progress_callback=_prog,
            )
        except TypeError:
            wf, lb, fo = ds.load_fn(
                classes, sr=16_000,
                duration=ds.default_duration,
                progress_callback=_prog,
            )

        all_wf.extend(wf)
        all_lb.extend(lb)
        all_folds.extend(fo)
        print(f"  {ds.name}: loaded {len(wf)} clips — {classes}")

    return all_wf, all_lb, all_folds