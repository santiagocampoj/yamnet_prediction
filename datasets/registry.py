"""
datasets/registry.py
--------------------
Unified registry for all supported audio datasets.

To add a new dataset:
  1. Create datasets/newdataset.py with:
       is_downloaded(), download(), available_classes(),
       load_dataset_with_folds()
  2. Register it in _build_registry() below.
"""

from dataclasses import dataclass
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


# ── Registry ───────────────────────────────────────────────────────────────────

def _build_registry() -> dict[str, DatasetInfo]:
    registry = {}

    try:
        from datasets.esc50 import (is_downloaded, download,
                                     available_classes,
                                     load_dataset_with_folds)
        registry["esc50"] = DatasetInfo(
            name="ESC-50",
            key="esc50",
            description="50 environmental sound classes · 2000 clips · 5s",
            classes_fn=available_classes,
            download_fn=download,
            is_downloaded_fn=is_downloaded,
            load_fn=load_dataset_with_folds,
            default_duration=5.0,
            color="#1D9E75",
        )
    except ImportError:
        pass

    try:
        from datasets.urbansound8k import (is_downloaded, download,
                                            available_classes,
                                            load_dataset_with_folds)
        registry["urbansound8k"] = DatasetInfo(
            name="UrbanSound8K",
            key="urbansound8k",
            description="10 urban sound classes · 8732 clips · ≤4s",
            classes_fn=available_classes,
            download_fn=download,
            is_downloaded_fn=is_downloaded,
            load_fn=load_dataset_with_folds,
            default_duration=4.0,
            color="#185FA5",
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
                        progress_callback=None
                        ) -> tuple[list, list, list]:
    """
    Load and merge clips from multiple datasets.

    Parameters
    ----------
    selections : {dataset_key: [class_name, ...]}
        e.g. {"esc50": ["dog","cat"], "urbansound8k": ["car_horn","siren"]}

    Returns
    -------
    waveforms, labels, folds  — all merged
    """
    all_wf, all_lb, all_folds = [], [], []

    for ds_key, classes in selections.items():
        if not classes:
            continue
        ds = DATASETS.get(ds_key)
        if ds is None or not ds.is_downloaded_fn():
            print(f"  [skip] {ds_key} not available")
            continue

        def _prog(i, t, key=ds_key):
            if progress_callback:
                progress_callback(key, i, t)

        wf, lb, fo = ds.load_fn(
            classes,
            sr=16_000,
            duration=ds.default_duration,
            progress_callback=_prog,
        )
        all_wf.extend(wf)
        all_lb.extend(lb)
        all_folds.extend(fo)
        print(f"  {ds.name}: loaded {len(wf)} clips — {classes}")

    return all_wf, all_lb, all_folds