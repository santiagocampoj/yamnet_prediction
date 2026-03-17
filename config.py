"""
config.py
---------
All constants, paths and default values for the YAMNet Prediction app.
Edit this file to change model paths, colours or UI defaults.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
ASSETS_DIR     = BASE_DIR / "assets"
WEIGHTS_PATH   = ASSETS_DIR / "yamnet.h5"
CLASS_MAP_PATH = ASSETS_DIR / "yamnet_class_map.csv"

# ── Model ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16_000   # Hz — required by YAMNet
TOP_N_DEFAULT  = 5        # default number of top classes to show
TOP_N_MAX      = 20

# ── Colours ────────────────────────────────────────────────────────────────────
PRIMARY_COLOR  = "#8B6F47"
CHART_COLORS   = [
    "#534AB7", "#7F77DD", "#AFA9EC", "#CECBF6",
    "#9FE1CB", "#1D9E75", "#D85A30", "#F0997B",
    "#185FA5", "#85B7EB",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
PAGE_TITLE     = "YAMNet Prediction"
PAGE_ICON      = "🎧"
PAGE_SUBTITLE  = "Analyse any audio with the original Google YAMNet model."
