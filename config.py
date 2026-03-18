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
SAMPLE_RATE    = 16_000   # Hz — required by YAMNet (resampled)
TOP_N_DEFAULT  = 5
TOP_N_MAX      = 20

# ── SPL analysis ───────────────────────────────────────────────────────────────
# Upper limit is set high — PyOctaveBand will cap it at Nyquist (sr/2)
# so the actual bands shown depend on the original audio sample rate
SPL_FREQ_LIMITS      = [16, 20000]
SPL_FRACTION_DEFAULT = 1            # 1 = octave, 3 = 1/3 octave
SPL_MODE_DEFAULT     = "spl"

# ── Colours ────────────────────────────────────────────────────────────────────
PRIMARY_COLOR  = "#8B6F47"
CHART_COLORS   = [
    "#534AB7", "#7F77DD", "#AFA9EC", "#CECBF6",
    "#9FE1CB", "#1D9E75", "#D85A30", "#F0997B",
    "#185FA5", "#85B7EB",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
PAGE_TITLE    = "YAMNet Prediction"
PAGE_ICON     = "🎧"
PAGE_SUBTITLE = "Analyse any audio with the original Google YAMNet model."