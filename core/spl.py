"""
core/spl.py
-----------
SPL analysis using PyOctaveBand.
Returns octave or 1/3-octave band levels in dBFS or dB SPL.
"""

import numpy as np
from pyoctaveband import octavefilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_spl(
    waveform: np.ndarray,
    sample_rate: int,
    fraction: int = 1,
    mode: str = "dbfs",
    calibration_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SPL per octave or 1/3-octave band.

    Parameters
    ----------
    waveform          : float32 mono waveform
    sample_rate       : sample rate in Hz
    fraction          : 1 = octave, 3 = 1/3 octave
    mode              : "dbfs" or "spl"
    calibration_factor: sensitivity factor for dB SPL (from calculate_sensitivity)

    Returns
    -------
    spl   : np.ndarray  level per band (dB)
    freqs : np.ndarray  center frequencies (Hz)
    """
    use_dbfs = (mode == "dbfs")

    spl, freqs = octavefilter(
        waveform,
        fs=sample_rate,
        fraction=fraction,
        order=6,
        limits=config.SPL_FREQ_LIMITS,
        filter_type="butter",
        dbfs=use_dbfs,
        calibration_factor=calibration_factor if not use_dbfs else 1.0,
        mode="rms",
    )

    return np.array(spl), np.array(freqs)