"""
core/spl.py
-----------
SPL analysis using PyOctaveBand.

Two analysis modes:
  - compute_spl       : RMS level per band (single value per band, averaged over time)
  - compute_spl_time  : Time-weighted level per band (level over time, Fast/Slow/Impulse)
"""

import numpy as np
from pyoctaveband import octavefilter, time_weighting

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
    Compute RMS SPL per octave or 1/3-octave band.
    Returns a single level per band averaged over the full clip.

    Returns
    -------
    spl   : (n_bands,)   level in dB per band
    freqs : (n_bands,)   center frequencies in Hz
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


def compute_spl_time(
    waveform: np.ndarray,
    sample_rate: int,
    fraction: int = 1,
    mode: str = "dbfs",
    time_mode: str = "fast",
    calibration_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute time-weighted SPL per band using IEC 61672-1 time constants.

    Parameters
    ----------
    time_mode : "fast" (125ms), "slow" (1000ms), or "impulse" (35ms rise / 1500ms decay)

    Returns
    -------
    levels : (n_bands, n_samples)  time-varying level per band (dB)
    freqs  : (n_bands,)            center frequencies in Hz
    t      : (n_samples,)          time axis in seconds
    """
    use_dbfs = (mode == "dbfs")
    ref      = 1.0 if use_dbfs else (20e-6 / calibration_factor)

    # get per-band filtered signals
    _, freqs, band_signals = octavefilter(
        waveform,
        fs=sample_rate,
        fraction=fraction,
        order=6,
        limits=config.SPL_FREQ_LIMITS,
        filter_type="butter",
        sigbands=True,
        dbfs=False,          # we handle dB conversion manually
        mode="rms",
    )

    freqs = np.array(freqs)
    t     = np.arange(len(waveform)) / sample_rate
    levels = []

    for band_signal in band_signals:
        # time_weighting returns the mean-square energy envelope
        envelope = time_weighting(band_signal, sample_rate, mode=time_mode)
        # convert to dB — clamp to avoid log(0)
        envelope = np.maximum(envelope, 1e-20)
        if use_dbfs:
            db = 10 * np.log10(envelope)
        else:
            db = 10 * np.log10(envelope / ref**2)
        levels.append(db)

    return np.array(levels), freqs, t