"""
core/audio.py
-------------
Audio loading and preprocessing utilities.
No Streamlit or model imports here — pure audio logic.
"""

import io
import numpy as np
import soundfile as sf
import resampy

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes, convert to mono float32 at SAMPLE_RATE.

    Returns
    -------
    waveform : np.ndarray  float32, shape (n_samples,)
    sr       : int         sample rate after resampling
    """
    wav_data, sr = sf.read(io.BytesIO(audio_bytes), dtype=np.int16)
    waveform     = (wav_data / 32768.0).astype("float32")

    # mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # resample to 16kHz if needed
    if sr != config.SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, config.SAMPLE_RATE)
        sr       = config.SAMPLE_RATE

    return waveform, sr


def duration(waveform: np.ndarray) -> float:
    """Return duration in seconds."""
    return len(waveform) / config.SAMPLE_RATE
