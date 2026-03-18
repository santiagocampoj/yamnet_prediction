"""
core/audio.py
-------------
Audio loading and preprocessing utilities.

Returns two independent waveforms:
  - waveform_yamnet : mono float32 at 16kHz  — for YAMNet
  - waveform_orig   : mono float32 at original sr — for SPL analysis
"""

import io
import numpy as np
import soundfile as sf
import resampy

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int, np.ndarray, int]:
    """
    Load audio from raw bytes.

    Returns
    -------
    waveform_yamnet : float32 mono at 16kHz
    sr_yamnet       : 16000
    waveform_orig   : float32 mono at original sample rate
    sr_orig         : original sample rate
    """
    wav_data, sr_orig = sf.read(io.BytesIO(audio_bytes), dtype=np.int16)
    waveform = (wav_data / 32768.0).astype("float32")

    # mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    waveform_orig = waveform.copy()

    # resample to 16kHz for YAMNet
    if sr_orig != config.SAMPLE_RATE:
        waveform_yamnet = resampy.resample(waveform, sr_orig, config.SAMPLE_RATE)
    else:
        waveform_yamnet = waveform.copy()

    return waveform_yamnet, config.SAMPLE_RATE, waveform_orig, sr_orig


def duration(waveform: np.ndarray, sr: int) -> float:
    """Return duration in seconds."""
    return len(waveform) / sr