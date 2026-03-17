"""
core/model.py
-------------
YAMNet model loading and inference.
No Streamlit imports here — pure ML logic.
"""

import numpy as np
import streamlit as st
import sys
import os

# make sure the root is in the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
# import params as yamnet_params
# import yamnet as yamnet_model
from core import params as yamnet_params
from core import yamnet as yamnet_model


@st.cache_resource(show_spinner="Loading YAMNet…")
def load_yamnet():
    """Load YAMNet once and cache it for the session."""
    p     = yamnet_params.Params()
    model = yamnet_model.yamnet_frames_model(p)
    model.load_weights(str(config.WEIGHTS_PATH))
    names = yamnet_model.class_names(str(config.CLASS_MAP_PATH))
    return model, names, p


def run_inference(waveform: np.ndarray):
    """
    Run YAMNet on a preprocessed waveform.

    Returns
    -------
    scores      : np.ndarray  (n_frames, 521)
    embeddings  : np.ndarray  (n_frames, 1024)
    spectrogram : np.ndarray  (n_frames, 64)
    """
    model, _, _ = load_yamnet()
    scores, embeddings, spectrogram = model(waveform)
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()
