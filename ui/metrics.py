"""
ui/metrics.py
-------------
Metrics row component.
"""

import numpy as np
import streamlit as st


def render_metrics(duration: float, sample_rate: int,
                   n_frames: int, top_class: str) -> None:
    """Render the 4 metric cards row."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",    f"{duration:.2f} s")
    c2.metric("Sample rate", f"{sample_rate // 1000} kHz")
    c3.metric("Frames",      n_frames)
    c4.metric("Top class",   top_class)
