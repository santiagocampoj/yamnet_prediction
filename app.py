"""
app.py — YAMNet Prediction
==========================
Entry point. Thin orchestrator — only UI wiring here.
All logic lives in core/ and ui/.

Run:
    streamlit run app.py
"""

import numpy as np
import streamlit as st

import config
from core.audio     import load_audio, duration
from core.model     import load_yamnet, run_inference
from core.visualize import (waveform_fig, spectrogram_fig,
                             scores_heatmap_fig, top_n_bars_fig)
from ui.upload      import audio_input
from ui.metrics     import render_metrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  div[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid rgba(128,128,128,0.15);
  }
  div[data-testid="metric-container"] label { font-size: 0.75rem !important; }
  div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,111,71,0.4);
    border-radius: 12px;
    padding: 12px;
    background: rgba(139,111,71,0.04);
  }
  div[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,111,71,0.8);
    background: rgba(139,111,71,0.08);
  }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
_, class_names, params = load_yamnet()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 1.5rem 0 1rem 0;">
  <h1 style="font-size:2.2rem; font-weight:600; margin-bottom:0.4rem;">
    {config.PAGE_ICON} {config.PAGE_TITLE}
  </h1>
  <p style="font-size:1rem; color:{config.PRIMARY_COLOR}; margin:0;">
    {config.PAGE_SUBTITLE}
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Audio input ────────────────────────────────────────────────────────────────
audio_bytes = audio_input()

st.markdown("---")

# ── Analysis ───────────────────────────────────────────────────────────────────
if audio_bytes:

    waveform, sr = load_audio(audio_bytes)

    with st.spinner("Running YAMNet…"):
        scores, embeddings, spectrogram = run_inference(waveform)

    mean_scores   = np.mean(scores, axis=0)
    patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds

    # audio player + metrics
    st.audio(audio_bytes)
    st.markdown(" ")
    render_metrics(
        duration=duration(waveform),
        sample_rate=sr,
        n_frames=scores.shape[0],
        top_class=class_names[np.argmax(mean_scores)],
    )

    st.markdown("---")

    # row 1 — waveform + spectrogram
    col_w, col_s = st.columns(2)
    with col_w:
        st.markdown("#### Waveform")
        st.plotly_chart(waveform_fig(waveform), use_container_width=True)
    with col_s:
        st.markdown("#### Log-mel spectrogram")
        st.plotly_chart(spectrogram_fig(spectrogram), use_container_width=True)

    st.markdown("---")

    # top-N slider — lives here, above the charts it controls
    top_n       = st.slider("Top-N classes", min_value=3,
                             max_value=config.TOP_N_MAX,
                             value=config.TOP_N_DEFAULT)
    top_indices = list(np.argsort(mean_scores)[::-1][:top_n])

    # row 2 — heatmap + bars
    col_h, col_b = st.columns([2, 1])
    with col_h:
        st.markdown("#### Class scores over time")
        st.plotly_chart(
            scores_heatmap_fig(scores, top_indices, class_names, patch_padding),
            use_container_width=True,
        )
    with col_b:
        st.markdown(f"#### Top {top_n} predictions")
        st.plotly_chart(
            top_n_bars_fig(mean_scores, top_indices, class_names),
            use_container_width=True,
        )

else:
    st.info("Drop a WAV file or record from your microphone above to get started.")
