"""
app.py — YAMNet Prediction
==========================
Entry point. Thin orchestrator — only UI wiring here.
All logic lives in core/ and ui/.

Audio pipeline:
  audio original ─┬─→ resamplear a 16kHz → YAMNet
                  └─→ sample rate original → SPL analysis

Run:
    streamlit run app.py
"""

import numpy as np
import streamlit as st

import config
from core.audio      import load_audio, duration
from core.model      import load_yamnet, run_inference
from core.spl        import compute_spl, compute_spl_time
from core.embeddings import project_with_references, has_umap
from core.visualize  import (waveform_fig, spectrogram_fig,
                              scores_heatmap_fig, top_n_bars_fig,
                              spl_fig, spl_time_fig, embedding_fig)
from ui.upload       import audio_input
from ui.metrics      import render_metrics

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

    waveform_yamnet, sr_yamnet, waveform_orig, sr_orig = load_audio(audio_bytes)

    with st.spinner("Running YAMNet…"):
        scores, embeddings, spectrogram = run_inference(waveform_yamnet)

    mean_scores   = np.mean(scores, axis=0)
    patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.audio(audio_bytes)
    st.markdown(" ")
    render_metrics(
        duration=duration(waveform_orig, sr_orig),
        sample_rate=sr_orig,
        n_frames=scores.shape[0],
        top_class=class_names[np.argmax(mean_scores)],
    )

    st.markdown("---")

    # ── Row 1: waveform + spectrogram ──────────────────────────────────────────
    col_w, col_s = st.columns(2)
    with col_w:
        st.markdown("#### Waveform")
        st.plotly_chart(waveform_fig(waveform_orig, sr_orig),
                        use_container_width=True)
    with col_s:
        st.markdown("#### Log-mel spectrogram")
        st.plotly_chart(spectrogram_fig(spectrogram), use_container_width=True)

    st.markdown("---")

    # ── Row 2: scores heatmap + top-N bars ─────────────────────────────────────
    top_n       = st.slider("Top-N classes", min_value=3,
                             max_value=config.TOP_N_MAX,
                             value=config.TOP_N_DEFAULT)
    top_indices = list(np.argsort(mean_scores)[::-1][:top_n])

    col_h, col_b = st.columns([3, 2])
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

    st.markdown("---")

    # ── Row 3: SPL analysis ────────────────────────────────────────────────────
    st.markdown("#### SPL analysis")
    st.caption(f"Analysing at original sample rate: **{sr_orig:,} Hz** "
               f"(max frequency: **{sr_orig // 2:,} Hz**)")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

    with ctrl1:
        fraction = st.radio(
            "Resolution",
            options=[1, 3],
            format_func=lambda x: "Octave (1/1)" if x == 1 else "1/3 Octave",
            index=0 if config.SPL_FRACTION_DEFAULT == 1 else 1,
        )

    with ctrl2:
        time_mode = st.radio(
            "Time weighting",
            options=["none", "fast", "slow", "impulse"],
            format_func=lambda x: {
                "none":    "None (RMS)",
                "fast":    "Fast (125ms)",
                "slow":    "Slow (1000ms)",
                "impulse": "Impulse (35ms)",
            }[x],
            index=0,
        )

    with ctrl3:
        st.info("For accurate dB SPL values you need a calibrated microphone. "
                "Without calibration, values are relative (uncalibrated dB SPL).")
        cal_factor = st.number_input(
            "Calibration factor",
            min_value=0.0001, max_value=100.0,
            value=1.0, step=0.001, format="%.4f",
        )

    if time_mode == "none":
        with st.spinner("Computing RMS SPL…"):
            spl_vals, spl_freqs = compute_spl(
                waveform_orig, sr_orig,
                fraction=fraction, mode="spl",
                calibration_factor=cal_factor,
            )
        st.plotly_chart(
            spl_fig(spl_vals, spl_freqs, "spl", fraction),
            use_container_width=True,
        )
    else:
        with st.spinner(f"Computing {time_mode.capitalize()} time-weighted SPL…"):
            levels, spl_freqs, t_axis = compute_spl_time(
                waveform_orig, sr_orig,
                fraction=fraction, mode="spl",
                time_mode=time_mode,
                calibration_factor=cal_factor,
            )
        st.plotly_chart(
            spl_time_fig(levels, spl_freqs, t_axis, "spl", fraction, time_mode),
            use_container_width=True,
        )
        with st.expander("Show RMS summary per band"):
            spl_vals, _ = compute_spl(
                waveform_orig, sr_orig,
                fraction=fraction, mode="spl",
                calibration_factor=cal_factor,
            )
            st.plotly_chart(
                spl_fig(spl_vals, spl_freqs, "spl", fraction),
                use_container_width=True,
            )

    st.markdown("---")

    # ── Row 4: Embedding analysis ──────────────────────────────────────────────
    st.markdown("#### Embedding analysis")
    st.caption("Each circle is a ~96ms frame of your audio projected in 2D. "
               "Diamonds are AudioSet reference points weighted by YAMNet scores. "
               "Frames that cluster together sound similar to YAMNet.")

    emb_col1, emb_col2 = st.columns([1, 3])

    with emb_col1:
        method = st.radio(
            "Projection method",
            options=["pca", "umap", "tsne"],
            format_func=lambda x: {
                "pca":  "PCA (fast)",
                "umap": "UMAP" + ("" if has_umap() else " ⚠ not installed"),
                "tsne": "t-SNE",
            }[x],
            index=0,
        )
        top_n_refs = st.slider("Reference classes", min_value=3,
                                max_value=20, value=10)

        if method == "umap":
            if not has_umap():
                st.error("Install with: `pip install umap-learn`")
                st.stop()
            n_neighbors = st.slider("n_neighbors", 2,
                                     min(50, len(embeddings) - 1), 10)
            min_dist    = st.slider("min_dist", 0.0, 0.9, 0.1, step=0.05)
            kw = {"n_neighbors": n_neighbors, "min_dist": min_dist}
        elif method == "tsne":
            perplexity = st.slider("Perplexity", 2,
                                    min(50, len(embeddings) - 1), 10)
            kw = {"perplexity": perplexity}
        else:
            kw = {}

    with emb_col2:
        with st.spinner(f"Projecting with {method.upper()}…"):
            proj_frames, proj_refs, frame_labels, ref_labels, expl_var = project_with_references(
                embeddings, scores, class_names,
                method=method, top_n_refs=top_n_refs, **kw,
            )

        st.plotly_chart(
            embedding_fig(proj_frames, frame_labels,
                          proj_refs, ref_labels, method,
                          explained_variance=expl_var),
            use_container_width=True,
        )

else:
    st.info("Drop a WAV file or record from your microphone above to get started.")