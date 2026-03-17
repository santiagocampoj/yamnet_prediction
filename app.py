"""
app.py — YAMNet Prediction Dashboard
=====================================
Run from inside the yamnet_prediction folder:
    streamlit run app.py

Requirements:
    pip install streamlit soundfile resampy plotly
"""

import io
import numpy as np
import soundfile as sf
import resampy
import plotly.graph_objects as go
import streamlit as st

import params as yamnet_params
import yamnet as yamnet_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YAMNet Prediction",
    page_icon="🎧",
    layout="wide",
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
  /* make the file uploader area bigger */
  div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(83,74,183,0.4);
    border-radius: 12px;
    padding: 12px;
    background: rgba(83,74,183,0.04);
  }
  div[data-testid="stFileUploader"]:hover {
    border-color: rgba(83,74,183,0.8);
    background: rgba(83,74,183,0.08);
  }
</style>
""", unsafe_allow_html=True)

PURPLE = "#534AB7"
COLORS = ["#534AB7", "#7F77DD", "#AFA9EC", "#CECBF6", "#9FE1CB",
          "#1D9E75", "#D85A30", "#F0997B", "#185FA5", "#85B7EB"]

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YAMNet…")
def load_model():
    p     = yamnet_params.Params()
    model = yamnet_model.yamnet_frames_model(p)
    model.load_weights("yamnet.h5")
    names = yamnet_model.class_names("yamnet_class_map.csv")
    return model, names, p

yamnet, class_names, params = load_model()


# ── Plotly helpers ─────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=30, l=50, r=10),
    font=dict(size=11),
)


def plot_waveform(waveform: np.ndarray) -> go.Figure:
    step = max(1, len(waveform) // 4000)
    w    = waveform[::step]
    t    = np.arange(len(w)) * step / params.sample_rate
    fig  = go.Figure(go.Scatter(
        x=t, y=w,
        line=dict(color=PURPLE, width=0.8),
        fill="tozeroy",
        fillcolor="rgba(83,74,183,0.12)",
        hovertemplate="t=%{x:.3f}s  amp=%{y:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Time (s)", showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)", zeroline=False)
    fig.update_yaxes(title="Amplitude", showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)",
                     zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_layout(**LAYOUT, height=230)
    return fig


def plot_spectrogram(spectrogram: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=spectrogram.T,
        colorscale="Magma",
        showscale=False,
        hovertemplate="frame=%{x}  mel=%{y}  val=%{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Frame", showgrid=False)
    fig.update_yaxes(title="Mel bin", showgrid=False)
    fig.update_layout(**LAYOUT, height=230)
    return fig


def plot_scores_heatmap(scores: np.ndarray, top_indices: list,
                        names: list, patch_padding: float) -> go.Figure:
    z      = scores[:, top_indices].T
    labels = [names[i] for i in top_indices]
    fig    = go.Figure(go.Heatmap(
        z=z,
        colorscale="gray_r",
        showscale=False,
        y=labels,
        hovertemplate="frame=%{x}  class=%{y}  score=%{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Frame",
                     range=[-patch_padding, scores.shape[0] + patch_padding],
                     showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(**LAYOUT, height=260)
    return fig


def plot_top_bars(mean_scores: np.ndarray, top_indices: list,
                  names: list) -> go.Figure:
    n      = len(top_indices)
    labels = [names[i] for i in reversed(top_indices)]
    values = [float(mean_scores[i]) for i in reversed(top_indices)]
    colors = [COLORS[i % len(COLORS)] for i in range(n)][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.15], showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)", title="Mean score")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(**LAYOUT, height=260)
    return fig


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size: 2.2rem; font-weight: 600; margin-bottom: 0.4rem;">
        🎧 YAMNet Prediction
    </h1>
    <p style="font-size: 1rem; color: #8B6F47; margin: 0;">
        Analyse any audio with the original Google YAMNet model.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── Audio input — drag & drop + mic ───────────────────────────────────────────
st.markdown("### Load audio")

col_up, col_mic = st.columns(2)

with col_up:
    st.markdown("**Drag & drop a WAV file**")
    uploaded = st.file_uploader(
        "Drop your file here or click to browse",
        type=["wav"],
        label_visibility="collapsed",
    )

with col_mic:
    st.markdown("**Or record from microphone**")
    recorded = st.audio_input("Click to record", label_visibility="collapsed")

# resolve which source to use — file takes priority
audio_source = uploaded or recorded
audio_bytes  = audio_source.read() if audio_source else None

st.markdown("---")


# ── Analysis ───────────────────────────────────────────────────────────────────
if audio_bytes:

    # load & preprocess
    wav_data, sr = sf.read(io.BytesIO(audio_bytes), dtype=np.int16)
    waveform     = (wav_data / 32768.0).astype("float32")
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # run model
    with st.spinner("Running YAMNet…"):
        scores, embeddings, spectrogram = yamnet(waveform)
        scores      = scores.numpy()
        spectrogram = spectrogram.numpy()

    mean_scores   = np.mean(scores, axis=0)
    patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
    duration      = len(waveform) / params.sample_rate

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.audio(audio_bytes)
    st.markdown(" ")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",    f"{duration:.2f} s")
    c2.metric("Sample rate", f"{params.sample_rate // 1000} kHz")
    c3.metric("Frames",      scores.shape[0])
    c4.metric("Top class",   class_names[np.argmax(mean_scores)])

    st.markdown("---")

    # ── Row 1: waveform + spectrogram ──────────────────────────────────────────
    col_w, col_s = st.columns(2)
    with col_w:
        st.markdown("#### Waveform")
        st.plotly_chart(plot_waveform(waveform), use_container_width=True)
    with col_s:
        st.markdown("#### Log-mel spectrogram")
        st.plotly_chart(plot_spectrogram(spectrogram), use_container_width=True)

    st.markdown("---")

    # ── Row 2: heatmap + top-N bars ────────────────────────────────────────────
    # Top-N control lives here, right above the plots it affects
    top_n       = st.slider("Top-N classes", min_value=3, max_value=20, value=5,
                            help="Controls both the heatmap and the bar chart")
    top_indices = list(np.argsort(mean_scores)[::-1][:top_n])

    col_h, col_b = st.columns([2, 1])
    with col_h:
        st.markdown(f"#### Class scores over time")
        st.plotly_chart(
            plot_scores_heatmap(scores, top_indices, class_names, patch_padding),
            use_container_width=True,
        )
    with col_b:
        st.markdown(f"#### Top {top_n} predictions")
        st.plotly_chart(
            plot_top_bars(mean_scores, top_indices, class_names),
            use_container_width=True,
        )

else:
    # empty state hint
    st.info("Drop a WAV file or record from your microphone above to get started.")