"""
core/visualize.py
-----------------
All Plotly chart builders.
No Streamlit imports here — returns go.Figure objects.
"""

import numpy as np
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=30, l=50, r=10),
    font=dict(size=11),
)


def waveform_fig(waveform: np.ndarray) -> go.Figure:
    step = max(1, len(waveform) // 4000)
    w    = waveform[::step]
    t    = np.arange(len(w)) * step / config.SAMPLE_RATE
    fig  = go.Figure(go.Scatter(
        x=t, y=w,
        line=dict(color=config.CHART_COLORS[0], width=0.8),
        fill="tozeroy",
        fillcolor="rgba(83,74,183,0.10)",
        hovertemplate="t=%{x:.3f}s  amp=%{y:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Time (s)", showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)", zeroline=False)
    fig.update_yaxes(title="Amplitude", showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)",
                     zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_layout(**_LAYOUT, height=230)
    return fig


def spectrogram_fig(spectrogram: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=spectrogram.T,
        colorscale="Magma",
        showscale=False,
        hovertemplate="frame=%{x}  mel=%{y}  val=%{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Frame", showgrid=False)
    fig.update_yaxes(title="Mel bin", showgrid=False)
    fig.update_layout(**_LAYOUT, height=230)
    return fig


def scores_heatmap_fig(scores: np.ndarray, top_indices: list,
                       class_names: list, patch_padding: float) -> go.Figure:
    z      = scores[:, top_indices].T
    labels = [class_names[i] for i in top_indices]
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
    fig.update_layout(**_LAYOUT, height=260)
    return fig


def top_n_bars_fig(mean_scores: np.ndarray, top_indices: list,
                   class_names: list) -> go.Figure:
    n      = len(top_indices)
    labels = [class_names[i] for i in reversed(top_indices)]
    values = [float(mean_scores[i]) for i in reversed(top_indices)]
    colors = [config.CHART_COLORS[i % len(config.CHART_COLORS)]
              for i in range(n)][::-1]
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
    fig.update_layout(**_LAYOUT, height=260)
    return fig


def spl_fig(spl: np.ndarray, freqs: np.ndarray,
            mode: str, fraction: int) -> go.Figure:
    """
    SPL bar chart per octave or 1/3-octave band.

    Parameters
    ----------
    spl      : level per band (dB)
    freqs    : center frequencies (Hz)
    mode     : "dbfs" or "spl"
    fraction : 1 or 3
    """
    y_label = "dBFS" if mode == "dbfs" else "dB SPL"
    title   = f"{'1/3-Octave' if fraction == 3 else 'Octave'} band levels ({y_label})"

    def fmt_freq(f):
        if f >= 1000:
            v = f"{f/1000:.1f}".rstrip("0").rstrip(".")
            return f"{v}k Hz"
        return f"{int(round(f))} Hz"

    labels = [fmt_freq(f) for f in freqs]

    # colour bars by relative energy level
    spl_min, spl_max = min(spl), max(spl)
    colors = []
    for v in spl:
        ratio = (v - spl_min) / (spl_max - spl_min + 1e-9)
        if ratio > 0.66:
            colors.append("#1D9E75")   # teal — high energy
        elif ratio > 0.33:
            colors.append("#534AB7")   # purple — mid
        else:
            colors.append("#B4B2A9")   # gray — low energy

    fig = go.Figure(go.Bar(
        x=labels,
        y=spl,
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in spl],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f} " + y_label + "<extra></extra>",
    ))
    fig.update_xaxes(title="Center frequency", tickangle=-45, showgrid=False)
    fig.update_yaxes(title=y_label, showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        title=dict(text=title, font=dict(size=13)),
        margin=dict(t=40, b=80, l=60, r=10),
        height=320,
    )
    return fig