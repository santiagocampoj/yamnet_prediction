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


def waveform_fig(waveform: np.ndarray, sr: int) -> go.Figure:
    """Waveform plot using the original sample rate for correct time axis."""
    step = max(1, len(waveform) // 4000)
    w    = waveform[::step]
    t    = np.arange(len(w)) * step / sr
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
    fig.update_layout(**_LAYOUT, height=180)
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
    fig.update_layout(**_LAYOUT, height=180)
    return fig


def spl_fig(spl: np.ndarray, freqs: np.ndarray,
            mode: str, fraction: int) -> go.Figure:
    """RMS SPL bar chart per octave or 1/3-octave band."""
    y_label = "dBFS" if mode == "dbfs" else "dB SPL"
    title   = f"{'1/3-Octave' if fraction == 3 else 'Octave'} band levels — RMS ({y_label})"

    def fmt_freq(f):
        if f >= 1000:
            v = f"{f/1000:.1f}".rstrip("0").rstrip(".")
            return f"{v}k Hz"
        return f"{int(round(f))} Hz"

    labels  = [fmt_freq(f) for f in freqs]
    spl_min = min(spl)
    spl_max = max(spl)

    colors = []
    for v in spl:
        ratio = (v - spl_min) / (spl_max - spl_min + 1e-9)
        if ratio > 0.66:
            colors.append("#1D9E75")
        elif ratio > 0.33:
            colors.append("#534AB7")
        else:
            colors.append("#B4B2A9")

    fig = go.Figure(go.Bar(
        x=labels, y=spl,
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


def spl_time_fig(levels: np.ndarray, freqs: np.ndarray,
                 t: np.ndarray, mode: str,
                 fraction: int, time_mode: str) -> go.Figure:
    """Time-weighted SPL heatmap (bands × time)."""
    y_label = "dBFS" if mode == "dbfs" else "dB SPL"
    title   = (f"Time-weighted SPL — {time_mode.capitalize()} "
               f"({'1/3-Oct' if fraction == 3 else 'Oct'}, {y_label})")

    def fmt_freq(f):
        if f >= 1000:
            v = f"{f/1000:.1f}".rstrip("0").rstrip(".")
            return f"{v}k Hz"
        return f"{int(round(f))} Hz"

    step     = max(1, levels.shape[1] // 2000)
    t_ds     = t[::step]
    z_ds     = levels[:, ::step]
    y_labels = [fmt_freq(f) for f in freqs]

    fig = go.Figure(go.Heatmap(
        x=t_ds,
        y=y_labels,
        z=z_ds,
        colorscale="Viridis",
        colorbar=dict(title=y_label, thickness=12),
        hovertemplate="t=%{x:.3f}s<br>band=%{y}<br>level=%{z:.1f} "
                      + y_label + "<extra></extra>",
    ))
    fig.update_xaxes(title="Time (s)", showgrid=False)
    fig.update_yaxes(title="Band", showgrid=False, autorange="reversed")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        title=dict(text=title, font=dict(size=13)),
        margin=dict(t=40, b=50, l=80, r=20),
        height=360,
    )
    return fig







def embedding_fig(
    proj_frames: np.ndarray,
    frame_labels: list,
    proj_refs: np.ndarray,
    ref_labels: list,
    method: str,
    explained_variance: list | None = None,
) -> go.Figure:
    import plotly.express as px

    fig = go.Figure()
    all_classes = sorted(set(frame_labels))
    single_class = len(all_classes) == 1

    # ── axis labels ────────────────────────────────────────────────────────────
    if explained_variance and len(explained_variance) >= 2:
        xaxis_title = f"PC 1 ({explained_variance[0]*100:.1f}% variance)"
        yaxis_title = f"PC 2 ({explained_variance[1]*100:.1f}% variance)"
    else:
        xaxis_title = "Dim 1"
        yaxis_title = "Dim 2"

    if single_class:
        fig.add_trace(go.Scatter(
            x=proj_frames[:, 0],
            y=proj_frames[:, 1],
            mode="markers",
            name="frames (time →)",
            marker=dict(
                size=7, opacity=0.8,
                color=list(range(len(proj_frames))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Frame", thickness=12,
                    x=1.18, xanchor="left",
                ),
            ),
            hovertemplate="frame %{text} — %{customdata}<extra></extra>",
            text=[str(i) for i in range(len(proj_frames))],
            customdata=frame_labels,
        ))
    else:
        palette   = px.colors.qualitative.Plotly
        color_map = {cls: palette[i % len(palette)]
                     for i, cls in enumerate(all_classes)}
        for cls in all_classes:
            mask = [i for i, l in enumerate(frame_labels) if l == cls]
            fig.add_trace(go.Scatter(
                x=proj_frames[mask, 0],
                y=proj_frames[mask, 1],
                mode="markers",
                name=cls,
                marker=dict(size=7, color=color_map[cls],
                            opacity=0.7, symbol="circle"),
                hovertemplate=f"<b>{cls}</b> — frame %{{text}}<extra></extra>",
                text=[str(m) for m in mask],
            ))

    # ── mean clip point (★) ────────────────────────────────────────────────────
    mean_x = proj_frames[:, 0].mean()
    mean_y = proj_frames[:, 1].mean()
    fig.add_trace(go.Scatter(
        x=[mean_x], y=[mean_y],
        mode="markers+text",
        name="★ your clip (mean)",
        marker=dict(size=22, color="gold", symbol="star",
                    line=dict(color="black", width=1.5), opacity=1.0),
        text=["★ your clip"],
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertemplate="<b>★ Your clip</b><br>mean of all frames<extra></extra>",
    ))

    # ── reference diamonds ─────────────────────────────────────────────────────
    palette   = px.colors.qualitative.Plotly
    color_map = {cls: palette[i % len(palette)]
                 for i, cls in enumerate(all_classes)}
    seen = set()
    for i, label in enumerate(ref_labels):
        if label in seen:
            continue
        seen.add(label)
        color = color_map.get(label, "#888888")
        fig.add_trace(go.Scatter(
            x=[proj_refs[i, 0]],
            y=[proj_refs[i, 1]],
            mode="markers+text",
            name=f"{label} ◆",
            marker=dict(size=18, color=color, opacity=1.0,
                        symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[label],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=f"<b>{label}</b> — AudioSet ref<extra></extra>",
        ))

    grad_or_class = "gradient = time" if single_class else "colour = class"
    fig.update_xaxes(title=xaxis_title, showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)", zeroline=False)
    fig.update_yaxes(title=yaxis_title, showgrid=True,
                     gridcolor="rgba(128,128,128,0.12)", zeroline=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        title=dict(
            text=f"Embedding space — {method.upper()} "
                 f"({grad_or_class} · diamonds = AudioSet refs · ★ = your clip mean)",
            font=dict(size=13),
        ),
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(t=40, b=40, l=50, r=220),
        height=500,
    )
    return fig


def confusion_matrix_fig(cm: list, classes: list) -> go.Figure:
    """
    Interactive Plotly confusion matrix with hover details.
    Shows both raw counts and row-normalized percentages on hover.
    """
    cm_arr   = np.array(cm)
    n        = len(classes)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm_arr / row_sums * 100, 0)

    hover = [[
        f"<b>True:</b> {classes[i]}<br>"
        f"<b>Predicted:</b> {classes[j]}<br>"
        f"<b>Count:</b> {cm_arr[i, j]}<br>"
        f"<b>Row %:</b> {cm_pct[i, j]:.1f}%"
        for j in range(n)]
        for i in range(n)]

    text = [[
        f"{cm_arr[i,j]}<br><span style='font-size:10px'>{cm_pct[i,j]:.0f}%</span>"
        for j in range(n)]
        for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=classes, y=classes,
        colorscale="Blues", zmin=0, zmax=100,
        text=text, texttemplate="%{text}",
        textfont=dict(size=12),
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title="Row %", thickness=12),
        showscale=True,
    ))
    fig.update_xaxes(title="Predicted", side="bottom",
                     tickangle=-35, showgrid=False)
    fig.update_yaxes(title="True", autorange="reversed", showgrid=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        margin=dict(t=20, b=80, l=100, r=60),
        height=max(350, n * 60),
    )
    return fig