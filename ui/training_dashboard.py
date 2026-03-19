"""
ui/training_dashboard.py
------------------------
Real-time training dashboard for Streamlit.
Shows live metrics, progress bar and updating loss/accuracy curves.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

BLUE   = "#1f77b4"   # train
ORANGE = "#ff7f0e"   # val


class StreamlitTrainingDashboard(tf.keras.callbacks.Callback):
    """
    Keras callback that updates a Streamlit dashboard in real time.

    Shows:
      - Progress bar with current epoch metrics
      - 4 metric cards (train/val acc + loss)
      - Two live charts side by side: Accuracy | Loss
      - Metrics log table with best epoch highlighted
    """

    def __init__(self, total_epochs: int, container=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.metrics_log  = []

        if container is None:
            container = st.container()

        self.progress_bar = container.progress(0, "Starting…")
        self.metrics_row  = container.empty()
        self.charts_ph    = container.empty()   # holds both charts side by side
        self.table_ph     = container.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs     = logs or {}
        acc      = logs.get("accuracy",     0) * 100
        val_acc  = logs.get("val_accuracy", 0) * 100
        loss     = logs.get("loss",         0)
        val_loss = logs.get("val_loss",     0)

        self.metrics_log.append({
            "Epoch":     epoch + 1,
            "Acc %":     round(acc,      1),
            "Val Acc %": round(val_acc,  1),
            "Loss":      round(loss,     4),
            "Val Loss":  round(val_loss, 4),
        })

        # ── Progress bar ───────────────────────────────────────────────────────
        pct = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(
            pct,
            f"Epoch {epoch+1}/{self.total_epochs} — "
            f"loss: {loss:.4f}  val_loss: {val_loss:.4f}  "
            f"acc: {acc:.1f}%  val_acc: {val_acc:.1f}%"
        )

        # ── Metric cards ───────────────────────────────────────────────────────
        with self.metrics_row.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train acc",  f"{acc:.1f}%")
            c2.metric("Val acc",    f"{val_acc:.1f}%",
                      delta=f"{val_acc - acc:.1f}%")
            c3.metric("Train loss", f"{loss:.4f}")
            c4.metric("Val loss",   f"{val_loss:.4f}",
                      delta=f"{val_loss - loss:.4f}")

        # ── Live charts (2 columns) ────────────────────────────────────────────
        if len(self.metrics_log) > 1:
            df     = pd.DataFrame(self.metrics_log)
            epochs = df["Epoch"].tolist()

            # Accuracy chart
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=epochs, y=df["Acc %"].tolist(),
                name="Train", mode="lines+markers",
                line=dict(color=BLUE, width=2),
                marker=dict(size=5),
            ))
            fig_acc.add_trace(go.Scatter(
                x=epochs, y=df["Val Acc %"].tolist(),
                name="Val", mode="lines+markers",
                line=dict(color=ORANGE, width=2),
                marker=dict(size=5),
            ))
            fig_acc.update_layout(
                title="Accuracy (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=40, b=40, l=50, r=20),
                font=dict(size=11),
                xaxis=dict(title="Epoch", showgrid=True,
                           gridcolor="rgba(128,128,128,0.12)"),
                yaxis=dict(title="Accuracy (%)", showgrid=True,
                           gridcolor="rgba(128,128,128,0.12)"),
                legend=dict(orientation="h", y=-0.3),
            )

            # Loss chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=df["Loss"].tolist(),
                name="Train", mode="lines+markers",
                line=dict(color=BLUE, width=2),
                marker=dict(size=5),
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=df["Val Loss"].tolist(),
                name="Val", mode="lines+markers",
                line=dict(color=ORANGE, width=2),
                marker=dict(size=5),
            ))
            fig_loss.update_layout(
                title="Loss",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=40, b=40, l=50, r=20),
                font=dict(size=11),
                xaxis=dict(title="Epoch", showgrid=True,
                           gridcolor="rgba(128,128,128,0.12)"),
                yaxis=dict(title="Loss", showgrid=True,
                           gridcolor="rgba(128,128,128,0.12)"),
                legend=dict(orientation="h", y=-0.3),
            )

            # render both side by side in the same placeholder
            with self.charts_ph.container():
                col_acc, col_loss = st.columns(2)
                with col_acc:
                    st.plotly_chart(fig_acc, use_container_width=True)
                with col_loss:
                    st.plotly_chart(fig_loss, use_container_width=True)

        # ── Metrics table ──────────────────────────────────────────────────────
        df = pd.DataFrame(self.metrics_log)
        self.table_ph.dataframe(
            df.style
              .highlight_max(subset=["Val Acc %"], color="#d4edda")
              .highlight_min(subset=["Val Loss"],  color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

    def on_train_end(self, logs=None):
        if self.metrics_log:
            best = max(self.metrics_log, key=lambda x: x["Val Acc %"])
            self.progress_bar.progress(
                1.0,
                f"✅ Training complete — "
                f"best val acc: {best['Val Acc %']}% at epoch {best['Epoch']}"
            )