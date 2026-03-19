"""
app.py — YAMNet Prediction
==========================
Three tabs:
  📊 Analysis  — YAMNet inference, SPL, embeddings
  🏋️ Training  — ESC-50 / custom data, classifier, fine-tune
  📈 Results   — curves, confusion matrix, predict, compare

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
from core.audio      import load_audio, duration
from core.model      import load_yamnet, run_inference
from core.spl        import compute_spl, compute_spl_time
from core.embeddings import project_with_references, has_umap
from core.visualize  import (waveform_fig, spectrogram_fig,
                              scores_heatmap_fig, top_n_bars_fig,
                              spl_fig, spl_time_fig, embedding_fig,
                              confusion_matrix_fig)
from ui.upload            import audio_input
from ui.metrics           import render_metrics
from ui.training_dashboard import StreamlitTrainingDashboard

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
    border-radius: 10px; padding: 12px 16px;
    border: 1px solid rgba(128,128,128,0.15);
  }
  div[data-testid="metric-container"] label { font-size: 0.75rem !important; }
  div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,111,71,0.4);
    border-radius: 12px; padding: 12px;
    background: rgba(139,111,71,0.04);
  }
  div[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,111,71,0.8);
    background: rgba(139,111,71,0.08);
  }
  button[data-baseweb="tab"] p {
    font-size: 1.6rem !important;
    font-weight: 500 !important;
  }
  button[data-baseweb="tab"] { padding: 14px 48px !important; }
  [data-baseweb="tab-list"] { justify-content: center !important; gap: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model once ────────────────────────────────────────────────────────────
_, class_names, params = load_yamnet()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:1.5rem 0 1rem 0;">
  <h1 style="font-size:2.2rem;font-weight:600;margin-bottom:0.4rem;">
    {config.PAGE_ICON} {config.PAGE_TITLE}
  </h1>
  <p style="font-size:1rem;color:{config.PRIMARY_COLOR};margin:0;">
    {config.PAGE_SUBTITLE}
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab_analysis, tab_training, tab_results = st.tabs([
    "📊  Analysis",
    "🏋️  Training",
    "📈  Results",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    audio_bytes = audio_input()
    st.markdown("---")

    if audio_bytes:
        st.session_state["audio_bytes"] = audio_bytes
    audio_bytes = st.session_state.get("audio_bytes")

    if audio_bytes:
        waveform_yamnet, sr_yamnet, waveform_orig, sr_orig = load_audio(audio_bytes)

        with st.spinner("Running YAMNet…"):
            scores, embeddings, spectrogram = run_inference(waveform_yamnet)

        mean_scores   = np.mean(scores, axis=0)
        patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds

        st.audio(audio_bytes)
        st.markdown(" ")
        render_metrics(
            duration=duration(waveform_orig, sr_orig),
            sample_rate=sr_orig,
            n_frames=scores.shape[0],
            top_class=class_names[np.argmax(mean_scores)],
        )
        st.markdown("---")

        col_w, col_s = st.columns(2)
        with col_w:
            st.markdown("#### Waveform")
            st.plotly_chart(waveform_fig(waveform_orig, sr_orig),
                            use_container_width=True)
        with col_s:
            st.markdown("#### Log-mel spectrogram")
            st.plotly_chart(spectrogram_fig(spectrogram),
                            use_container_width=True)

        st.markdown("---")

        top_n       = st.slider("Top-N classes", 3, config.TOP_N_MAX,
                                 config.TOP_N_DEFAULT, key="an_topn")
        top_indices = list(np.argsort(mean_scores)[::-1][:top_n])
        col_h, col_b = st.columns([3, 2])
        with col_h:
            st.markdown("#### Class scores over time")
            st.plotly_chart(
                scores_heatmap_fig(scores, top_indices, class_names,
                                   patch_padding),
                use_container_width=True)
        with col_b:
            st.markdown(f"#### Top {top_n} predictions")
            st.plotly_chart(
                top_n_bars_fig(mean_scores, top_indices, class_names),
                use_container_width=True)

        st.markdown("---")

        st.markdown("#### SPL analysis")
        st.caption(f"Original sample rate: **{sr_orig:,} Hz** — "
                   f"max frequency: **{sr_orig//2:,} Hz**")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            fraction = st.radio("Resolution", [1, 3],
                                 format_func=lambda x:
                                     "Octave" if x==1 else "1/3 Octave",
                                 key="an_frac")
        with c2:
            time_mode = st.radio(
                "Time weighting",
                ["none","fast","slow","impulse"],
                format_func=lambda x: {
                    "none":"None (RMS)","fast":"Fast 125ms",
                    "slow":"Slow 1s","impulse":"Impulse 35ms"}[x],
                key="an_tw")
        with c3:
            st.info("Without a calibrated mic, values are relative dB SPL.")
            cal_factor = st.number_input(
                "Calibration factor", 0.0001, 100.0, 1.0, 0.001,
                format="%.4f", key="an_cal")

        if time_mode == "none":
            sv, sf = compute_spl(waveform_orig, sr_orig, fraction=fraction,
                                  mode="spl", calibration_factor=cal_factor)
            st.plotly_chart(spl_fig(sv, sf, "spl", fraction),
                            use_container_width=True)
        else:
            lv, sf, ta = compute_spl_time(
                waveform_orig, sr_orig, fraction=fraction, mode="spl",
                time_mode=time_mode, calibration_factor=cal_factor)
            st.plotly_chart(
                spl_time_fig(lv, sf, ta, "spl", fraction, time_mode),
                use_container_width=True)
            with st.expander("Show RMS summary"):
                sv, _ = compute_spl(waveform_orig, sr_orig,
                                     fraction=fraction, mode="spl",
                                     calibration_factor=cal_factor)
                st.plotly_chart(spl_fig(sv, sf, "spl", fraction),
                                use_container_width=True)

        st.markdown("---")

        st.markdown("#### Embedding analysis")
        st.caption("Circles = ~96ms frames · Diamonds = AudioSet refs · "
                   "★ = mean of your clip")
        ec1, ec2 = st.columns([1, 3])
        with ec1:
            method = st.radio(
                "Projection", ["pca","umap","tsne"],
                format_func=lambda x: {
                    "pca":"PCA (fast)",
                    "umap":"UMAP" + ("" if has_umap() else " ⚠"),
                    "tsne":"t-SNE"}[x],
                key="an_proj")
            top_n_refs = st.slider("Reference classes", 3, 20, 10,
                                    key="an_refs")
            kw = {}
            if method == "umap":
                if not has_umap():
                    st.error("pip install umap-learn"); st.stop()
                kw["n_neighbors"] = st.slider(
                    "n_neighbors", 2, min(50, len(embeddings)-1), 10,
                    key="an_nn")
                kw["min_dist"] = st.slider(
                    "min_dist", 0.0, 0.9, 0.1, step=0.05, key="an_md")
            elif method == "tsne":
                kw["perplexity"] = st.slider(
                    "Perplexity", 2, min(50, len(embeddings)-1), 10,
                    key="an_pp")
        with ec2:
            with st.spinner(f"Projecting with {method.upper()}…"):
                proj_frames, proj_refs, frame_labels, ref_labels, expl_var = \
                    project_with_references(
                        embeddings, scores, class_names,
                        method=method, top_n_refs=top_n_refs, **kw)
            st.plotly_chart(
                embedding_fig(proj_frames, frame_labels, proj_refs,
                              ref_labels, method,
                              explained_variance=expl_var),
                use_container_width=True)
    else:
        st.info("Upload or record audio above to start the analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_training:

    st.markdown("### 🏋️ Train a custom classifier")
    st.caption("Clips are always split at clip level before extracting embeddings "
               "— no frames from the same clip appear in different splits.")
    st.markdown("---")

    # ── Step 1: Data source ────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Data source")
    data_source = st.radio("Source",
                            ["ESC-50 (download)", "My own audio folder"],
                            horizontal=True, key="tr_source")

    if data_source == "ESC-50 (download)":
        from core.esc50 import (is_downloaded, download, available_classes,
                                 prepare_train_folder, get_metadata,
                                 load_dataset_with_folds)

        if is_downloaded():
            st.success("ESC-50 already downloaded ✓")
        else:
            st.info("~600 MB download from GitHub.")
            if st.button("Download ESC-50", type="primary", key="tr_dl"):
                prog = st.progress(0, "Downloading…")
                ok   = download(progress_callback=lambda p:
                                prog.progress(p, f"{p*100:.0f}%"))
                prog.empty()
                if ok:
                    st.success("Downloaded ✓"); st.rerun()
                else:
                    st.error("Download failed.")

        if is_downloaded():
            df      = get_metadata()
            all_cls = available_classes()

            with st.expander("Browse dataset"):
                dist = df["category"].value_counts().reset_index()
                dist.columns = ["class","count"]
                fig_bar = go.Figure(go.Bar(
                    x=dist["class"], y=dist["count"],
                    marker=dict(color=config.CHART_COLORS[0])))
                fig_bar.update_layout(
                    height=200, margin=dict(t=10,b=60,l=40,r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickangle=-45))
                st.plotly_chart(fig_bar, use_container_width=True)

            preset = st.selectbox("Quick preset", [
                "Custom selection",
                "Animals (dog, cat, crow, rooster, hen)",
                "Nature (rain, sea_waves, crackling_fire, crickets, chirping_birds)",
                "Urban (chainsaw, airplane, helicopter, train, hand_saw)",
                "Human (crying_baby, laughing, clapping, sneezing, coughing)",
            ], key="tr_preset")

            preset_map = {
                "Animals (dog, cat, crow, rooster, hen)":
                    ["dog","cat","crow","rooster","hen"],
                "Nature (rain, sea_waves, crackling_fire, crickets, chirping_birds)":
                    ["rain","sea_waves","crackling_fire","crickets","chirping_birds"],
                "Urban (chainsaw, airplane, helicopter, train, hand_saw)":
                    ["chainsaw","airplane","helicopter","train","hand_saw"],
                "Human (crying_baby, laughing, clapping, sneezing, coughing)":
                    ["crying_baby","laughing","clapping","sneezing","coughing"],
            }
            default  = preset_map.get(preset, ["dog","cat"])
            selected = st.multiselect("Classes", all_cls, default=default,
                                       key="tr_classes")

            if selected:
                cnt  = df[df["category"].isin(selected)]["category"].value_counts()
                cols = st.columns(min(len(selected), 6))
                for i, cls in enumerate(selected):
                    cols[i % len(cols)].metric(cls, cnt.get(cls, 0))
                st.info("ESC-50 folds: **1-3 = train · 4 = val · 5 = test**")
                st.session_state["tr_selected_classes"] = selected
                st.session_state["tr_use_esc50_folds"]  = True

    else:
        st.info("Point to a folder with one subfolder per class:\n"
                "`my_data/dog/*.wav`, `my_data/cat/*.wav`, etc.")
        custom_dir = st.text_input("Folder path", "./data/train",
                                    key="tr_custom")
        if os.path.isdir(custom_dir):
            classes_found = [d for d in os.listdir(custom_dir)
                             if os.path.isdir(os.path.join(custom_dir, d))]
            if classes_found:
                st.success(f"Found {len(classes_found)} classes: "
                           f"{', '.join(sorted(classes_found))}")
                st.session_state["tr_data_dir"]        = custom_dir
                st.session_state["tr_use_esc50_folds"] = False
        elif custom_dir:
            st.error("Folder not found.")

    st.markdown("---")

    # ── Step 2: Training mode ──────────────────────────────────────────────────
    st.markdown("#### Step 2 — Training mode")
    train_mode = st.radio(
        "Mode", ["Feature extraction", "Fine-tune YAMNet"],
        horizontal=True, key="tr_mode",
        help="Feature extraction: frozen YAMNet + dense head. "
             "Fine-tune: unfreeze last N YAMNet layers.")

    c1, c2 = st.columns(2)
    with c1:
        epochs     = st.slider("Max epochs", 5, 100, 20, key="tr_epochs")
        use_frames = st.toggle(
            "Use all frames per clip (Google tutorial style)",
            value=True, key="tr_frames")
        out_model  = st.text_input(
            "Save model to",
            "./models/classifier" if train_mode == "Feature extraction"
            else "./models/finetune",
            key="tr_model_out")
    with c2:
        if train_mode == "Fine-tune YAMNet":
            n_unfreeze = st.slider("Layers to unfreeze", 0, 14, 3,
                                    key="tr_unfreeze")
            lr_map     = {"1e-5": 1e-5, "1e-4": 1e-4, "1e-3": 1e-3}
            lr_label   = st.selectbox("Learning rate", list(lr_map.keys()),
                                       index=1, key="tr_lr")
            st.markdown("**Layer map** (purple = training)")
            lcols = st.columns(14)
            for i, lc in enumerate(lcols):
                active = i >= (14 - n_unfreeze)
                lc.markdown(
                    f"<div style='text-align:center;padding:4px 1px;"
                    f"border-radius:4px;"
                    f"background:{'#534AB7' if active else '#eee'};"
                    f"color:{'white' if active else '#aaa'};"
                    f"font-size:10px;'>L{i+1}</div>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    st.markdown("#### Step 3 — Train")

    ready = st.session_state.get("tr_use_esc50_folds") is not None
    if not ready:
        st.warning("Complete Step 1 first.")
    else:
        use_esc50 = st.session_state.get("tr_use_esc50_folds", False)

        if st.button("🚀 Start training", type="primary", key="tr_run"):
            from core.classifier import (train, get_report, save_model,
                                          load_dataset_from_folder)

            if use_esc50:
                from core.esc50 import load_dataset_with_folds
                selected = st.session_state["tr_selected_classes"]
                st.info(f"Loading ESC-50: {selected}")
                prog = st.progress(0, "Loading audio…")
                waveforms, labels, folds = load_dataset_with_folds(
                    selected,
                    progress_callback=lambda i,t:
                        prog.progress(i/t, f"Loading {i}/{t}…"))
                prog.empty()
            else:
                data_dir  = st.session_state["tr_data_dir"]
                waveforms, labels = load_dataset_from_folder(data_dir)
                folds = None

            if not waveforms:
                st.error("No audio files found."); st.stop()

            classes = sorted(set(labels))
            st.success(f"Loaded {len(waveforms)} clips · "
                       f"{len(classes)} classes: {classes}")

            # ── Live training dashboard ────────────────────────────────────────
            st.markdown("##### Training progress")
            dashboard_container = st.container()
            dashboard = StreamlitTrainingDashboard(
                total_epochs=epochs,
                container=dashboard_container,
            )

            metrics_list = []

            if train_mode == "Feature extraction":
                prog2 = st.progress(0, "Extracting embeddings…")
                model, le, history, X_tr, y_tr, report = train(
                    waveforms, labels,
                    folds=folds,
                    epochs=epochs,
                    use_frame_embeddings=use_frames,
                    metrics_list=metrics_list,
                    progress_callback=lambda i,t:
                        prog2.progress(i/t, f"Embedding {i}/{t}…"),
                )
                prog2.empty()
                save_model(model, le, X_tr, y_tr, out_model)
                st.session_state["res_metrics_clf"] = metrics_list

            else:
                from core.finetuning import train_finetune
                prog2 = st.progress(0, "Extracting embeddings…")
                model, le, history, report = train_finetune(
                    waveforms, labels,
                    folds=folds,
                    n_unfreeze=n_unfreeze,
                    epochs=epochs,
                    learning_rate=lr_map[lr_label],
                    metrics_list=metrics_list,
                    progress_callback=lambda i,t:
                        prog2.progress(i/t, f"Embedding {i}/{t}…"),
                )
                prog2.empty()
                os.makedirs(out_model, exist_ok=True)
                model.save(os.path.join(out_model, "finetune_model.keras"))
                st.session_state["res_metrics_ft"] = metrics_list

            st.session_state.update({
                "res_metrics":  metrics_list,
                "res_report":   report,
                "res_classes":  report["classes"],
                "res_cm":       report["confusion_matrix"],
                "res_mode":     train_mode,
                "res_model":    model,
                "res_le":       le,
                "res_trained":  True,
            })

            test_acc  = report.get("test_accuracy", "—")
            test_loss = report.get("test_loss", "—")
            st.success(
                f"✅ Done! Test accuracy: **{test_acc}%** · "
                f"Test loss: **{test_loss}** → go to **📈 Results**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:

    if not st.session_state.get("res_trained"):
        st.info("Train a model in the **🏋️ Training** tab first.")
    else:
        r       = st.session_state["res_report"]
        metrics = st.session_state["res_metrics"]
        classes = st.session_state["res_classes"]
        cm      = st.session_state["res_cm"]
        report  = r.get("report", {})
        mode    = st.session_state["res_mode"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{r.get('test_accuracy','—')}%")
        c2.metric("Test loss",     r.get("test_loss", "—"))
        c3.metric("Mode",          mode)
        st.markdown("---")

        r1, r2, r3, r4 = st.tabs([
            "📉 Training curves",
            "🔲 Confusion matrix",
            "🎵 Predict new audio",
            "⚖️ Compare strategies",
        ])

        # ── Training curves ─────────────────────────────────────────────────
        with r1:
          if metrics:
              df   = pd.DataFrame(metrics)
              best = df.loc[df["val_acc"].idxmax()]
              c1, c2, c3 = st.columns(3)
              c1.metric("Best val accuracy", f"{best['val_acc']:.1f}%")
              c2.metric("At epoch",          int(best["epoch"]))
              c3.metric("Val loss",          f"{best['val_loss']:.4f}")

              col_acc, col_loss = st.columns(2)

              with col_acc:
                  fig_acc = go.Figure()
                  fig_acc.add_trace(go.Scatter(
                      x=df["epoch"], y=df["acc"],
                      name="Train", mode="lines+markers",
                      line=dict(color="#1f77b4", width=2),
                      marker=dict(size=5),
                  ))
                  fig_acc.add_trace(go.Scatter(
                      x=df["epoch"], y=df["val_acc"],
                      name="Val", mode="lines+markers",
                      line=dict(color="#ff7f0e", width=2),
                      marker=dict(size=5),
                  ))
                  fig_acc.update_layout(
                      title="Accuracy (%)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=300,
                      margin=dict(t=40, b=40, l=50, r=20),
                      xaxis=dict(title="Epoch", showgrid=True,
                                gridcolor="rgba(128,128,128,0.12)"),
                      yaxis=dict(title="Accuracy (%)", showgrid=True,
                                gridcolor="rgba(128,128,128,0.12)"),
                      legend=dict(orientation="h", y=-0.2),
                  )
                  st.plotly_chart(fig_acc, use_container_width=True)

              with col_loss:
                  fig_loss = go.Figure()
                  fig_loss.add_trace(go.Scatter(
                      x=df["epoch"], y=df["loss"],
                      name="Train", mode="lines+markers",
                      line=dict(color="#1f77b4", width=2),
                      marker=dict(size=5),
                  ))
                  fig_loss.add_trace(go.Scatter(
                      x=df["epoch"], y=df["val_loss"],
                      name="Val", mode="lines+markers",
                      line=dict(color="#ff7f0e", width=2),
                      marker=dict(size=5),
                  ))
                  fig_loss.update_layout(
                      title="Loss",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=300,
                      margin=dict(t=40, b=40, l=50, r=20),
                      xaxis=dict(title="Epoch", showgrid=True,
                                gridcolor="rgba(128,128,128,0.12)"),
                      yaxis=dict(title="Loss", showgrid=True,
                                gridcolor="rgba(128,128,128,0.12)"),
                      legend=dict(orientation="h", y=-0.2),
                  )
                  st.plotly_chart(fig_loss, use_container_width=True)

              st.dataframe(
                  df.rename(columns={
                      "epoch":"Epoch","acc":"Train acc %",
                      "val_acc":"Val acc %","loss":"Train loss",
                      "val_loss":"Val loss"})
                  .style.highlight_max(subset=["Val acc %"], color="#d4edda")
                        .highlight_min(subset=["Val loss"],  color="#d4edda"),
                  use_container_width=True, hide_index=True)

        # ── Confusion matrix ─────────────────────────────────────────────────
        with r2:
            if cm:
                st.plotly_chart(
                    confusion_matrix_fig(cm, classes),
                    use_container_width=True)

                rows = [{"Class": c,
                         "Precision": f"{report[c]['precision']:.2f}",
                         "Recall":    f"{report[c]['recall']:.2f}",
                         "F1":        f"{report[c]['f1-score']:.2f}",
                         "Support":   int(report[c]["support"])}
                        for c in classes if c in report]
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)

        # ── Predict new audio ─────────────────────────────────────────────────
        with r3:
            st.markdown("Upload or record a clip to classify it.")
            pred_bytes = None
            p1, p2 = st.columns(2)
            with p1:
                pf = st.file_uploader("Upload WAV", type=["wav"],
                                       key="res_upload")
                if pf: pred_bytes = pf.read()
            with p2:
                pm = st.audio_input("Record", key="res_mic",
                                     label_visibility="collapsed")
                if pm: pred_bytes = pm.read()

            if pred_bytes:
                from core.classifier import predict_file
                wf_y, _, _, _ = load_audio(pred_bytes)
                pred = predict_file(st.session_state["res_model"],
                                    st.session_state["res_le"], wf_y)
                st.audio(pred_bytes)
                st.metric("Predicted class", pred["label"],
                          f"confidence {pred['confidence']*100:.1f}%")
                for cls, prob in sorted(pred["all"].items(),
                                         key=lambda x: -x[1]):
                    st.progress(prob,
                                text=f"**{cls}** — {prob*100:.1f}%")

        # ── Compare strategies ────────────────────────────────────────────────
        with r4:
            st.markdown("Run both strategies in Training to compare here.")
            rows = []
            clf_m = st.session_state.get("res_metrics_clf", [])
            ft_m  = st.session_state.get("res_metrics_ft",  [])
            if clf_m:
                best = max(clf_m, key=lambda x: x["val_acc"])
                rows.append({"Strategy": "Feature extraction",
                             "Best val acc": f"{best['val_acc']:.1f}%",
                             "Epoch": int(best["epoch"]),
                             "Val loss": f"{best['val_loss']:.4f}"})
            if ft_m:
                best = max(ft_m, key=lambda x: x["val_acc"])
                rows.append({"Strategy": "Fine-tune",
                             "Best val acc": f"{best['val_acc']:.1f}%",
                             "Epoch": int(best["epoch"]),
                             "Val loss": f"{best['val_loss']:.4f}"})
            if rows:
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)
                fig = go.Figure(go.Bar(
                    x=[r["Strategy"] for r in rows],
                    y=[float(r["Best val acc"].replace("%",""))
                       for r in rows],
                    marker=dict(color=[config.CHART_COLORS[0],
                                       config.CHART_COLORS[1]]),
                    text=[r["Best val acc"] for r in rows],
                    textposition="outside",
                ))
                fig.update_yaxes(range=[0, 105],
                                  title="Val accuracy (%)")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=40, l=50, r=20),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train at least one model first.")