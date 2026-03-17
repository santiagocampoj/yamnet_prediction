"""
ui/upload.py
------------
Audio input component: drag & drop + microphone.
"""

import streamlit as st


def audio_input() -> bytes | None:
    """
    Render the audio input section (upload + mic).
    Returns raw audio bytes or None if no input yet.
    File upload takes priority over mic recording.
    """
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
        recorded = st.audio_input(
            "Click to record",
            label_visibility="collapsed",
        )

    source = uploaded or recorded
    return source.read() if source else None
