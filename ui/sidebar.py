"""
ui/sidebar.py
-------------
Sidebar controls.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def render_sidebar() -> int:
    """
    Render the sidebar and return the selected top_n value.
    """
    with st.sidebar:
        st.markdown(f"## {config.PAGE_ICON} {config.PAGE_TITLE}")
        st.markdown("---")
        st.markdown("### Settings")
        top_n = st.slider(
            "Top-N classes (default)",
            min_value=3,
            max_value=config.TOP_N_MAX,
            value=config.TOP_N_DEFAULT,
            help="Default value — you can also change it inline above the charts.",
        )
        st.markdown("---")
        st.caption("Powered by Google YAMNet · AudioSet")
    return top_n
