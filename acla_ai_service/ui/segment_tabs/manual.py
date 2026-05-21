import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_display_labels, get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    GRAPH_CONFIGS, LABEL_CATEGORIES
)
from .components.manual_feature_visualization import render_feature_visualization
from .components.manual_track_map import render_manual_track_map
from .components.manual_annotation_manager import render_manual_annotation_manager
from .components.manual_lap_agent import render_manual_lap_agent

def render_manual_annotation(selected_annotation_key, selected_session_key, available_sessions):
    """
    Renders the Telemetry Segment Annotation tab.
    """
    
    # Shared helper for formatting session options - local to this view or passed in? 
    # We can re-fetch this here to ensure it's up to date
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    # Calculate index to maintain selection across reruns
    index = 0
    if "manual_session_selector" in st.session_state:
        try:
            if st.session_state.manual_session_selector in available_sessions:
                index = available_sessions.index(st.session_state.manual_session_selector)
        except ValueError:
            pass

    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Session", 
            options=available_sessions,
            format_func=format_session_option,
            index=index,
            key="manual_session_selector"
        )
    
    with st.spinner(f"Loading session {session_id}..."):
        df = load_session_data(selected_session_key, session_id)
        
        # Load existing annotations for this session if we switched sessions
        if ("last_session_id" not in st.session_state or 
            st.session_state.last_session_id != session_id or 
            "last_annotation_key" not in st.session_state or
            st.session_state.last_annotation_key != selected_annotation_key):
            
                st.session_state.current_annotations = load_annotations(session_id, selected_annotation_key)
                st.session_state.last_session_id = session_id
                st.session_state.last_annotation_key = selected_annotation_key
    
    if df.empty:
        st.warning("Selected session has no data.")
        st.stop()

    # Get metadata from the store for display - we don't have direct access to store object here easily unless we import get_store
    # Let's import get_store from shared if needed, or just skip chunk count display if not critical. 
    # Or import get_store.
    from .shared import get_store
    store = get_store()
    metadata = store.get_cache_metadata(selected_session_key)
    session_count = len(available_sessions)
    st.write(f"Loaded {len(df)} records from session {session_id} (Total sessions: {session_count}).")

    # Display Track Name if available
    if "Static_track" in df.columns:
         track_name = df["Static_track"].iloc[0]
         st.markdown(f"**Track:** {track_name}")
    
    # --- Common Definitions ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]
    # Global visualization range control
    st.caption("Visualization Range (Graphs & Track Map)")
    
    # Callback to sync inputs with slider
    def update_global_slider_range():
        s = st.session_state.get("manual_global_viz_start_input", 0)
        e = st.session_state.get("manual_global_viz_end_input", 0)
        if s <= e:
            st.session_state.manual_global_viz_range = (s, e)

    def update_global_inputs_from_slider():
        val = st.session_state.get("manual_global_viz_range", (0, 0))
        st.session_state.manual_global_viz_start_input = val[0]
        st.session_state.manual_global_viz_end_input = val[1]

    if "manual_global_viz_range" not in st.session_state:
        st.session_state.manual_global_viz_range = (0, min(len(df), 5000))
    if "manual_global_viz_start_input" not in st.session_state:
        st.session_state.manual_global_viz_start_input = st.session_state.manual_global_viz_range[0]
    if "manual_global_viz_end_input" not in st.session_state:
        st.session_state.manual_global_viz_end_input = st.session_state.manual_global_viz_range[1]

    col_global_slider, col_global_inputs = st.columns([3, 1])
    with col_global_slider:
        viz_start_idx, viz_end_idx = st.slider(
            "Select Range",
            min_value=0,
            max_value=len(df),
            key="manual_global_viz_range",
            label_visibility="collapsed",
            on_change=update_global_inputs_from_slider
        )

    with col_global_inputs:
         c_input1, c_input2 = st.columns(2)
         with c_input1:
             st.number_input("Start", min_value=0, max_value=len(df), key="manual_global_viz_start_input", on_change=update_global_slider_range)
         with c_input2:
             st.number_input("End", min_value=0, max_value=len(df), key="manual_global_viz_end_input", on_change=update_global_slider_range)

    # Feature selection for visualization
    render_feature_visualization(df, numeric_cols, viz_start_idx, viz_end_idx, session_id)

    render_manual_track_map(df, viz_start_idx, viz_end_idx, session_id)

    # --- Unified Annotation Management ---
    render_manual_annotation_manager(df, numeric_cols, session_id, selected_annotation_key)

    # --- Lap-to-Segment Excerpter (AI Agent) ---
    render_manual_lap_agent(df, session_id, selected_annotation_key)
