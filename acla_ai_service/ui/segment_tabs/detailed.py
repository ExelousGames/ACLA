import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import queue
import threading
import copy
import json
import traceback
import os
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_display_labels, get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES
)

try:
    from ..gemini_analyzer import GeminiAnalyzer, GRAPH_DEFINITIONS
except ImportError:
    # Fallback if relative import fails structure
    try:
        from ui.gemini_analyzer import GeminiAnalyzer, GRAPH_DEFINITIONS
    except ImportError:
        GeminiAnalyzer = None
        GRAPH_DEFINITIONS = []



try:
    from ..services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver
except ImportError:
    try:
        from ui.services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver
    except ImportError:
        BatchAnnotationService = None
        StreamlitBatchObserver = None


def render_detailed_labeling(selected_annotation_key, selected_session_key, available_sessions):
    """
    Renders the Telemetry Segment Annotation tab.
    """
    
    # Shared helper for formatting session options
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    # Calculate index to maintain selection across reruns
    index = 0
    if "detailed_session_selector" in st.session_state:
        try:
            if st.session_state.detailed_session_selector in available_sessions:
                index = available_sessions.index(st.session_state.detailed_session_selector)
        except ValueError:
            pass

    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Session", 
            options=available_sessions,
            format_func=format_session_option,
            index=index,
            key="detailed_session_selector"
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

    from .shared import get_store
    store = get_store()
    metadata = store.get_cache_metadata(selected_session_key)
    chunk_count = metadata.chunk_count if metadata else len(available_sessions)
    st.write(f"Loaded {len(df)} records from session {session_id} (Total sessions: {chunk_count}).")

    # Display Track Name if available
    if "Static_track" in df.columns:
         track_name = df["Static_track"].iloc[0]
         st.markdown(f"**Track:** {track_name}")
    
    # --- Common Definitions ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]

    from .components.detailed_annotation_manager import render_annotation_manager
    render_annotation_manager(df, session_id, selected_annotation_key, numeric_cols)

    # --- Visualization Range & Graphs (MOVED DOWN) ---
    st.markdown("---")
    st.caption("Visualization Range (Graphs & Track Map)")
    
    # Initialize visualization range state if not set
    if "detailed_global_viz_range" not in st.session_state:
        default_viz_start = 0
        default_viz_end = min(100, len(df)-1)
        st.session_state.detailed_global_viz_range = (default_viz_start, default_viz_end)
        st.session_state.detailed_global_viz_start_input = default_viz_start
        st.session_state.detailed_global_viz_end_input = default_viz_end
    
    # Callback to sync inputs with slider
    def update_global_slider_range():
        s = st.session_state.get("detailed_global_viz_start_input", 0)
        e = st.session_state.get("detailed_global_viz_end_input", 0)
        if s <= e:
            st.session_state.detailed_global_viz_range = (s, e)

    col_global_slider, col_global_inputs = st.columns([3, 1])
    with col_global_slider:
        viz_start_idx, viz_end_idx = st.slider(
            "Select Range",
            min_value=0,
            max_value=len(df),
            key="detailed_global_viz_range",
            label_visibility="collapsed"
        )
    
    with col_global_inputs:
         c_input1, c_input2 = st.columns(2)
         with c_input1:
             st.number_input("Start", min_value=0, max_value=len(df), value=viz_start_idx, key="detailed_global_viz_start_input", on_change=update_global_slider_range)
         with c_input2:
             st.number_input("End", min_value=0, max_value=len(df), value=viz_end_idx, key="detailed_global_viz_end_input", on_change=update_global_slider_range)

    from .components.detailed_feature_visualization import render_feature_visualization
    render_feature_visualization(df, viz_start_idx, viz_end_idx, session_id, numeric_cols, default_cols)

    # --- Track Map Visualization ---
    from .components.detailed_track_map import render_track_map
    render_track_map(df, viz_start_idx, viz_end_idx, session_id)

    # --- List View ---
    from .components.detailed_list_view import render_list_view
    render_list_view(session_id, selected_annotation_key)
