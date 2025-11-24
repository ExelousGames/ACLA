"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Ensure app module is on path
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

from app.services.zarr_telemetry_store import get_shared_zarr_store

# Constants
ANNOTATION_CACHE_KEY = "manual_segment_annotations"
DEFAULT_LABELS = ["Overtaking", "Tire Strategy", "Expert Line Adherence", "Mistake", "Recovery"]

def _run_async(func, *args, **kwargs):
    """Execute an async function from a synchronous context."""
    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

@st.cache_resource
def get_store():
    return get_shared_zarr_store()

def load_session_keys(store) -> List[str]:
    """Load available session cache keys."""
    keys = store.list_cache_keys()
    # Filter for session keys if possible, or just return all
    return [k for k in keys if "racing_sessions" in k and "processed" not in k]

@st.cache_data(max_entries=1)
def load_session_data(cache_key: str) -> pd.DataFrame:
    """Load session data from Zarr."""
    store = get_store()
    chunks = store.get_cached_data_chunks(cache_key)
    all_records = []
    for chunk in chunks:
        if isinstance(chunk, dict) and "data" in chunk:
             all_records.extend(chunk["data"])
        elif isinstance(chunk, list):
            all_records.extend(chunk)
    
    if not all_records:
        return pd.DataFrame()
    
    return pd.DataFrame(all_records)

def save_annotations(annotations: List[Dict[str, Any]]):
    """Save annotations to Zarr store."""
    store = get_store()
    
    # We append new annotations as a new chunk
    if annotations:
        _run_async(store.cache_chunks_streaming, ANNOTATION_CACHE_KEY, [annotations])
        st.success(f"Saved {len(annotations)} annotations to {ANNOTATION_CACHE_KEY}")

def main():
    st.set_page_config(page_title="Segment Annotation App", layout="wide")
    st.title("Telemetry Segment Annotation")

    store = get_store()
    session_keys = load_session_keys(store)

    if not session_keys:
        st.warning("No racing sessions found in Zarr store.")
        return

    selected_session_key = st.selectbox("Select Session", session_keys)
    
    if selected_session_key:
        with st.spinner("Loading session data..."):
            df = load_session_data(selected_session_key)
        
        if df.empty:
            st.warning("Selected session has no data.")
            return

        st.write(f"Loaded {len(df)} records.")
        
        # Feature selection for visualization
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]
        selected_cols = [c for c in default_cols if c in numeric_cols]
        if not selected_cols:
            selected_cols = numeric_cols[:3]
            
        viz_cols = st.multiselect("Features to Visualize", numeric_cols, default=selected_cols)
        
        if viz_cols:
            # Downsample for plotting if too large
            plot_df = df
            if len(df) > 10000:
                plot_df = df.iloc[::len(df)//5000] # Approx 5000 points
            
            fig = px.line(plot_df, x=plot_df.index, y=viz_cols, title="Telemetry Data")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Add Annotation")
        
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=0)
        with col2:
            end_idx = st.number_input("End Index", min_value=0, max_value=len(df)-1, value=min(100, len(df)-1))
            
        selected_labels = st.multiselect("Labels", DEFAULT_LABELS)
        
        if st.button("Add Annotation"):
            if start_idx >= end_idx:
                st.error("Start index must be less than end index.")
            elif not selected_labels:
                st.error("Please select at least one label.")
            else:
                annotation = {
                    "session_key": selected_session_key,
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "labels": selected_labels,
                    "timestamp": datetime.now().isoformat(),
                    "segment_length": int(end_idx - start_idx)
                }
                
                if "current_annotations" not in st.session_state:
                    st.session_state.current_annotations = []
                st.session_state.current_annotations.append(annotation)
                st.success("Annotation added to list.")

        st.subheader("Current Session Annotations")
        if "current_annotations" in st.session_state and st.session_state.current_annotations:
            st.dataframe(pd.DataFrame(st.session_state.current_annotations))
            
            if st.button("Save All Annotations to Zarr"):
                save_annotations(st.session_state.current_annotations)
                st.session_state.current_annotations = [] # Clear after save
        else:
            st.info("No annotations added yet.")

if __name__ == "__main__":
    main()
