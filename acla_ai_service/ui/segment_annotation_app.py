"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import json
import zarr
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
from app.services.full_dataset_ml_service import PipelineConfig

# Constants
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

@st.cache_data(max_entries=1, show_spinner=False)
def load_chunk_data(cache_key: str, chunk_index: int) -> pd.DataFrame:
    """Load a specific chunk of session data from Zarr."""
    store = get_store()
    # Accessing internal method _group_path to avoid iterating all chunks
    group_path = store._group_path(cache_key)
    
    if not group_path.exists():
        return pd.DataFrame()
        
    try:
        # Open in read-only mode
        group = zarr.open_group(str(group_path), mode="r")
        chunk_name = f"chunk_{chunk_index:06d}"
        
        if chunk_name not in group:
            return pd.DataFrame()
            
        raw_bytes = bytes(group[chunk_name][:])
        chunk = json.loads(raw_bytes.decode("utf-8"))
        
        # Robust DataFrame creation
        if isinstance(chunk, list):
            df = pd.DataFrame(chunk)
        elif isinstance(chunk, dict):
            if "data" in chunk and isinstance(chunk["data"], list):
                df = pd.DataFrame(chunk["data"])
            else:
                try:
                    df = pd.DataFrame(chunk)
                except ValueError:
                    df = pd.DataFrame([chunk])
        else:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        print(f"Error loading chunk {chunk_index}: {e}")
        return pd.DataFrame()

def load_existing_annotations(session_key: str, chunk_index: int) -> List[Dict[str, Any]]:
    """Load existing annotations for the specific session and chunk."""
    store = get_store()
    pipeline_config = PipelineConfig()
    
    if not store.has_cached_data(pipeline_config.annotation_cache_key):
        return []
        
    annotations = []
    try:
        chunks_iterator = store.get_cached_data_chunks(pipeline_config.annotation_cache_key)
        for chunk in chunks_iterator:
            if isinstance(chunk, list):
                for ann in chunk:
                    if isinstance(ann, dict):
                        if ann.get("session_key") == session_key and ann.get("chunk_index") == chunk_index:
                            annotations.append(ann)
    except Exception as e:
        print(f"Error loading annotations: {e}")
        
    return annotations

def save_annotations(annotations: List[Dict[str, Any]]):
    """Save annotations to Zarr store."""
    store = get_store()
    pipeline_config = PipelineConfig()
    
    # We append new annotations as a new chunk
    if annotations:
        _run_async(store.cache_chunks_streaming, pipeline_config.annotation_cache_key, [annotations])
        st.success(f"Saved {len(annotations)} annotations to {pipeline_config.annotation_cache_key}")

def main():
    st.set_page_config(page_title="Segment Annotation App", layout="wide")
    st.title("Telemetry Segment Annotation")

    store = get_store()
    
    pipeline_config = PipelineConfig()
    selected_session_key = pipeline_config.enriched_sessions_cache_key

    if selected_session_key not in store.list_cache_keys():
        st.error(f"Data key '{selected_session_key}' not found. Please run the data preparation pipeline first.")
        return
    
    st.info(f"Annotating data from: {selected_session_key}")
    
    if selected_session_key:
        # Get metadata to determine available chunks
        metadata = store.get_cache_metadata(selected_session_key)
        if not metadata or metadata.chunk_count == 0:
            st.warning("Selected session has no data chunks.")
            return

        # Chunk selection
        col_sel1, col_sel2 = st.columns([1, 3])
        with col_sel1:
            chunk_index = st.number_input(
                "Select Data Chunk", 
                min_value=1, 
                max_value=metadata.chunk_count, 
                value=1,
                help=f"Total chunks: {metadata.chunk_count}"
            )
        
        with st.spinner(f"Loading chunk {chunk_index}..."):
            df = load_chunk_data(selected_session_key, chunk_index)
        
        if df.empty:
            st.warning("Selected chunk has no data.")
            return

        st.write(f"Loaded {len(df)} records from chunk {chunk_index} (Total chunks: {metadata.chunk_count}).")
        
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
            
            # Visualize existing annotations
            existing_anns = load_existing_annotations(selected_session_key, chunk_index)
            
            # Also visualize currently added (unsaved) annotations for this chunk
            current_anns = []
            if "current_annotations" in st.session_state:
                current_anns = [
                    a for a in st.session_state.current_annotations 
                    if a.get("session_key") == selected_session_key and a.get("chunk_index") == chunk_index
                ]
            
            # Combine and plot
            for ann in existing_anns + current_anns:
                start = ann.get("start_index", 0)
                end = ann.get("end_index", 0)
                labels = ann.get("labels", [])
                label_text = ", ".join(labels) if isinstance(labels, list) else str(labels)
                
                # Use a different color for unsaved vs saved
                is_unsaved = ann in current_anns
                fill_color = "rgba(255, 165, 0, 0.2)" if is_unsaved else "rgba(0, 255, 0, 0.2)"
                
                fig.add_vrect(
                    x0=start, 
                    x1=end,
                    fillcolor=fill_color,
                    layer="below",
                    line_width=0,
                    annotation_text=label_text,
                    annotation_position="top left"
                )

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
                    "chunk_index": int(chunk_index),
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
