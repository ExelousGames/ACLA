"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
"""

import torch # Must import torch before streamlit to avoid "Examining the path of torch.classes" errors
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json
import zarr
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

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
from app.config.pipeline_config import PipelineConfig
import app.models.segment_models
import importlib
# Force reload to pick up model changes (e.g. new fields)
importlib.reload(app.models.segment_models)
from app.models.segment_models import AnnotatedSegment, LABEL_MAPPING, LABEL_NAME_TO_ID, SegmentFeatureCatalog
from app.services.segment_updater import SegmentUpdater
from app.services.local_vlm_service import LocalVLMService, LocalVLMConfig, VLMProcessManager

LABEL_DESCRIPTIONS = {
    "Overtaking": "The driver is actively passing or attempting to pass another car.",
    "Missing data": "The telemetry data has gaps or invalid values.",
    "Expert Adherence": "The driver is closely following the optimal racing line and speed profile as defined by the expert.",
    "Recovery & Merge": "The driver is recovering and rejoining the expert racing line.",
    "Superior Expert": "The driver is performing better than the calculated expert baseline (e.g., higher speed in corners, better braking).",
    "Unexpected driving behavior": "The driver's behavior is erratic or does not match typical racing patterns.",
    "Mistake": "The driver makes a clear error, such as braking too late, missing an apex, or going off-track."
}

@st.cache_resource
def get_vlm_service():
    """Load the VLM service for inference."""
    try:
        # Explicitly set device to cuda or cpu to avoid accelerate's "auto" which can be unstable on some setups
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use 4-bit quantization for efficiency
        config = LocalVLMConfig(
            load_in_4bit=True, 
            load_in_8bit=False,
            device_map=device_map
        )
        service = VLMProcessManager(config)
        return service
    except Exception as e:
        print(f"Failed to load VLM service: {e}")
        return None

def get_display_labels(labels):
    """Convert label IDs or strings to display strings."""
    if not isinstance(labels, list):
        labels = [labels]
    
    display_labels = []
    for l in labels:
        if isinstance(l, int):
            display_labels.append(LABEL_MAPPING.get(l, f"Unknown({l})"))
        else:
            display_labels.append(str(l))
    return display_labels

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

def get_available_sessions(cache_key: str) -> List[str]:
    """Get list of available session IDs from the store."""
    store = get_store()
    group_path = store._group_path(cache_key)
    if not group_path.exists():
        return []
    try:
        group = zarr.open_group(str(group_path), mode="r")
        # Filter out metadata chunk
        sessions = [k for k in group.array_keys() if k != "chunk_000000"]
        return sorted(sessions)
    except Exception:
        return []

@st.cache_data(max_entries=1, show_spinner=False)
def load_session_data(cache_key: str, session_id: str) -> pd.DataFrame:
    """Load a specific session of data from Zarr."""
    store = get_store()
    # Accessing internal method _group_path to avoid iterating all chunks
    group_path = store._group_path(cache_key)
    
    if not group_path.exists():
        return pd.DataFrame()
        
    try:
        # Open in read-only mode
        group = zarr.open_group(str(group_path), mode="r")
        chunk_name = session_id
        
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
        print(f"Error loading session {session_id}: {e}")
        return pd.DataFrame()

def load_annotations(session_id: str, annotation_key: str) -> List[AnnotatedSegment]:
    """Load annotations for a specific session."""
    store = get_store()
    chunk_data = store.get_chunk(annotation_key, session_id)
    
    raw_data = []
    if chunk_data:
        if isinstance(chunk_data, list):
            raw_data = chunk_data
        elif isinstance(chunk_data, dict) and "data" in chunk_data:
             raw_data = chunk_data["data"]
             
    segments = [AnnotatedSegment.from_dict(d) for d in raw_data]
    # Ensure chunk_index is set for loaded segments
    for s in segments:
        if s.chunk_index is None:
            s.chunk_index = session_id
    return segments

def save_annotations(session_id: str, annotations: List[AnnotatedSegment], annotation_key: str):
    """Save annotations to Zarr store."""
    store = get_store()
    
    # Save to specific chunk index
    data_to_save = [a.to_dict() for a in annotations]
    store.save_chunk(annotation_key, session_id, data_to_save)
    st.success(f"Saved {len(annotations)} annotations to {annotation_key} (session {session_id})")

def main():
    st.set_page_config(page_title="Segment Annotation App", layout="wide")
    
    store = get_store()
    pipeline_config = PipelineConfig()

    # Add sidebar controls
    with st.sidebar:
        st.header("App Controls")
        if st.button("Finish & Exit", type="primary", help="Close the app and return to the pipeline"):
            st.success("Exiting...")
            import time
            time.sleep(0.5)
            os._exit(0)
        
        st.markdown("---")
        st.header("Annotation Dataset")
        
        # Dataset Selection Logic
        default_ann_key = pipeline_config.annotation_cache_key
        
        # Get all keys and filter/sort
        all_keys = store.list_cache_keys()
        
        dataset_mode = st.radio("Dataset Mode", ["Select Existing", "Create New"], key="dataset_mode")
        
        selected_annotation_key = default_ann_key
        
        if dataset_mode == "Select Existing":
            # Filter keys to only those containing the annotation cache key
            options = sorted([k for k in all_keys if default_ann_key in k])
            
            # Put default at top if exists
            index = 0
            if default_ann_key in options:
                index = options.index(default_ann_key)
            elif options:
                index = 0
            
            selected_annotation_key = st.selectbox("Select Dataset", options, index=index, key="dataset_select")
        else:
            new_dataset_suffix = st.text_input("New Dataset Name Suffix", value="custom_v1")
            if new_dataset_suffix:
                # Auto-prepend the default key if not present
                if new_dataset_suffix.startswith(default_ann_key):
                    selected_annotation_key = new_dataset_suffix
                else:
                    # Add underscore if needed
                    sep = "_" if not default_ann_key.endswith("_") and not new_dataset_suffix.startswith("_") else ""
                    selected_annotation_key = f"{default_ann_key}{sep}{new_dataset_suffix}"
                
                st.info(f"Will create/use dataset: **{selected_annotation_key}**")
            else:
                st.warning("Please enter a dataset name suffix.")
                selected_annotation_key = None

        # --- Maintenance Section ---
        if selected_annotation_key:
            st.markdown("---")
            st.header("Maintenance")
            if st.button("Update Features (All Sessions)", help="Re-extract telemetry data for all annotations in this dataset to include new features from the source data."):
                updater = SegmentUpdater()
                source_key = pipeline_config.enriched_sessions_cache_key
                
                # Get all sessions that have annotations
                annotated_sessions = get_available_sessions(selected_annotation_key)
                
                if not annotated_sessions:
                    st.warning("No annotated sessions found to update.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, sess_id in enumerate(annotated_sessions):
                        status_text.text(f"Updating session {sess_id} ({i+1}/{len(annotated_sessions)})...")
                        
                        # Load
                        anns = load_annotations(sess_id, selected_annotation_key)
                        if anns:
                            # Update
                            updated_anns = updater.update_segments(source_key, anns)
                            # Save directly to avoid UI spam
                            data_to_save = [a.to_dict() for a in updated_anns]
                            store.save_chunk(selected_annotation_key, sess_id, data_to_save)
                        
                        progress_bar.progress((i + 1) / len(annotated_sessions))
                    
                    status_text.text("Update complete!")
                    st.success(f"Updated features for {len(annotated_sessions)} sessions.")
                    
                    # Force reload of current session if it was updated
                    if "last_session_id" in st.session_state:
                        del st.session_state.last_session_id 
                        st.rerun()

    st.title("Telemetry Segment Annotation")

    if not selected_annotation_key:
        st.warning("Please select or create an annotation dataset in the sidebar.")
        return

    st.info(f"Using Annotation Dataset: **{selected_annotation_key}**")
    
    selected_session_key = pipeline_config.enriched_sessions_cache_key

    if selected_session_key not in store.list_cache_keys():
        st.error(f"Data key '{selected_session_key}' not found. Please run the data preparation pipeline first.")
        return
    
    st.info(f"Annotating data from: {selected_session_key}")
    
    if selected_session_key:
        # Get metadata to determine available sessions
        metadata = store.get_cache_metadata(selected_session_key)
        if not metadata or metadata.chunk_count == 0:
            st.warning("Selected session key has no data.")
            return

        # Session selection
        available_sessions = get_available_sessions(selected_session_key)
        
        if not available_sessions:
             st.warning("Selected session key has no data.")
             return

        # Get annotated sessions for the selected dataset
        annotated_sessions = set(get_available_sessions(selected_annotation_key))

        def format_session_option(s):
            status = "✅" if s in annotated_sessions else "⭕"
            return f"{status} {s}"

        # Calculate index to maintain selection across reruns
        index = 0
        if "session_selector" in st.session_state:
            try:
                if st.session_state.session_selector in available_sessions:
                    index = available_sessions.index(st.session_state.session_selector)
            except ValueError:
                pass

        col_sel1, col_sel2 = st.columns([1, 3])
        with col_sel1:
            session_id = st.selectbox(
                "Select Session", 
                options=available_sessions,
                format_func=format_session_option,
                index=index,
                key="session_selector"
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
            return

        st.write(f"Loaded {len(df)} records from session {session_id} (Total sessions: {metadata.chunk_count}).")
        
        # Global visualization range control
        viz_start_idx, viz_end_idx = st.slider(
            "Global Visualization Range",
            min_value=0,
            max_value=len(df),
            value=(0, min(len(df), 5000)),
            key="global_viz_range"
        )

        # Feature selection for visualization
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]
        
        if "graph_ids" not in st.session_state:
            st.session_state.graph_ids = [0]
            st.session_state.next_graph_id = 1

        if st.button("Add Graph"):
            st.session_state.graph_ids.append(st.session_state.next_graph_id)
            st.session_state.next_graph_id += 1

        graphs_to_remove = []

        for graph_id in st.session_state.graph_ids:
            col_viz, col_btn = st.columns([6, 1])
            
            with col_viz:
                # Default selection logic for the first graph (id 0)
                current_default = []
                if graph_id == 0:
                    current_default = [c for c in default_cols if c in numeric_cols]
                    if not current_default:
                        current_default = numeric_cols[:3]
                
                viz_cols = st.multiselect(
                    f"Features to Visualize (Graph {graph_id})", 
                    numeric_cols, 
                    default=current_default,
                    key=f"viz_cols_{graph_id}"
                )
            
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True) # Spacing
                if st.button("Remove", key=f"remove_btn_{graph_id}"):
                    graphs_to_remove.append(graph_id)

            if viz_cols:
                # Apply global range filter
                sliced_df = df.iloc[viz_start_idx:viz_end_idx]

                # Plot without downsampling
                fig = px.line(sliced_df, x=sliced_df.index, y=viz_cols, title=f"Telemetry Data - Graph {graph_id}")
                # Visualize existing annotations
                if "current_annotations" in st.session_state and st.session_state.current_annotations:
                    for ann in st.session_state.current_annotations:
                        # Ensure we only visualize annotations for the current session
                        ann_chunk = getattr(ann, "chunk_index", None)
                        if ann_chunk is not None and ann_chunk != session_id:
                            continue

                        start = getattr(ann, "start_index", None)
                        end = getattr(ann, "end_index", None)

                        # Skip if annotation is completely outside the visualization range
                        if start is not None and end is not None:
                            if end <= viz_start_idx or start >= viz_end_idx:
                                continue

                        labels = ann.labels
                        display_labels = get_display_labels(labels)
                        label_str = ", ".join(display_labels)
                        
                        if start is not None and end is not None:
                            fig.add_vrect(
                                x0=start, 
                                x1=end, 
                                fillcolor="green", 
                                opacity=0.1, 
                                layer="below", 
                                line_width=0,
                                annotation_text=label_str,
                                annotation_position="top left"
                            )

                st.plotly_chart(fig, use_container_width=True)
        
        if graphs_to_remove:
            for gid in graphs_to_remove:
                if gid in st.session_state.graph_ids:
                    st.session_state.graph_ids.remove(gid)
            st.rerun()

        # --- Track Map Visualization ---
        st.subheader("Track Map & Positions")
        
        # Check if we have position data
        has_player_pos = "Graphics_player_pos_x" in df.columns and "Graphics_player_pos_y" in df.columns
        has_player_pos_z = "Graphics_player_pos_z" in df.columns
        
        has_opponent_pos = any(f"Opponent_{i}_pos_x" in df.columns for i in range(1, 6))
        
        has_expert_pos = "expert_optimal_player_pos_x" in df.columns and "expert_optimal_player_pos_y" in df.columns
        has_expert_pos_z = "expert_optimal_player_pos_z" in df.columns
        
        if has_player_pos or has_opponent_pos or has_expert_pos:
            # View controls
            col_ctrl1, col_ctrl2 = st.columns([3, 1])
            with col_ctrl1:
                # Callback to sync inputs with slider
                def update_slider_range():
                    s = st.session_state.get("track_map_start_input", 0)
                    e = st.session_state.get("track_map_end_input", 0)
                    if s <= e:
                        st.session_state.track_map_slider = (s, e)

                # Slider for selecting timestamp range
                start_idx, end_idx = st.slider(
                    "Select Timestamp for Position View", 
                    min_value=0, 
                    max_value=len(df)-1, 
                    value=(0, min(len(df)-1, 200)),
                    key="track_map_slider"
                )
                
                # Manual input for range
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.number_input("Start", min_value=0, max_value=len(df)-1, value=start_idx, key="track_map_start_input", on_change=update_slider_range)
                with mc2:
                    st.number_input("End", min_value=0, max_value=len(df)-1, value=end_idx, key="track_map_end_input", on_change=update_slider_range)

            with col_ctrl2:
                st.caption("Axis Settings")
                invert_x = st.checkbox("Invert X", value=False)
                invert_y = st.checkbox("Invert Y", value=False)
                invert_z = st.checkbox("Invert Z", value=False)
            
            # Create windowed dataframe for trajectory plotting
            map_plot_df = df.iloc[start_idx:end_idx+1]
            
            selected_time_idx = end_idx if end_idx < len(df) else len(df) - 1
            current_row = df.iloc[selected_time_idx]
            start_row = df.iloc[start_idx]
            map_data = []
            
            # Add Player Position
            if has_player_pos:
                # End Position
                p_data = {
                    "x": current_row["Graphics_player_pos_x"],
                    "y": current_row["Graphics_player_pos_y"],
                    "Type": "Player",
                    "ID": "Player End",
                    "Marker": "End"
                }
                if has_player_pos_z:
                    p_data["z"] = current_row["Graphics_player_pos_z"]
                map_data.append(p_data)

                # Start Position
                p_start = {
                    "x": start_row["Graphics_player_pos_x"],
                    "y": start_row["Graphics_player_pos_y"],
                    "Type": "Player",
                    "ID": "Player Start",
                    "Marker": "Start"
                }
                if has_player_pos_z:
                    p_start["z"] = start_row["Graphics_player_pos_z"]
                map_data.append(p_start)

            # Add Expert Position
            if has_expert_pos:
                # End Position
                e_data = {
                    "x": current_row["expert_optimal_player_pos_x"],
                    "y": current_row["expert_optimal_player_pos_y"],
                    "Type": "Expert",
                    "ID": "Expert End",
                    "Marker": "End"
                }
                if has_expert_pos_z:
                    e_data["z"] = current_row["expert_optimal_player_pos_z"]
                map_data.append(e_data)

                # Start Position
                e_start = {
                    "x": start_row["expert_optimal_player_pos_x"],
                    "y": start_row["expert_optimal_player_pos_y"],
                    "Type": "Expert",
                    "ID": "Expert Start",
                    "Marker": "Start"
                }
                if has_expert_pos_z:
                    e_start["z"] = start_row["expert_optimal_player_pos_z"]
                map_data.append(e_start)
            
            # Add Opponent Positions
            for i in range(1, 6):
                opp_x_col = f"Opponent_{i}_pos_x"
                opp_y_col = f"Opponent_{i}_pos_y"
                opp_z_col = f"Opponent_{i}_pos_z"
                opp_id_col = f"Opponent_{i}_car_id"
                
                if opp_x_col in df.columns and opp_y_col in df.columns:
                    # Filter out inactive opponents (usually 0,0 coordinates)
                    if current_row[opp_x_col] != 0 or current_row[opp_y_col] != 0:
                        opp_id = current_row[opp_id_col] if opp_id_col in df.columns else f"Opponent {i}"
                        o_data = {
                            "x": current_row[opp_x_col],
                            "y": current_row[opp_y_col],
                            "Type": "Opponent",
                            "ID": str(opp_id),
                            "Marker": "End"
                        }
                        if opp_z_col in df.columns:
                            o_data["z"] = current_row[opp_z_col]
                        map_data.append(o_data)
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                use_3d = "z" in map_df.columns
                
                if use_3d:
                    fig_map = px.scatter_3d(
                        map_df, 
                        x="x", 
                        y="y", 
                        z="z",
                        color="Type", 
                        symbol="Marker",
                        hover_data=["ID"],
                        title=f"Positions (Start: {start_idx}, End: {selected_time_idx}) (3D)",
                        color_discrete_map={"Player": "green", "Opponent": "red", "Expert": "blue"},
                        symbol_map={"Start": "diamond", "End": "circle"}
                    )
                    fig_map.update_traces(marker=dict(size=5))
                    
                    scene_dict = dict(aspectmode='data')
                    if invert_x: scene_dict['xaxis'] = dict(autorange="reversed")
                    if invert_y: scene_dict['yaxis'] = dict(autorange="reversed")
                    if invert_z: scene_dict['zaxis'] = dict(autorange="reversed")
                    fig_map.update_layout(scene=scene_dict)
                else:
                    fig_map = px.scatter(
                        map_df, 
                        x="x", 
                        y="y", 
                        color="Type", 
                        symbol="Marker",
                        hover_data=["ID"],
                        title=f"Positions (Start: {start_idx}, End: {selected_time_idx})",
                        color_discrete_map={"Player": "green", "Opponent": "red", "Expert": "blue"},
                        symbol_map={"Start": "x", "End": "circle"}
                    )
                    if invert_x: fig_map.update_xaxes(autorange="reversed")
                    if invert_y: fig_map.update_yaxes(autorange="reversed")

                # Add Trajectories
                # Player
                if has_player_pos:
                    if use_3d and has_player_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=map_plot_df["Graphics_player_pos_x"], 
                            y=map_plot_df["Graphics_player_pos_y"],
                            z=map_plot_df["Graphics_player_pos_z"],
                            mode="lines",
                            name="Player Trajectory",
                            line=dict(color="green", width=2),
                            opacity=0.5,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=map_plot_df["Graphics_player_pos_x"], 
                            y=map_plot_df["Graphics_player_pos_y"],
                            mode="lines",
                            name="Player Trajectory",
                            line=dict(color="green", width=1, dash="dot"),
                            opacity=0.5,
                            showlegend=True
                        ))

                # Expert
                if has_expert_pos:
                    if use_3d and has_expert_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=map_plot_df["expert_optimal_player_pos_x"], 
                            y=map_plot_df["expert_optimal_player_pos_y"],
                            z=map_plot_df["expert_optimal_player_pos_z"],
                            mode="lines",
                            name="Expert Trajectory",
                            line=dict(color="blue", width=2),
                            opacity=0.5,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=map_plot_df["expert_optimal_player_pos_x"], 
                            y=map_plot_df["expert_optimal_player_pos_y"],
                            mode="lines",
                            name="Expert Trajectory",
                            line=dict(color="blue", width=1, dash="dot"),
                            opacity=0.5,
                            showlegend=True
                        ))
                
                # Opponents
                for i in range(1, 6):
                    opp_x_col = f"Opponent_{i}_pos_x"
                    opp_y_col = f"Opponent_{i}_pos_y"
                    opp_z_col = f"Opponent_{i}_pos_z"
                    
                    if opp_x_col in df.columns and opp_y_col in df.columns:
                        # Filter out inactive (0,0) points for cleaner trajectories
                        opp_df = map_plot_df[(map_plot_df[opp_x_col] != 0) | (map_plot_df[opp_y_col] != 0)]
                        if not opp_df.empty:
                            if use_3d and opp_z_col in df.columns:
                                fig_map.add_trace(go.Scatter3d(
                                    x=opp_df[opp_x_col], 
                                    y=opp_df[opp_y_col],
                                    z=opp_df[opp_z_col],
                                    mode="lines",
                                    name=f"Opponent {i} Trajectory",
                                    line=dict(color="red", width=2),
                                    opacity=0.3,
                                    showlegend=True
                                ))
                            else:
                                fig_map.add_trace(go.Scatter(
                                    x=opp_df[opp_x_col], 
                                    y=opp_df[opp_y_col],
                                    mode="lines",
                                    name=f"Opponent {i} Trajectory",
                                    line=dict(color="red", width=1, dash="dot"),
                                    opacity=0.3,
                                    showlegend=True
                                ))

                if not use_3d:
                    fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
                
                fig_map.update_layout(uirevision=session_id)
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No active cars found at this timestamp.")
        else:
            st.info("Position data (Graphics_player_pos_x/y) not available in this dataset.")

        # --- Unified Annotation Management ---
        st.markdown("---")
        st.subheader("Manage Annotations")

        # Ensure annotations list exists
        if "current_annotations" not in st.session_state:
            st.session_state.current_annotations = []



        # 1. Select Mode/Annotation
        annotation_options = ["Create New"]
        if st.session_state.current_annotations:
             annotation_options.extend(range(len(st.session_state.current_annotations)))

        def format_func(option):
            if option == "Create New":
                return "➕ Create New Annotation"
            else:
                ann = st.session_state.current_annotations[option]
                labels = ", ".join(get_display_labels(ann.labels))
                return f"#{option}: {labels} (Start: {ann.start_index}, End: {ann.end_index})"

        selected_option = st.selectbox(
            "Select Action / Annotation",
            options=annotation_options,
            format_func=format_func,
            key="annotation_selector"
        )

        # 2. The Form
        with st.container():
            # Determine default values based on selection
            if selected_option == "Create New":
                form_title = "Add New Annotation"
                default_start = 0
                default_end = min(100, len(df)-1)
                default_labels = []
                submit_label = "Add Annotation"
                is_edit = False
            else:
                form_title = f"Edit Annotation #{selected_option}"
                ann = st.session_state.current_annotations[selected_option]
                default_start = ann.start_index
                default_end = ann.end_index
                default_labels = [l for l in get_display_labels(ann.labels) if l in LABEL_MAPPING.values()]
                submit_label = "Update Annotation"
                is_edit = True

            st.markdown(f"**{form_title}**")
            
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                form_start = st.number_input(
                    "Start Index", 
                    min_value=0, 
                    max_value=len(df)-1, 
                    value=default_start,
                    key=f"form_start_{selected_option}"
                )
            with col_form2:
                form_end = st.number_input(
                    "End Index", 
                    min_value=0, 
                    max_value=len(df)-1, 
                    value=default_end,
                    key=f"form_end_{selected_option}"
                )
            
            form_labels = st.multiselect(
                "Labels", 
                list(LABEL_MAPPING.values()), 
                default=default_labels,
                key=f"form_labels_{selected_option}"
            )

            # Feature Change Calculator
            with st.expander("Feature Change Calculator"):
                f_col1, f_col2 = st.columns([1, 2])
                with f_col1:
                    # Default to speed or gas if available
                    default_calc_idx = 0
                    if "speed_kmh" in numeric_cols:
                        default_calc_idx = numeric_cols.index("speed_kmh")
                    
                    calc_feature = st.selectbox(
                        "Select Feature", 
                        numeric_cols, 
                        index=default_calc_idx,
                        key=f"calc_feat_{selected_option}"
                    )
                
                with f_col2:
                    if calc_feature and form_start < form_end and int(form_end) < len(df):
                        # Calculate changes
                        calc_slice = df.iloc[int(form_start):int(form_end)+1][calc_feature]
                        
                        # Comprehensive Statistical Analysis
                        min_val = calc_slice.min()
                        max_val = calc_slice.max()
                        mean_val = calc_slice.mean()
                        median_val = calc_slice.median()
                        std_val = calc_slice.std()
                        var_val = calc_slice.var()
                        
                        # Derivative Stats (Rate of Change)
                        diffs = calc_slice.diff().dropna()
                        max_rate = diffs.max() if not diffs.empty else 0
                        min_rate = diffs.min() if not diffs.empty else 0
                        avg_abs_rate = diffs.abs().mean() if not diffs.empty else 0
                        
                        # Integral (Area under curve approximation)
                        area = np.trapz(calc_slice.values)
                        
                        # Total Change
                        total_change = calc_slice.iloc[-1] - calc_slice.iloc[0]

                        st.markdown("##### Statistical Analysis")
                        
                        # Row 1: Range & Central Tendency
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Minimum", f"{min_val:.2f}")
                        c2.metric("Maximum", f"{max_val:.2f}")
                        c3.metric("Mean", f"{mean_val:.2f}")
                        c4.metric("Median", f"{median_val:.2f}")

                        # Row 2: Variability & Dynamics
                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Std Dev", f"{std_val:.2f}")
                        c6.metric("Max Rate (Δ)", f"{max_rate:.2f}")
                        c7.metric("Min Rate (Δ)", f"{min_rate:.2f}")
                        c8.metric("Avg Volatility", f"{avg_abs_rate:.2f}", help="Average absolute change between consecutive points")
                        
                        # Row 3: Cumulative
                        c9, c10, c11, c12 = st.columns(4)
                        c9.metric("Integral (Area)", f"{area:.2f}", help="Area under the curve (Trapezoidal rule)")
                        c10.metric("Sum", f"{calc_slice.sum():.2f}")
                        c11.metric("Variance", f"{var_val:.2f}")
                        c12.metric("Total Change", f"{total_change:.2f}", help="Difference between end and start value")

            # AI Label Suggestion (Use Container instead of Expander to avoid nesting issues with Status)
            with st.container():
                st.markdown("---")
                st.subheader("AI Label Suggestion (VLM)")
                st.markdown("Use the local VLM to visually analyze the selected telemetry segment and suggest a label.")

                valid_default_cols = [c for c in default_cols if c in numeric_cols]
                if not valid_default_cols and numeric_cols:
                    valid_default_cols = numeric_cols[:min(3, len(numeric_cols))]

                llm_features = st.multiselect(
                    "Select Features for VLM Analysis",
                    numeric_cols,
                    default=valid_default_cols,
                    key=f"llm_features_{selected_option}"
                )

                include_traj = st.checkbox("Include 2D Trajectory in Analysis", value=True, key=f"include_traj_{selected_option}")
                
                if st.button("Analyze Segment with VLM", key=f"btn_analyze_{selected_option}"):
                    # Check range validity
                    a_start = st.session_state.get(f"form_start_{selected_option}", default_start)
                    a_end = st.session_state.get(f"form_end_{selected_option}", default_end)
                    
                    if a_start >= a_end:
                         st.error("Invalid range for analysis.")
                    elif not llm_features:
                         st.error("Please select at least one feature for analysis.")
                    else:
                        with st.status("Analyzing segment telemetry...", expanded=True) as status:
                            # Get VLM
                            status.write("Loading VLM model...")
                            vlm_service = get_vlm_service()
                            
                            if vlm_service:
                                status.write("Extracting telemetry data...")
                                seg_slice = df.iloc[int(a_start):int(a_end)]
                                
                                # Telemetry Data
                                telemetry_csv_str = seg_slice[llm_features].to_csv(index=False)
                                
                                # Trajectory Data
                                trajectory_csv_str = None
                                if include_traj:
                                    traj_cols = [
                                        'Graphics_player_pos_x', 'Graphics_player_pos_y', 'Graphics_player_pos_z',
                                        'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z'
                                    ]
                                    available_traj_cols = [c for c in traj_cols if c in seg_slice.columns]
                                    if available_traj_cols:
                                        trajectory_csv_str = seg_slice[available_traj_cols].to_csv(index=False)
                                
                                # Build detailed label descriptions
                                label_desc_list = []
                                for name in sorted(list(set(LABEL_MAPPING.values()))):
                                    desc = LABEL_DESCRIPTIONS.get(name, "No description available.")
                                    label_desc_list.append(f"- {name}: {desc}")
                                available_labels_str = "\n".join(label_desc_list)
                                
                                prompt = (
                                    f"The graphs show telemetry data from a racing car session. The graphs contain a comparison between the driver and an expert reference.\n"
                                    f"Based on the trends in the data, suggest the most appropriate behavioral label from the following list:\n{available_labels_str}\n\n"
                                    f"Respond in English. Be concise and professional. Explain your reasoning based on the visual patterns in the graphs."
                                )
                                
                                status.write("Starting VLM analysis...")
                                
                                # Use a mutable reference to hold the placeholder for streaming output
                                stream_ph_ref = [None]
                                
                                def status_callback_handler(msg):
                                    if msg.startswith("Generating:"):
                                        text_content = msg[len("Generating: "):]
                                        if stream_ph_ref[0] is None:
                                            stream_ph_ref[0] = status.empty()
                                        stream_ph_ref[0].markdown(f"**Generating:**\n\n{text_content}")
                                    else:
                                        status.write(msg)

                                try:
                                    response, img = vlm_service.analyze_data(
                                        telemetry_csv_str, 
                                        prompt, 
                                        trajectory_csv_data=trajectory_csv_str,
                                        status_callback=status_callback_handler
                                    )
                                    status.update(label="Analysis Complete", state="complete", expanded=False)
                                    st.markdown("### VLM Analysis & Suggestion")
                                    st.image(img, caption="VLM Visualization (Combined)", use_column_width=True)
                                    st.info(response)
                                except Exception as e:
                                    status.update(label="Analysis Failed", state="error")
                                    st.error(f"Error during VLM analysis: {e}")
                                    st.error(f"Inference failed: {e}")
                            else:
                                status.update(label="LLM Load Failed", state="error")
                                st.error("Could not load LLM. Check server logs or ensure model is downloaded.")

            # Auto-Segmentation (Agent Mode)
            with st.expander("Auto-Segment Range (Agent Mode)", expanded=False):
                st.markdown("Iteratively analyze the specified range to find and extract sequential segments.")
                
                as_col1, as_col2 = st.columns(2)
                with as_col1:
                    auto_start_idx = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=viz_start_idx, key="as_start")
                with as_col2:
                    auto_end_idx = st.number_input("End Index", min_value=0, max_value=len(df), value=viz_end_idx, key="as_end")
                
                # Check for cols defined in previous block, else re-derive
                as_default_cols = numeric_cols[:min(3, len(numeric_cols))]
                if "valid_default_cols" in locals():
                    as_default_cols = valid_default_cols

                as_features = st.multiselect(
                    "Features for Auto-Segmentation",
                    numeric_cols,
                    default=as_default_cols,
                    key="as_features"
                )
                
                if st.button("Start Auto-Segmentation Agent", type="primary"):
                    if auto_start_idx >= auto_end_idx:
                        st.error("Invalid range.")
                    else:
                        vlm_service = get_vlm_service()
                        if not vlm_service:
                            st.error("VLM Service not available.")
                        else:
                            current_cursor = int(auto_start_idx)
                            limit_cursor = int(auto_end_idx)
                            
                            status_container = st.status("Running Auto-Segmentation Agent...", expanded=True)
                            results_container = st.container()
                            
                            full_labels_str = ", ".join(sorted(list(set(LABEL_MAPPING.values()))))
                            
                            found_count = 0
                            
                            # Reset proposed annotations
                            st.session_state.proposed_auto_annotations = []
                            
                            import re
                            import json
                            
                            while current_cursor < limit_cursor:
                                chunk_len = limit_cursor - current_cursor
                                # Minimum chunk size to analyze
                                if chunk_len < 10: 
                                    status_container.write("Remaining chunk too small. Stopping.")
                                    break
                                
                                status_container.write(f"Analyzing window: {current_cursor} to {limit_cursor} (Len: {chunk_len})")
                                
                                # Prepare data
                                current_slice = df.iloc[current_cursor:limit_cursor]
                                slice_csv = current_slice[as_features].to_csv(index=False)
                                
                                # Construct Prompt
                                analysis_prompt = (
                                    f"Analyze the telemetry graph (left to right) representing a driving session.\n"
                                    f"Identify the **first/left-most** distinct driving behavior segment that matches one of these labels: [{full_labels_str}].\n"
                                    f"Ignore any 'Missing data' or unclear regions at the very beginning if they differ from the first clear segment.\n"
                                    f"Return the result ONLY as a JSON object with this format:\n"
                                    f'{{\n  "found": true,\n  "label": "LabelName",\n  "start_percentage": <float 0.0-1.0 representing start of segment in this window>,\n  "end_percentage": <float 0.0-1.0 representing end of segment in this window>,\n  "confidence": <float 0.0-1.0>\n}}\n'
                                    f"If no distinct segment is found or the data is just noise/empty, return {{ \"found\": false }}.\n"
                                    f"Focus on finding the START and END of the FIRST valid segment."
                                )
                                
                                try:
                                    # Call VLM - we can reuse the status callback for partial updates if we want, 
                                    # but inside a loop it might be noisy. We'll skip detailed streaming for the loop.
                                    vlm_resp, vlm_img = vlm_service.analyze_data(slice_csv, analysis_prompt)
                                    
                                    # Parse JSON
                                    json_match = re.search(r'\{.*\}', vlm_resp, re.DOTALL)
                                    segment_info = None
                                    if json_match:
                                        try:
                                            segment_info = json.loads(json_match.group(0))
                                        except:
                                            pass
                                    
                                    if segment_info and segment_info.get("found"):
                                        lbl = segment_info.get("label")
                                        start_pct = segment_info.get("start_percentage", 0.0)
                                        end_pct = segment_info.get("end_percentage", 0.1)
                                        
                                        # Clamp
                                        start_pct = max(0.0, min(1.0, start_pct))
                                        end_pct = max(0.0, min(1.0, end_pct))
                                        
                                        if end_pct <= start_pct:
                                             # Fallback if VLM gives bad range, advance by small amount
                                             end_pct = start_pct + 0.1
                                        
                                        # Convert to absolute indices
                                        seg_start_abs = int(current_cursor + (start_pct * chunk_len))
                                        seg_end_abs = int(current_cursor + (end_pct * chunk_len))
                                        
                                        # Ensure progress: If VLM says start is 0 and end is very small, or start > end
                                        # We force a move to avoid infinite loop
                                        if seg_end_abs <= current_cursor + 5:
                                            seg_end_abs = current_cursor + max(20, int(chunk_len * 0.1))
                                            
                                        # Map Label
                                        label_id = -1
                                        best_match_name = "Unknown"
                                        for lid, name in LABEL_MAPPING.items():
                                            if name.lower() == lbl.lower():
                                                label_id = lid
                                                best_match_name = name
                                                break
                                        
                                        if label_id == -1:
                                            # Try finding closest match
                                            for lid, name in LABEL_MAPPING.items():
                                                if name.lower() in lbl.lower() or lbl.lower() in name.lower():
                                                    label_id = lid
                                                    best_match_name = name
                                                    break
                                            
                                            if label_id == -1:
                                                status_container.write(f"Warning: Unknown label '{lbl}', defaulting to 'Unexpected driving behavior'.")
                                                best_match_name = "Unexpected driving behavior"
                                                label_id = LABEL_NAME_TO_ID.get("Unexpected driving behavior", 0)

                                        # Create Annotation
                                        seg_length = seg_end_abs - seg_start_abs
                                        
                                        # Extract segment specific data for storage
                                        seg_data_slice = df.iloc[seg_start_abs:seg_end_abs]
                                        telemetry_data_dict = seg_data_slice.to_dict(orient="records")
                                        
                                        new_ann = AnnotatedSegment(
                                            labels=[label_id],
                                            segment_length=seg_length,
                                            start_index=seg_start_abs,
                                            end_index=seg_end_abs,
                                            chunk_index=session_id,
                                            telemetry_data=telemetry_data_dict,
                                            notes=f"Auto-generated (Conf: {segment_info.get('confidence', 'NA')})"
                                        )
                                        
                                        if "proposed_auto_annotations" not in st.session_state:
                                            st.session_state.proposed_auto_annotations = []

                                        st.session_state.proposed_auto_annotations.append(new_ann)
                                        found_count += 1
                                        
                                        results_container.success(f"Proposed: **{best_match_name}** ({seg_start_abs} - {seg_end_abs})")
                                        
                                        # Advance cursor to the END of the found segment to continue searching after it
                                        current_cursor = seg_end_abs
                                        
                                    else:
                                        status_container.warning("No segment returned by VLM or JSON parse failed. Stopping.")
                                        if not segment_info:
                                            status_container.text(f"Raw VLM Response: {vlm_resp}")
                                        break
                                        
                                except Exception as e:
                                    status_container.error(f"Error in loop: {e}")
                                    break
                            
                            status_container.update(label=f"Auto-Segmentation Complete. Found {found_count} segments.", state="complete")
                            if found_count > 0:
                                st.session_state.review_auto_segments = True
                                st.rerun()

            # Actions
            # --- Auto-Segmentation Review Block ---
            if st.session_state.get("review_auto_segments", False) and st.session_state.get("proposed_auto_annotations"):
                st.divider()
                st.subheader("Creation Review")
                st.info(f"The Auto-Segmentation Agent proposed {len(st.session_state.proposed_auto_annotations)} segments. Please review them below.")
                
                # Convert to dataframe for display
                prop_data = []
                for i, ann in enumerate(st.session_state.proposed_auto_annotations):
                    d = ann.to_dict()
                    d["labels"] = ", ".join(get_display_labels(ann.labels))
                    # Remove bulk data for display
                    if "telemetry_data" in d: del d["telemetry_data"]
                    d["_temp_id"] = i
                    prop_data.append(d)
                
                st.dataframe(pd.DataFrame(prop_data), use_container_width=True)
                
                r_col1, r_col2 = st.columns([1, 5])
                with r_col1:
                    if st.button("✅ Accept All", type="primary"):
                        st.session_state.current_annotations.extend(st.session_state.proposed_auto_annotations)
                        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                        # Cleanup
                        st.session_state.proposed_auto_annotations = []
                        st.session_state.review_auto_segments = False
                        st.success("All segments added!")
                        time.sleep(1)
                        st.rerun()
                
                with r_col2:
                    if st.button("❌ Discard Results"):
                        st.session_state.proposed_auto_annotations = []
                        st.session_state.review_auto_segments = False
                        st.warning("Proposed segments discarded.")
                        st.rerun()
            
            st.divider()

            # Actions
            col_actions = st.columns([1, 1, 1, 3])
            def handle_submit():
                # Access values from session state
                s_start = st.session_state[f"form_start_{selected_option}"]
                s_end = st.session_state[f"form_end_{selected_option}"]
                s_labels = st.session_state[f"form_labels_{selected_option}"]
                
                if s_start >= s_end:
                    st.session_state.temp_error = "Start index must be less than end index."
                    return
                if not s_labels:
                    st.session_state.temp_error = "Please select at least one label."
                    return
                
                label_ids = [LABEL_NAME_TO_ID[l] for l in s_labels if l in LABEL_NAME_TO_ID]
                
                # Extract telemetry data
                segment_df = df.iloc[int(s_start):int(s_end)]
                telemetry_data = segment_df.to_dict(orient="records")

                if is_edit:
                    # Update existing
                    ann = st.session_state.current_annotations[selected_option]
                    ann.start_index = int(s_start)
                    ann.end_index = int(s_end)
                    ann.segment_length = int(s_end - s_start)
                    ann.labels = label_ids
                    ann.telemetry_data = telemetry_data
                    st.session_state.temp_success = "Annotation updated!"
                else:
                    # Create new
                    annotation = AnnotatedSegment(
                        labels=label_ids,
                        segment_length=int(s_end - s_start),
                        start_index=int(s_start),
                        end_index=int(s_end),
                        chunk_index=session_id,
                        telemetry_data=telemetry_data
                    )
                    st.session_state.current_annotations.append(annotation)
                    st.session_state.temp_success = "Annotation added!"
                
                save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)

            with col_actions[0]:
                st.button(submit_label, type="primary", key=f"submit_{selected_option}", on_click=handle_submit)
            
            if "temp_error" in st.session_state:
                st.error(st.session_state.temp_error)
                del st.session_state.temp_error
            if "temp_success" in st.session_state:
                st.success(st.session_state.temp_success)
                del st.session_state.temp_success

            with col_actions[1]:
                if is_edit:
                    if st.button("Delete", type="secondary", key=f"delete_{selected_option}"):
                        st.session_state.current_annotations.pop(selected_option)
                        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                        st.success("Annotation deleted!")
                        st.rerun()
                elif selected_option == "Create New":
                    if st.button("Auto-Detect", help="Detect segments in the specified range matching selected labels"):
                        st.session_state.show_auto_detect_confirm = True

                    if st.session_state.get("show_auto_detect_confirm", False):
                        st.warning(f"⚠️ This will remove existing annotations in the range {form_start}-{form_end} before running detection. Are you sure?")
                        col_confirm, col_cancel = st.columns(2)
                        
                        if col_confirm.button("Yes, Clear Range & Detect"):
                             st.session_state.show_auto_detect_confirm = False
                             st.session_state.run_auto_detect = True
                             st.rerun()
                        
                        if col_cancel.button("Cancel"):
                            st.session_state.show_auto_detect_confirm = False
                            st.rerun()

                    if st.session_state.get("run_auto_detect", False):
                        st.session_state.run_auto_detect = False
                        
                        # Clear annotations in range first
                        st.session_state.current_annotations = [
                            a for a in st.session_state.current_annotations
                            if a.end_index <= form_start or a.start_index >= form_end
                        ]
                        
                        if form_start >= form_end:
                            st.error("Start index must be less than end index.")
                        else:
                            with st.spinner("Running classifier..."):
                                from app.services.segment_classifier_service import segment_classifier
                                try:
                                    # Slice the dataframe
                                    scan_df = df.iloc[int(form_start):int(form_end)]
                                    detected = segment_classifier.scan_telemetry_data(scan_df)
                                    
                                    new_anns = []
                                    if detected:
                                        for d in detected:
                                            # Filter by selected labels if any are selected
                                            relevant_labels = []
                                            if form_labels:
                                                relevant_labels = [l for l in d.labels if l in form_labels]
                                            else:
                                                relevant_labels = d.labels

                                            if relevant_labels:
                                                # Convert to IDs
                                                label_ids = []
                                                for name in relevant_labels:
                                                    if name in LABEL_NAME_TO_ID:
                                                        label_ids.append(LABEL_NAME_TO_ID[name])
                                                
                                                if label_ids:
                                                    # Calculate absolute indices within the session
                                                    # d.start_index and d.end_index are relative to scan_df
                                                    abs_start = int(form_start) + (d.start_index if d.start_index is not None else 0)
                                                    abs_end = int(form_start) + (d.end_index if d.end_index is not None else len(d.telemetry_data))

                                                    ann = AnnotatedSegment(
                                                        labels=label_ids,
                                                        segment_length=len(d.telemetry_data),
                                                        telemetry_data=d.telemetry_data,
                                                        chunk_index=session_id,
                                                        start_index=abs_start,
                                                        end_index=abs_end
                                                    )
                                                    new_anns.append(ann)
                                        
                                        if new_anns:
                                            st.session_state.current_annotations.extend(new_anns)
                                            st.success(f"Added {len(new_anns)} detected segments in range {form_start}-{form_end}.")
                                        else:
                                            st.warning(f"No segments found matching selected labels in range {form_start}-{form_end}.")
                                    else:
                                        st.info(f"No segments detected in range {form_start}-{form_end}.")
                                    
                                    # Always save because we cleared the annotations
                                    save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error: {e}")

        # 3. List View
        st.subheader("Current Session Annotations List")
        if st.session_state.current_annotations:
            display_data = []
            for ann in st.session_state.current_annotations:
                d = ann.to_dict()
                d["labels"] = ", ".join(get_display_labels(ann.labels))
                if "telemetry_data" in d:
                    del d["telemetry_data"]
                display_data.append(d)
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        else:
            st.info("No annotations added yet.")
            
        if st.button("Force Save All to Zarr"):
            save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)

if __name__ == "__main__":
    main()
