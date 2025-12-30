"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
"""

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
                        st.warning("⚠️ This will remove ALL existing annotations for this session before running detection. Are you sure?")
                        col_confirm, col_cancel = st.columns(2)
                        
                        if col_confirm.button("Yes, Clear & Detect"):
                             st.session_state.show_auto_detect_confirm = False
                             st.session_state.run_auto_detect = True
                             st.rerun()
                        
                        if col_cancel.button("Cancel"):
                            st.session_state.show_auto_detect_confirm = False
                            st.rerun()

                    if st.session_state.get("run_auto_detect", False):
                        st.session_state.run_auto_detect = False
                        
                        # Clear annotations first
                        st.session_state.current_annotations = []
                        
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
                                            st.success(f"Replaced annotations with {len(new_anns)} detected segments.")
                                        else:
                                            st.warning("No segments found matching selected labels. Annotations cleared.")
                                    else:
                                        st.info("No segments detected. Annotations cleared.")
                                    
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
