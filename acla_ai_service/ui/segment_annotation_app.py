"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def load_annotations(chunk_index: int) -> List[Dict[str, Any]]:
    """Load annotations for a specific chunk."""
    store = get_store()
    pipeline_config = PipelineConfig()
    annotation_key = pipeline_config.annotation_cache_key
    chunk_data = store.get_chunk(annotation_key, chunk_index)
    
    if chunk_data:
        if isinstance(chunk_data, list):
            return chunk_data
        if isinstance(chunk_data, dict) and "data" in chunk_data:
             return chunk_data["data"]
             
    return []

def save_annotations(chunk_index: int, annotations: List[Dict[str, Any]]):
    """Save annotations to Zarr store."""
    store = get_store()
    pipeline_config = PipelineConfig()
    annotation_key = pipeline_config.annotation_cache_key
    
    # Save to specific chunk index
    store.save_chunk(annotation_key, chunk_index, annotations)
    st.success(f"Saved {len(annotations)} annotations to {annotation_key} (chunk {chunk_index})")

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
                help=f"Total chunks: {metadata.chunk_count}",
                key="chunk_selector"
            )
        
        with st.spinner(f"Loading chunk {chunk_index}..."):
            df = load_chunk_data(selected_session_key, chunk_index)
            
            # Load existing annotations for this chunk if we switched chunks
            if "last_chunk_index" not in st.session_state or st.session_state.last_chunk_index != chunk_index:
                 st.session_state.current_annotations = load_annotations(chunk_index)
                 st.session_state.last_chunk_index = chunk_index
        
        if df.empty:
            st.warning("Selected chunk has no data.")
            return

        st.write(f"Loaded {len(df)} records from chunk {chunk_index} (Total chunks: {metadata.chunk_count}).")
        
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
                # Downsample for plotting if too large
                plot_df = df
                if len(df) > 10000:
                    plot_df = df.iloc[::len(df)//5000] # Approx 5000 points
                
                fig = px.line(plot_df, x=plot_df.index, y=viz_cols, title=f"Telemetry Data - Graph {graph_id}")
                fig.update_layout(uirevision=f"{chunk_index}_{graph_id}")
                
                # Visualize existing annotations
                if "current_annotations" in st.session_state and st.session_state.current_annotations:
                    for ann in st.session_state.current_annotations:
                        start = ann.get("start_index")
                        end = ann.get("end_index")
                        labels = ann.get("labels", [])
                        label_str = ", ".join(labels) if isinstance(labels, list) else str(labels)
                        
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
                # Slider for selecting timestamp range
                start_idx, end_idx = st.slider(
                    "Select Timestamp for Position View", 
                    min_value=0, 
                    max_value=len(df)-1, 
                    value=(0, min(len(df)-1, 200)),
                    key="track_map_slider"
                )

            with col_ctrl2:
                st.caption("Axis Settings")
                invert_x = st.checkbox("Invert X", value=False)
                invert_y = st.checkbox("Invert Y", value=False)
                invert_z = st.checkbox("Invert Z", value=False)
            
            # Create windowed dataframe for trajectory plotting
            map_plot_df = df.iloc[start_idx:end_idx]
            
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
                
                fig_map.update_layout(uirevision=chunk_index)
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No active cars found at this timestamp.")
        else:
            st.info("Position data (Graphics_player_pos_x/y) not available in this dataset.")

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
        if "current_annotations" in st.session_state:
            if st.session_state.current_annotations:
                st.dataframe(pd.DataFrame(st.session_state.current_annotations))
                
                # Delete annotation functionality
                col1, col2 = st.columns([3, 1])
                
                def delete_callback():
                    idx = st.session_state.delete_annotation_idx
                    if 0 <= idx < len(st.session_state.current_annotations):
                        st.session_state.current_annotations.pop(idx)
                        # Auto-save to ensure persistence
                        save_annotations(st.session_state.chunk_selector, st.session_state.current_annotations)

                with col1:
                    st.selectbox(
                        "Select annotation to delete",
                        options=range(len(st.session_state.current_annotations)),
                        format_func=lambda x: f"{x}: {st.session_state.current_annotations[x]['labels']} ({st.session_state.current_annotations[x]['start_index']}-{st.session_state.current_annotations[x]['end_index']})",
                        key="delete_annotation_idx"
                    )
                with col2:
                    st.write("")
                    st.write("")
                    st.button("Delete Selected", on_click=delete_callback)
            else:
                st.info("No annotations added yet.")
            
            if st.button("Save All Annotations to Zarr"):
                save_annotations(chunk_index, st.session_state.current_annotations)
                # We don't clear annotations after save so user can see what is saved
        else:
            st.info("No annotations added yet.")

if __name__ == "__main__":
    main()
