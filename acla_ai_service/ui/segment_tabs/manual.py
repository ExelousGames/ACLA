import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_vlm_service, get_display_labels, get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment, LABEL_DESCRIPTIONS,
    GRAPH_CONFIGS, LABEL_CATEGORIES
)

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
    chunk_count = metadata.chunk_count if metadata else len(available_sessions)
    st.write(f"Loaded {len(df)} records from session {session_id} (Total sessions: {chunk_count}).")

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

    col_global_slider, col_global_inputs = st.columns([3, 1])
    with col_global_slider:
        viz_start_idx, viz_end_idx = st.slider(
            "Select Range",
            min_value=0,
            max_value=len(df),
            value=(0, min(len(df), 5000)),
            key="manual_global_viz_range",
            label_visibility="collapsed"
        )
    
    with col_global_inputs:
         c_input1, c_input2 = st.columns(2)
         with c_input1:
             st.number_input("Start", min_value=0, max_value=len(df), value=viz_start_idx, key="manual_global_viz_start_input", on_change=update_global_slider_range)
         with c_input2:
             st.number_input("End", min_value=0, max_value=len(df), value=viz_end_idx, key="manual_global_viz_end_input", on_change=update_global_slider_range)

    # Feature selection for visualization
    if "graph_ids" not in st.session_state:
        st.session_state.graph_ids = [0]
        st.session_state.next_graph_id = 1

    if st.button("Add Graph", key="manual_add_graph_btn"):
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
                key=f"manual_viz_cols_{graph_id}"
            )
        
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing
            if st.button("Remove", key=f"manual_remove_btn_{graph_id}"):
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
                        # Add hoverable invisible marker for segment stats
                        hover_summary = [f"<b>Segment: {label_str}</b>", f"Range: {start}-{end}"]
                        for col in viz_cols:
                            if col in df.columns:
                                try:
                                    s_idx = max(0, min(start, len(df)-1))
                                    e_idx = max(0, min(end, len(df)-1))
                                    val_start = df[col].iloc[s_idx]
                                    val_end = df[col].iloc[e_idx]
                                    diff = val_end - val_start
                                    hover_summary.append(f"Total {col} Δ: {diff:+.2f}")
                                except Exception:
                                    pass
                        
                        # Create hover trace for the inner segment (start+1 to end-1)
                        if viz_cols:
                             s_inner = start + 1
                             e_inner = end - 1
                             s_safe = max(0, min(s_inner, len(df)-1))
                             e_safe = max(0, min(e_inner, len(df)-1))

                             if s_safe <= e_safe:
                                 # Anchor to the first visualized column
                                 anchor_col = viz_cols[0]
                                 # Extract path
                                 x_path = df.index[s_safe : e_safe+1]
                                 y_path = df[anchor_col].iloc[s_safe : e_safe+1]
                                 
                                 # Generate per-point hover text
                                 segment_hover_texts = []
                                 for i in range(s_safe, e_safe + 1):
                                     point_lines = hover_summary.copy()
                                     point_lines.append(f"<b>Index: {i}</b>")
                                     
                                     for col in viz_cols:
                                         if col in df.columns:
                                             val = df[col].iloc[i]
                                             prev_val = df[col].iloc[i-1] if i > 0 else val
                                             step_diff = val - prev_val
                                             point_lines.append(f"{col}: {val:.2f} (Δ {step_diff:+.4f})")
                                     
                                     segment_hover_texts.append("<br>".join(point_lines))

                                 fig.add_trace(go.Scatter(
                                    x=x_path,
                                    y=y_path,
                                    mode="lines",
                                    line=dict(color="rgba(0,0,0,0)", width=4), # Transparent but clickable
                                    hoverinfo="text",
                                    hovertext=segment_hover_texts,
                                    showlegend=False,
                                    hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.9)")
                                 ))

                        fig.add_vrect(
                            x0=start, 
                            x1=end, 
                            fillcolor="green", 
                            opacity=0.1, 
                            layer="below", 
                            line_width=1,
                            line_color="green",
                            annotation_text=f"{label_str} [{start}-{end}]",
                            annotation_position="top left"
                        )

            st.plotly_chart(fig, use_container_width=True)
    
    if graphs_to_remove:
        for gid in graphs_to_remove:
            if gid in st.session_state.graph_ids:
                st.session_state.graph_ids.remove(gid)
        st.rerun()

    # --- Track Map Visualization ---
    show_track_map = st.checkbox("Show Track Map & Positions", value=True, key="manual_track_map_visible")

    if show_track_map:
        # Check if we have position data
        has_player_pos = "Graphics_player_pos_x" in df.columns and "Graphics_player_pos_y" in df.columns
        has_player_pos_z = "Graphics_player_pos_z" in df.columns
        
        has_opponent_pos = any(f"Opponent_{i}_pos_x" in df.columns for i in range(1, 6))
        
        has_expert_pos = "expert_optimal_player_pos_x" in df.columns and "expert_optimal_player_pos_y" in df.columns
        has_expert_pos_z = "expert_optimal_player_pos_z" in df.columns
        
        if has_player_pos or has_opponent_pos or has_expert_pos:
            # View controls
            st.caption("Axis Settings")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                invert_x = st.checkbox("Invert X", value=False, key="manual_invert_x")
            with col_ctrl2:
                invert_y = st.checkbox("Invert Y", value=False, key="manual_invert_y")
            with col_ctrl3:
                invert_z = st.checkbox("Invert Z", value=False, key="manual_invert_z")
            
            # Create windowed dataframe for trajectory plotting using Global Range
            start_idx = min(viz_start_idx, len(df) - 1)
            
            # Ensure indices are within bounds
            safe_end_idx = min(viz_end_idx, len(df))
            if safe_end_idx <= start_idx:
                 # Handle empty range selection gracefully
                 map_plot_df = pd.DataFrame(columns=df.columns)
                 selected_time_idx = start_idx
            else:
                 map_plot_df = df.iloc[start_idx:safe_end_idx]
                 selected_time_idx = safe_end_idx - 1
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

    # Sort annotations by start_index
    if st.session_state.current_annotations:
        st.session_state.current_annotations.sort(key=lambda x: getattr(x, "start_index", 0))

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
        key="manual_annotation_selector"
    )

    # Manual Annotation Logic
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
            key=f"manual_form_start_{selected_option}"
        )
    with col_form2:
        form_end = st.number_input(
            "End Index", 
            min_value=0, 
            max_value=len(df)-1, 
            value=default_end,
            key=f"manual_form_end_{selected_option}"
        )

    def copy_range_to_viz():
        s_start = st.session_state.get(f"manual_form_start_{selected_option}", 0)
        s_end = st.session_state.get(f"manual_form_end_{selected_option}", 0)
        st.session_state.manual_global_viz_range = (int(s_start), int(s_end))
        st.session_state.manual_global_viz_start_input = int(s_start)
        st.session_state.manual_global_viz_end_input = int(s_end)

    st.button("Copy Range to Visualization", 
        help="Update the global visualization range to match these start/end indices", 
        key=f"manual_copy_range_{selected_option}",
        on_click=copy_range_to_viz
    )
    
    # Filter for Main Labels
    main_label_ids = LABEL_CATEGORIES.get("Main Labels", [])
    main_label_options = [LABEL_MAPPING[lid] for lid in main_label_ids if lid in LABEL_MAPPING]
    valid_defaults = [l for l in default_labels if l in main_label_options]

    form_labels = st.multiselect(
        "Labels", 
        main_label_options, 
        default=valid_defaults,
        key=f"manual_form_labels_{selected_option}"
    )

    # Feature Change Calculator
    show_calculator = st.toggle("Show Feature Change Calculator", key="manual_show_calculator")

    if show_calculator:
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            # Ensure the persisted selection is valid for the current dataset
            current_selection = st.session_state.get("manual_calc_feature_global")
            if current_selection and current_selection not in numeric_cols:
                # Reset if invalid
                if "speed_kmh" in numeric_cols:
                    st.session_state["manual_calc_feature_global"] = "speed_kmh"
                elif numeric_cols:
                    st.session_state["manual_calc_feature_global"] = numeric_cols[0]

            # Default to speed or gas if available
            default_calc_idx = 0
            if "speed_kmh" in numeric_cols:
                default_calc_idx = numeric_cols.index("speed_kmh")
            
            calc_feature = st.selectbox(
                "Select Feature", 
                numeric_cols, 
                index=default_calc_idx,
                key="manual_calc_feature_global"
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

                st.markdown("##### Rate of Change Over Time")
                if not diffs.empty:
                    scr_col1, scr_col2 = st.columns([1, 1])
                    with scr_col1:
                         # Smoothing Control
                         smooth_window = st.slider(
                             "Smoothing (Moving Average)", 
                             min_value=1, 
                             max_value=max(2, min(50, len(diffs))), 
                             value=1, 
                             key=f"manual_roc_smooth_{selected_option}"
                         )
                    
                    data_to_plot = diffs
                    if smooth_window > 1:
                        data_to_plot = diffs.rolling(window=smooth_window, center=True).mean()

                    fig_roc = px.line(
                        x=data_to_plot.index, 
                        y=data_to_plot.values, 
                        labels={'x': 'Index', 'y': 'Change'}, 
                        title=f"Rate of Change (Δ) - {calc_feature} (Window: {smooth_window})"
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

    # Classifier Probability Check
    with st.expander("Classifier Probabilities (AI Check)"):
        if form_start < form_end and int(form_end) < len(df):
            if st.button("Check Probabilities for Range", key="manual_check_probs_btn"):
                with st.spinner("Analyzing segment with Classifier..."):
                    try:
                        # Import here to avoid circular dependencies during initial load
                        from app.services.segment_classifier_service import segment_classifier
                        
                        # Extract segment
                        snippet = df.iloc[int(form_start):int(form_end)]
                        probs = segment_classifier.predict_segment_probabilities(snippet)
                        
                        st.write("Confidence per Label:")
                        # Filter and display
                        has_results = False
                        for label, score in probs.items():
                            if score > 0.01:
                                has_results = True
                                c_lab, c_prog = st.columns([1, 2])
                                with c_lab:
                                    label_str = LABEL_MAPPING.get(label, str(label))
                                    st.caption(f"{label_str} ({score:.1%})")
                                with c_prog:
                                    st.progress(score)
                        
                        if not has_results:
                            st.info("No labels detected with significant probability (>1%)")
                            
                    except Exception as e:
                        st.error(f"Error calling classifier: {str(e)}")
        else:
            st.info("Select a valid range (min length 1) to check probabilities.")

    # VLM Analysis Section
    with st.expander("AI Label Analysis"):
        st.markdown("Use VLM to analyze why the selected labels fit this segment.")
        analyze_vlm = st.button("Analyze Reason with VLM")

    if analyze_vlm:
        if form_start >= form_end:
             st.error("Invalid range selected.")
        elif not form_labels:
             st.error("Please select labels to analyze.")
        else:
             service = get_vlm_service()
             if not service:
                 st.error("VLM Service not available.")
             else:
                 # Prepare data
                 segment_df = df.iloc[int(form_start):int(form_end)+1]
                 
                 with st.status("Initializing VLM Analysis...", expanded=True) as status:
                     # Prepare CSVs based on GRAPH_CONFIGS
                     csv_inputs = []
                     support_lines_list = []
                     
                     st.write("Preparing Input Data...")
                     
                     graph_descriptions = []
                     for g_conf in GRAPH_CONFIGS:
                         # Filter columns
                         cols = [c for c in g_conf.features if c in segment_df.columns]
                         if cols:
                             # Create sub-df with these columns
                             sub_df = segment_df[cols].copy()
                             csv_inputs.append(sub_df.to_csv(index=False))
                             support_lines_list.append(g_conf.reference_lines)
                             graph_descriptions.append(g_conf.description)
                     
                     if not csv_inputs:
                         # Fallback if specific features not found
                         st.warning("No specific graph configurations matched. Using all numeric data.")
                         sub_df = segment_df.select_dtypes(include=['number'])
                         csv_inputs.append(sub_df.to_csv(index=False))
                         support_lines_list.append([])

                     # Trajectory
                     traj_cols = ['Graphics_player_pos_x', 'Graphics_player_pos_y', 
                                  'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y']
                     traj_cols = [c for c in traj_cols if c in segment_df.columns]
                     traj_csv = None
                     if len(traj_cols) >= 2:
                         traj_csv = segment_df[traj_cols].to_csv(index=False)

                     # Prompt
                     selected_labels_str = ", ".join(form_labels)
                     descriptions_str = ""
                     for l in form_labels:
                         if l in LABEL_DESCRIPTIONS:
                             descriptions_str += f"- {l}: {LABEL_DESCRIPTIONS[l]}\n"
                     
                     graph_context_str = "\n".join([f"- Graph {i+1}: {desc}" for i, desc in enumerate(graph_descriptions)])
                     
                     prompt = (
                         f"I have labeled this telemetry segment as: {selected_labels_str}.\n"
                         f"Here are the descriptions for these labels:\n{descriptions_str}\n"
                         f"Here are the descriptions of the telemetry graphs provided:\n{graph_context_str}\n"
                         "Based on the telemetry data and vehicle trajectory graphs, explain why these labels are appropriate for this segment.  "
                     )
                     
                     st.markdown("### VLM Reasoning")
                     response_placeholder = st.empty()
                     current_response_text = ""

                     def update_progress(msg):
                         nonlocal current_response_text
                         if msg.startswith("__STREAM__"):
                             token = msg.replace("__STREAM__", "", 1)
                             current_response_text += token
                             response_placeholder.markdown(current_response_text)
                         else:
                             status.update(label=f"VLM Analysis: {msg}")

                     try:
                         status.update(label="Running VLM Inference...")
                         response, img = service.analyze_data(
                             csv_data=csv_inputs,
                             prompt=prompt,
                             trajectory_csv_data=traj_csv,
                             support_lines=support_lines_list,
                             status_callback=update_progress
                         )
                         
                         status.update(label="Analysis Complete!", state="complete", expanded=False)
                         
                         # Overwrite with final response to ensure consistency
                         response_placeholder.markdown(response)
                         
                     except Exception as e:
                         status.update(label="Analysis Failed", state="error")
                         st.error(f"Analysis failed: {str(e)}")

    # Form Actions
    col_actions = st.columns([1, 1, 1, 3])
    def handle_submit():
        # Access values from session state
        s_start = st.session_state[f"manual_form_start_{selected_option}"]
        s_end = st.session_state[f"manual_form_end_{selected_option}"]
        s_labels = st.session_state[f"manual_form_labels_{selected_option}"]
        
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
        st.button(submit_label, type="primary", key=f"manual_submit_{selected_option}", on_click=handle_submit)
    
    if "temp_error" in st.session_state:
        st.error(st.session_state.temp_error)
        del st.session_state.temp_error
    if "temp_success" in st.session_state:
        st.success(st.session_state.temp_success)
        del st.session_state.temp_success

    with col_actions[1]:
        if is_edit:
            if st.button("Delete", type="secondary", key=f"manual_delete_{selected_option}"):
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

    # List View (Inside Tab 1)
    if st.toggle("Show Current Session Annotations List"):
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

            if st.button("Delete All Segments for Session", type="primary", key="manual_btn_del_all_seg"):
                st.session_state.manual_show_delete_all_confirm = True
            
            if st.session_state.get("manual_show_delete_all_confirm", False):
                st.warning(f"⚠️ Are you sure you want to DELETE ALL {len(st.session_state.current_annotations)} segments for session '{session_id}'? This cannot be undone.")
                col_confirm_del, col_cancel_del = st.columns(2)
                
                with col_confirm_del:
                    if st.button("Yes, Delete All", key="manual_confirm_del_all"):
                        st.session_state.current_annotations = []
                        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                        st.session_state.manual_show_delete_all_confirm = False
                        st.success(f"All annotations for session {session_id} have been deleted.")
                        st.rerun()
                
                with col_cancel_del:
                    if st.button("Cancel", key="manual_cancel_del_all"):
                        st.session_state.manual_show_delete_all_confirm = False
                        st.rerun()
        else:
            st.info("No annotations added yet.")
        
    if st.button("Force Save All to Zarr"):
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
