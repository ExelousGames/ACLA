import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..shared import get_display_labels, GRAPH_CONFIGS
from .track_sections import add_track_section_bands, track_sections_available


def _smooth_display_columns(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    if window <= 1:
        return df

    display_df = df.copy()
    display_df[columns] = display_df[columns].rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()
    return display_df


def render_feature_visualization(df: pd.DataFrame, numeric_cols: list, viz_start_idx: int, viz_end_idx: int, session_id: str):
    # Feature selection for visualization
    if "graph_ids" not in st.session_state:
        # Initialize with all configured graphs by default
        st.session_state.graph_ids = list(range(len(GRAPH_CONFIGS)))
        st.session_state.next_graph_id = len(GRAPH_CONFIGS) + 1

    if st.button("Add Graph", key="manual_add_graph_btn"):
        st.session_state.graph_ids.append(st.session_state.next_graph_id)
        st.session_state.next_graph_id += 1

    show_sections = st.checkbox(
        "Show Track Sections on Graphs",
        value=False,
        key="manual_graph_show_track_sections",
        disabled=not track_sections_available(df),
    )

    graphs_to_remove = []

    for graph_id in st.session_state.graph_ids:
        col_viz, col_smooth, col_btn = st.columns([5, 1.4, 1])
        
        # Determine configuration for this graph
        config = GRAPH_CONFIGS[graph_id] if graph_id < len(GRAPH_CONFIGS) else None

        with col_viz:
            # Default selection logic
            current_default = []
            if config:
                current_default = [c for c in config.features if c in numeric_cols]
            
            # Fallback for manually added graphs (or if config features missing)
            if not current_default and graph_id >= len(GRAPH_CONFIGS):
                 # Classic default (Speed, Gas, Brake) only for new custom graphs
                 default_cols_ref = ["speed_kmh", "gas", "brake", "steer_angle"]
                 current_default = [c for c in default_cols_ref if c in numeric_cols]
            
            # Show description if available
            if config and config.description:
                st.caption(config.description)

            viz_cols = st.multiselect(
                f"Features to Visualize (Graph {graph_id})", 
                numeric_cols, 
                default=current_default,
                key=f"manual_viz_cols_{graph_id}"
            )

        with col_smooth:
            st.markdown("<br>", unsafe_allow_html=True)
            smooth_window = st.slider(
                "Smooth",
                min_value=1,
                max_value=101,
                value=st.session_state.get(f"manual_viz_smooth_{graph_id}", 1),
                step=2,
                help="Display-only rolling window for this graph.",
                key=f"manual_viz_smooth_{graph_id}",
            )
        
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing
            if st.button("Remove", key=f"manual_remove_btn_{graph_id}"):
                graphs_to_remove.append(graph_id)

        if viz_cols:
            # Apply global range filter (include end index)
            sliced_df = df.iloc[viz_start_idx:min(viz_end_idx + 1, len(df))]
            display_df = _smooth_display_columns(sliced_df, viz_cols, smooth_window)

            # Plot without downsampling
            title = f"Telemetry Data - Graph {graph_id}"
            if smooth_window > 1:
                title += f" (Smooth: {smooth_window})"
            fig = px.line(display_df, x=display_df.index, y=viz_cols, title=title)

            if show_sections:
                add_track_section_bands(fig, sliced_df, viz_cols[0])

            # Add reference lines if configured
            if config and config.reference_lines:
                for ref in config.reference_lines:
                    fig.add_hline(
                        y=ref["value"], 
                        line_dash="dash", 
                        line_color=ref.get("color", "gray"),
                        annotation_text=ref.get("name", ""),
                        annotation_position="bottom right"
                    )

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
                                 visible_start = max(s_safe, viz_start_idx)
                                 visible_end = min(e_safe, viz_end_idx)
                                 if visible_start > visible_end:
                                     continue
                                 x_path = df.index[visible_start : visible_end+1]
                                 y_path = display_df[anchor_col].reindex(x_path)
                                 
                                 # Generate per-point hover text
                                 segment_hover_texts = []
                                 for i in range(visible_start, visible_end + 1):
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

            st.plotly_chart(fig, width='stretch')
    
    if graphs_to_remove:
        for gid in graphs_to_remove:
            if gid in st.session_state.graph_ids:
                st.session_state.graph_ids.remove(gid)
        st.rerun()
