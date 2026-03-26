import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..shared import LABEL_CATEGORIES, LABEL_MAPPING

def render_feature_visualization(df, viz_start_idx, viz_end_idx, session_id, numeric_cols, default_cols):
    # Feature selection for visualization
    if "detailed_graph_ids" not in st.session_state:
        st.session_state.detailed_graph_ids = [0, 1, 2, 3, 4, 5]
        st.session_state.detailed_next_graph_id = 6

    def on_add_graph():
        st.session_state.detailed_graph_ids.append(st.session_state.detailed_next_graph_id)
        st.session_state.detailed_next_graph_id += 1

    st.button("Add Graph", key="detailed_add_graph_btn", on_click=on_add_graph)

    graphs_to_remove = []

    for graph_id in st.session_state.detailed_graph_ids:
        col_viz, col_btn = st.columns([6, 1])
        
        with col_viz:
            # Default selection logic
            current_default = []
            
            # Define requested defaults
            defaults_map = {
                0: ["expert_optimal_throttle", "Physics_gas"],
                1: ["expert_optimal_brake", "Physics_brake"],
                2: ["expert_time_difference"],
                3: ["speed_difference"],
                4: ["expert_optimal_speed", "Physics_speed_kmh"],
                5: ["driver_push_to_limit"]
            }
            
            # Use specific defaults if available for this graph_id
            if graph_id in defaults_map:
                current_default = [c for c in defaults_map[graph_id] if c in numeric_cols]
            elif graph_id == 0: # Fallback for legacy or if graph 0 not in map (though it is)
                current_default = [c for c in default_cols if c in numeric_cols]
                if not current_default:
                    current_default = numeric_cols[:3]
            
            viz_cols = st.multiselect(
                f"Features to Visualize (Graph {graph_id})", 
                numeric_cols, 
                default=current_default,
                key=f"detailed_viz_cols_{graph_id}"
            )
        
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing
            if st.button("Remove", key=f"detailed_remove_btn_{graph_id}"):
                graphs_to_remove.append(graph_id)

        if viz_cols:
            # Use global slider for visualization range, even if an annotation is selected
            plot_start = viz_start_idx
            plot_end = viz_end_idx

            # Apply range filter (include end index)
            sliced_df = df.iloc[plot_start:min(plot_end + 1, len(df))]

            # Plot without downsampling
            fig = px.line(sliced_df, x=sliced_df.index, y=viz_cols, title=f"Telemetry Data - Graph {graph_id}")

            # Enhance hover with detailed stats (Index & Delta) for all points to match Manual UI capabilities
            if viz_cols and not sliced_df.empty:
                full_deltas = df[viz_cols].diff().iloc[plot_start:min(plot_end + 1, len(df))]
                hover_texts = []
                for idx_val, row in sliced_df.iterrows():
                    lines = [f"<b>Index: {idx_val}</b>"]
                    for col in viz_cols:
                        if col in row:
                            d = full_deltas.at[idx_val, col]
                            d_str = f"{d:+.4f}" if pd.notna(d) else "N/A"
                            lines.append(f"{col}: {row[col]:.2f} (Δ {d_str})")
                    hover_texts.append("<br>".join(lines))
                
                fig.update_traces(hovertemplate="%{text}", text=hover_texts)

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
                        if end <= plot_start or start >= plot_end:
                            continue

                    labels = ann.labels
                    
                    # Group labels by Main Labels
                    main_labels_present = [l for l in labels if l in LABEL_CATEGORIES.get("Main Labels", [])]
                    segment_types_present = [l for l in labels if l in LABEL_CATEGORIES.get("Segment Type", [])]
                    
                    grouped_labels = []
                    for ml in main_labels_present:
                        sub_labels = [l for l in labels if l in LABEL_CATEGORIES.get(ml, [])]
                        ml_name = LABEL_MAPPING.get(ml, str(ml))
                        if sub_labels:
                            sub_names = [LABEL_MAPPING.get(sl, str(sl)) for sl in sub_labels]
                            grouped_labels.append(f"{ml_name}: {', '.join(sub_names)}")
                        else:
                            grouped_labels.append(ml_name)
                            
                    if segment_types_present:
                        st_names = [LABEL_MAPPING.get(st, str(st)) for st in segment_types_present]
                        grouped_labels.append(f"Segment Type: {', '.join(st_names)}")
                        
                    accounted_for = set(main_labels_present + segment_types_present)
                    for ml in main_labels_present:
                        accounted_for.update(LABEL_CATEGORIES.get(ml, []))
                    
                    other_labels = [l for l in labels if l not in accounted_for]
                    if other_labels:
                        other_names = [LABEL_MAPPING.get(l, str(l)) for l in other_labels]
                        grouped_labels.append(f"Other: {', '.join(other_names)}")
                        
                    label_str = "<br>".join(grouped_labels) if grouped_labels else "No Labels"
                    
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
                            line_width=0,
                            annotation_text=label_str,
                            annotation_position="top left",
                            annotation_align="left"
                        )

            st.plotly_chart(fig, width='stretch')
    
    if graphs_to_remove:
        for gid in graphs_to_remove:
            if gid in st.session_state.detailed_graph_ids:
                st.session_state.detailed_graph_ids.remove(gid)
        st.rerun()
