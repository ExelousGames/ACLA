import streamlit as st
import pandas as pd
import time
from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_vlm_service, extract_json_from_response, 
    get_display_labels, get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    LABEL_DESCRIPTIONS
)

def render_agent_mode(selected_annotation_key, selected_session_key, available_sessions):
    """
    Renders the Auto-Segment Range (Agent Mode) tab.
    """
    st.markdown("Iteratively analyze telemetry to find behavioral segments.")
    
    # Shared helper for formatting session options
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    agent_mode = st.radio("Mode", ["Single Session Range", "Batch (Multiple Sessions)"], horizontal=True)
    
    # Setup Tasks
    sessions_to_process = []
    cols_ref_df = pd.DataFrame()

    if agent_mode == "Single Session Range":
        # Reuse available_sessions from outer scope
        as_sess_id = st.selectbox(
            "Select Session for Agent", 
            available_sessions, 
            format_func=format_session_option,
            key="as_sess_single"
        )
        
        col_load_container = st.container()
        
        # Load immediately to show range params
        as_df = load_session_data(selected_session_key, as_sess_id)
        
        if not as_df.empty:
            cols_ref_df = as_df
            as_len = len(as_df)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                as_start = st.number_input("Start Index", 0, as_len-1, 0, key="as_start_s")
            with c2:
                as_end = st.number_input("End Index", 0, as_len, min(as_len, 2000), key="as_end_s")
            with c3:
                as_win = st.number_input("Window Size", 1, 2000, 200, step=50, key="as_win_s")
            with c4:
                as_shift = st.number_input("Shift Size", 1, 2000, 50, step=10, key="as_shift_s", help="Samples to skip if no segment found.")
            
            sessions_to_process.append({
                "id": as_sess_id,
                "df": as_df,
                "start": as_start,
                "end": as_end,
                "window": as_win,
                "shift": as_shift
            })
        else:
            st.warning("Selected session has no data.")

    else:
        # Batch Mode
        as_sess_ids = st.multiselect("Select Sessions", available_sessions, key="as_sess_multi")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            as_limit = st.number_input("Max Samples per Session", 100, 100000, 5000, step=500, key="as_limit_m", help="Process from index 0 to this limit (or end of file).")
        with c2:
            as_win = st.number_input("Window Size", 1, 2000, 200, step=50, key="as_win_m")
        with c3:
            as_shift = st.number_input("Shift Size", 1, 2000, 50, step=10, key="as_shift_m", help="Samples to skip if no segment found.")
        
        if as_sess_ids:
            # Load first session to determine columns
            cols_ref_df = load_session_data(selected_session_key, as_sess_ids[0])
            
            for sid in as_sess_ids:
                sessions_to_process.append({
                    "id": sid,
                    "lazy": True,
                    "start": 0,
                    "end": as_limit,
                    "window": as_win,
                    "shift": as_shift
                })
    
    # Feature Selection
    as_feature_sets = []
    if not cols_ref_df.empty:
        as_numeric = cols_ref_df.select_dtypes(include=['number']).columns.tolist()
        
        # Default config per user request
        default_graph_config = {
            0: ["speed_difference"],
            1: ["expert_optimal_throttle", "Physics_gas"],
            2: ["expert_optimal_brake", "Physics_brake"]
        }
        
        num_graphs = st.number_input("Number of Graphs (Feature Sets)", min_value=1, max_value=5, value=3, step=1, key="num_graphs_agent")
        
        for i in range(num_graphs):
            # Determine defaults for this graph
            target_defaults = default_graph_config.get(i, [])
            # Filter to what is actually available in the data
            defaults = [c for c in target_defaults if c in as_numeric]
            
            # Fallback for Graph 1 if configured defaults are missing: try standard basic columns
            if i == 0 and not defaults:
                defaults = [c for c in ["speed_kmh", "gas", "brake", "steer_angle"] if c in as_numeric]
            
            cols = st.multiselect(f"Features for Graph {i+1}", as_numeric, default=defaults, key=f"as_feat_sel_{i}")
            if cols:
                as_feature_sets.append(cols)
        
        as_include_traj = st.checkbox("Include 2D Trajectory in Analysis", value=True, key="as_include_traj")

        def start_agent_callback():
            st.session_state.run_auto_segment = True
            st.session_state.proposed_auto_annotations = []
            st.session_state.review_auto_segments = False

        st.button("Start Auto-Segmentation Agent", type="primary", disabled=not sessions_to_process, on_click=start_agent_callback)
    
    # --- Execution ---
    if st.session_state.get("run_auto_segment", False):
        # Immediate safety shutoff - prevents infinite loops
        st.session_state.run_auto_segment = False
        
        vlm_service = get_vlm_service()
        if not vlm_service:
            st.error("VLM Service not available.")
        else:
            st.session_state.proposed_auto_annotations = []
            found_total = 0
            
            status_container = st.status("Running Agent...", expanded=True)
            
            try:
                full_labels_str = ", ".join(sorted(list(set(LABEL_MAPPING.values()))))
                
                for task in sessions_to_process:
                    sess_id = task["id"]
                    status_container.write(f"**Processing Session: {sess_id}**")
                    
                    # Load DF
                    if task.get("lazy"):
                        task_df = load_session_data(selected_session_key, sess_id)
                    else:
                        task_df = task["df"]
                    
                    if task_df.empty:
                        status_container.write(f"Skipping {sess_id} (No data)")
                        continue
                        
                    t_start = task["start"]
                    t_end = min(task["end"], len(task_df))
                    t_win = task["window"]
                    t_shift = task.get("shift", t_win)
                    
                    if t_start >= t_end:
                        continue

                    current_cursor = int(t_start)
                    limit_cursor = int(t_end)
                    
                    # Log start
                    print(f"--- Starting Analysis for {sess_id} from {current_cursor} to {limit_cursor} ---")
                    
                    # Existing Agent Loop Logic adapted for multi-session
                    while current_cursor < limit_cursor:
                        # Define window
                        active_limit = min(current_cursor + t_win, limit_cursor)
                        chunk_len = active_limit - current_cursor
                        
                        if chunk_len < 10: 
                            break
                        
                        # Optimization: Update status label instead of appending log lines
                        status_container.update(label=f"Processing {sess_id}: Analyzing window {current_cursor} - {active_limit}...")
                        
                        # Prepare data
                        current_slice = task_df.iloc[current_cursor:active_limit]
                        
                        # Generate CSV(s) for the feature sets
                        slice_csv_input = []
                        valid_feature_sets = [fs for fs in as_feature_sets if fs]
                        if not valid_feature_sets and not as_feature_sets:
                            # Fallback if somehow nothing selected, though UI shouldn't allow start
                            slice_csv_input = ""
                        else:
                            for feats in valid_feature_sets:
                                slice_csv_input.append(current_slice[feats].to_csv(index=False))
                        
                        # If single graph, pass as string for backward compat or simplicity
                        if isinstance(slice_csv_input, list) and len(slice_csv_input) == 1:
                            slice_csv_input = slice_csv_input[0]
                        elif isinstance(slice_csv_input, list) and len(slice_csv_input) == 0:
                             slice_csv_input = ""
                        # If multiple, slice_csv_input remains a List[str]
                        
                        # Trajectory Data
                        trajectory_csv_str = None
                        if as_include_traj:
                            traj_cols = [
                                'Graphics_player_pos_x', 'Graphics_player_pos_y', 'Graphics_player_pos_z',
                                'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z'
                            ]
                            available_traj_cols = [c for c in traj_cols if c in current_slice.columns]
                            if available_traj_cols:
                                trajectory_csv_str = current_slice[available_traj_cols].to_csv(index=False)

                        # Dynamic prompt header based on number of graphs
                        prompt_intro = "Analyze the telemetry graph (left to right) representing a driving session."
                        feature_desc = ""

                        if isinstance(slice_csv_input, list):
                            prompt_intro = "Analyze the set of synchronized telemetry graphs (stacked vertically). They all share the same time axis (x-axis)."
                            # Add descriptions for each graph to help VLM distinguish them
                            desc_lines = []
                            for idx, feats in enumerate(valid_feature_sets):
                                desc_lines.append(f"- Graph {idx+1}: {', '.join(feats)}")
                            feature_desc = "The graphs display the following signals (from top to bottom):\n" + "\n".join(desc_lines)
                        elif valid_feature_sets:
                            # Single graph case
                            feature_desc = f"The graph displays: {', '.join(valid_feature_sets[0])}"

                        # Label Definitions
                        label_context = "Definitions:\n"
                        for lname in sorted(list(set(LABEL_MAPPING.values()))):
                            if lname in LABEL_DESCRIPTIONS:
                                label_context += f"- {lname}: {LABEL_DESCRIPTIONS[lname]}\n"

                        analysis_prompt = (
                            f"{prompt_intro}\n"
                            f"{feature_desc}\n"
                            f"{label_context}\n"
                            f"Identify the **first/left-most** distinct driving behavior segment that matches one of these labels: [{full_labels_str}].\n"
                            f"Return the result ONLY as a JSON object (no markdown, no explanation) with this format:\n"
                            f'{{\n  "found": true,\n  "label": "LabelName",\n  "start_percentage": <float 0.0-1.0 representing start of segment in this window>,\n  "end_percentage": <float 0.0-1.0 representing end of segment in this window>,\n  "confidence": <float 0.0-1.0>\n}}\n'
                            f"If no distinct segment is found or the data is just noise/empty, return {{ \"found\": false }}.\n"
                            f"Focus on finding the START and END of the FIRST valid segment."
                        )
                        
                        try:
                            vlm_resp, _ = vlm_service.analyze_data(slice_csv_input, analysis_prompt, trajectory_csv_data=trajectory_csv_str)
                            
                            # Optimization: Log to console instead of UI to prevent lag
                            print(f"[{sess_id} {current_cursor}] Inference output: {vlm_resp}")

                            segment_info = extract_json_from_response(vlm_resp)
                            
                            if segment_info and segment_info.get("found"):
                                lbl = segment_info.get("label")
                                start_pct = segment_info.get("start_percentage", 0.0)
                                end_pct = segment_info.get("end_percentage", 0.1)
                                
                                # Clamp
                                start_pct = max(0.0, min(1.0, start_pct))
                                end_pct = max(0.0, min(1.0, end_pct))
                                if end_pct <= start_pct: end_pct = start_pct + 0.1
                                
                                seg_start_abs = int(current_cursor + (start_pct * chunk_len))
                                seg_end_abs = int(current_cursor + (end_pct * chunk_len))
                                
                                if seg_end_abs <= current_cursor + 5:
                                    seg_end_abs = current_cursor + max(20, int(chunk_len * 0.1))
                                
                                # Label Match
                                label_id = -1
                                match_name = "Unknown"
                                for lid, name in LABEL_MAPPING.items():
                                    if name.lower() == lbl.lower():
                                        label_id = lid
                                        match_name = name
                                        break
                                if label_id == -1:
                                    for lid, name in LABEL_MAPPING.items():
                                        if name.lower() in lbl.lower() or lbl.lower() in name.lower():
                                            label_id = lid
                                            match_name = name
                                            break
                                if label_id == -1:
                                    label_id = LABEL_NAME_TO_ID.get("Unexpected driving behavior", 0)
                                    match_name = "Unexpected driving behavior"
                                
                                new_ann = AnnotatedSegment(
                                    labels=[label_id],
                                    segment_length=int(seg_end_abs - seg_start_abs),
                                    start_index=seg_start_abs,
                                    end_index=seg_end_abs,
                                    chunk_index=sess_id,
                                    telemetry_data=task_df.iloc[seg_start_abs:seg_end_abs].to_dict(orient="records"),
                                    notes=f"Auto (Conf: {segment_info.get('confidence', 'NA')})"
                                )
                                
                                if "proposed_auto_annotations" not in st.session_state:
                                    st.session_state.proposed_auto_annotations = []
                                st.session_state.proposed_auto_annotations.append(new_ann)
                                found_total += 1
                                
                                # Optimization: Log found segment to console instead of UI stacking
                                print(f"Found [{match_name}] in {sess_id}: {seg_start_abs}-{seg_end_abs}")
                                status_container.update(label=f"Processing {sess_id} (Found {found_total} segments so far)...")
                                
                                current_cursor = seg_end_abs
                            else:
                                # Not found or parsing failed, move window forward
                                if active_limit == limit_cursor: break
                                current_cursor += t_win
                                
                        except Exception as e:
                            print(f"Error in VLM loop: {e}")
                            status_container.error(f"Error: {e}")
                            # Move forward to avoid stuck loop on error
                            current_cursor += t_win
                            if current_cursor >= limit_cursor:
                                break
            
            except Exception as critical_e:
                st.error(f"Critical Agent Error: {critical_e}")
                
            status_container.update(label=f"Done. Found {found_total} segments.", state="complete")
            
            # Store completion flag to prevent loop
            st.session_state.agent_run_completed = True
            
            if found_total > 0:
                st.session_state.review_auto_segments = True
                # No st.rerun() to avoid potential infinite loops.
                # The app will propagate to the Review & Save section below naturally.

    # --- Review & Save ---
    if st.session_state.get("review_auto_segments", False) and st.session_state.get("proposed_auto_annotations"):
        st.divider()
        st.subheader("Review Proposed Segments")
        
        props = st.session_state.proposed_auto_annotations
        
        # Table
        disp = []
        for p in props:
            d = p.to_dict()
            d["labels"] = ", ".join(get_display_labels(p.labels))
            if "telemetry_data" in d: del d["telemetry_data"]
            disp.append(d)
        
        st.dataframe(pd.DataFrame(disp), use_container_width=True)
        
        col_save, col_discard = st.columns([1, 4])
        if col_save.button("✅ Accept & Save All", type="primary"):
            # Group by session
            from collections import defaultdict
            grouped = defaultdict(list)
            for p in props:
                grouped[p.chunk_index].append(p)
            
            progress = st.progress(0)
            for i, (sid, new_anns) in enumerate(grouped.items()):
                # Load existing
                existing = load_annotations(sid, selected_annotation_key)
                existing.extend(new_anns)
                save_annotations(sid, existing, selected_annotation_key)
                progress.progress((i+1)/len(grouped))
            
            st.session_state.proposed_auto_annotations = []
            st.session_state.review_auto_segments = False
            st.success("Saved!")
            time.sleep(1)
            st.rerun()
        
        if col_discard.button("❌ Discard"):
            st.session_state.proposed_auto_annotations = []
            st.session_state.review_auto_segments = False
            st.rerun()
