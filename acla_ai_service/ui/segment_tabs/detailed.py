import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_display_labels, get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    LABEL_CATEGORIES
)

try:
    from ..gemini_analyzer import GeminiAnalyzer
except ImportError:
    # Fallback if relative import fails structure
    try:
        from ui.gemini_analyzer import GeminiAnalyzer
    except ImportError:
        GeminiAnalyzer = None


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

    # --- Unified Annotation Management (MOVED UP) ---
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
        key="detailed_annotation_selector"
    )
    
    # Auto-update visualization range when selection changes
    if "last_detailed_selection" not in st.session_state:
        st.session_state.last_detailed_selection = None
    
    if selected_option != st.session_state.last_detailed_selection:
        st.session_state.last_detailed_selection = selected_option
        # If switching to an edit, auto-set the visualization range components to show the segment
        if selected_option != "Create New":
             ann_sel = st.session_state.current_annotations[selected_option]
             st.session_state.detailed_global_viz_start_input = ann_sel.start_index
             st.session_state.detailed_global_viz_end_input = ann_sel.end_index
             st.session_state.detailed_global_viz_range = (ann_sel.start_index, ann_sel.end_index)

    # Manual Annotation Logic
    is_sub_segment = False
    input_min = 0
    input_max = len(df)-1

    if selected_option == "Create New":
        form_title = "Add New Annotation"
        default_start = 0
        default_end = min(100, len(df)-1)
        default_labels = []
        submit_label = "Add Annotation"
        is_edit = False
    else:
        # Existing Annotation Selected
        ann = st.session_state.current_annotations[selected_option]
        
        # Mode Selection
        mode = st.radio("Action Mode", ["Edit Segment", "Add Detail (Sub-segment)"], key=f"detailed_mode_{selected_option}")
        
        if mode == "Edit Segment":
            form_title = f"Edit Annotation #{selected_option}"
            default_start = ann.start_index
            default_end = ann.end_index
            default_labels = [l for l in get_display_labels(ann.labels) if l in LABEL_MAPPING.values()]
            submit_label = "Update Annotation"
            is_edit = True
        else:
            # Add Detail Mode
            form_title = f"Add Detail to Segment #{selected_option}"
            input_min = ann.start_index
            input_max = ann.end_index
            default_start = ann.start_index
            default_end = ann.end_index
            default_labels = []
            submit_label = "Add Detailed Segment"
            is_edit = False
            is_sub_segment = True

    st.markdown(f"**{form_title}**")
    
    col_form1, col_form2 = st.columns(2)
    with col_form1:
        form_start = st.number_input(
            "Start Index", 
            min_value=input_min, 
            max_value=input_max, 
            value=default_start,
            key=f"detailed_form_start_{selected_option}",
            disabled=False
        )
    with col_form2:
        form_end = st.number_input(
            "End Index", 
            min_value=input_min, 
            max_value=input_max, 
            value=default_end,
            key=f"detailed_form_end_{selected_option}",
            disabled=False
        )
    
    st.markdown("##### Labels")
    selected_labels_all = []
    
    if is_sub_segment:
        # Restricted Workflow: Only allow selecting sub-labels for existing Main Labels
        current_ann = st.session_state.current_annotations[selected_option]
        current_ids = current_ann.labels
        
        # Identify "parent" categories present in the annotation
        exposed_label_names = set()
        categories_to_show = []
        
        for cat, ids in LABEL_CATEGORIES.items():
            # If category is an ID key, it is a sub-label group (e.g. 28 -> [29, 30...])
            # Show this group ONLY if the parent ID (cat) is in current labels
            if cat in current_ids:
                 categories_to_show.append((cat, ids))

        # Render selectors
        for cat, ids in categories_to_show:
             display_name = LABEL_MAPPING.get(cat, str(cat))
             cat_names = [LABEL_MAPPING[lid] for lid in ids if lid in LABEL_MAPPING]
             exposed_label_names.update(cat_names)
             
             # Pre-select existing sub-labels
             cat_defaults = [l for l in default_labels if l in cat_names]
             
             cat_selected = st.multiselect(
                f"{display_name} Specifics",
                cat_names,
                default=cat_defaults,
                key=f"detailed_form_labels_{selected_option}_{cat}"
            )
             selected_labels_all.extend(cat_selected)

        # Preserve any labels that were NOT exposed in the selectors (e.g. Main Labels or other categories)
        for d_l in default_labels:
            if d_l not in exposed_label_names:
                 selected_labels_all.append(d_l)
                 
        if not categories_to_show:
            st.info("No sub-labels available for the assigned main labels.")
            
    else:
        # Create New or Edit Mode
        
        # 1. Main Labels
        main_cat = "Main Labels"
        main_ids = LABEL_CATEGORIES.get(main_cat, [])
        main_names = [LABEL_MAPPING[lid] for lid in main_ids if lid in LABEL_MAPPING]
        
        # Determine defaults
        main_defaults = [l for l in default_labels if l in main_names]
        
        main_selected = st.multiselect(
            main_cat,
            main_names,
            default=main_defaults,
            key=f"detailed_form_labels_{selected_option}_{main_cat}"
        )
        selected_labels_all.extend(main_selected)
        
        # Determine which sub-categories to show based on Main Labels selection
        selected_main_ids = {LABEL_NAME_TO_ID.get(name) for name in main_selected if name in LABEL_NAME_TO_ID}

        # 2. Sub-categories
        for category, category_ids in LABEL_CATEGORIES.items():
            if category == main_cat:
                continue
            
            # Only show if the category (which is a parent ID) is selected in Main Labels, or if it is the "Other Labels" group
            if category == "Other Labels" or category in selected_main_ids:
                display_name = category
                if category in LABEL_MAPPING:
                    display_name = LABEL_MAPPING[category]

                cat_label_names = [LABEL_MAPPING[lid] for lid in category_ids if lid in LABEL_MAPPING]
                if cat_label_names:
                    cat_defaults = [l for l in default_labels if l in cat_label_names]
                    
                    cat_selected = st.multiselect(
                        f"{display_name}",
                        cat_label_names,
                        default=cat_defaults,
                        key=f"detailed_form_labels_{selected_option}_{category}"
                    )
                    selected_labels_all.extend(cat_selected)
        
        # Also handle uncategorized labels for Create Mode
        all_cat_ids = [lid for ids in LABEL_CATEGORIES.values() for lid in ids]
        uncategorized_ids = [lid for lid in LABEL_MAPPING.keys() if lid not in all_cat_ids]
        if uncategorized_ids:
            uncat_names = [LABEL_MAPPING[lid] for lid in uncategorized_ids if lid in LABEL_MAPPING]
            if uncat_names:
                uncat_defaults = [l for l in default_labels if l in uncat_names]
                uncat_selected = st.multiselect(
                    "Other",
                    uncat_names,
                    default=uncat_defaults,
                    key=f"detailed_form_labels_{selected_option}_other"
                )
                selected_labels_all.extend(uncat_selected)
            
    form_labels = selected_labels_all

    # Classifier Probability Check (Only for New Annotations)
    if not is_edit:
        with st.expander("Classifier Probabilities (AI Check)"):
            if form_start < form_end and int(form_end) < len(df):
                if st.button("Check Probabilities for Range", key="detailed_check_probs_btn"):
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

    # Gemini AI Analysis
    with st.expander("Advanced AI Analysis (Gemini)"):
        # API Key management
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            gemini_api_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_api_key_input")
        
        if gemini_api_key:
            if st.button("Identify Sub-Labels with Gemini", key="gemini_identify_btn"):
                if form_start >= form_end:
                    st.error("Invalid range selected.")
                else:
                    with st.spinner("Preparing graphs and asking Gemini..."):
                        if GeminiAnalyzer is None:
                             st.error("GeminiAnalyzer module could not be imported.")
                        else:
                            try:
                                analyzer = GeminiAnalyzer(gemini_api_key)
                                
                                # 1. Prepare Data
                                analysis_df = df.iloc[int(form_start):int(form_end)]
                                
                                # 2. Gather Feature Graphs
                                graph_configs = {}
                                # Check session state for graph configs or use defaults if not yet rendered
                                chart_ids = st.session_state.get("detailed_graph_ids", [0, 1, 2, 3, 4, 5])
                                
                                # Default mappings from line 630
                                defaults_map = {
                                    0: ["expert_optimal_throttle", "Physics_gas"],
                                    1: ["expert_optimal_brake", "Physics_brake"],
                                    2: ["expert_time_difference"],
                                    3: ["speed_difference"],
                                    4: ["expert_optimal_speed", "Physics_speed_kmh"],
                                    5: ["driver_push_to_limit"]
                                }
                                default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]

                                for gid in chart_ids:
                                    key = f"detailed_viz_cols_{gid}"
                                    # If in session state, use it
                                    if key in st.session_state and st.session_state[key]:
                                        graph_configs[gid] = st.session_state[key]
                                    elif gid in defaults_map:
                                        # Use defaults if not in session state (first load)
                                        # Filter for valid columns
                                        available = [c for c in defaults_map[gid] if c in df.columns]
                                        if available:
                                            graph_configs[gid] = available
                                    elif gid == 0:
                                        # Fallback for graph 0
                                        available = [c for c in default_cols if c in df.columns]
                                        if available:
                                            graph_configs[gid] = available

                                
                                # 3. Track Config
                                track_config = {
                                    "player_x": "Graphics_player_pos_x",
                                    "player_y": "Graphics_player_pos_y",
                                    "expert_x": "expert_optimal_player_pos_x",
                                    "expert_y": "expert_optimal_player_pos_y"
                                }
                                
                                # 4. Current Labels
                                current_labels_display = form_labels
                                
                                # 5. Contextual Sub-labels (e.g. MS -> MS1..MS30)
                                sub_label_context = []
                                for lname in current_labels_display:
                                    lid = LABEL_NAME_TO_ID.get(lname)
                                    # If the selected label is a parent category (e.g. 'MS', 'RM')
                                    if lid and lid in LABEL_CATEGORIES:
                                        child_ids = LABEL_CATEGORIES[lid]
                                        if child_ids:
                                            # Format list of children
                                            child_docs = []
                                            for child_id in child_ids:
                                                child_name = LABEL_MAPPING.get(child_id, child_id)
                                                child_docs.append(f"- {child_id}: {child_name}")
                                            
                                            block = f"Sub-labels for '{lname}' ({lid}):\n" + "\n".join(child_docs)
                                            sub_label_context.append(block)

                                # Analyze
                                result = analyzer.analyze_segment(
                                    analysis_df, 
                                    graph_configs, 
                                    track_config, 
                                    current_labels_display,
                                    available_sub_labels_context=sub_label_context
                                )
                                
                                st.markdown("### Gemini Analysis Results")
                                st.markdown(result)
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

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
                key=f"detailed_calc_feat_{selected_option}"
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
                             key=f"detailed_roc_smooth_{selected_option}"
                         )

                    data_to_plot = diffs
                    if smooth_window > 1:
                        data_to_plot = diffs.rolling(window=smooth_window, center=True).mean()

                    fig_roc = px.line(
                        x=data_to_plot.index, 
                        y=data_to_plot.values, 
                        labels={'x': 'Index', 'y': f'Change'}, 
                        title=f"Rate of Change (Δ) - {calc_feature} (Window: {smooth_window})"
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)



    # Form Actions
    col_actions = st.columns([1, 1, 1, 3])
    def handle_submit(go_next=False):
        # Access values from session state
        s_start = st.session_state[f"detailed_form_start_{selected_option}"]
        s_end = st.session_state[f"detailed_form_end_{selected_option}"]
        s_labels = form_labels # Use locally aggregated variable
        
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

            if go_next:
                if isinstance(selected_option, int) and selected_option + 1 < len(st.session_state.current_annotations):
                    st.session_state.detailed_annotation_selector = selected_option + 1
                    st.session_state.temp_success = f"Updated #{selected_option}. Moving to #{selected_option + 1}."
                else:
                    st.session_state.temp_success = "Updated annotation. (End of list)"
            else:
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

            if go_next and isinstance(selected_option, int):
                if selected_option + 1 < len(st.session_state.current_annotations) - 1:
                    st.session_state.detailed_annotation_selector = selected_option + 1
                    st.session_state.temp_success = f"Added Detail. Moving to #{selected_option + 1}."
                else:
                    st.session_state.temp_success = "Added Detail. (End of list)"
            else:
                st.session_state.temp_success = "Annotation added!"
        
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)

    if is_edit:
        with col_actions[0]:
             st.button("Update & Next ⏭️", type="primary", key=f"detailed_submit_next_{selected_option}", on_click=handle_submit, args=(True,))
        
        with col_actions[1]:
            st.button(submit_label, type="secondary", key=f"detailed_submit_{selected_option}", on_click=handle_submit, args=(False,))

        with col_actions[2]:
            if st.button("Delete", type="secondary", key=f"detailed_delete_{selected_option}"):
                st.session_state.current_annotations.pop(selected_option)
                save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                st.success("Annotation deleted!")
                st.rerun()
    else:
        with col_actions[0]:
            st.button(submit_label, type="primary", key=f"detailed_submit_{selected_option}", on_click=handle_submit, args=(False,))
    
    if "temp_error" in st.session_state:
        st.error(st.session_state.temp_error)
        del st.session_state.temp_error
    if "temp_success" in st.session_state:
        st.success(st.session_state.temp_success)
        del st.session_state.temp_success

    if selected_option == "Create New":
        with col_actions[1]:
            if st.button("Auto-Detect", help="Detect segments in the specified range matching selected labels", key="detailed_auto_detect"):
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

    if not is_edit and isinstance(selected_option, int):
        with col_actions[1]:
             st.button("Add & Next ⏭️", type="secondary", key=f"detailed_submit_next_sub_{selected_option}", on_click=handle_submit, args=(True,))

    # --- Visualization Range & Graphs (MOVED DOWN) ---
    st.markdown("---")
    st.caption("Visualization Range (Graphs & Track Map)")
    
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
            value=(0, min(len(df), 5000)),
            key="detailed_global_viz_range",
            label_visibility="collapsed"
        )
    
    with col_global_inputs:
         c_input1, c_input2 = st.columns(2)
         with c_input1:
             st.number_input("Start", min_value=0, max_value=len(df), value=viz_start_idx, key="detailed_global_viz_start_input", on_change=update_global_slider_range)
         with c_input2:
             st.number_input("End", min_value=0, max_value=len(df), value=viz_end_idx, key="detailed_global_viz_end_input", on_change=update_global_slider_range)

    # Feature selection for visualization
    if "detailed_graph_ids" not in st.session_state:
        st.session_state.detailed_graph_ids = [0, 1, 2, 3, 4, 5]
        st.session_state.detailed_next_graph_id = 6

    if st.button("Add Graph", key="detailed_add_graph_btn"):
        st.session_state.detailed_graph_ids.append(st.session_state.detailed_next_graph_id)
        st.session_state.detailed_next_graph_id += 1

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

            # Apply range filter
            sliced_df = df.iloc[plot_start:plot_end]

            # Plot without downsampling
            fig = px.line(sliced_df, x=sliced_df.index, y=viz_cols, title=f"Telemetry Data - Graph {graph_id}")

            # Enhance hover with detailed stats (Index & Delta) for all points to match Manual UI capabilities
            if viz_cols and not sliced_df.empty:
                full_deltas = df[viz_cols].diff().iloc[plot_start:plot_end]
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
                            line_width=0,
                            annotation_text=label_str,
                            annotation_position="top left"
                        )

            st.plotly_chart(fig, use_container_width=True)
    
    if graphs_to_remove:
        for gid in graphs_to_remove:
            if gid in st.session_state.detailed_graph_ids:
                st.session_state.detailed_graph_ids.remove(gid)
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
        st.caption("Axis Settings")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            invert_x = st.checkbox("Invert X", value=False, key="detailed_invert_x")
        with col_ctrl2:
            invert_y = st.checkbox("Invert Y", value=False, key="detailed_invert_y")
        with col_ctrl3:
            invert_z = st.checkbox("Invert Z", value=False, key="detailed_invert_z")
        
        # Create windowed dataframe for trajectory plotting using Global Range
        start_idx = min(viz_start_idx, len(df) - 1)
        
        # Ensure indices are within bounds
        safe_end_idx = min(viz_end_idx, len(df))
        
        context_plot_df = pd.DataFrame(columns=df.columns)

        if safe_end_idx <= start_idx:
             # Handle empty range selection gracefully
             map_plot_df = pd.DataFrame(columns=df.columns)
             selected_time_idx = start_idx
        else:
             map_plot_df = df.iloc[start_idx:safe_end_idx]
             selected_time_idx = safe_end_idx - 1
             
             # Calculate extended range for context
             segment_len = safe_end_idx - start_idx
             padding = max(100, int(segment_len * 0.5)) 
             ext_start_idx = max(0, start_idx - padding)
             ext_end_idx = min(len(df), safe_end_idx + padding)
             context_plot_df = df.iloc[ext_start_idx:ext_end_idx]

        current_row = df.iloc[selected_time_idx]
        start_row = df.iloc[start_idx]
        map_data = []
        
        # Helper for Max Curvature Calculation
        def get_max_curvature_point(df_in, x_col, y_col, z_col=None, speed_col=None, label_type="Player"):
            if df_in.empty or len(df_in) <= 5:
                return None
            try:
                xs = df_in[x_col].values
                ys = df_in[y_col].values
                
                dx = np.gradient(xs)
                dy = np.gradient(ys)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                
                # Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
                numerator = np.abs(dx * ddy - dy * ddx)
                denominator = np.power(dx**2 + dy**2, 1.5)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    curvature = np.where(denominator > 1e-6, numerator / denominator, 0)
                
                curvature = np.nan_to_num(curvature)

                # Ignore low speed noise
                if speed_col and speed_col in df_in.columns:
                    s_vals = df_in[speed_col].values
                    curvature[s_vals < 10] = 0.0

                max_k = np.max(curvature)
                
                # If we detect a corner (significant curvature)
                if max_k > 0.002: # Threshold for corner detection
                    # Find point with minimum speed
                    if speed_col and speed_col in df_in.columns:
                        min_speed_idx_local = np.argmin(df_in[speed_col].values)
                        target_idx = df_in.index[min_speed_idx_local]
                        target_row = df_in.loc[target_idx]
                        marker_label = "Corner Apex (Min Speed)"
                    else:
                        # Fallback to max curvature if no speed
                        max_k_idx_local = np.argmax(curvature)
                        target_idx = df_in.index[max_k_idx_local]
                        target_row = df_in.loc[target_idx]
                        marker_label = "Max Curvature"
                    
                    p_geo = {
                        "x": target_row[x_col],
                        "y": target_row[y_col],
                        "Type": label_type,
                        "ID": f"{label_type} {marker_label}",
                        "Marker": marker_label,
                        "Speed": target_row[speed_col] if speed_col and speed_col in df_in.columns else None
                    }
                    if z_col and z_col in df_in.columns:
                        p_geo["z"] = target_row[z_col]
                    return p_geo
            except Exception:
                pass
            return None

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

            # Max Curvature Point
            max_curvature_data = get_max_curvature_point(
                map_plot_df, 
                "Graphics_player_pos_x", 
                "Graphics_player_pos_y", 
                "Graphics_player_pos_z" if has_player_pos_z else None,
                "Physics_speed_kmh",
                "Player"
            )
            if max_curvature_data:
                map_data.append(max_curvature_data)

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

            # Expert Max Curvature Point
            e_max_curvature_data = get_max_curvature_point(
                map_plot_df,
                "expert_optimal_player_pos_x",
                "expert_optimal_player_pos_y",
                "expert_optimal_player_pos_z" if has_expert_pos_z else None,
                "expert_optimal_speed",  # Use dedicated column
                "Expert"
            )
            if e_max_curvature_data:
                map_data.append(e_max_curvature_data)
        
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
                    symbol_map={"Start": "diamond", "End": "circle", "Max Curvature": "x"}
                )
                fig_map.update_traces(marker=dict(size=5))
                
                scene_dict = dict(
                    aspectmode='data',
                    # dragmode='turntable',
                    camera=dict(
                        projection=dict(type='orthographic'),
                        up=dict(x=0, y=0, z=1)  # Fix Z-axis as up for easier yaw rotation
                    )
                )
                if invert_x: scene_dict['xaxis'] = dict(autorange="reversed")
                if invert_y: scene_dict['yaxis'] = dict(autorange="reversed")
                if invert_z: scene_dict['zaxis'] = dict(autorange="reversed")
                fig_map.update_layout(scene=scene_dict, dragmode='turntable')
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
                    symbol_map={"Start": "x", "End": "circle", "Max Curvature": "star"}
                )
                if invert_x: fig_map.update_xaxes(autorange="reversed")
                if invert_y: fig_map.update_yaxes(autorange="reversed")

            # Add Trajectories
            # Player
            if has_player_pos:
                # Add Context (Extended Trajectory) first so it renders below
                if not context_plot_df.empty:
                    if use_3d and has_player_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=context_plot_df["Graphics_player_pos_x"], 
                            y=context_plot_df["Graphics_player_pos_y"],
                            z=context_plot_df["Graphics_player_pos_z"],
                            mode="lines",
                            name="Player (Context)",
                            line=dict(color="magenta", width=3),
                            opacity=0.3,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=context_plot_df["Graphics_player_pos_x"], 
                            y=context_plot_df["Graphics_player_pos_y"],
                            mode="lines",
                            name="Player (Context)",
                            line=dict(color="magenta", width=2),
                            opacity=0.3,
                            showlegend=True
                        ))

                # Add Current Segment Trajectory
                if use_3d and has_player_pos_z:
                    fig_map.add_trace(go.Scatter3d(
                        x=map_plot_df["Graphics_player_pos_x"], 
                        y=map_plot_df["Graphics_player_pos_y"],
                        z=map_plot_df["Graphics_player_pos_z"],
                        mode="lines",
                        name="Player Trajectory",
                        line=dict(color="green", width=5),
                        opacity=1.0,
                        showlegend=True
                    ))
                else:
                    fig_map.add_trace(go.Scatter(
                        x=map_plot_df["Graphics_player_pos_x"], 
                        y=map_plot_df["Graphics_player_pos_y"],
                        mode="lines",
                        name="Player Trajectory",
                        line=dict(color="green", width=3),
                        opacity=1.0,
                        showlegend=True
                    ))

            # Expert
            if has_expert_pos:
                # Context
                if not context_plot_df.empty:
                    if use_3d and has_expert_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=context_plot_df["expert_optimal_player_pos_x"], 
                            y=context_plot_df["expert_optimal_player_pos_y"],
                            z=context_plot_df["expert_optimal_player_pos_z"],
                            mode="lines",
                            name="Expert (Context)",
                            line=dict(color="orange", width=3),
                            opacity=0.3,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=context_plot_df["expert_optimal_player_pos_x"], 
                            y=context_plot_df["expert_optimal_player_pos_y"],
                            mode="lines",
                            name="Expert (Context)",
                            line=dict(color="orange", width=2),
                            opacity=0.3,
                            showlegend=True
                        ))

                # Segment
                if use_3d and has_expert_pos_z:
                    fig_map.add_trace(go.Scatter3d(
                        x=map_plot_df["expert_optimal_player_pos_x"], 
                        y=map_plot_df["expert_optimal_player_pos_y"],
                        z=map_plot_df["expert_optimal_player_pos_z"],
                        mode="lines",
                        name="Expert Trajectory",
                        line=dict(color="blue", width=5),
                        opacity=1.0,
                        showlegend=True
                    ))
                else:
                    fig_map.add_trace(go.Scatter(
                        x=map_plot_df["expert_optimal_player_pos_x"], 
                        y=map_plot_df["expert_optimal_player_pos_y"],
                        mode="lines",
                        name="Expert Trajectory",
                        line=dict(color="blue", width=3),
                        opacity=1.0,
                        showlegend=True
                    ))
            
            # Opponents
            for i in range(1, 6):
                opp_x_col = f"Opponent_{i}_pos_x"
                opp_y_col = f"Opponent_{i}_pos_y"
                opp_z_col = f"Opponent_{i}_pos_z"
                
                if opp_x_col in df.columns and opp_y_col in df.columns:
                    # Filter out inactive (0,0) points for cleaner trajectories
                    
                    # Context
                    if not context_plot_df.empty:
                        opp_ctx = context_plot_df[(context_plot_df[opp_x_col] != 0) | (context_plot_df[opp_y_col] != 0)]
                        if not opp_ctx.empty:
                            if use_3d and opp_z_col in df.columns:
                                fig_map.add_trace(go.Scatter3d(
                                    x=opp_ctx[opp_x_col], 
                                    y=opp_ctx[opp_y_col],
                                    z=opp_ctx[opp_z_col],
                                    mode="lines",
                                    name=f"Opponent {i} (Context)",
                                    line=dict(color="aqua", width=3),
                                    opacity=0.3,
                                    showlegend=True
                                ))
                            else:
                                fig_map.add_trace(go.Scatter(
                                    x=opp_ctx[opp_x_col], 
                                    y=opp_ctx[opp_y_col],
                                    mode="lines",
                                    name=f"Opponent {i} (Context)",
                                    line=dict(color="aqua", width=2),
                                    opacity=0.3,
                                    showlegend=True
                                ))

                    # Segment
                    opp_df = map_plot_df[(map_plot_df[opp_x_col] != 0) | (map_plot_df[opp_y_col] != 0)]
                    if not opp_df.empty:
                        if use_3d and opp_z_col in df.columns:
                            fig_map.add_trace(go.Scatter3d(
                                x=opp_df[opp_x_col], 
                                y=opp_df[opp_y_col],
                                z=opp_df[opp_z_col],
                                mode="lines",
                                name=f"Opponent {i} Trajectory",
                                line=dict(color="red", width=5),
                                opacity=1.0,
                                showlegend=True
                            ))
                        else:
                            fig_map.add_trace(go.Scatter(
                                x=opp_df[opp_x_col], 
                                y=opp_df[opp_y_col],
                                mode="lines",
                                name=f"Opponent {i} Trajectory",
                                line=dict(color="red", width=3),
                                opacity=1.0,
                                showlegend=True
                            ))

            if not use_3d:
                fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
            
            fig_map.update_layout(uirevision=session_id, height=800)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No active cars found at this timestamp.")
    else:
        st.info("Position data (Graphics_player_pos_x/y) not available in this dataset.")

    # List View (Inside Tab 1)
    if st.toggle("Show Current Session Annotations List"):
        st.subheader("Current Session Annotations List")
        if st.session_state.current_annotations:
            display_data = []
            
            # Determine filter range if a segment is selected
            filter_range = None
            if selected_option != "Create New":
                sel_ann = st.session_state.current_annotations[selected_option]
                filter_range = (sel_ann.start_index, sel_ann.end_index)

            for i, ann in enumerate(st.session_state.current_annotations):
                # Apply filter if set
                if filter_range:
                    if not (ann.start_index >= filter_range[0] and ann.end_index <= filter_range[1]):
                        continue

                d = ann.to_dict()
                d["Annotation ID"] = i
                d["labels"] = ", ".join(get_display_labels(ann.labels))
                if "telemetry_data" in d:
                    del d["telemetry_data"]
                display_data.append(d)
            
            if display_data:
                st.dataframe(pd.DataFrame(display_data).set_index("Annotation ID"), use_container_width=True)
            else:
                st.info("No segments found inside the selected annotation.")

            if st.button("Delete All Segments for Session", type="primary", key="detailed_btn_del_all_seg"):
                st.session_state.detailed_show_delete_all_confirm = True
            
            if st.session_state.get("detailed_show_delete_all_confirm", False):
                st.warning(f"⚠️ Are you sure you want to DELETE ALL {len(st.session_state.current_annotations)} segments for session '{session_id}'? This cannot be undone.")
                col_confirm_del, col_cancel_del = st.columns(2)
                
                with col_confirm_del:
                    if st.button("Yes, Delete All", key="detailed_confirm_del_all"):
                        st.session_state.current_annotations = []
                        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                        st.session_state.detailed_show_delete_all_confirm = False
                        st.success(f"All annotations for session {session_id} have been deleted.")
                        st.rerun()
                
                with col_cancel_del:
                    if st.button("Cancel", key="detailed_cancel_del_all"):
                        st.session_state.detailed_show_delete_all_confirm = False
                        st.rerun()
        else:
            st.info("No annotations added yet.")
        
    if st.button("Force Save All to Zarr", key="detailed_force_save_zarr"):
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
