import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .manual_classifier_check import render_classifier_probability_check
from .manual_feature_calculator import render_feature_calculator
from ..shared import (
    save_annotations, get_display_labels, build_segment,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    LABEL_CATEGORIES
)

def render_manual_annotation_manager(df, numeric_cols, session_id, selected_annotation_key):
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

    if "manual_annotation_selector" not in st.session_state or st.session_state.manual_annotation_selector not in annotation_options:
        st.session_state.manual_annotation_selector = annotation_options[0] if annotation_options else "Create New"

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
        ann = None
        submit_label = "Add Annotation"
        is_edit = False
    else:
        form_title = f"Edit Annotation #{selected_option}"
        ann = st.session_state.current_annotations[selected_option]
        default_start = ann.start_index
        default_end = ann.end_index
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
    
    # Two-step label selection: parent labels first, then sub-labels per parent.
    main_label_ids = LABEL_CATEGORIES.get("Main Labels", [])
    existing_label_ids = list(ann.labels) if is_edit else []

    parent_options = [LABEL_MAPPING[lid] for lid in main_label_ids if lid in LABEL_MAPPING]
    parent_defaults = [LABEL_MAPPING[lid] for lid in existing_label_ids
                       if lid in main_label_ids and lid in LABEL_MAPPING]

    selected_parent_names = st.multiselect(
        "Parent Labels",
        parent_options,
        default=parent_defaults,
        key=f"manual_form_parent_labels_{selected_option}"
    )

    form_labels = list(selected_parent_names)
    for parent_name in selected_parent_names:
        parent_id = LABEL_NAME_TO_ID.get(parent_name)
        sub_ids = LABEL_CATEGORIES.get(parent_id, []) if parent_id else []
        if not sub_ids:
            continue
        sub_options = [LABEL_MAPPING[lid] for lid in sub_ids if lid in LABEL_MAPPING]
        sub_defaults = [LABEL_MAPPING[lid] for lid in existing_label_ids
                        if lid in sub_ids and lid in LABEL_MAPPING]
        selected_sub_names = st.multiselect(
            f"Sub-labels for {parent_name}",
            sub_options,
            default=sub_defaults,
            key=f"manual_form_sub_labels_{parent_id}_{selected_option}"
        )
        form_labels.extend(selected_sub_names)

    # Segment Type is always enabled (independent of parent selection)
    segment_type_ids = LABEL_CATEGORIES.get("Segment Type", [])
    if segment_type_ids:
        segment_type_options = [LABEL_MAPPING[lid] for lid in segment_type_ids if lid in LABEL_MAPPING]
        segment_type_defaults = [LABEL_MAPPING[lid] for lid in existing_label_ids
                                 if lid in segment_type_ids and lid in LABEL_MAPPING]
        selected_segment_type_names = st.multiselect(
            "Segment Type",
            segment_type_options,
            default=segment_type_defaults,
            key=f"manual_form_segment_type_{selected_option}"
        )
        form_labels.extend(selected_segment_type_names)

    # Feature Change Calculator
    render_feature_calculator(df, numeric_cols, form_start, form_end, selected_option)

    # Classifier Probability Check
    render_classifier_probability_check(df, form_start, form_end, LABEL_MAPPING)



    # Form Actions
    col_actions = st.columns([1, 1, 1, 3])
    def handle_submit(s_start, s_end, s_labels):
        if s_start >= s_end:
            st.session_state.temp_error = "Start index must be less than end index."
            return
        if not s_labels:
            st.session_state.temp_error = "Please select at least one label."
            return
        
        label_ids = [LABEL_NAME_TO_ID[l] for l in s_labels if l in LABEL_NAME_TO_ID]

        if is_edit:
            # Update existing — keep the same id so downstream refs stay stable.
            ann = st.session_state.current_annotations[selected_option]
            updated = build_segment(
                df,
                start=int(s_start), end=int(s_end), label_ids=label_ids,
                notes=ann.notes, parent_id=ann.parent_id,
                chunk_index=ann.chunk_index, id=ann.id,
            )
            st.session_state.current_annotations[selected_option] = updated
            st.session_state.temp_success = "Annotation updated!"
        else:
            annotation = build_segment(
                df,
                start=int(s_start), end=int(s_end), label_ids=label_ids,
                chunk_index=session_id,
            )
            st.session_state.current_annotations.append(annotation)
            st.session_state.temp_success = "Annotation added!"
        
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
        st.rerun()

    with col_actions[0]:
        if st.button(submit_label, type="primary", key=f"manual_submit_{selected_option}"):
            handle_submit(form_start, form_end, form_labels)
    
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
                        from app.ml.segment_classifier.service import segment_classifier
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
            st.dataframe(pd.DataFrame(display_data), width='stretch')

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
        
    if st.button("Force Save All"):
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)