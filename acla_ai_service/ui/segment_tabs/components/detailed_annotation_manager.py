import streamlit as st
import pandas as pd
from ..shared import (
    save_annotations, get_display_labels,
    LABEL_MAPPING, LABEL_NAME_TO_ID, AnnotatedSegment,
    LABEL_CATEGORIES
)

def render_annotation_manager(df, session_id, selected_annotation_key, numeric_cols):
        # --- Unified Annotation Management (MOVED UP) ---
        st.markdown("---")
        st.subheader("Manage Annotations")
    
        # Ensure annotations list exists
        if "current_annotations" not in st.session_state:
            st.session_state.current_annotations = []
            
        if not st.session_state.current_annotations:
            st.warning("No segments found in this session.")
            return
            
        # 1. Select Mode/Annotation
        existing_ids = {getattr(a, 'id', None) for a in st.session_state.current_annotations if getattr(a, 'id', None)}
        annotation_options = [
            i for i, ann in enumerate(st.session_state.current_annotations) 
            if not getattr(ann, 'parent_id', None) or getattr(ann, 'parent_id', None) not in existing_ids
        ]
        
        if not annotation_options:
            st.warning("No root segments found in this session.")
            return
        
    
    
        def format_func(option):
            ann = st.session_state.current_annotations[option]
            labels = ", ".join(get_display_labels(ann.labels))
            return f"#{option}: {labels} (Start: {ann.start_index}, End: {ann.end_index})"
    
        # Determine the default index for the selectbox
        default_index = 0
        if "last_detailed_selection" in st.session_state and st.session_state.last_detailed_selection in annotation_options:
            default_index = annotation_options.index(st.session_state.last_detailed_selection)
        
        selected_option = st.selectbox(
            "Select Action / Annotation",
            options=annotation_options,
            format_func=format_func,
            index=default_index,
            key="detailed_annotation_selector"
        )
        
        # Auto-update visualization range when selection changes
        if "last_detailed_selection" not in st.session_state:
            st.session_state.last_detailed_selection = None
        
        if selected_option != st.session_state.last_detailed_selection:
            st.session_state.last_detailed_selection = selected_option
            
            # Clear form inputs to force refresh of values
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith("detailed_form_")]
            for k in keys_to_clear:
                del st.session_state[k]
    
            # Auto-set the visualization range to show the segment
            ann_sel = st.session_state.current_annotations[selected_option]
            st.session_state.detailed_global_viz_start_input = ann_sel.start_index
            st.session_state.detailed_global_viz_end_input = ann_sel.end_index
            st.session_state.detailed_global_viz_range = (ann_sel.start_index, ann_sel.end_index)
            
        # Manual Annotation Logic
        input_min = 0
        input_max = len(df)-1
    
        # Existing Annotation Selected - Edit Mode
        ann = st.session_state.current_annotations[selected_option]
        form_title = f"Edit Annotation #{selected_option}"
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
            
            # Only show if the category (which is a parent ID) is selected in Main Labels, or if it is the "Segment Type" group
            if category == "Segment Type" or category in selected_main_ids:
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
    
        # Gemini AI Analysis
        from .detailed_gemini_analysis import render_gemini_analysis
        render_gemini_analysis(df, form_start, form_end, form_labels)
    
        # Feature Change Calculator
        from .detailed_feature_calculator import render_feature_calculator
        render_feature_calculator(df, form_start, form_end, numeric_cols, selected_option)
    
        # Form Actions
        col_actions = st.columns([1, 1, 1, 3])
        def update_selection_state(next_selection_key):
            st.session_state.last_detailed_selection = next_selection_key
            st.session_state.detailed_annotation_selector = next_selection_key
            
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith("detailed_form_")]
            for k in keys_to_clear:
                del st.session_state[k]
                
            if isinstance(next_selection_key, int) and next_selection_key < len(st.session_state.current_annotations):
                ann_sel = st.session_state.current_annotations[next_selection_key]
                st.session_state.detailed_global_viz_start_input = ann_sel.start_index
                st.session_state.detailed_global_viz_end_input = ann_sel.end_index
                st.session_state.detailed_global_viz_range = (ann_sel.start_index, ann_sel.end_index)
            else:
                 default_start = 0
                 default_end = min(100, len(df)-1)
                 st.session_state.detailed_global_viz_start_input = default_start
                 st.session_state.detailed_global_viz_end_input = default_end
                 st.session_state.detailed_global_viz_range = (default_start, default_end)
    
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
    
            # Update existing
            ann = st.session_state.current_annotations[selected_option]
            ann.start_index = int(s_start)
            ann.end_index = int(s_end)
            ann.segment_length = int(s_end - s_start)
            ann.labels = label_ids
            ann.telemetry_data = telemetry_data

            if go_next:
                root_options = [i for i, a in enumerate(st.session_state.current_annotations) if not getattr(a, 'parent_id', None)]
                try:
                    curr_idx = root_options.index(selected_option)
                    if curr_idx + 1 < len(root_options):
                        next_option = root_options[curr_idx + 1]
                        st.session_state.temp_success = f"Updated #{selected_option}. Moving to #{next_option}."
                        update_selection_state(next_option)
                    else:
                        st.session_state.temp_success = "Updated annotation. (End of list)"
                except ValueError:
                    st.session_state.temp_success = "Updated annotation."
            else:
                st.session_state.temp_success = "Annotation updated!"
            
            save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
    
        def handle_delete():
            """Delete the currently selected annotation, its children, and move to the next one."""
            if isinstance(selected_option, int) and selected_option < len(st.session_state.current_annotations):
                # Remove the annotation
                deleted_ann = st.session_state.current_annotations.pop(selected_option)
                labels = ", ".join(get_display_labels(deleted_ann.labels))
                
                # Remove any child segments
                original_len = len(st.session_state.current_annotations)
                st.session_state.current_annotations = [
                    ann for ann in st.session_state.current_annotations
                    if not (hasattr(ann, "parent_id") and getattr(ann, "parent_id") == deleted_ann.id)
                ]
                children_deleted = original_len - len(st.session_state.current_annotations)
                child_msg = f" and {children_deleted} child segment(s)" if children_deleted > 0 else ""
                
                # Determine next selection
                # We need to recalculate annotation_options because indexes might have changed
                root_options = [i for i, ann in enumerate(st.session_state.current_annotations) if not getattr(ann, 'parent_id', None)]

                if len(root_options) == 0:
                    # No annotations left
                    update_selection_state(0)
                    st.session_state.temp_success = f"Deleted annotation ({labels}){child_msg}. No annotations remaining."
                else:
                    # Try to select the same index, or the last available if we were at the end
                    next_idx = root_options[min(selected_option, len(root_options) - 1)]
                    update_selection_state(next_idx)
                    st.session_state.temp_success = f"Deleted annotation ({labels}){child_msg}. Moved to next annotation."
                
                # Save changes
                save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
    
        with col_actions[0]:
             st.button("Update & Next ⏭️", type="primary", key=f"detailed_submit_next_{selected_option}", on_click=handle_submit, args=(True,))
        
        with col_actions[1]:
            st.button(submit_label, type="secondary", key=f"detailed_submit_{selected_option}", on_click=handle_submit, args=(False,))
        
        with col_actions[2]:
            st.button("🗑️ Delete", type="secondary", key=f"detailed_delete_{selected_option}", on_click=handle_delete, help="Delete this annotation")
        
        if "temp_error" in st.session_state:
            st.error(st.session_state.temp_error)
            del st.session_state.temp_error
        if "temp_success" in st.session_state:
            st.success(st.session_state.temp_success)
            del st.session_state.temp_success
    
