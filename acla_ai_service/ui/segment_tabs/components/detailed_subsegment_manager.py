import streamlit as st
import uuid
import copy
from ..shared import save_annotations, AnnotatedSegment, get_display_labels, LABEL_MAPPING, MAIN_LABEL_GUIDELINES

def render_subsegment_manager(df, session_id, selected_annotation_key):
    """
    Renders UI for managing sub-segments under an existing parent segment.
    """
    st.markdown("---")
    st.subheader("Manage Sub-Segment")
    
    if "current_annotations" not in st.session_state or not st.session_state.current_annotations:
        st.info("No parent segments available. Create a segment first.")
        return
        
    # Determine the parent segment based on the active selection in detailed_annotation_manager
    selected_index = st.session_state.get("detailed_annotation_selector", None)
    
    if selected_index is None or selected_index >= len(st.session_state.current_annotations):
        st.info("Please select a valid primary segment in 'Manage Annotations' to manage sub-segments.")
        return
        
    parent_seg = st.session_state.current_annotations[selected_index]
    parent_id = parent_seg.id
    
    # Check if the selected segment is already a sub-segment (optional logic based on your needs)
    if getattr(parent_seg, 'parent_id', None):
        st.info("The selected segment is already a sub-segment. Select a primary segment to manage sub-segments.")
        return

    # Find existing sub-segments for this parent
    sub_segments = [seg for seg in st.session_state.current_annotations if getattr(seg, 'parent_id', None) == parent_id]
    
    subsegment_options = ["Create New Sub-Segment"] + [f"{i}: {', '.join(get_display_labels(seg.labels))} ({seg.start_index}-{seg.end_index})" for i, seg in enumerate(sub_segments)]
    
    selected_subsegment_str = st.selectbox("Select Sub-Segment", options=subsegment_options, key="manage_subsegment_selector")
    
    is_new = selected_subsegment_str == "Create New Sub-Segment"
    selected_sub_idx = -1 if is_new else int(selected_subsegment_str.split(":")[0])
    selected_sub_seg = None if is_new else sub_segments[selected_sub_idx]

    p_start = parent_seg.start_index or 0
    p_end = parent_seg.end_index or len(df) - 1

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Parent Segment:** {', '.join(get_display_labels(parent_seg.labels))}")
        st.caption(f"Parent Range: {p_start} to {p_end}")
        
        default_start = p_start if is_new else (selected_sub_seg.start_index if getattr(selected_sub_seg, 'start_index', None) is not None else p_start)
        default_end = min(p_start + 10, p_end) if is_new else (selected_sub_seg.end_index if getattr(selected_sub_seg, 'end_index', None) is not None else p_end)
        
        input_key_suffix = selected_sub_seg.id if not is_new else "new"
        
        sub_start = st.number_input("Sub-Segment Start", min_value=p_start, max_value=p_end, value=default_start, key=f"sub_start_{input_key_suffix}")
        sub_end = st.number_input("Sub-Segment End", min_value=p_start, max_value=p_end, value=default_end, key=f"sub_end_{input_key_suffix}")
        
    with col2:
        default_labels = [] if is_new else selected_sub_seg.labels
        sub_labels = st.multiselect(
            "Sub-Segment Labels",
            options=list(LABEL_MAPPING.keys()),
            default=default_labels,
            format_func=lambda x: LABEL_MAPPING.get(str(x), str(x)),
            key=f"sub_labels_{input_key_suffix}"
        )
        
        default_notes = "" if is_new else getattr(selected_sub_seg, 'notes', "")
        sub_notes = st.text_area("Sub-Segment Notes (Optional)", value=default_notes, key=f"sub_notes_{input_key_suffix}")
        
        def save_sub_segment_callback():
            s_start = st.session_state[f"sub_start_{input_key_suffix}"]
            s_end = st.session_state[f"sub_end_{input_key_suffix}"]
            s_labels = st.session_state[f"sub_labels_{input_key_suffix}"]
            s_notes = st.session_state[f"sub_notes_{input_key_suffix}"]
            if s_start >= s_end:
                st.toast("Error: Start index must be less than end index.", icon="❌")
                return
            if not s_labels:
                st.toast("Error: Please select at least one label.", icon="❌")
                return
                
            if is_new:
                new_sub_id = str(uuid.uuid4())
                new_sub_seg = AnnotatedSegment(
                    id=new_sub_id,
                    labels=s_labels,
                    segment_length=s_end - s_start,
                    start_index=s_start,
                    end_index=s_end,
                    notes=s_notes,
                    parent_id=parent_id
                )
                st.session_state.current_annotations.append(new_sub_seg)
                st.toast("Sub-segment added successfully!", icon="✅")
            else:
                for idx, ann in enumerate(st.session_state.current_annotations):
                    if ann.id == selected_sub_seg.id:
                        st.session_state.current_annotations[idx].start_index = s_start
                        st.session_state.current_annotations[idx].end_index = s_end
                        st.session_state.current_annotations[idx].labels = s_labels
                        st.session_state.current_annotations[idx].notes = s_notes
                        st.session_state.current_annotations[idx].segment_length = s_end - s_start
                        break
                st.toast("Sub-segment updated successfully!", icon="✅")
            
            st.session_state.has_unsaved_changes = True
            save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)

            # Update visualization range to match parent segment
            st.session_state.detailed_global_viz_range = (p_start, p_end)
            st.session_state.detailed_global_viz_start_input = p_start
            st.session_state.detailed_global_viz_end_input = p_end

        def delete_sub_segment_callback():
            if not is_new and selected_sub_seg:
                st.session_state.current_annotations = [ann for ann in st.session_state.current_annotations if ann.id != selected_sub_seg.id]
                st.session_state.has_unsaved_changes = True
                save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
                st.toast("Sub-segment deleted successfully!", icon="✅")

        c1, c2 = st.columns(2)
        with c1:
            st.button("Save Sub-Segment" if is_new else "Update Sub-Segment", use_container_width=True, type="primary", on_click=save_sub_segment_callback)
        with c2:
            if not is_new:
                st.button("Delete", use_container_width=True, type="secondary", on_click=delete_sub_segment_callback)
