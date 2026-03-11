import streamlit as st
import uuid
import copy
from ..shared import save_annotations, AnnotatedSegment, get_display_labels, LABEL_MAPPING, MAIN_LABEL_GUIDELINES

def render_subsegment_manager(df, session_id, selected_annotation_key):
    """
    Renders UI for adding sub-segments under an existing parent segment.
    """
    st.markdown("---")
    st.subheader("Add Sub-Segment")
    
    if "current_annotations" not in st.session_state or not st.session_state.current_annotations:
        st.info("No parent segments available. Create a segment first.")
        return
        
    # Determine the parent segment based on the active selection in detailed_annotation_manager
    selected_index = st.session_state.get("detailed_annotation_selector", None)
    
    if selected_index is None or selected_index >= len(st.session_state.current_annotations):
        st.info("Please select a valid primary segment in 'Manage Annotations' to add sub-segments.")
        return
        
    parent_seg = st.session_state.current_annotations[selected_index]
    parent_id = parent_seg.id
    
    # Check if the selected segment is already a sub-segment (optional logic based on your needs)
    if parent_seg.parent_id:
        st.info("The selected segment is already a sub-segment. Select a primary segment to add sub-segments.")
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Parent Segment:** {', '.join(get_display_labels(parent_seg.labels))}")
        st.caption(f"Parent Range: {parent_seg.start_index} to {parent_seg.end_index}")
        
        p_start = parent_seg.start_index or 0
        p_end = parent_seg.end_index or len(df) - 1
        
        sub_start = st.number_input("Sub-Segment Start", min_value=p_start, max_value=p_end, value=p_start, key="sub_start")
        sub_end = st.number_input("Sub-Segment End", min_value=p_start, max_value=p_end, value=min(p_start + 10, p_end), key="sub_end")
            
    with col2:
        sub_labels = st.multiselect(
            "Sub-Segment Labels",
            options=list(LABEL_MAPPING.keys()),
            format_func=lambda x: LABEL_MAPPING.get(str(x), str(x)),
            key="sub_labels"
        )
        
        sub_notes = st.text_area("Sub-Segment Notes (Optional)", key="sub_notes")
        
        if st.button("Add Sub-Segment", use_container_width=True, type="primary"):
            if sub_start >= sub_end:
                st.error("Start index must be less than end index.")
                return
            if not sub_labels:
                st.error("Please select at least one label.")
                return
                
            new_sub_id = str(uuid.uuid4())
            new_sub_seg = AnnotatedSegment(
                id=new_sub_id,
                labels=sub_labels,
                segment_length=sub_end - sub_start,
                start_index=sub_start,
                end_index=sub_end,
                notes=sub_notes,
                parent_id=parent_id
            )
            
            st.session_state.current_annotations.append(new_sub_seg)
            st.session_state.has_unsaved_changes = True
            save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
            st.success("Sub-segment added successfully!")
            st.rerun()
