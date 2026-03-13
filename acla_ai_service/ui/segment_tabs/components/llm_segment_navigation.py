import streamlit as st
from typing import List, Dict, Any, Tuple, Callable

def handle_segment_selection_and_filtering(
    available_sessions: List[str],
    selected_key: str,
    load_annotated_segments: Callable[[str, str], List[Dict[str, Any]]],
    label_mapping: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """
    Handles the 'Select Session (Chunk)', 'Filter by Labels', and 'Select Segment' UI.
    Returns:
        filtered_segments (List[Dict]): the currently filtered segments.
        selected_session (str): the chosen session.
        all_segments (List[Dict]): the un-filtered segments for the chosen session.
    """
    col1, col2 = st.columns(2)
    selected_session = None
    
    with col1:
        if available_sessions:
            selected_session = st.selectbox("Select Session (Chunk)", options=available_sessions)
        else:
            st.warning("No sessions found in this dataset.")
            return [], None, []
    
    if not selected_key or not selected_session:
        st.warning("Please select a dataset and a session chunk to begin.")
        return [], None, []

    state_key = f"{selected_key}_{selected_session}"
    
    # Session State Initialization
    if "segments" not in st.session_state or st.session_state.get("current_state_key") != state_key:
        with st.spinner(f"Loading segments from {selected_session}..."):
            st.session_state.segments = load_annotated_segments(selected_session, selected_key)
            st.session_state.current_state_key = state_key
            st.session_state.current_index = 0
            st.rerun()
            
    segments = st.session_state.segments
    
    # --- Filter Configuration ---
    all_segment_labels = set()
    label_counts = {}
    
    if segments:
        for seg in segments:
            for l in seg.get("labels", []):
                 l_str = str(l)
                 all_segment_labels.add(l_str)
                 label_counts[l_str] = label_counts.get(l_str, 0) + 1
        
        # Create options list with counts
        filter_options = ["All"]
        sorted_label_ids = sorted(list(all_segment_labels), key=lambda x: label_counts[x], reverse=True)
        option_to_id = {"All": None}
        
        for l_id in sorted_label_ids:
            name = label_mapping.get(l_id, l_id)
            count = label_counts[l_id]
            option_str = f"{name} ({count})"
            filter_options.append(option_str)
            option_to_id[option_str] = l_id
            
        with col2:
            selected_filter_options = st.multiselect("Filter by Labels", options=filter_options[1:])
        target_label_ids = [option_to_id[opt] for opt in selected_filter_options if opt in option_to_id]
    else:
        target_label_ids = []
        with col2:
            st.info("No labels found to filter.")

    # Apply Filter
    filtered_segments = []
    if target_label_ids:
        for s in segments:
            segment_labels = [str(l) for l in s.get("labels", [])]
            if all(t_id in segment_labels for t_id in target_label_ids):
                filtered_segments.append(s)
    else:
        filtered_segments = segments
    
    if "last_filter_ids" not in st.session_state:
        st.session_state.last_filter_ids = target_label_ids
        
    if st.session_state.last_filter_ids != target_label_ids:
        st.session_state.current_index = 0
        st.session_state.last_filter_ids = target_label_ids
        
    total_segments = len(filtered_segments)
    
    if total_segments == 0:
        if segments:
            st.warning("No segments match the current filter.")
        else:
            st.warning(f"No annotated segments found in chunk {selected_session}.")
        return [], selected_session, segments

    if st.session_state.current_index >= total_segments:
        st.session_state.current_index = 0
    
    # Render previously separated segment navigation inline
    render_segment_navigation(filtered_segments, total_segments, len(segments))
    
    return filtered_segments, selected_session, segments

def render_segment_navigation(filtered_segments: List[Dict[str, Any]], total_segments: int, total_unfiltered: int):
    segment_options = [f"Segment {i+1} of {total_segments} (ID: {seg.get('id', 'Unknown')})" for i, seg in enumerate(filtered_segments)]
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("Previous"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col_nav3:
        if st.button("Next"):
            st.session_state.current_index = min(total_segments - 1, st.session_state.current_index + 1)
            
    with col_nav2:
        selected_opt = st.selectbox(
            "Select Segment",
            options=segment_options,
            index=st.session_state.current_index,
            label_visibility="collapsed"
        )
        if selected_opt:
            st.session_state.current_index = segment_options.index(selected_opt)
            
        st.markdown(f"*(Filtered from {total_unfiltered} total)*")
