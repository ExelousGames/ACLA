import streamlit as st
from typing import Any, Callable, Dict, List, Tuple


def _unit_all_label_ids(unit: Dict[str, Any]) -> List[str]:
    out = list(unit["parent_label_ids"])
    for child in unit["children_label_ids"]:
        out.extend(child)
    return out


def handle_segment_selection_and_filtering(
    available_sessions: List[str],
    selected_key: str,
    load_units_fn: Callable[[str, str], List[Dict[str, Any]]],
    label_mapping: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """Session picker + label filter for training units.

    Returns (filtered_units, selected_session, all_units_in_session).
    """
    col1, col2 = st.columns(2)
    selected_session = None

    with col1:
        if available_sessions:
            selected_session = st.selectbox(
                "Select Session (Chunk)", options=available_sessions,
            )
        else:
            st.warning("No sessions found in this dataset.")
            return [], None, []

    if not selected_key or not selected_session:
        st.warning("Please select a dataset and a session chunk to begin.")
        return [], None, []

    state_key = f"{selected_key}_{selected_session}"
    if "units" not in st.session_state or st.session_state.get("current_state_key") != state_key:
        with st.spinner(f"Loading units from {selected_session}..."):
            st.session_state.units = load_units_fn(selected_session, selected_key)
            st.session_state.current_state_key = state_key
            st.session_state.current_index = 0
            st.rerun()

    units = st.session_state.units

    label_counts: Dict[str, int] = {}
    for u in units:
        for l in _unit_all_label_ids(u):
            label_counts[l] = label_counts.get(l, 0) + 1

    if label_counts:
        sorted_ids = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
        option_to_id: Dict[str, str] = {}
        options: List[str] = []
        for lid in sorted_ids:
            name = label_mapping.get(lid, lid)
            opt = f"{name} ({label_counts[lid]})"
            options.append(opt)
            option_to_id[opt] = lid

        with col2:
            selected_opts = st.multiselect("Filter by Labels", options=options)
        target_ids = [option_to_id[o] for o in selected_opts]
    else:
        target_ids = []
        with col2:
            st.info("No labels found to filter.")

    if target_ids:
        filtered = [
            u for u in units
            if all(t in _unit_all_label_ids(u) for t in target_ids)
        ]
    else:
        filtered = units

    if "last_filter_ids" not in st.session_state:
        st.session_state.last_filter_ids = target_ids
    if st.session_state.last_filter_ids != target_ids:
        st.session_state.current_index = 0
        st.session_state.last_filter_ids = target_ids

    total = len(filtered)
    if total == 0:
        if units:
            st.warning("No units match the current filter.")
        else:
            st.warning(f"No annotated units found in chunk {selected_session}.")
        return [], selected_session, units

    if st.session_state.current_index >= total:
        st.session_state.current_index = 0

    render_unit_navigation(filtered, total, len(units))
    return filtered, selected_session, units


def render_unit_navigation(
    filtered_units: List[Dict[str, Any]],
    total_units: int,
    total_unfiltered: int,
) -> None:
    options = [
        f"Unit {i + 1} of {total_units} (ID: {u.get('unit_id', 'Unknown')}, kind: {u.get('kind')})"
        for i, u in enumerate(filtered_units)
    ]

    col_prev, col_pick, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col_next:
        if st.button("Next"):
            st.session_state.current_index = min(
                total_units - 1, st.session_state.current_index + 1,
            )
    with col_pick:
        chosen = st.selectbox(
            "Select Unit",
            options=options,
            index=st.session_state.current_index,
            label_visibility="collapsed",
        )
        if chosen:
            st.session_state.current_index = options.index(chosen)
        st.markdown(f"*(Filtered from {total_unfiltered} total)*")
