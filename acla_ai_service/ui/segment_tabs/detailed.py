import copy
from typing import Iterable

import pandas as pd
import streamlit as st

from .shared import (
    AnnotatedSegment,
    get_available_sessions,
    load_annotations,
    load_session_segments,
)


def _root_segment_indices(annotations: list[AnnotatedSegment]) -> list[int]:
    existing_ids = {
        getattr(annotation, "id", None)
        for annotation in annotations
        if getattr(annotation, "id", None)
    }
    return [
        index
        for index, annotation in enumerate(annotations)
        if not getattr(annotation, "parent_id", None)
        or getattr(annotation, "parent_id", None) not in existing_ids
    ]


def _annotation_signature(annotation: AnnotatedSegment) -> tuple:
    return (
        getattr(annotation, "id", None),
        getattr(annotation, "parent_id", None),
        getattr(annotation, "start_index", None),
        getattr(annotation, "end_index", None),
        getattr(annotation, "segment_length", None),
        getattr(annotation, "chunk_index", None),
        tuple(getattr(annotation, "labels", []) or []),
        getattr(annotation, "notes", None),
        getattr(annotation, "opponent_interaction", None),
        len(getattr(annotation, "telemetry_data", None) or []),
    )


def _annotations_signature(annotations: list[AnnotatedSegment]) -> tuple:
    return tuple(_annotation_signature(annotation) for annotation in annotations)


def _resolve_loaded_annotation_selection(
    annotations: list[AnnotatedSegment],
    requested_selection: int | None,
) -> int | None:
    root_indices = _root_segment_indices(annotations)
    if not root_indices:
        return None

    if not isinstance(requested_selection, int):
        return root_indices[0]

    for root_index in root_indices:
        if root_index >= requested_selection:
            return root_index

    return root_indices[-1]


def _clear_detailed_form_state() -> None:
    form_keys = [
        key for key in st.session_state.keys()
        if key.startswith("detailed_form_")
    ]
    for key in form_keys:
        st.session_state.pop(key, None)


def _segments_to_positioned_dataframe(
    segments: Iterable[AnnotatedSegment],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    max_position = 0

    for segment in segments:
        rows = getattr(segment, "telemetry_data", None) or []
        start = getattr(segment, "start_index", None)
        end = getattr(segment, "end_index", None)
        if end is not None:
            max_position = max(max_position, int(end))
        if not rows or start is None:
            continue

        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame.index = range(int(start), int(start) + len(frame))
        frames.append(frame)
        max_position = max(max_position, int(start) + len(frame))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.reindex(range(max_position + 1))


def _safe_load_annotations(
    session_id: str,
    selected_annotation_key: str,
) -> list[AnnotatedSegment]:
    try:
        return load_annotations(session_id, selected_annotation_key)
    except Exception:
        return []


def _set_visualization_range(start: int, end: int, max_index: int) -> None:
    safe_start = max(0, min(int(start), max_index))
    safe_end = max(safe_start, min(int(end), max_index))
    st.session_state.detailed_global_viz_range = (safe_start, safe_end)
    st.session_state.detailed_global_viz_start_input = safe_start
    st.session_state.detailed_global_viz_end_input = safe_end


def _set_selection_visualization(
    annotations: list[AnnotatedSegment],
    selected_annotation: int | None,
    max_index: int,
) -> None:
    if isinstance(selected_annotation, int) and selected_annotation < len(annotations):
        annotation = annotations[selected_annotation]
        _set_visualization_range(
            getattr(annotation, "start_index", 0) or 0,
            getattr(annotation, "end_index", max_index) or max_index,
            max_index,
        )
    else:
        _set_visualization_range(0, min(100, max_index), max_index)


def _segment_session_counts(
    selected_session_key: str,
    available_sessions: list[str],
) -> dict[str, int]:
    return {
        session_id: len(load_session_segments(selected_session_key, session_id))
        for session_id in available_sessions
    }


def render_detailed_labeling(
    selected_annotation_key,
    selected_session_key,
    available_sessions,
):
    """Render segment-level detailed annotation."""
    annotated_sessions = set(get_available_sessions(selected_annotation_key))
    segment_counts = _segment_session_counts(selected_session_key, available_sessions)
    segment_sessions = [
        session_id for session_id, count in segment_counts.items() if count > 0
    ]
    if not segment_sessions:
        st.subheader("Detailed Segment Annotation")
        st.error(
            "Detailed Annotation only works on session chunks that contain "
            "segments. This input dataset has no segment chunks."
        )
        return
    session_options = segment_sessions

    def format_session_option(session_id: str) -> str:
        status = "✅" if session_id in annotated_sessions else "⭕"
        count = segment_counts.get(session_id, 0)
        if count:
            return f"{status} {session_id} | {count} segments"
        return f"{status} {session_id}"

    st.subheader("Detailed Segment Annotation")
    st.caption(
        "Refine imported parent segments into detailed labels and child sub-segments."
    )

    top_cols = st.columns([1.2, 1, 1, 1])
    with top_cols[0]:
        if st.session_state.get("detailed_session_selector") not in session_options:
            st.session_state.detailed_session_selector = session_options[0]

        session_id = st.selectbox(
            "Session / segment chunk",
            options=session_options,
            format_func=format_session_option,
            key="detailed_session_selector",
        )
    with top_cols[1]:
        st.metric("Input segments", segment_counts.get(session_id, 0))
    with top_cols[2]:
        st.metric("Chunks", len(session_options))
    with top_cols[3]:
        st.metric("Annotated chunks", len(annotated_sessions))

    st.info(f"Importing session segments from `{selected_session_key}`.")
    input_segments = load_session_segments(selected_session_key, session_id)
    if not input_segments:
        st.error("Selected session chunk has no segments.")
        return

    saved_annotations = _safe_load_annotations(session_id, selected_annotation_key)
    if input_segments and not saved_annotations:
        current_annotations = copy.deepcopy(input_segments)
    else:
        current_annotations = saved_annotations
    st.session_state.current_annotations = current_annotations

    loaded_signature = (
        selected_session_key,
        selected_annotation_key,
        session_id,
        _annotations_signature(current_annotations),
    )
    data_reloaded = st.session_state.get("detailed_loaded_signature") != loaded_signature
    pending_selection = st.session_state.pop("pending_detailed_selection", None)
    requested_selection = (
        pending_selection
        if pending_selection is not None
        else st.session_state.get("detailed_annotation_selector")
    )
    resolved_selection = _resolve_loaded_annotation_selection(
        current_annotations,
        requested_selection,
    )
    if resolved_selection is not None:
        selection_changed = (
            st.session_state.get("detailed_annotation_selector") != resolved_selection
        )
        st.session_state.detailed_annotation_selector = resolved_selection
        if data_reloaded or selection_changed:
            _clear_detailed_form_state()

    st.session_state.detailed_loaded_signature = loaded_signature
    st.session_state.last_session_id = session_id
    st.session_state.last_annotation_key = selected_annotation_key
    if data_reloaded:
        st.session_state.pop("detailed_last_focus_segment", None)
        st.session_state.pop("detailed_focus_segment_selector", None)

    df = _segments_to_positioned_dataframe(input_segments)

    if df.empty:
        st.warning("Selected chunk has no telemetry rows to display.")
        st.stop()

    max_index = max(0, len(df) - 1)
    if data_reloaded or pending_selection is not None:
        _set_selection_visualization(
            current_annotations,
            st.session_state.get("detailed_annotation_selector"),
            max_index,
        )

    root_count = len(_root_segment_indices(current_annotations))
    child_count = max(0, len(current_annotations) - root_count)
    st.write(
        f"Loaded {len(df)} telemetry positions from `{session_id}` "
        f"({root_count} parent segments, {child_count} child segments)."
    )

    if "Static_track" in df.columns:
        track_names = df["Static_track"].dropna()
        if not track_names.empty:
            st.markdown(f"**Track:** {track_names.iloc[0]}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    default_cols = ["speed_kmh", "gas", "brake", "steer_angle"]

    viz_col, manager_col = st.columns([3, 1])

    from .components.detailed_annotation_manager import render_annotation_manager

    with manager_col:
        render_annotation_manager(df, session_id, selected_annotation_key, numeric_cols)

        from .components.detailed_subsegment_manager import render_subsegment_manager

        render_subsegment_manager(df, session_id, selected_annotation_key)

    selected_annotation = st.session_state.get("detailed_annotation_selector")
    root_indices = _root_segment_indices(current_annotations)
    if "detailed_global_viz_range" not in st.session_state:
        if selected_annotation in root_indices:
            annotation = current_annotations[selected_annotation]
            _set_visualization_range(
                getattr(annotation, "start_index", 0) or 0,
                getattr(annotation, "end_index", max_index) or max_index,
                max_index,
            )
        else:
            _set_visualization_range(0, min(100, max_index), max_index)
    else:
        start, end = st.session_state.detailed_global_viz_range
        _set_visualization_range(start, end, max_index)

    with viz_col:
        viz_scroll = st.container(height=1200)

    with viz_scroll:
        st.markdown("---")
        st.caption("Visualization Range")

        def update_global_inputs_from_slider():
            start, end = st.session_state.detailed_global_viz_range
            st.session_state.detailed_global_viz_start_input = start
            st.session_state.detailed_global_viz_end_input = end

        def update_global_slider_range():
            start = st.session_state.get("detailed_global_viz_start_input", 0)
            end = st.session_state.get("detailed_global_viz_end_input", 0)
            if start <= end:
                st.session_state.detailed_global_viz_range = (start, end)

        col_global_slider, col_global_inputs = st.columns([3, 1])
        with col_global_slider:
            viz_start_idx, viz_end_idx = st.slider(
                "Select Range",
                min_value=0,
                max_value=max_index,
                key="detailed_global_viz_range",
                on_change=update_global_inputs_from_slider,
                label_visibility="collapsed",
            )

        with col_global_inputs:
            c_input1, c_input2 = st.columns(2)
            with c_input1:
                st.number_input(
                    "Start",
                    min_value=0,
                    max_value=max_index,
                    key="detailed_global_viz_start_input",
                    on_change=update_global_slider_range,
                )
            with c_input2:
                st.number_input(
                    "End",
                    min_value=0,
                    max_value=max_index,
                    key="detailed_global_viz_end_input",
                    on_change=update_global_slider_range,
                )

        from .components.detailed_feature_visualization import render_feature_visualization

        render_feature_visualization(
            df,
            viz_start_idx,
            viz_end_idx,
            session_id,
            numeric_cols,
            default_cols,
        )

        from .components.detailed_track_map import render_track_map

        render_track_map(df, viz_start_idx, viz_end_idx, session_id)

        from .components.detailed_list_view import render_list_view

        render_list_view(session_id, selected_annotation_key)
