import copy
import time

import streamlit as st

from .shared import (
    get_display_labels,
    load_annotations,
    load_session_segments,
    save_annotations,
)


def _safe_load_annotations(session_id: str, selected_annotation_key: str):
    try:
        return load_annotations(session_id, selected_annotation_key)
    except Exception:
        return []


def _parents_by_id(annotations, source_segments):
    parents = {
        getattr(segment, "id", None): segment
        for segment in source_segments
        if getattr(segment, "id", None)
    }
    parents.update({
        getattr(segment, "id", None): segment
        for segment in annotations
        if getattr(segment, "id", None)
    })
    return parents


def _missing_parent_labels(child, parents):
    parent = parents.get(getattr(child, "parent_id", None))
    if parent is None:
        return []

    child_labels = list(getattr(child, "labels", []) or [])
    child_label_set = set(child_labels)
    missing_labels = []
    for label in getattr(parent, "labels", []) or []:
        if label in child_label_set:
            continue
        missing_labels.append(label)
        child_label_set.add(label)
    return missing_labels


def _propagate_parent_labels(annotations, source_segments):
    parents = _parents_by_id(annotations, source_segments)
    children = [
        segment for segment in annotations
        if getattr(segment, "parent_id", None)
    ]
    missing_parent_refs = 0
    children_updated = 0
    labels_added = 0

    for child in children:
        if getattr(child, "parent_id", None) not in parents:
            missing_parent_refs += 1
            continue

        missing = _missing_parent_labels(child, parents)
        if not missing:
            continue

        child.labels = list(getattr(child, "labels", []) or []) + missing
        children_updated += 1
        labels_added += len(missing)

    return {
        "children": len(children),
        "missing_parent_refs": missing_parent_refs,
        "children_updated": children_updated,
        "labels_added": labels_added,
    }


def _load_annotation_chunk(selected_annotation_key, selected_session_key, session_id):
    source_segments = load_session_segments(selected_session_key, session_id)
    saved_annotations = _safe_load_annotations(session_id, selected_annotation_key)
    annotations = saved_annotations if saved_annotations else copy.deepcopy(source_segments)
    return annotations, source_segments, bool(saved_annotations)


def _chunk_preview(selected_annotation_key, selected_session_key, session_id):
    annotations, source_segments, has_saved_annotations = _load_annotation_chunk(
        selected_annotation_key,
        selected_session_key,
        session_id,
    )
    preview_annotations = copy.deepcopy(annotations)
    summary = _propagate_parent_labels(preview_annotations, source_segments)
    summary.update({
        "session_id": session_id,
        "segments": len(annotations),
        "source_segments": len(source_segments),
        "has_saved_annotations": has_saved_annotations,
    })
    return summary


def _format_session(summary):
    status = "OK" if summary["children_updated"] == 0 else "Needs update"
    source = "saved" if summary["has_saved_annotations"] else "input"
    return (
        f"{status} {summary['session_id']} | {summary['children_updated']} child "
        f"segment(s) need labels | {summary['segments']} {source} segment(s)"
    )


def _render_missing_label_examples(annotations, source_segments):
    parents = _parents_by_id(annotations, source_segments)
    rows = []
    for child in annotations:
        if not getattr(child, "parent_id", None):
            continue
        missing = _missing_parent_labels(child, parents)
        if not missing:
            continue
        rows.append({
            "child_id": getattr(child, "id", ""),
            "parent_id": getattr(child, "parent_id", ""),
            "range": f"{getattr(child, 'start_index', '')}-{getattr(child, 'end_index', '')}",
            "missing_labels": ", ".join(get_display_labels(missing)),
        })
        if len(rows) >= 25:
            break

    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")


def _apply_chunk(selected_annotation_key, selected_session_key, session_id):
    annotations, source_segments, _ = _load_annotation_chunk(
        selected_annotation_key,
        selected_session_key,
        session_id,
    )
    summary = _propagate_parent_labels(annotations, source_segments)
    if summary["children_updated"]:
        save_annotations(session_id, annotations, selected_annotation_key, silent=True)
    return summary


def render_parent_label_propagation(
    selected_annotation_key,
    selected_session_key,
    available_sessions,
):
    st.header("Parent Label Propagation")
    st.caption(
        "Append any missing parent segment labels to direct child sub-segments "
        "matched by parent_id."
    )

    segment_sessions = [
        session_id for session_id in available_sessions
        if load_session_segments(selected_session_key, session_id)
    ]
    if not segment_sessions:
        st.error(
            "Parent Label Propagation only works on input chunks that contain "
            "segments. Select a segment-output dataset as the source."
        )
        return

    summaries = {
        session_id: _chunk_preview(
            selected_annotation_key,
            selected_session_key,
            session_id,
        )
        for session_id in segment_sessions
    }
    needs_update = [
        summary for summary in summaries.values()
        if summary["children_updated"]
    ]
    missing_parent_refs = sum(
        summary["missing_parent_refs"] for summary in summaries.values()
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Chunks", len(segment_sessions))
    metric_cols[1].metric("Chunks needing updates", len(needs_update))
    metric_cols[2].metric(
        "Child segments needing labels",
        sum(summary["children_updated"] for summary in needs_update),
    )
    metric_cols[3].metric(
        "Parent labels to add",
        sum(summary["labels_added"] for summary in needs_update),
    )
    if missing_parent_refs:
        st.warning(
            f"{missing_parent_refs} child segment(s) reference a parent id "
            "that was not found in the selected input/output chunks."
        )

    previous_selection = st.session_state.get("parent_labels_session_selector")
    index = 0
    if previous_selection in segment_sessions:
        index = segment_sessions.index(previous_selection)

    col_select, col_apply_all = st.columns([2, 1])
    with col_select:
        session_id = st.selectbox(
            "Session / segment chunk",
            options=segment_sessions,
            format_func=lambda value: _format_session(summaries[value]),
            index=index,
            key="parent_labels_session_selector",
        )
    with col_apply_all:
        st.write("")
        st.write("")
        apply_all = st.button(
            "Apply to all chunks",
            key="parent_labels_apply_all",
            disabled=not needs_update,
        )

    annotations, source_segments, has_saved_annotations = _load_annotation_chunk(
        selected_annotation_key,
        selected_session_key,
        session_id,
    )
    selected_summary = summaries[session_id]
    source_label = "saved output" if has_saved_annotations else "input source"
    st.caption(
        f"{session_id}: {selected_summary['children']} child segment(s), "
        f"{selected_summary['children_updated']} need parent labels. "
        f"Using {source_label} annotations."
    )
    _render_missing_label_examples(annotations, source_segments)

    apply_selected = st.button(
        "Add parent labels to selected chunk",
        key="parent_labels_apply_selected",
        type="primary",
        disabled=selected_summary["children_updated"] == 0,
    )

    if apply_selected:
        summary = _apply_chunk(
            selected_annotation_key,
            selected_session_key,
            session_id,
        )
        st.success(
            f"Added {summary['labels_added']} parent label(s) to "
            f"{summary['children_updated']} child segment(s)."
        )
        time.sleep(1)
        st.rerun()

    if apply_all:
        chunks_updated = 0
        children_updated = 0
        labels_added = 0
        for update_summary in needs_update:
            summary = _apply_chunk(
                selected_annotation_key,
                selected_session_key,
                update_summary["session_id"],
            )
            if summary["children_updated"]:
                chunks_updated += 1
                children_updated += summary["children_updated"]
                labels_added += summary["labels_added"]

        st.success(
            f"Updated {chunks_updated} chunk(s): added {labels_added} parent "
            f"label(s) to {children_updated} child segment(s)."
        )
        time.sleep(1)
        st.rerun()
