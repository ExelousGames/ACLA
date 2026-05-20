"""
Streamlit UI for annotating LLM training units.

A training unit is either a parent segment with its children, or a lone
segment with no parent. Parent/child is detected from
``AnnotatedSegment.parent_id``. The annotator captures two completions
per unit — critique-mode and guide-mode — which the
``app.pipelines.training.dataset_builder`` Step-2 pipeline fans out
into chat-format training rows.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
import time
from typing import List, Dict, Any


# Ensure app module is on path
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

try:
    from segment_tabs.shared import get_store, get_available_sessions, load_annotations
    from app.domain.labels import LABEL_MAPPING
    from segment_tabs.components.llm_ai_annotation import render_ai_assisted_annotation_tab
    from segment_tabs.components.llm_dataset_editor import render_dataset_editor_tab
    from segment_tabs.components.llm_segment_labels import render_unit_labels_section
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


def save_training_unit(
    unit: Dict[str, Any],
    completion_critique: str,
    completion_guide: str,
    output_path: str,
) -> None:
    """Append one annotated unit to the JSONL file in the Step-2 input schema."""
    record = {
        "unit_id": unit["unit_id"],
        "kind": unit["kind"],
        "parent_label_ids": unit["parent_label_ids"],
        "children_label_ids": unit["children_label_ids"],
        "completion_critique": completion_critique,
        "completion_guide": completion_guide,
        "timestamp": time.time(),
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_existing_units(output_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing unit annotations keyed by unit_id (latest record wins)."""
    records: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(output_path):
        return records
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = record.get("unit_id")
            if uid:
                records[uid] = record
    return records


def load_all_annotations_df(output_path: str) -> pd.DataFrame:
    """Load all annotations into a DataFrame for the inline editor."""
    records = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(records)


def _unit_has_labels(unit: Dict[str, Any]) -> bool:
    if unit["parent_label_ids"]:
        return True
    return any(cl for cl in unit["children_label_ids"])


def load_training_units(session_id: str, cache_key: str) -> List[Dict[str, Any]]:
    """Group annotated segments into training units.

    Top-level segments (``parent_id is None``) become unit roots. A root
    with children is ``kind="parent_with_children"``; one with no
    children is ``kind="isolated"``. Children's label_ids are kept
    per-child (not flattened) so the Step-2 builder can render them as
    structured context.
    """
    try:
        annotations = load_annotations(session_id, cache_key)
    except Exception as e:
        import traceback
        st.error(f"Error loading segments: {e}\n{traceback.format_exc()}")
        return []

    segments_by_id: Dict[str, Dict[str, Any]] = {}
    children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    roots: List[Dict[str, Any]] = []

    for item in annotations:
        seg = item.to_dict()
        sid = seg.get("id")
        if not sid:
            continue
        segments_by_id[sid] = seg
        parent_id = seg.get("parent_id")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(seg)
        else:
            roots.append(seg)

    units: List[Dict[str, Any]] = []
    for root in roots:
        rid = root["id"]
        kids = children_by_parent.get(rid, [])
        # Keep children in telemetry order so the prompt mirrors lap flow
        kids.sort(key=lambda c: (c.get("start_index") if c.get("start_index") is not None else 0))
        unit = {
            "unit_id": rid,
            "kind": "parent_with_children" if kids else "isolated",
            "parent_label_ids": [str(l) for l in (root.get("labels") or [])],
            "children_label_ids": [
                [str(l) for l in (c.get("labels") or [])] for c in kids
            ],
            "session_id": session_id,
            "start_index": root.get("start_index"),
            "end_index": root.get("end_index"),
            "child_count": len(kids),
            "_root_segment": root,
            "_child_segments": kids,
        }
        if _unit_has_labels(unit):
            units.append(unit)

    return units


def _annotation_status(unit: Dict[str, Any],
                       existing_units: Dict[str, Dict[str, Any]]):
    """Return (annotated, labels_changed, existing_record)."""
    existing = existing_units.get(unit["unit_id"])
    if not existing:
        return False, False, None
    saved_parent = sorted(existing.get("parent_label_ids") or [])
    saved_children = [sorted(c) for c in (existing.get("children_label_ids") or [])]
    cur_parent = sorted(unit["parent_label_ids"])
    cur_children = [sorted(c) for c in unit["children_label_ids"]]
    labels_changed = (saved_parent != cur_parent) or (saved_children != cur_children)
    return True, labels_changed, existing


def main():
    st.set_page_config(page_title="LLM Dataset Annotation", layout="wide")

    st.title("LLM Training Unit Annotation")
    st.markdown(
        "Capture **critique** and **guide** completions per training unit "
        "(parent segment + children, or isolated segment). Saved records "
        "feed `app.pipelines.training.dataset_builder` (Step 2)."
    )

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")

        store = get_store()
        all_keys = store.list_cache_keys()
        annotated_keys = [k for k in all_keys if "annotation" in k.lower()] or all_keys

        selected_key = st.selectbox(
            "Select Input Dataset (Segments)",
            options=annotated_keys,
            index=0 if annotated_keys else None
        )

        available_sessions = []
        if selected_key:
            available_sessions = get_available_sessions(selected_key)

        output_dir = Path("models/llm_datasets")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = st.text_input("Output Filename", "telemetry_descriptions_v1.jsonl")
        output_path = output_dir / output_filename

        st.info(f"Saving to: `{output_path}`")

        existing_units = load_existing_units(str(output_path))
        st.write(f"Loaded {len(existing_units)} existing unit annotations.")

    # --- Main Content ---
    main_tab_single, main_tab_batch = st.tabs(["Single Unit Annotation", "Batch AI Annotation"])

    with main_tab_single:
        st.header("Single Unit Annotation")

        from segment_tabs.components.llm_segment_navigation import handle_segment_selection_and_filtering

        filtered_units, _, _ = handle_segment_selection_and_filtering(
            available_sessions,
            selected_key,
            load_training_units,
            LABEL_MAPPING
        )

        if not filtered_units:
            return

        total_units = len(filtered_units)
        current_unit = filtered_units[st.session_state.current_index]

        annotated, labels_changed, existing_record = _annotation_status(
            current_unit, existing_units,
        )

        if labels_changed:
            st.warning("⚠️ **Labels Updated**: the labels for this unit have changed since last annotation.")
        elif annotated:
            st.success("✅ **Already Annotated**: critique + guide saved.")
        else:
            st.info("🆕 **New Unit**: requires critique + guide.")

        render_unit_labels_section(current_unit)

        tab_edit, tab_ai = st.tabs(["Dataset Editor", "AI Assisted Annotation"])

        crit_widget_key = f"input_crit_{st.session_state.current_index}"
        guide_widget_key = f"input_guide_{st.session_state.current_index}"

        with tab_edit:
            render_dataset_editor_tab(
                unit=current_unit,
                existing_record=existing_record,
                crit_widget_key=crit_widget_key,
                guide_widget_key=guide_widget_key,
                output_path=str(output_path),
                total_units=total_units,
                save_training_unit_fn=save_training_unit,
                load_all_annotations_df_fn=load_all_annotations_df,
            )

        with tab_ai:
            render_ai_assisted_annotation_tab(
                unit=current_unit,
                crit_widget_key=crit_widget_key,
                guide_widget_key=guide_widget_key,
            )

        with st.expander("View Raw Unit Data"):
            st.json({k: v for k, v in current_unit.items()
                     if not k.startswith("_")})

    with main_tab_batch:
        from segment_tabs.components.llm_batch_ai_annotation import render_batch_ai_annotation_tab

        batch_units_list = []
        if selected_key and available_sessions:
            for session_id in available_sessions:
                batch_units_list.extend(load_training_units(session_id, selected_key))

        render_batch_ai_annotation_tab(
            units=batch_units_list,
            existing_units=existing_units,
            output_path=str(output_path),
            available_sessions=available_sessions,
            save_training_unit_fn=save_training_unit,
        )


if __name__ == "__main__":
    main()
