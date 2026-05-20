import streamlit as st
import json
import time
from typing import Any, Callable, Dict

from app.pipelines.training.dataset_builder import render_labels_text


def render_dataset_editor_tab(
    unit: Dict[str, Any],
    existing_record: Dict[str, Any],
    crit_widget_key: str,
    guide_widget_key: str,
    output_path: str,
    total_units: int,
    save_training_unit_fn: Callable,
    load_all_annotations_df_fn: Callable,
):
    st.subheader("Manual Annotation")

    labels_text = render_labels_text(
        unit["parent_label_ids"], unit["children_label_ids"],
    )

    st.text_area(
        "System prompt — critique mode (preview)",
        value=f"critique user, {labels_text}",
        height=80,
        disabled=True,
    )
    st.text_area(
        "System prompt — guide mode (preview)",
        value=f"guide user, {labels_text}",
        height=80,
        disabled=True,
    )

    # Pending drafts injected by the AI tab
    for pending_key, target_key in (
        (f"pending_{crit_widget_key}", crit_widget_key),
        (f"pending_{guide_widget_key}", guide_widget_key),
    ):
        if pending_key in st.session_state:
            st.session_state[target_key] = st.session_state.pop(pending_key)

    if crit_widget_key not in st.session_state:
        st.session_state[crit_widget_key] = (
            existing_record.get("completion_critique", "") if existing_record else ""
        )
    if guide_widget_key not in st.session_state:
        st.session_state[guide_widget_key] = (
            existing_record.get("completion_guide", "") if existing_record else ""
        )

    completion_critique = st.text_area(
        "Critique completion (Assistant response, critique mode)",
        value=st.session_state[crit_widget_key],
        height=150,
        key=crit_widget_key,
    )
    completion_guide = st.text_area(
        "Guide completion (Assistant response, guide mode)",
        value=st.session_state[guide_widget_key],
        height=150,
        key=guide_widget_key,
    )

    if st.button("Save Unit Annotation", type="primary"):
        if not completion_critique.strip() or not completion_guide.strip():
            st.error("Both critique and guide completions are required.")
        else:
            save_training_unit_fn(
                unit, completion_critique, completion_guide, str(output_path),
            )
            st.success("Saved!")
            time.sleep(0.5)
            st.session_state.current_index = min(
                total_units - 1, st.session_state.current_index + 1,
            )
            st.rerun()

    st.divider()

    st.subheader("Dataset Editor")
    st.markdown("Edit or delete saved unit annotations.")

    df = load_all_annotations_df_fn(str(output_path))
    if df.empty or "unit_id" not in df.columns:
        st.info("The dataset is currently empty.")
        return

    unit_df = df[df["unit_id"] == unit["unit_id"]].copy()
    if unit_df.empty:
        st.info("No annotation saved for this unit yet.")
        return

    unit_df.insert(0, "Delete", False)
    edited_df = st.data_editor(
        unit_df,
        num_rows="dynamic",
        width="stretch",
        height=300,
        key=f"dataset_editor_{unit['unit_id']}",
        disabled=(
            "unit_id", "kind", "parent_label_ids",
            "children_label_ids", "timestamp",
        ),
    )
    if st.button("Save Dataset Changes", type="secondary"):
        other_df = df[df["unit_id"] != unit["unit_id"]].copy()
        with open(output_path, "w") as f:
            for _, row in other_df.iterrows():
                f.write(json.dumps(row.dropna().to_dict()) + "\n")
            for _, row in edited_df.iterrows():
                if row.get("Delete", False):
                    continue
                if "Delete" in row:
                    row = row.drop("Delete")
                f.write(json.dumps(row.dropna().to_dict()) + "\n")
        st.success("Successfully updated dataset!")
        time.sleep(1)
        st.rerun()
