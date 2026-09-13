"""Dispatcher for deterministic sub-segment discovery."""

from ._agent_annotation_shared import (
    collect_parent_info,
    execute_pipeline_run,
    render_followup_chat,
    render_staged_review,
)
from app.local_annotation_agent.workflow import AnnotationPipelineConfig

import streamlit as st


def render_agent_annotation(
    df,
    form_start,
    form_end,
    form_labels,
    session_id,
    selected_annotation_key,
):
    """Render deterministic discovery and the shared staged review."""
    with st.expander("Deterministic Sub-Segment Discovery"):
        config = AnnotationPipelineConfig(provider_id="deterministic")
        parent_id, parent_main_label_ids, existing_children = collect_parent_info(
            form_labels,
        )
        if existing_children:
            st.caption(
                f"{len(existing_children)} existing child sub-segment(s) "
                "will be used to avoid exact duplicate proposals."
            )
        if st.button(
            "Calculate Sub-Segment Labels",
            key="agent_annot_provider_run",
            type="primary",
        ):
            execute_pipeline_run(
                df=df,
                form_start=form_start,
                form_end=form_end,
                session_id=session_id,
                parent_main_label_ids=parent_main_label_ids,
                existing_children=existing_children,
                config=config,
            )

    parent_id, _, _ = collect_parent_info(form_labels)
    render_staged_review(
        parent_id, session_id, selected_annotation_key, form_start, form_end,
        df=df,
    )
    render_followup_chat()
