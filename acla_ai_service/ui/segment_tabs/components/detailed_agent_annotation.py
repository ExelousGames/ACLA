"""Dispatcher for AI sub-segment discovery."""

from ._agent_annotation_shared import (
    collect_parent_info,
    execute_pipeline_run,
    render_followup_chat,
    render_staged_review,
)
from .annotation_provider_controls import render_annotation_provider_config

import streamlit as st


def render_agent_annotation(
    df,
    form_start,
    form_end,
    form_labels,
    session_id,
    selected_annotation_key,
):
    """Render provider-selected AI discovery and the shared staged review."""
    with st.expander("AI Sub-Segment Discovery"):
        config = render_annotation_provider_config(
            key_prefix="agent_annot_provider",
            default_temperature=0.7,
            default_max_new_tokens=1500,
            default_tool_budget=3,
        )
        parent_id, parent_main_label_ids, existing_children = collect_parent_info(
            form_labels,
        )
        if existing_children:
            st.caption(
                f"{len(existing_children)} existing child sub-segment(s) "
                "will be provided to the AI provider to avoid duplicates."
            )
        disabled = config is None
        if st.button(
            "Run AI Sub-Segment Discovery",
            key="agent_annot_provider_run",
            type="primary",
            disabled=disabled,
        ) and config is not None:
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
