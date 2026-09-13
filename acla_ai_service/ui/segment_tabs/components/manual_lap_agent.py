"""Dispatcher for the provider-selected lap-to-segment excerpter."""

from __future__ import annotations

import streamlit as st

from ._lap_agent_shared import (
    KEY_LAP_RANGE, KEY_LAP_CIRCUIT,
    execute_lap_agent_run,
    render_lap_panel, render_lap_staged_review,
    reset_lap_agent_state_for_context, track_name_to_circuit_id,
)
from app.local_annotation_agent.workflow import AnnotationPipelineConfig, run_annotation


def render_manual_lap_agent(
    df, session_id, selected_annotation_key, selected_session_key=None,
):
    """Render the lap-to-segment excerpter section."""
    session_context = (selected_session_key, selected_annotation_key, session_id)
    reset_lap_agent_state_for_context(session_context)

    st.markdown("---")
    st.subheader("Lap-to-Segment Deterministic Annotation")
    st.caption(
        "Pick a lap range; the deterministic `split_lap_by_circuit_sections` "
        "tool rough-splits solo laps into per-`circuit_section` sub-ranges. "
        "When opponent data is present, it emits only close racing-interaction "
        "windows. Requirements then annotate **one section per click**."
    )

    track_name = (
        df["Static_track"].iloc[0]
        if "Static_track" in df.columns and not df.empty else None
    )
    circuit_id = track_name_to_circuit_id(track_name)

    # Lap range picker + rough split + current-section view render ONCE here
    # so the provider panel uses one widget key namespace.
    head = render_lap_panel(df, circuit_id, session_context=session_context)

    with st.expander("Deterministic Lap-to-Segment Annotation"):
        config = AnnotationPipelineConfig(provider_id="deterministic")

        if head is None:
            st.caption(
                "Pick a valid lap range above - the splitter fills the array automatically."
            )
        else:
            existing = _collect_existing_lap_annotations()
            if st.button(
                "Calculate current section labels",
                key="lap_provider_run",
                type="primary",
            ):
                def _run_lap(**kw):
                    return run_annotation(flow="lap", config=config, **kw)

                lap_start, lap_end = st.session_state[KEY_LAP_RANGE]
                execute_lap_agent_run(
                    run_fn=_run_lap,
                    df=df,
                    lap_start=int(lap_start),
                    lap_end=int(lap_end),
                    head_segment=head,
                    circuit_id=st.session_state[KEY_LAP_CIRCUIT],
                    existing=existing,
                    extra_kwargs={},
                )

    render_lap_staged_review(session_id, selected_annotation_key, df=df)


def _collect_existing_lap_annotations():
    lap_range = st.session_state.get(KEY_LAP_RANGE)
    if not lap_range:
        return []
    lap_start, lap_end = int(lap_range[0]), int(lap_range[1])
    annotations = st.session_state.get("current_annotations", [])
    out = []
    for ann in annotations:
        s = int(getattr(ann, "start_index", 0))
        e = int(getattr(ann, "end_index", 0))
        if e <= lap_start or s >= lap_end:
            continue
        out.append({
            "start_index": s,
            "end_index": e,
            "labels": list(getattr(ann, "labels", [])),
        })
    return out
