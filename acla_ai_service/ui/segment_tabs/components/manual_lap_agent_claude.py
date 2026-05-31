"""Claude lap-section annotation expander (manual.py).

One agent session per click. Annotates ONE rough-split section at a time,
with tools to revise the boundary before submitting. Mirrors the layout of
``detailed_agent_annotation_claude.py`` so the user gets a familiar
configuration surface.
"""

from __future__ import annotations

import streamlit as st

from ._lap_agent_shared import (
    KEY_LAP_SEGMENTS, KEY_LAP_RANGE, KEY_LAP_CIRCUIT,
    execute_lap_agent_run,
)


def render_lap_agent_claude(df, session_id, selected_annotation_key, circuit_id, head):
    """Render the Claude lap-section excerpter expander.

    ``head`` is the current head segment from the shared lap panel
    (rendered once by the dispatcher). When None, the user hasn't run
    the split yet — show the expander controls but disable the run
    button.
    """
    with st.expander("☁️ Claude — Lap-to-Segment Excerpter"):
        st.markdown(
            "Annotates **one rough-split section per click** using an "
            "agentic Claude session. Tools available: `render_graph`, "
            "`query_telemetry`, `compute_expert_phases`, "
            "`locate_circuit_section`, `classify_opponent_interaction`, "
            "`revise_range`, `submit_result`. The agent inspects the section, "
            "shrinks / extends the boundary when a rule fires, then "
            "submits parent labels (circuit + circuit_section + ST1-ST6 + "
            "optional main)."
        )

        max_iterations = st.number_input(
            "Tool-call budget (×10)", min_value=1, max_value=10, value=3,
            help="Caps the agent loop at this many tool calls × 10.",
            key="lap_claude_max_iter",
        )

        from app.claude.backend import CLAUDE_VLM_MODELS
        model_options = list(CLAUDE_VLM_MODELS.keys())
        claude_model = st.selectbox(
            "Claude model", options=model_options,
            format_func=lambda x: CLAUDE_VLM_MODELS[x]["label"],
            index=0,
            help="Sonnet 4.6 recommended.", key="lap_claude_model",
        )
        use_thinking = st.checkbox(
            "Use extended thinking", value=False,
            help="Adds step-by-step reasoning to each turn.",
            key="lap_claude_thinking",
        )
        st.caption(
            "ℹ️ Routed through `claude-agent-sdk` → your local `claude` CLI "
            "login. Subject to Max-plan rate limits."
        )

        if head is None:
            st.caption("Pick a valid lap range above — the splitter fills the array automatically.")
            return

        existing = _collect_existing_lap_annotations()

        if st.button(
            "▶ Run Claude on current section",
            key="lap_claude_run", type="primary",
        ):
            try:
                from app.local_annotation_agent.workflow import (
                    AnnotationPipelineConfig,
                    run_annotation,
                )
            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}\n\n"
                    "Install with `pip install claude-agent-sdk`."
                )
                return

            config = AnnotationPipelineConfig(
                backend="claude",
                max_iterations=int(max_iterations),
                claude_model=claude_model,
                claude_use_thinking=bool(use_thinking),
            )

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


def _collect_existing_lap_annotations():
    """Existing annotations on this lap (for the agent to avoid duplicates).

    Pulls every current annotation whose range overlaps the picked lap
    range. Sent into the agent as the ``existing_section_annotations``
    list so it can see what was already labelled.
    """
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
