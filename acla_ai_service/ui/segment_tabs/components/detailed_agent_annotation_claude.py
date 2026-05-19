"""Claude sub-segment discovery (claude-agent-sdk → Claude Code login).

Renders the **☁️ Claude** expander below the analysis section. Unlike the
local-VLM variant — which chains 10-30 stateless VLM calls through a
LangGraph pipeline — this backend runs ONE Claude session with telemetry
tools (`render_graph`, `query_telemetry`, `compute_expert_phases`,
`submit_proposal`) so Claude can iterate naturally instead of paying
subprocess startup cost per step. See
``app.services.llm.claude_annotation_runner`` for the runner.

Routed through ``claude-agent-sdk`` → the user's local ``claude`` CLI
login (Claude Max). No API key needed; subject to Max-plan rate limits.

Shares the streaming UI, result panel, and staged-review panel with the
local component via ``_agent_annotation_shared.py``.
"""

import streamlit as st

from ._agent_annotation_shared import (
    collect_parent_info,
    execute_pipeline_run,
)


def render_agent_annotation_claude(
    df,
    form_start,
    form_end,
    form_labels,
    session_id,
    selected_annotation_key,
):
    """Render the Claude sub-segment discovery expander."""
    with st.expander("☁️ Claude Sub-Segment Discovery"):
        st.markdown(
            "Run **one agentic Claude session** with telemetry tools "
            "(`render_graph`, `query_telemetry`, `compute_expert_phases`, "
            "`submit_proposal`). Claude iterates: render → look → query → "
            "reason → submit. One subprocess start vs. 10-30 in the chained "
            "pipeline."
        )

        # --- Pipeline settings ---
        # The runner caps total tool calls at max_iterations * 10. Default 3
        # gives ~30 tool calls per session — usually enough for a thorough
        # investigation of one parent segment.
        max_iterations = st.number_input(
            "Tool-call budget (×10)",
            min_value=1,
            max_value=10,
            value=3,
            help=(
                "Caps the agent loop at this many tool calls × 10. Raise for "
                "complex segments where Claude needs to render many zooms or "
                "run more queries."
            ),
            key="agent_annot_claude_max_iter",
        )

        from app.agents.backends.claude_sdk import CLAUDE_VLM_MODELS

        claude_model_options = list(CLAUDE_VLM_MODELS.keys())
        claude_model = st.selectbox(
            "Claude model",
            options=claude_model_options,
            format_func=lambda x: CLAUDE_VLM_MODELS[x]["label"],
            index=0,
            help=(
                "Sonnet 4.6 is recommended — Opus 4.7 has a much tighter "
                "weekly cap on Max plans."
            ),
            key="agent_annot_claude_model",
        )
        claude_use_thinking = st.checkbox(
            "Use extended thinking",
            value=False,
            help=(
                "Adds a step-by-step reasoning instruction to each prompt. "
                "Slower and uses more output tokens against your Max quota, "
                "but tends to improve grounding on hard segments."
            ),
            key="agent_annot_claude_thinking",
        )
        st.caption(
            "ℹ️ Routed through `claude-agent-sdk` → your local `claude` CLI "
            "login. No API key needed; subject to Max-plan rate limits."
        )

        # --- Collect parent info ---
        parent_id, parent_main_label_ids, existing_children = collect_parent_info(
            form_labels,
        )

        if existing_children:
            st.caption(
                f"ℹ️ {len(existing_children)} existing child sub-segment(s) "
                "will be provided to Claude to avoid duplicates."
            )

        # --- Run button ---
        if st.button(
            "▶ Run Claude Sub-Segment Discovery",
            key="agent_annot_claude_run",
            type="primary",
        ):
            try:
                from app.pipelines.annotation import (
                    AnnotationPipelineConfig,
                )
            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}\n\n"
                    "Install with: `pip install langgraph langchain-core`"
                )
                return

            config = AnnotationPipelineConfig(
                max_iterations=int(max_iterations),
                backend="claude",
                claude_model=claude_model,
                claude_use_thinking=bool(claude_use_thinking),
            )

            execute_pipeline_run(
                df=df,
                form_start=form_start,
                form_end=form_end,
                session_id=session_id,
                parent_main_label_ids=parent_main_label_ids,
                existing_children=existing_children,
                config=config,
            )
