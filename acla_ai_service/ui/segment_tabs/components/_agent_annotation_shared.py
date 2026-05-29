"""Shared helpers for the VLM sub-segment discovery components.

Used by ``detailed_agent_annotation_local.py`` and
``detailed_agent_annotation_claude.py``. Owns:

- ``LiveVlmOutput``: streaming output renderer (collapsible per-step UI)
- ``execute_pipeline_run``: wraps the LangGraph call with progress / live UI
- ``render_pipeline_result``: post-run banner, proposals, reasoning, graphs
- ``render_staged_review``: editable per-row review panel + atomic save
- ``collect_parent_info``: parent_id / main-label hints / existing children

Both backends share ``st.session_state['agent_annot_result']`` so the
staged-review panel is last-run-wins.
"""

import io
import time
import traceback

import streamlit as st
from PIL import Image

from ..shared import (
    LABEL_CATEGORIES,
    LABEL_MAPPING,
    LABEL_NAME_TO_ID,
    save_annotations,
)

from app.domain.segment import AnnotatedSegment
_NODE_ICONS = {
    "planner": "🧠",
    "step_solver": "🛠️",
    "describe_graphs": "🔍",
    "zoom": "🔬",
    "label_verifier": "✅",
    "proposal_synthesizer": "📝",
}

_ATTACHMENT_ICONS = {
    "text": "📎",
    "structured": "📋",
    "image_set": "🖼️",
}


class LiveVlmOutput:
    """Hold streaming state for one annotation run and render it.

    Each completed VLM call collapses into an expander with a summary
    header (icon + node name + detail + duration). The active / streaming
    call stays expanded at the bottom, in its own placeholder so it can
    be cleared independently of the completed expanders.
    """

    def __init__(self) -> None:
        # Header carries a live total-elapsed counter (wall-clock from
        # analysis start) — refreshed on every render().
        self.header_area = st.empty()
        self.analysis_start_time = time.time()
        # Two separate placeholders so the streaming "active" block can
        # be cleared independently of the completed expanders. Mixing
        # st.expander widgets with bare st.markdown inside a single
        # placeholder leaves the markdown elements stuck in the DOM
        # when the active state empties out at end-of-run.
        self.completed_area = st.empty()
        self.active_area = st.empty()

        # Each section: {meta: {node_name, phase, iteration, total},
        #                prompt, reasoning, text, duration}
        self.completed_sections: list[dict] = []
        self.vlm_buffer: list[str] = []        # response chunks for active call
        self.reasoning_buffer: list[str] = []  # chain-of-thought (thinking models)
        self.active_prompt: str = ""
        self.active_meta: dict = {}            # set by on_vlm_prompt from pipeline stage
        self.step_start_time: float = 0.0

        # Progress widgets attached by the caller.
        self.progress_bar = None
        self.status_text = None

        self._render_header()

    def attach_progress(self, progress_bar, status_text) -> None:
        self.progress_bar = progress_bar
        self.status_text = status_text

    @staticmethod
    def _format_header(meta: dict, duration: float | None = None) -> str:
        """Build a section header like '🧠 Planner (1/2) — main (3.2s)'."""
        node = meta.get("node_name") or "Processing"
        icon = _NODE_ICONS.get(node, "●")
        title = node.replace("_", " ").title()
        graphs = meta.get("graphs") or []
        graph_tag = f" [{', '.join(graphs)}]" if graphs else ""
        iteration = meta.get("iteration")
        total = meta.get("total")
        iter_tag = f" ({iteration}/{total})" if iteration and total else ""
        phase = meta.get("phase") or ""
        phase_tag = f" — {phase}" if phase else ""
        dur_tag = f"  ({duration:.1f}s)" if duration else ""
        return f"{icon} {title}{graph_tag}{iter_tag}{phase_tag}{dur_tag}"

    @staticmethod
    def _render_attachments(meta: dict) -> None:
        """Show a one-line chip row of attachments fed to this VLM call."""
        atts = meta.get("attachments") or []
        if not atts:
            st.caption(
                "_📎 (no upstream attachments — built from raw run inputs)_"
            )
            return
        chips: list[str] = []
        for att in atts:
            icon = _ATTACHMENT_ICONS.get(att.get("kind", ""), "📎")
            label = att.get("label") or att.get("name", "?")
            count = att.get("count")
            chip = f"{icon} {label}"
            if count is not None:
                chip = f"{chip} ({count})"
            chips.append(chip)
        st.caption(" · ".join(chips))

    def _finalize_active_section(self) -> None:
        """Move the active VLM section to ``completed_sections``."""
        if self.active_prompt or self.vlm_buffer or self.reasoning_buffer:
            elapsed = (
                time.time() - self.step_start_time
                if self.step_start_time else 0
            )
            self.completed_sections.append({
                "meta": dict(self.active_meta),
                "prompt": self.active_prompt,
                "reasoning": "".join(self.reasoning_buffer),
                "text": "".join(self.vlm_buffer),
                "duration": elapsed,
            })
            self.vlm_buffer.clear()
            self.reasoning_buffer.clear()
            self.active_prompt = ""
            self.active_meta.clear()
            self.step_start_time = 0.0

    def _render_completed(self) -> None:
        with self.completed_area.container():
            for s in self.completed_sections:
                header = self._format_header(s["meta"], s.get("duration", 0))
                with st.expander(header, expanded=False):
                    self._render_attachments(s["meta"])
                    has_response = bool(s.get("text"))
                    st.markdown(
                        "**Prompt:**" if has_response else "**Summary:**"
                    )
                    with st.container(border=True):
                        st.markdown(s["prompt"])
                    if s.get("reasoning"):
                        st.markdown(
                            f"**💭 Thinking:**\n\n{s['reasoning']}"
                        )
                    if has_response:
                        st.markdown(f"**Response:**\n\n{s['text']}")

    def _render_active(self) -> None:
        if not (self.active_prompt or self.vlm_buffer or self.reasoning_buffer):
            self.active_area.empty()
            return
        with self.active_area.container():
            elapsed = (
                time.time() - self.step_start_time
                if self.step_start_time else 0
            )
            st.markdown(
                f"**{self._format_header(self.active_meta)}** "
                f"_{elapsed:.0f}s …_"
            )
            self._render_attachments(self.active_meta)
            st.markdown("*Prompt:*")
            with st.container(border=True):
                st.markdown(self.active_prompt)
            if self.reasoning_buffer:
                st.markdown(
                    f"*💭 Thinking (streaming…)*\n\n"
                    f"{''.join(self.reasoning_buffer)}"
                )
            st.markdown(
                f"*Response (streaming…)*\n\n"
                f"{''.join(self.vlm_buffer)}"
            )

    def _render_header(self) -> None:
        total_elapsed = time.time() - self.analysis_start_time
        self.header_area.markdown(
            f"**Live VLM Output** _(total {total_elapsed:.1f}s)_"
        )

    def render(self) -> None:
        self._render_header()
        self._render_completed()
        self._render_active()

    # ---- pipeline callbacks ----

    def on_vlm_prompt(self, prompt: str, stage: dict) -> None:
        # Finalize any prior active call before starting the next.
        self._finalize_active_section()
        self.active_prompt = prompt
        self.active_meta.clear()
        self.active_meta.update(stage)
        self.step_start_time = time.time()
        self.render()

    def on_vlm_stream(self, chunk: str) -> None:
        self.vlm_buffer.append(chunk)
        self.render()

    def on_vlm_reasoning(self, chunk: str) -> None:
        self.reasoning_buffer.append(chunk)
        self.render()

    def on_step_event(self, summary: str, stage: dict) -> None:
        # Non-VLM event (e.g. zoom rendering). Finalize any in-flight
        # active section, then append a response-less completed section.
        self._finalize_active_section()
        self.completed_sections.append({
            "meta": dict(stage),
            "prompt": summary,
            "reasoning": "",
            "text": "",
            "duration": 0,
        })
        self.render()

    def on_progress(self, node_name: str, detail: str) -> None:
        if self.progress_bar is not None:
            # Rough progress: each completed VLM section nudges the bar.
            pct = min((len(self.completed_sections) + 1) / 12, 0.99)
            self.progress_bar.progress(pct)
        if self.status_text is not None:
            self.status_text.markdown(
                f"**Status:** _{node_name}: {detail[:200]}_"
            )

    def finalize(self) -> None:
        """Flush any in-flight section and re-render once at end-of-run."""
        self._finalize_active_section()
        self.render()


def collect_parent_info(form_labels):
    """Resolve parent metadata used by both backends.

    Returns ``(parent_id, parent_main_label_ids, existing_children)``.
    Only Main Labels are forwarded as hints to the VLM; existing
    children are passed so the VLM can avoid duplicate proposals.
    """
    main_label_set = set(LABEL_CATEGORIES.get("Main Labels", []))
    parent_main_label_ids = [
        LABEL_NAME_TO_ID[l] for l in form_labels
        if l in LABEL_NAME_TO_ID and LABEL_NAME_TO_ID[l] in main_label_set
    ]

    annotations = st.session_state.get("current_annotations", [])
    selected_option = st.session_state.get("detailed_annotation_selector")
    parent_ann = annotations[selected_option] if (
        selected_option is not None and selected_option < len(annotations)
    ) else None
    parent_id = parent_ann.id if parent_ann else None

    existing_children = []
    if parent_id:
        for ann in annotations:
            if getattr(ann, "parent_id", None) == parent_id:
                existing_children.append({
                    "start_index": ann.start_index,
                    "end_index": ann.end_index,
                    "labels": list(ann.labels),
                })

    return parent_id, parent_main_label_ids, existing_children


def execute_pipeline_run(
    *,
    df,
    form_start,
    form_end,
    session_id,
    parent_main_label_ids,
    existing_children,
    config,
):
    """Run the annotation pipeline with live UI feedback.

    Renders the live VLM output panel and the post-run result panel.
    Caches the result under ``st.session_state['agent_annot_result']``
    so the shared staged-review panel below can pick it up.
    """
    # One unified entry dispatches on config.backend internally; the agent
    # box picks the LangGraph runner for "local" and the agentic Claude
    # runner for "claude" via run_agent.
    backend = getattr(config, "backend", "local")
    try:
        from app.local_annotation_agent.workflow import run_annotation
    except ImportError as e:
        st.error(
            f"Missing dependency: {e}\n\n"
            "Install with: `pip install langgraph langchain-core` "
            "(or `pip install claude-agent-sdk` for the Claude backend)."
        )
        return

    progress_area = st.container()
    status_text = progress_area.empty()
    progress_bar = progress_area.progress(0)

    live = LiveVlmOutput()
    live.attach_progress(progress_bar, status_text)

    spinner_msg = (
        "Claude is analysing the segment…"
        if backend == "claude"
        else "Running sub-segment discovery pipeline…"
    )
    try:
        with st.spinner(spinner_msg):
            result = run_annotation(
                flow="detailed",
                df=df,
                start_index=int(form_start),
                end_index=int(form_end),
                session_id=session_id,
                parent_main_labels=parent_main_label_ids,
                existing_children=existing_children,
                config=config,
                progress_callback=live.on_progress,
                vlm_stream_callback=live.on_vlm_stream,
                vlm_prompt_callback=live.on_vlm_prompt,
                vlm_reasoning_callback=live.on_vlm_reasoning,
                step_event_callback=live.on_step_event,
            )
    except Exception as e:
        live.finalize()
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
        return

    # Finalise any still-active call and force a clean re-render so the
    # streaming placeholder is emptied even if no chunks were buffered.
    live.finalize()
    progress_bar.progress(1.0)

    render_pipeline_result(result, form_start, form_end)
    st.session_state["agent_annot_result"] = result

    # Claude-only: stash the context the follow-up chat needs so it can
    # re-create the same parent / tool environment on every rerun.
    if backend == "claude":
        st.session_state["agent_annot_followup_ctx"] = {
            "df": df,
            "parent_start": int(form_start),
            "parent_end": int(form_end),
            "parent_main_labels": list(parent_main_label_ids),
            "existing_children": list(existing_children),
            "claude_model": getattr(config, "claude_model", "claude-sonnet-4-6"),
            "use_thinking": bool(getattr(config, "claude_use_thinking", False)),
            "max_turns": int(getattr(config, "max_iterations", 3)) * 10,
        }
        # Fresh chat per new annotation run.
        st.session_state["agent_annot_followup_chat"] = []
    else:
        # Local-VLM run replaces a prior Claude result — drop the stale
        # follow-up state so the chat doesn't dangle over an unrelated run.
        st.session_state.pop("agent_annot_followup_ctx", None)
        st.session_state.pop("agent_annot_followup_chat", None)


def render_pipeline_result(result, form_start, form_end) -> None:
    """Show banner, proposed sub-segments, reasoning, graphs, log."""
    if result.accepted:
        st.success(
            f"✅ Accepted after {result.iterations} iteration(s)."
        )
    else:
        st.warning(
            f"⚠️ Max iterations reached ({result.iterations}). "
            "Showing best proposal."
        )

    grouped_preview = group_proposals_by_range(result)
    st.markdown(
        f"##### Proposed Sub-Segments ({len(grouped_preview)})  "
        f"_(parent: [{form_start}, {form_end}])_"
    )
    for (gs, ge), anns in grouped_preview:
        label_names = ", ".join(
            LABEL_MAPPING.get(a["label_id"], a["label_id"]) for a in anns
        )
        st.markdown(f"- **[{gs}, {ge}]** — {label_names}")

    st.markdown("##### Reasoning")
    st.markdown(result.final_reasoning)

    if result.graph_images:
        with st.expander(
            f"📊 Telemetry Graphs ({len(result.graph_images)} images)",
            expanded=False,
        ):
            cols = st.columns(min(len(result.graph_images), 3))
            for idx, img_bytes in enumerate(result.graph_images):
                with cols[idx % len(cols)]:
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        if img.width == 0 or img.height == 0:
                            st.warning(
                                f"Graph {idx + 1}: image has zero dimensions "
                                "and cannot be displayed."
                            )
                        else:
                            st.image(img, width='stretch')
                    except Exception as e:
                        st.error(f"Graph {idx + 1}: failed to load image — {e}")

    with st.expander("Full agent conversation log"):
        for msg in result.messages:
            role = msg.get("role", "unknown").replace("_", " ").title()
            iteration = msg.get("iteration", "?")
            content = msg.get("content", "")
            verdict = msg.get("verdict", "")
            header = f"**[Iter {iteration}] {role}**"
            if verdict:
                header += f" — _{verdict.upper()}_"
            st.markdown(header)
            st.text(content[:2000])
            st.markdown("---")


def render_staged_review(
    parent_id,
    session_id,
    selected_annotation_key,
    form_start,
    form_end,
    df=None,
) -> None:
    """Editable per-row review panel for the most recent pipeline result.

    Reads ``st.session_state['agent_annot_result']`` — shared across
    both backends, last-run-wins. No-op when no result is pending.
    """
    if "agent_annot_result" not in st.session_state:
        return

    result = st.session_state["agent_annot_result"]
    grouped = group_proposals_by_range(result)

    st.info(
        f"Pending {len(grouped)} sub-segment(s) "
        f"({result.iterations} iter, "
        f"{'accepted' if result.accepted else 'max-iter'})"
    )

    st.markdown("##### Review & Edit Before Saving")
    st.caption(
        "Each row is one AI-discovered sub-segment. Labels sharing the "
        "same range are grouped. Edit ranges/labels per row, or remove "
        "a row by clearing its labels."
    )

    all_label_options = sorted(LABEL_MAPPING.values())
    staged_segments: list[dict] = []
    for i, ((gs, ge), anns) in enumerate(grouped):
        with st.container(border=True):
            st.markdown(f"**Sub-segment {i + 1}**")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                seg_start = st.number_input(
                    "Start",
                    min_value=int(form_start),
                    max_value=int(form_end),
                    value=int(gs),
                    key=f"agent_staged_start_{i}",
                )
            with col_r2:
                seg_end = st.number_input(
                    "End",
                    min_value=int(form_start),
                    max_value=int(form_end),
                    value=int(ge),
                    key=f"agent_staged_end_{i}",
                )

            default_labels = [
                LABEL_MAPPING.get(a["label_id"], a["label_id"])
                for a in anns
                if a["label_id"] in LABEL_MAPPING
            ]
            seg_labels = st.multiselect(
                "Labels",
                options=all_label_options,
                default=default_labels,
                key=f"agent_staged_labels_{i}",
            )

            default_notes = "; ".join(
                a.get("reasoning", "") for a in anns if a.get("reasoning")
            )[:500]
            seg_notes = st.text_input(
                "Notes (optional)",
                value=default_notes,
                key=f"agent_staged_notes_{i}",
            )

            staged_segments.append({
                "start": int(seg_start),
                "end": int(seg_end),
                "labels": seg_labels,
                "notes": seg_notes,
            })

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            f"✅ Confirm & Save All ({len(staged_segments)})",
            key="agent_annot_confirm",
            type="primary",
        ):
            _persist_staged_subsegments(
                staged_segments=staged_segments,
                parent_id=parent_id,
                session_id=session_id,
                selected_annotation_key=selected_annotation_key,
                df=df,
            )
    with col_btn2:
        if st.button(
            "❌ Discard Proposals",
            key="agent_annot_discard",
        ):
            del st.session_state["agent_annot_result"]
            st.session_state.pop("agent_annot_followup_ctx", None)
            st.session_state.pop("agent_annot_followup_chat", None)
            st.rerun()


def render_followup_chat() -> None:
    """Chat panel for interrogating the just-finished Claude annotation.

    Renders only when both ``agent_annot_result`` and the Claude-specific
    ``agent_annot_followup_ctx`` are present in session_state. Each user
    turn starts a fresh Claude session pre-seeded with the same parent
    context and prior proposals, with the same telemetry tools available
    so Claude can re-inspect the data while answering — only
    ``submit_proposal`` is excluded.
    """
    if (
        "agent_annot_result" not in st.session_state
        or "agent_annot_followup_ctx" not in st.session_state
    ):
        return

    ctx = st.session_state["agent_annot_followup_ctx"]
    result = st.session_state["agent_annot_result"]
    chat: list[dict] = st.session_state.setdefault("agent_annot_followup_chat", [])

    st.markdown("---")
    st.markdown("##### 💬 Ask Claude about these proposals")
    st.caption(
        "Use this to interrogate the just-finished annotation so you can "
        "refine the skill text (label catalog / graph `how_to_analyze` "
        "blocks). Claude can still call `render_graph`, `query_telemetry`, "
        "and `compute_expert_phases` to look at the data while answering."
    )

    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_question = st.chat_input(
        "e.g. 'why didn't `late_braking` fit here?'",
        key="agent_annot_followup_input",
    )
    if not user_question:
        return

    with st.chat_message("user"):
        st.markdown(user_question)
    chat.append({"role": "user", "content": user_question})

    from app.local_annotation_agent.workflow import run_claude_followup

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        tool_placeholder = st.empty()
        buffer: list[str] = []
        tool_events: list[str] = []

        def on_text_chunk(chunk: str) -> None:
            buffer.append(chunk)
            text_placeholder.markdown("".join(buffer) + "▌")

        def on_tool_event(name: str, inp: dict) -> None:
            short = str(inp)
            if len(short) > 160:
                short = short[:160] + "…"
            tool_events.append(f"🔧 `{name}` {short}")
            tool_placeholder.caption("\n\n".join(tool_events))

        try:
            response = run_claude_followup(
                df=ctx["df"],
                start_index=ctx["parent_start"],
                end_index=ctx["parent_end"],
                parent_main_labels=ctx["parent_main_labels"],
                existing_children=ctx["existing_children"],
                claude_model=ctx["claude_model"],
                use_thinking=ctx["use_thinking"],
                max_turns=ctx["max_turns"],
                prior_result=result,
                chat_history=chat[:-1],  # prior turns; current user turn passes via user_question
                user_question=user_question,
                on_text_chunk=on_text_chunk,
                on_tool_event=on_tool_event,
            )
        except Exception as e:
            text_placeholder.error(f"Follow-up chat error: {e}")
            chat.pop()  # roll the user turn back so they can retry
            return

        final_text = response or "".join(buffer) or "(empty response)"
        text_placeholder.markdown(final_text)
        chat.append({"role": "assistant", "content": final_text})


def group_proposals_by_range(result) -> list[tuple[tuple[int, int], list[dict]]]:
    """Group per-label proposals into one entry per unique (start, end).

    Falls back to a single entry covering ``[sub_start, sub_end]`` with
    all final_labels if the pipeline didn't expose per-label annotations
    (e.g. older result objects).
    """
    label_annotations = list(getattr(result, "label_annotations", None) or [])

    if not label_annotations and result.final_labels:
        s = result.sub_start if result.sub_start is not None else 0
        e = result.sub_end if result.sub_end is not None else 0
        label_annotations = [
            {
                "label_id": l,
                "start_index": s,
                "end_index": e,
                "reasoning": result.final_reasoning,
            }
            for l in result.final_labels
        ]

    grouped: dict[tuple[int, int], list[dict]] = {}
    order: list[tuple[int, int]] = []
    for ann in label_annotations:
        try:
            key = (int(ann["start_index"]), int(ann["end_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(ann)

    order.sort()
    return [(k, grouped[k]) for k in order]


def _persist_staged_subsegments(
    staged_segments: list[dict],
    parent_id: str | None,
    session_id: str,
    selected_annotation_key: str,
    df=None,
) -> None:
    """Validate every staged segment, then persist them all atomically."""
    from ..shared import build_segment

    new_children: list[AnnotatedSegment] = []
    errors: list[str] = []

    for i, seg in enumerate(staged_segments, start=1):
        start = seg["start"]
        end = seg["end"]
        label_names = seg["labels"]

        # Rows with no labels are treated as user-removed — skip silently.
        if not label_names:
            continue

        if start >= end:
            errors.append(
                f"Sub-segment {i}: start ({start}) must be less than end ({end})."
            )
            continue

        label_ids = [
            LABEL_NAME_TO_ID[n] for n in label_names if n in LABEL_NAME_TO_ID
        ]
        if not label_ids:
            errors.append(f"Sub-segment {i}: no valid labels resolved.")
            continue

        new_children.append(build_segment(
            df,
            start=int(start), end=int(end), label_ids=label_ids,
            notes=seg.get("notes", ""), parent_id=parent_id,
        ))

    if errors:
        for e in errors:
            st.error(e)
        return

    if not new_children:
        st.warning("Nothing to save — every row had its labels cleared.")
        return

    annotations = list(st.session_state.get("current_annotations", []))
    annotations.extend(new_children)
    st.session_state["current_annotations"] = annotations

    save_annotations(session_id, annotations, selected_annotation_key)

    if "agent_annot_result" in st.session_state:
        del st.session_state["agent_annot_result"]
    st.session_state.pop("agent_annot_followup_ctx", None)
    st.session_state.pop("agent_annot_followup_chat", None)

    st.success(f"Saved {len(new_children)} sub-segment(s).")
    st.rerun()
