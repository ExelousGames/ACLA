"""Shared helpers for the lap-to-segment excerpter UI.

Owns:
  - Session-state keys for the rough-split array + staged proposal.
  - ``run_split`` — calls the deterministic splitter and stores the array.
  - ``rebuild_remaining_segments`` — after a revise_segment_range result,
    re-runs the splitter from the new boundary onward (used by both
    backends).
  - ``render_lap_panel`` — picks the lap range, runs the split, shows the
    array, and exposes the "current section" + "skip" / "remove" actions.
  - ``render_lap_staged_review`` — staged review for the last agent
    result; user edits labels / range before saving.
  - ``execute_lap_agent_run`` — wraps a single agent invocation with live
    UI feedback (mirrors detailed flow's ``execute_pipeline_run``).
  - ``track_name_to_circuit_id`` — Static_track string → catalog circuit id.
"""

from __future__ import annotations

import io
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from PIL import Image

from app.domain.segment import AnnotatedSegment
from ..shared import (
    LABEL_MAPPING, LABEL_NAME_TO_ID, save_annotations,
)


# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------

KEY_LAP_RANGE = "lap_agent_lap_range"            # (start, end) of the lap the user picked
KEY_LAP_SEGMENTS = "lap_agent_segments"          # list[dict] from split_lap_by_circuit_sections
KEY_LAP_CIRCUIT = "lap_agent_circuit_id"         # circuit id detected from Static_track
KEY_LAP_CURSOR = "lap_agent_cursor"              # index of the "next" section in segments
KEY_LAP_STAGED = "lap_agent_staged_result"       # LapAnnotationResult-ish dict awaiting review
KEY_LAP_LIVE = "lap_agent_live_output"           # last live-output snapshot for re-render


# ---------------------------------------------------------------------------
# Track name → circuit id
# ---------------------------------------------------------------------------

def track_name_to_circuit_id(track_name: Optional[str]) -> Optional[str]:
    """Normalise the ``Static_track`` value to a catalog circuit id."""
    if not track_name:
        return None
    canon = str(track_name).strip().lower().replace(" ", "_").replace("-", "_")
    # The label catalog uses these IDs; expand as more circuits get added.
    return canon if canon in {"brands_hatch", "silverstone"} else canon


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def run_split(df, start: int, end: int, circuit_id: Optional[str]) -> List[Dict[str, Any]]:
    """Run the deterministic splitter and return the segments list."""
    from app.agents.tools import split_lap_by_circuit_sections
    att = split_lap_by_circuit_sections(df, int(start), int(end), circuit_id=circuit_id)
    content = att.content or {}
    return list(content.get("segments") or [])


def rebuild_remaining_segments(
    df, lap_start: int, lap_end: int, circuit_id: Optional[str],
    revised_end: int,
) -> List[Dict[str, Any]]:
    """Re-run the splitter from ``revised_end`` to ``lap_end``.

    Used after an agent calls ``revise_segment_range`` — the head of the
    array gets replaced with the agent-decided range, and everything
    downstream is rebuilt from the new boundary so neighbouring sections
    pick up the shift.
    """
    if revised_end >= lap_end:
        return []
    return run_split(df, int(revised_end), int(lap_end), circuit_id)


# ---------------------------------------------------------------------------
# Live-output panel (light version of _agent_annotation_shared.LiveVlmOutput)
# ---------------------------------------------------------------------------

class LapLiveOutput:
    """Compact streaming-output renderer for the lap flow.

    Lap sessions are short (one section per click — typically 3-6 VLM /
    tool calls for Claude, one VLM call for local). We don't need the
    detailed flow's per-step expander UI — a header + a single live block
    plus a tool-event chip row is enough.
    """

    def __init__(self) -> None:
        self.start_time = time.time()
        self.header = st.empty()
        self.tool_chip_area = st.empty()
        self.text_area = st.empty()
        self.thinking_area = st.empty()
        self.tool_events: List[str] = []
        self.text_chunks: List[str] = []
        self.thinking_chunks: List[str] = []
        self._render()

    def _render(self) -> None:
        elapsed = time.time() - self.start_time
        self.header.markdown(f"**Lap agent run** _(elapsed {elapsed:.1f}s)_")
        if self.tool_events:
            self.tool_chip_area.caption(" · ".join(self.tool_events[-8:]))
        if self.thinking_chunks:
            self.thinking_area.markdown(
                f"_💭 Thinking:_\n\n{''.join(self.thinking_chunks)}"
            )
        if self.text_chunks:
            self.text_area.markdown(f"_Response:_\n\n{''.join(self.text_chunks)}")

    def on_prompt(self, prompt: str, stage: Dict[str, Any]) -> None:  # noqa: ARG002
        # Prompt is large; show only that we started.
        self.tool_events.append("🟢 prompt sent")
        self._render()

    def on_text_chunk(self, chunk: str) -> None:
        self.text_chunks.append(chunk)
        self._render()

    def on_reasoning_chunk(self, chunk: str) -> None:
        self.thinking_chunks.append(chunk)
        self._render()

    def on_step_event(self, summary: str, stage: Dict[str, Any]) -> None:
        phase = stage.get("phase") or ""
        chip = phase.replace("tool:", "🔧 ") if phase.startswith("tool:") else f"📎 {phase}"
        self.tool_events.append(chip)
        self._render()

    def on_progress(self, node_name: str, detail: str) -> None:
        self.tool_events.append(f"⏱ {detail}")
        self._render()


# ---------------------------------------------------------------------------
# Agent invocation wrapper
# ---------------------------------------------------------------------------

def execute_lap_agent_run(
    *,
    run_fn: Callable,
    df,
    lap_start: int,
    lap_end: int,
    head_segment: Dict[str, Any],
    circuit_id: str,
    existing: List[Dict[str, Any]],
    extra_kwargs: Dict[str, Any],
) -> None:
    """Invoke a lap-annotation runner with live UI feedback.

    On success the result is stashed in ``st.session_state[KEY_LAP_STAGED]``
    for the staged-review panel below. On revision, the segments array is
    rebuilt from the revised end onward.
    """
    progress_area = st.container()
    status_text = progress_area.empty()
    progress_bar = progress_area.progress(0)
    live = LapLiveOutput()

    section_id = head_segment["circuit_section_id"]
    section_start = int(head_segment["start_index"])
    section_end = int(head_segment["end_index"])

    try:
        with st.spinner(f"Annotating section {section_id} …"):
            result = run_fn(
                df=df,
                lap_start=int(lap_start),
                lap_end=int(lap_end),
                section_id=section_id,
                section_start=section_start,
                section_end=section_end,
                circuit_id=circuit_id,
                existing_section_annotations=existing,
                progress_callback=live.on_progress,
                vlm_stream_callback=live.on_text_chunk,
                vlm_prompt_callback=live.on_prompt,
                vlm_reasoning_callback=live.on_reasoning_chunk,
                step_event_callback=live.on_step_event,
                **extra_kwargs,
            )
    except Exception as e:
        status_text.error(f"Agent error: {e}")
        st.code(traceback.format_exc())
        return

    progress_bar.progress(1.0)

    # Rebuild downstream segments if the agent revised the boundary.
    if result.revised:
        downstream = rebuild_remaining_segments(
            df, int(lap_start), int(lap_end), circuit_id, int(result.end_index),
        )
        # Replace the head segment with the agent's revised one and prepend
        # the rebuilt tail. The head is now the just-annotated section; the
        # UI advances the cursor when the user confirms the staged review.
        segments = list(st.session_state.get(KEY_LAP_SEGMENTS, []))
        cursor = int(st.session_state.get(KEY_LAP_CURSOR, 0))
        # Mutate the head and discard everything after it; then append the rebuilt tail.
        if 0 <= cursor < len(segments):
            head = dict(segments[cursor])
            head["start_index"] = int(result.start_index)
            head["end_index"] = int(result.end_index)
            head["circuit_section_id"] = result.section_id
            segments = segments[: cursor + 1] + downstream
            segments[cursor] = head
            st.session_state[KEY_LAP_SEGMENTS] = segments

    st.session_state[KEY_LAP_STAGED] = {
        "section_id": result.section_id,
        "start_index": int(result.start_index),
        "end_index": int(result.end_index),
        "label_ids": list(result.label_ids),
        "reasoning": result.reasoning,
        "revised": bool(result.revised),
        "submitted": bool(result.submitted),
        "rough_start": int(result.rough_start),
        "rough_end": int(result.rough_end),
        "rejected": list(result.rejected_proposals),
        "rendered_images": list(result.rendered_images),
        "tool_calls": int(result.tool_calls),
    }


# ---------------------------------------------------------------------------
# Lap panel (range picker + split + array view)
# ---------------------------------------------------------------------------

def render_lap_panel(df, circuit_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Render the lap-range picker + array view.

    The split runs automatically: once when the user first sets the range
    (or changes it), and again after every successful save (the saved
    section's `end` becomes the boundary the next split starts from). The
    user does not press a button — the array is always derived fresh from
    the current ``Graphics_normalized_car_position`` slice.

    Returns the *current head segment* (or None) for the caller to feed
    into ``execute_lap_agent_run``.
    """
    if not circuit_id:
        st.warning(
            "Cannot detect the circuit from `Static_track`. The "
            "lap-to-segment excerpter needs a recognised circuit "
            "(brands_hatch / silverstone). Skipping."
        )
        return None

    st.caption(
        f"Detected circuit: `{circuit_id}` — the splitter partitions the "
        "picked range automatically using each section's "
        "`normalized_position_range`."
    )

    default_start = 0
    default_end = min(len(df), 5000)
    prev = st.session_state.get(KEY_LAP_RANGE)
    if isinstance(prev, (list, tuple)) and len(prev) == 2:
        default_start, default_end = int(prev[0]), int(prev[1])

    col1, col2 = st.columns(2)
    with col1:
        lap_start = st.number_input(
            "Lap start index", min_value=0, max_value=max(len(df) - 1, 0),
            value=min(default_start, max(len(df) - 1, 0)),
            key="lap_agent_lap_start",
        )
    with col2:
        lap_end = st.number_input(
            "Lap end index", min_value=1, max_value=len(df),
            value=min(default_end, len(df)),
            key="lap_agent_lap_end",
        )

    # ----------------------------------------------------------------------
    # Auto-split trigger.
    #
    # We re-run the deterministic splitter whenever any of the inputs that
    # define the array have changed — the picked range, the circuit, or the
    # split version stamp (bumped by the persist hook after a successful
    # save so the next click sees a freshly partitioned tail).
    # ----------------------------------------------------------------------
    if lap_end - lap_start < 3:
        st.warning(
            f"Lap range too short — pick at least 3 ilocs "
            f"(currently {lap_end - lap_start})."
        )
        return None

    desired_key = (int(lap_start), int(lap_end), circuit_id,
                   int(st.session_state.get("lap_agent_split_version", 0)))
    last_key = st.session_state.get("lap_agent_split_key")
    if desired_key != last_key:
        st.session_state[KEY_LAP_RANGE] = (int(lap_start), int(lap_end))
        st.session_state[KEY_LAP_CIRCUIT] = circuit_id
        # The "auto re-split" hook (set by _persist_lap_annotation) tells us
        # to keep the cursor where it is and re-split only the downstream
        # tail. Otherwise (fresh range / circuit change) we replace the
        # whole array and reset the cursor.
        rebuild_from = st.session_state.pop("lap_agent_rebuild_from_iloc", None)
        if rebuild_from is None:
            segments = run_split(df, int(lap_start), int(lap_end), circuit_id)
            st.session_state[KEY_LAP_SEGMENTS] = segments
            st.session_state[KEY_LAP_CURSOR] = 0
            st.session_state.pop(KEY_LAP_STAGED, None)
        else:
            cursor = int(st.session_state.get(KEY_LAP_CURSOR, 0))
            existing = list(st.session_state.get(KEY_LAP_SEGMENTS, []))
            tail = run_split(
                df, int(rebuild_from), int(lap_end), circuit_id,
            )
            st.session_state[KEY_LAP_SEGMENTS] = existing[:cursor] + tail
            st.session_state[KEY_LAP_CURSOR] = cursor
        st.session_state["lap_agent_split_key"] = desired_key

    segments: List[Dict[str, Any]] = st.session_state.get(KEY_LAP_SEGMENTS, [])
    if not segments:
        st.info(
            "The splitter produced zero sections — typically because the "
            "circuit's `normalized_position_range` values in the label "
            "catalog are not yet measured for the picked range."
        )
        return None

    cursor = int(st.session_state.get(KEY_LAP_CURSOR, 0))

    st.markdown(f"##### Rough split — {len(segments)} section(s)")
    st.caption(
        "Auto-split from `split_lap_by_circuit_sections`. The cursor advances "
        "after each saved annotation; the tail re-splits using the saved "
        "end as the new boundary."
    )
    rows = []
    for i, seg in enumerate(segments):
        marker = "▶" if i == cursor else "  "
        rows.append(
            f"{marker} `#{i}` `{seg['circuit_section_id']}` "
            f"({seg.get('circuit_section_name', '')}) — "
            f"[{seg['start_index']}, {seg['end_index']}] "
            f"(coverage {seg.get('coverage_fraction', 0):.0%})"
        )
    st.code("\n".join(rows), language="text")

    if cursor >= len(segments):
        st.success("All sections annotated. Pick a new lap range above to continue.")
        if st.button("↺ Reset cursor", key="lap_agent_reset_cursor"):
            st.session_state[KEY_LAP_CURSOR] = 0
            st.rerun()
        return None

    head = segments[cursor]
    st.markdown(
        f"**Current section ({cursor + 1} / {len(segments)}):** "
        f"`{head['circuit_section_id']}` — "
        f"[{head['start_index']}, {head['end_index']}]"
    )

    col_skip, col_drop = st.columns(2)
    with col_skip:
        if st.button("⏭ Skip this section (no annotation)", key="lap_agent_skip"):
            st.session_state[KEY_LAP_CURSOR] = cursor + 1
            st.session_state.pop(KEY_LAP_STAGED, None)
            st.rerun()
    with col_drop:
        if st.button("🗑 Remove section from array", key="lap_agent_drop"):
            new_segments = segments[:cursor] + segments[cursor + 1:]
            st.session_state[KEY_LAP_SEGMENTS] = new_segments
            st.session_state.pop(KEY_LAP_STAGED, None)
            st.rerun()

    return head


# ---------------------------------------------------------------------------
# Staged review (post-agent)
# ---------------------------------------------------------------------------

def render_lap_staged_review(session_id: str, selected_annotation_key: str) -> None:
    """Editable staged-review panel for the last agent run."""
    staged = st.session_state.get(KEY_LAP_STAGED)
    if not staged:
        return

    st.markdown("---")
    st.markdown("##### Review & Edit Before Saving")
    if staged.get("revised"):
        st.caption(
            f"ℹ️ Agent revised the boundary: "
            f"rough [{staged['rough_start']}, {staged['rough_end']}] → "
            f"new [{staged['start_index']}, {staged['end_index']}]. "
            "Downstream array has been rebuilt."
        )
    if staged.get("rejected"):
        st.caption(
            f"⚠️ {len(staged['rejected'])} label_id(s) rejected by the runner: "
            + ", ".join(str(r.get("value")) for r in staged["rejected"])
        )

    with st.container(border=True):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            seg_start = st.number_input(
                "Start", min_value=0, max_value=10_000_000,
                value=int(staged["start_index"]), key="lap_staged_start",
            )
        with col_r2:
            seg_end = st.number_input(
                "End", min_value=0, max_value=10_000_000,
                value=int(staged["end_index"]), key="lap_staged_end",
            )

        all_label_options = sorted(LABEL_MAPPING.values())
        default_labels = [
            LABEL_MAPPING.get(l, l) for l in staged["label_ids"] if l in LABEL_MAPPING
        ]
        seg_labels = st.multiselect(
            "Labels", options=all_label_options, default=default_labels,
            key="lap_staged_labels",
        )
        seg_notes = st.text_area(
            "Reasoning / notes", value=staged.get("reasoning", "")[:1500],
            key="lap_staged_notes", height=100,
        )

    if staged.get("rendered_images"):
        with st.expander(
            f"📊 Rendered graphs ({len(staged['rendered_images'])} image(s))",
            expanded=False,
        ):
            cols = st.columns(min(len(staged["rendered_images"]), 3))
            for idx, img_bytes in enumerate(staged["rendered_images"]):
                with cols[idx % len(cols)]:
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, width="stretch")
                    except Exception as e:
                        st.error(f"Graph {idx + 1} failed to render: {e}")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button(
            "✅ Save & advance",
            key="lap_staged_save",
            type="primary",
            disabled=not seg_labels,
        ):
            _persist_lap_annotation(
                start=int(seg_start), end=int(seg_end),
                label_names=seg_labels, notes=seg_notes,
                session_id=session_id,
                selected_annotation_key=selected_annotation_key,
            )
    with col_btn2:
        if st.button("⏭ Skip & advance (don't save)", key="lap_staged_skip"):
            _advance_cursor_and_clear_staged()
    with col_btn3:
        if st.button("❌ Discard agent result", key="lap_staged_discard"):
            st.session_state.pop(KEY_LAP_STAGED, None)
            st.rerun()


def _persist_lap_annotation(
    *, start: int, end: int, label_names: List[str], notes: str,
    session_id: str, selected_annotation_key: str,
) -> None:
    if start >= end:
        st.error("Start must be less than end.")
        return
    label_ids = [LABEL_NAME_TO_ID[n] for n in label_names if n in LABEL_NAME_TO_ID]
    if not label_ids:
        st.error("No valid labels resolved.")
        return

    new_ann = AnnotatedSegment(
        id=str(uuid.uuid4()),
        labels=label_ids,
        segment_length=end - start,
        start_index=start,
        end_index=end,
        notes=notes,
    )
    annotations = list(st.session_state.get("current_annotations", []))
    annotations.append(new_ann)
    st.session_state["current_annotations"] = annotations
    save_annotations(session_id, annotations, selected_annotation_key)

    # Advance the cursor past the just-saved head, then ask the lap panel
    # to re-split the tail from the saved end on the next render. The
    # version stamp bump invalidates ``lap_agent_split_key`` so the auto-
    # split branch fires; ``rebuild_from_iloc`` makes it a tail-only
    # rebuild (cursor preserved) instead of a full reset.
    cursor = int(st.session_state.get(KEY_LAP_CURSOR, 0))
    st.session_state[KEY_LAP_CURSOR] = cursor + 1
    st.session_state["lap_agent_rebuild_from_iloc"] = int(end)
    st.session_state["lap_agent_split_version"] = (
        int(st.session_state.get("lap_agent_split_version", 0)) + 1
    )
    st.session_state.pop(KEY_LAP_STAGED, None)
    st.rerun()


def _advance_cursor_and_clear_staged() -> None:
    cursor = int(st.session_state.get(KEY_LAP_CURSOR, 0))
    st.session_state[KEY_LAP_CURSOR] = cursor + 1
    st.session_state.pop(KEY_LAP_STAGED, None)
    st.rerun()
