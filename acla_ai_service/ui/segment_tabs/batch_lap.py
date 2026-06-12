import streamlit as st
import time
import traceback
from pprint import pformat

from .batch import _load_batch_session
from .components.annotation_provider_controls import render_annotation_provider_config
from .components.opponent_interaction import format_targeted_car
from .shared import build_segment, save_annotations, LABEL_MAPPING
from app.local_annotation_agent import ClaudeUsageExhausted


_USAGE_EXHAUSTED_WARNING = (
    "⚠️ Claude usage is exhausted (Max-plan quota / 5-hour window / "
    "credit balance). Batch halted — try again later."
)

_ERROR_DETAIL_ATTRS = (
    "result",
    "error",
    "message",
    "details",
    "data",
    "payload",
    "stdout",
    "stderr",
    "returncode",
    "exit_code",
    "code",
    "status",
)


def _format_error_details(exc: BaseException) -> str:
    """Return an inspectable error report for SDK exceptions."""
    lines = [
        f"type: {type(exc).__module__}.{type(exc).__qualname__}",
        f"str: {str(exc)!r}",
        f"repr: {repr(exc)}",
    ]
    if exc.args:
        lines.append(f"args: {pformat(exc.args)}")

    seen = set()
    for attr in _ERROR_DETAIL_ATTRS:
        if hasattr(exc, attr):
            try:
                value = getattr(exc, attr)
            except Exception as attr_exc:
                value = f"<failed to read: {attr_exc!r}>"
            lines.append(f"{attr}: {pformat(value)}")
            seen.add(attr)

    public_attrs = {
        key: value
        for key, value in getattr(exc, "__dict__", {}).items()
        if not key.startswith("_") and key not in seen
    }
    if public_attrs:
        lines.append(f"public_attrs: {pformat(public_attrs)}")

    if exc.__cause__ is not None:
        lines.append(
            "cause: "
            f"{type(exc.__cause__).__module__}.{type(exc.__cause__).__qualname__} "
            f"{exc.__cause__!r}"
        )
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        lines.append(
            "context: "
            f"{type(exc.__context__).__module__}.{type(exc.__context__).__qualname__} "
            f"{exc.__context__!r}"
        )

    tb = traceback.format_exc().rstrip()
    if tb:
        lines.append("traceback:")
        lines.append(tb)
    return "\n".join(lines)


def _targeted_car_suffix(opponent_interaction) -> str:
    target = format_targeted_car(opponent_interaction)
    return f", target={target}" if target else ""


def _render_provider_config(key_prefix: str, *, default_temperature: float, default_max_new_tokens: int):
    return render_annotation_provider_config(
        key_prefix=key_prefix,
        default_temperature=default_temperature,
        default_max_new_tokens=default_max_new_tokens,
        default_tool_budget=3,
    )


def render_batch_lap_agent_claude(df, session_id, selected_annotation_key):
    """Batch provider-selected Lap-to-Segment Excerpter."""
    from .components._lap_agent_shared import (
        track_name_to_circuit_id, run_split, rebuild_remaining_segments,
    )

    st.header("Batch Lap-to-Segment Excerpter (AI Provider)")
    st.write(
        "Pick a lap range; the deterministic splitter partitions it into "
        "per-`circuit_section` sub-ranges. If opponent data is present, it "
        "emits only close racing-interaction windows. "
        "The selected AI provider annotates every section automatically and auto-saves each "
        "result as a new segment."
    )

    range_context = (session_id, selected_annotation_key, len(df))
    if st.session_state.get("batch_lap_claude_range_context") != range_context:
        st.session_state["batch_lap_claude_range_context"] = range_context
        st.session_state["batch_lap_claude_start"] = 0
        st.session_state["batch_lap_claude_end"] = min(len(df), 5000)

    track_name = (
        df["Static_track"].iloc[0]
        if "Static_track" in df.columns and not df.empty else None
    )
    circuit_id = track_name_to_circuit_id(track_name)
    if not circuit_id:
        st.warning(
            "Cannot detect the circuit from `Static_track`. The lap "
            "excerpter needs a recognised circuit. Skipping."
        )
        return
    st.caption(f"Detected circuit: `{circuit_id}`")

    col1, col2 = st.columns(2)
    with col1:
        lap_start = st.number_input(
            "Lap start index", min_value=0, max_value=max(len(df) - 1, 0),
            key="batch_lap_claude_start",
        )
    with col2:
        lap_end = st.number_input(
            "Lap end index", min_value=1, max_value=len(df),
            key="batch_lap_claude_end",
        )

    if lap_end - lap_start < 3:
        st.warning(f"Lap range too short — pick at least 3 ilocs (currently {lap_end - lap_start}).")
        return

    coverage_slot = st.empty()
    _render_lap_coverage_bar(
        coverage_slot,
        int(lap_start),
        int(lap_end),
        chart_key="batch_lap_claude_coverage_initial",
    )

    st.markdown("---")
    config = _render_provider_config(
        "batch_lap_provider",
        default_temperature=0.3,
        default_max_new_tokens=1500,
    )

    st.markdown("---")
    skip_overlap = st.checkbox(
        "Skip sections that overlap existing annotations",
        value=True, key="batch_lap_claude_skip_overlap",
        help=(
            "Recommended — avoids re-annotating sections you've already "
            "labelled. When unchecked, the agent runs on every section "
            "and replaces any overlapping annotations and their children."
        ),
    )

    if "batch_lap_claude_logs" not in st.session_state:
        st.session_state["batch_lap_claude_logs"] = []

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "▶ Run Batch Lap Excerpter",
            key="batch_lap_claude_run", type="primary",
        )
    with col_clear:
        if st.button("Clear log", key="batch_lap_claude_clear_log"):
            st.session_state["batch_lap_claude_logs"] = []
            st.rerun()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_area = st.empty()
    logs = st.session_state["batch_lap_claude_logs"]

    def _flush_log():
        if logs:
            log_area.code("\n".join(logs), language="text", line_numbers=True)

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] {msg}")
        if len(logs) > 1000:
            del logs[: len(logs) - 1000]
        _flush_log()

    _flush_log()

    if not run_clicked:
        return
    if config is None:
        st.error("No annotation AI provider is available.")
        return

    try:
        from app.local_annotation_agent.workflow import run_annotation
    except ImportError as e:
        st.error(
            f"Missing dependency: {e}"
        )
        return

    segments = run_split(df, int(lap_start), int(lap_end), circuit_id)
    if not segments:
        split_meta = st.session_state.get("lap_agent_split_meta", {}) or {}
        if split_meta.get("opponent_session"):
            st.info(
                "Opponent data is present, but no close overtake offence / "
                "defense engagement window was found in the picked range."
            )
        else:
            st.info(
                "The splitter produced zero sections — the circuit's "
                "`normalized_position_range` values may not cover the picked range."
            )
        return

    log(f"Starting batch lap excerpter: {len(segments)} section(s), "
        f"lap=[{int(lap_start)}, {int(lap_end)}], circuit={circuit_id}, "
        f"provider={config.provider_id}")

    saved_count = 0
    skipped_count = 0
    error_count = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        sec_id = seg["circuit_section_id"]
        sec_start = int(seg["start_index"])
        sec_end = int(seg["end_index"])
        target_suffix = _targeted_car_suffix(seg.get("opponent_interaction"))

        if skip_overlap and _section_overlaps_existing(sec_start, sec_end):
            log(
                f"Section #{i} `{sec_id}` [{sec_start}, {sec_end}]"
                f"{target_suffix}: skipped (overlaps existing annotation)."
            )
            skipped_count += 1
            i += 1
            progress_bar.progress((i) / len(segments))
            continue

        existing = _collect_existing_lap_annotations(int(lap_start), int(lap_end))

        status_text.markdown(
            f"**Section #{i + 1}/{len(segments)}** `{sec_id}`"
            f"{target_suffix} — [{sec_start}, {sec_end}], running {config.provider_id}..."
        )
        log(f"Section #{i} `{sec_id}`{target_suffix}: running [{sec_start}, {sec_end}]")

        try:
            result = run_annotation(
                flow="lap",
                df=df,
                config=config,
                session_id=session_id,
                lap_start=int(lap_start),
                lap_end=int(lap_end),
                section_id=sec_id,
                section_start=sec_start,
                section_end=sec_end,
                circuit_id=circuit_id,
                section_split_basis=seg.get("split_basis"),
                opponent_interaction=seg.get("opponent_interaction"),
                existing_section_annotations=existing,
            )
        except ClaudeUsageExhausted as e:
            log(f"Section #{i} `{sec_id}`: HALTED — Claude usage exhausted: {e}")
            st.warning(_USAGE_EXHAUSTED_WARNING)
            i += 1
            progress_bar.progress(i / len(segments))
            break
        except Exception as e:
            error_count += 1
            log(f"Section #{i} `{sec_id}`: ERROR — {e}")
            log(
                f"Section #{i} `{sec_id}`: detailed error result\n"
                f"{_format_error_details(e)}"
            )
            i += 1
            progress_bar.progress(i / len(segments))
            continue

        label_ids = [l for l in result.label_ids if l in LABEL_MAPPING]
        if not label_ids:
            log(f"Section #{i} `{sec_id}`{target_suffix}: no valid labels resolved — skipped.")
            error_count += 1
            i += 1
            progress_bar.progress(i / len(segments))
            continue

        new_ann = build_segment(
            df,
            start=int(result.start_index),
            end=int(result.end_index),
            label_ids=label_ids,
            notes=(result.reasoning or "")[:1500],
            opponent_interaction=seg.get("opponent_interaction"),
        )
        annotations = list(st.session_state.get("current_annotations", []))
        removed_parents = 0
        removed_children = 0
        if _section_overlaps_existing(int(result.start_index), int(result.end_index)):
            if skip_overlap:
                log(
                    f"Section #{i} `{sec_id}`{target_suffix}: skipped after agent "
                    f"(final range [{result.start_index}, {result.end_index}] "
                    "overlaps existing annotation)."
                )
                skipped_count += 1
                i += 1
                progress_bar.progress(i / len(segments))
                continue

            annotations, removed_parents, removed_children = (
                _remove_overlapping_annotations_and_children(
                    annotations,
                    int(result.start_index),
                    int(result.end_index),
                )
            )
        annotations.append(new_ann)
        st.session_state["current_annotations"] = annotations
        save_annotations(session_id, annotations, selected_annotation_key, silent=True)
        saved_count += 1

        if result.revised:
            tail = rebuild_remaining_segments(
                df, int(lap_start), int(lap_end), circuit_id, int(result.end_index),
            )
            segments = segments[: i + 1] + tail
            segments[i] = {
                **segments[i],
                "start_index": int(result.start_index),
                "end_index": int(result.end_index),
                "circuit_section_id": result.section_id,
            }
            log(f"Section #{i} `{sec_id}`{target_suffix}: saved (revised → "
                f"[{result.start_index}, {result.end_index}]); tail rebuilt → "
                f"{len(tail)} downstream section(s).")
        else:
            log(f"Section #{i} `{sec_id}`{target_suffix}: saved [{result.start_index}, {result.end_index}] "
                f"with {len(label_ids)} label(s).")
        if removed_parents or removed_children:
            log(
                f"Section #{i} `{sec_id}`{target_suffix}: replaced "
                f"{removed_parents} overlapping annotation(s) and "
                f"{removed_children} child annotation(s)."
            )

        i += 1
        progress_bar.progress(i / len(segments))

    progress_bar.progress(1.0)
    status_text.markdown(
        f"**Done.** Saved: {saved_count}, skipped: {skipped_count}, errors: {error_count}."
    )
    log(f"Finished. {saved_count} saved, {skipped_count} skipped, {error_count} error(s).")

    _render_lap_coverage_bar(
        coverage_slot,
        int(lap_start),
        int(lap_end),
        chart_key="batch_lap_claude_coverage_final",
    )


def _section_overlaps_existing(sec_start: int, sec_end: int) -> bool:
    """True if any current annotation overlaps the section range."""
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        if e > sec_start and s < sec_end:
            return True
    return False


def _remove_overlapping_annotations_and_children(
    annotations: list,
    sec_start: int,
    sec_end: int,
) -> tuple[list, int, int]:
    """Drop top-level annotations overlapping a range, plus their children."""
    parent_ids_to_remove = set()
    for ann in annotations:
        if not _annotation_overlaps(ann, sec_start, sec_end):
            continue
        parent_id = getattr(ann, "parent_id", None)
        if parent_id:
            parent_ids_to_remove.add(parent_id)
            continue
        ann_id = getattr(ann, "id", None)
        if ann_id:
            parent_ids_to_remove.add(ann_id)

    kept = []
    removed_parents = 0
    removed_children = 0
    for ann in annotations:
        parent_id = getattr(ann, "parent_id", None)
        if parent_id and parent_id in parent_ids_to_remove:
            removed_children += 1
            continue
        if not parent_id and _annotation_overlaps(ann, sec_start, sec_end):
            removed_parents += 1
            continue
        kept.append(ann)
    return kept, removed_parents, removed_children


def _annotation_overlaps(ann, sec_start: int, sec_end: int) -> bool:
    s = int(getattr(ann, "start_index", 0) or 0)
    e = int(getattr(ann, "end_index", 0) or 0)
    return e > sec_start and s < sec_end


def _normalise_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raw = [(int(s), int(e)) for s, e in ranges if int(e) > int(s)]
    raw.sort()
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _compute_interval_coverage(
    target_ranges: list[tuple[int, int]],
    annotation_ranges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    targets = _normalise_ranges(target_ranges)
    annotations = _normalise_ranges(annotation_ranges)

    raw_covered: list[tuple[int, int]] = []
    for ts, te in targets:
        for s, e in annotations:
            if e <= ts or s >= te:
                continue
            raw_covered.append((max(s, ts), min(e, te)))
    covered = _normalise_ranges(raw_covered)

    gaps: list[tuple[int, int]] = []
    for ts, te in targets:
        pos = ts
        for s, e in covered:
            if e <= ts or s >= te:
                continue
            cs, ce = max(s, ts), min(e, te)
            if cs > pos:
                gaps.append((pos, cs))
            pos = max(pos, ce)
        if pos < te:
            gaps.append((pos, te))
    return covered, gaps


def _render_coverage_bar(
    slot,
    target_ranges: list[tuple[int, int]],
    annotation_ranges: list[tuple[int, int]],
    *,
    title: str,
    legend_note: str,
    chart_key: str,
) -> None:
    import plotly.graph_objects as go

    targets = _normalise_ranges(target_ranges)
    if not targets:
        slot.empty()
        return

    covered, gaps = _compute_interval_coverage(targets, annotation_ranges)
    total = sum(e - s for s, e in targets)
    covered_len = sum(e - s for s, e in covered)
    gap_len = total - covered_len
    coverage_pct = (covered_len / total * 100) if total else 0.0
    longest_gap = max((e - s for s, e in gaps), default=0)
    x_min = min(s for s, _ in targets)
    x_max = max(e for _, e in targets)

    fig = go.Figure()
    for s, e in targets:
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(220, 53, 69, 0.75)", line=dict(width=0), layer="below",
        )
    for s, e in covered:
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(40, 167, 69, 0.9)", line=dict(width=0), layer="below",
        )
    if gaps:
        fig.add_trace(go.Scatter(
            x=[(s + e) / 2 for s, e in gaps],
            y=[0.5] * len(gaps),
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            hovertext=[f"gap [{s}, {e}] · {e - s} iloc(s)" for s, e in gaps],
            hoverinfo="text",
            showlegend=False,
        ))
    fig.update_layout(
        height=110,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(range=[x_min, x_max], title="iloc index", fixedrange=True),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        showlegend=False,
    )

    with slot.container():
        st.caption(
            f"**{title}:** {coverage_pct:.1f}% covered · "
            f"{len(gaps)} gap(s) · {gap_len} iloc(s) uncovered · "
            f"longest gap {longest_gap} iloc(s)  "
            f"{legend_note}"
        )
        st.plotly_chart(fig, width="stretch", key=chart_key)


def _render_lap_coverage_bar(
    slot,
    lap_start: int,
    lap_end: int,
    *,
    chart_key: str,
) -> None:
    annotation_ranges = []
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        annotation_ranges.append((s, e))
    _render_coverage_bar(
        slot,
        [(lap_start, lap_end)],
        annotation_ranges,
        title="Annotation coverage",
        legend_note="(🟩 annotated · 🟥 not yet reached)",
        chart_key=chart_key,
    )


def _collect_existing_lap_annotations(lap_start: int, lap_end: int):
    """Annotations overlapping the lap range — passed as dup-avoidance hints."""
    out = []
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        if e <= lap_start or s >= lap_end:
            continue
        out.append({
            "start_index": s,
            "end_index": e,
            "labels": list(getattr(ann, "labels", [])),
        })
    return out


def render_batch_lap(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_batch_lap_agent_claude(df, session_id, selected_annotation_key)
