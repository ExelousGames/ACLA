import streamlit as st
import time
import traceback
import copy
import pandas as pd

from .shared import (
    build_segment,
    get_available_sessions,
    load_annotations,
    load_session_segments,
    save_annotations,
    LABEL_CATEGORIES,
    LABEL_MAPPING,
)
from app.local_annotation_agent.workflow import AnnotationPipelineConfig
_MIN_DISCOVERED_CHILD_LENGTH = 3


def _render_provider_config(key_prefix: str, *, default_temperature: float, default_max_new_tokens: int):
    return AnnotationPipelineConfig(provider_id="deterministic")


def _persist_children_for_parent(parent, result, session_id, selected_annotation_key, df):
    """Auto-save deterministically discovered children under ``parent``."""
    from .components._agent_annotation_shared import (
        group_proposals_by_range,
        with_parent_label_ids,
    )

    grouped = group_proposals_by_range(result)
    main_label_ids = set(LABEL_CATEGORIES.get("Main Labels", []))
    parent_main_label_ids = [
        label for label in getattr(parent, "labels", [])
        if label in main_label_ids
    ]

    new_children = []
    for (gs, ge), anns in grouped:
        if ge - gs < _MIN_DISCOVERED_CHILD_LENGTH:
            continue
        label_ids = [a["label_id"] for a in anns if a.get("label_id") in LABEL_MAPPING]
        if not label_ids:
            continue
        notes = "; ".join(a.get("reasoning", "") for a in anns if a.get("reasoning"))
        new_children.append(build_segment(
            df,
            start=int(gs),
            end=int(ge),
            label_ids=with_parent_label_ids(label_ids, parent_main_label_ids),
            notes=notes,
            parent_id=parent.id,
        ))

    if not new_children:
        return 0

    annotations = list(st.session_state.get("current_annotations", []))
    annotations.extend(new_children)
    st.session_state["current_annotations"] = annotations
    save_annotations(session_id, annotations, selected_annotation_key, silent=True)
    return len(new_children)


def _delete_selected_parent_subsegments(
    session_id,
    selected_annotation_key,
    selected_parent_ids: set[str],
) -> int:
    """Remove child sub-segments belonging to the selected parents."""
    annotations = list(st.session_state.get("current_annotations", []))
    remaining_annotations = [
        ann for ann in annotations
        if getattr(ann, "parent_id", None) not in selected_parent_ids
    ]
    deleted = len(annotations) - len(remaining_annotations)
    if deleted:
        st.session_state["current_annotations"] = remaining_annotations
        save_annotations(
            session_id,
            remaining_annotations,
            selected_annotation_key,
            silent=True,
        )
    return deleted


def _segments_to_positioned_dataframe(segments) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    max_position = 0

    for segment in segments:
        rows = getattr(segment, "telemetry_data", None) or []
        start = getattr(segment, "start_index", None)
        end = getattr(segment, "end_index", None)
        if end is not None:
            max_position = max(max_position, int(end))
        if not rows or start is None:
            continue

        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame.index = range(int(start), int(start) + len(frame))
        frames.append(frame)
        max_position = max(max_position, int(start) + len(frame))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.reindex(range(max_position + 1))


def _load_batch_segment_input(
    selected_annotation_key,
    selected_session_key,
    available_sessions,
):
    """Load segment chunks only; batch sub-segment discovery does not run on raw laps."""
    chunk_summaries = {}
    for session_id in available_sessions:
        input_segments = load_session_segments(selected_session_key, session_id)
        total_count = len([
            segment for segment in input_segments
            if not getattr(segment, "parent_id", None)
        ])
        if not total_count:
            total_count = len(input_segments)
        if total_count <= 0:
            continue

        try:
            saved_annotations = load_annotations(session_id, selected_annotation_key)
        except Exception:
            saved_annotations = []

        parent_ids = {
            getattr(segment, "id", None)
            for segment in input_segments
            if not getattr(segment, "parent_id", None) and getattr(segment, "id", None)
        }
        if not parent_ids:
            parent_ids = {
                getattr(segment, "id", None)
                for segment in input_segments
                if getattr(segment, "id", None)
            }
        annotated_parent_ids = {
            getattr(segment, "parent_id", None)
            for segment in saved_annotations
            if getattr(segment, "parent_id", None) in parent_ids
        }
        annotated_count = min(len(annotated_parent_ids), total_count)
        chunk_summaries[session_id] = {
            "total": total_count,
            "annotated": annotated_count,
            "unannotated": total_count - annotated_count,
        }

    segment_sessions = list(chunk_summaries)
    if not segment_sessions:
        st.subheader("Batch Auto-Annotation (Sub-Segment Discovery)")
        st.error(
            "Batch Sub-Segment Discovery only works on input chunks that "
            "contain segments. Select a segment-output dataset, such as a "
            "parent annotation node's output."
        )
        return None, None

    def format_session_option(session_id: str) -> str:
        summary = chunk_summaries[session_id]
        if summary["unannotated"] == 0:
            status = "✅"
        elif summary["annotated"] > 0:
            status = "🟡"
        else:
            status = "⭕"
        return (
            f"{status} {session_id} | {summary['total']} segments | "
            f"{summary['annotated']} annotated / {summary['unannotated']} unannotated"
        )

    index = 0
    previous_selection = st.session_state.get("batch_subseg_chunk_selector")
    if previous_selection in segment_sessions:
        index = segment_sessions.index(previous_selection)

    col_sel1, _ = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Segment Chunk",
            options=segment_sessions,
            format_func=format_session_option,
            index=index,
            key="batch_subseg_chunk_selector",
        )

    with st.spinner(f"Loading segment chunk {session_id}..."):
        input_segments = load_session_segments(selected_session_key, session_id)
        if not input_segments:
            st.error("Selected input chunk has no segments.")
            return None, None

        try:
            saved_annotations = load_annotations(session_id, selected_annotation_key)
        except Exception:
            saved_annotations = []
        state_key = (
            selected_session_key,
            selected_annotation_key,
            session_id,
            len(input_segments),
            len(saved_annotations),
        )
        if st.session_state.get("batch_subseg_loaded_state_key") != state_key:
            if saved_annotations:
                st.session_state.current_annotations = saved_annotations
            else:
                st.session_state.current_annotations = copy.deepcopy(input_segments)
            st.session_state.batch_subseg_loaded_state_key = state_key
            st.session_state.last_session_id = session_id
            st.session_state.last_annotation_key = selected_annotation_key

    df = _segments_to_positioned_dataframe(input_segments)
    if df.empty:
        st.warning("Selected segment chunk has no telemetry rows.")
        return None, None

    if "Static_track" in df.columns:
        track_names = df["Static_track"].dropna()
        if not track_names.empty:
            st.markdown(f"**Track:** {track_names.iloc[0]}")

    return df, session_id


def render_batch_auto_annotation(df, selected_annotation_key):
    """Batch deterministic sub-segment discovery."""
    st.header("Batch Auto-Annotation (Sub-Segment Discovery)")
    st.write(
        "For each parent segment in the selected range, run the **Sub-Segment Discovery** "
        "deterministic requirements and auto-save discovered children."
    )

    if not st.session_state.get("current_annotations"):
        st.info("No segments available to process. Please create segments first.")
        return

    annotations = st.session_state.current_annotations
    parent_annotations = [
        ann for ann in annotations
        if not getattr(ann, "parent_id", None)
    ]
    total_parents = len(parent_annotations)
    if not parent_annotations:
        st.info("No parent segments available to process.")
        return

    if total_parents > 1:
        batch_range = st.slider("Select Parent Segment Range", 0, total_parents - 1,
                                (0, total_parents - 1), step=1)
    else:
        batch_range = (0, 0)
        st.write("1 parent segment available.")
    process_indices = list(range(batch_range[0], batch_range[1] + 1))
    selected_parent_ids = {
        parent_annotations[idx].id
        for idx in process_indices
        if 0 <= idx < len(parent_annotations)
        and getattr(parent_annotations[idx], "id", None)
    }
    st.write(f"Selected {len(process_indices)} parent segment(s) for analysis.")
    selected_parent_spans = _selected_parent_spans(parent_annotations, process_indices, len(df))
    coverage_slot = st.empty()
    _render_subsegment_coverage_bar(
        coverage_slot,
        selected_parent_spans,
        chart_key="batch_agent_subsegment_coverage_initial",
    )

    st.markdown("---")
    config = _render_provider_config(
        "batch_agent_provider",
        default_temperature=0.7,
        default_max_new_tokens=1500,
    )

    st.markdown("---")
    delete_existing_subsegments = st.checkbox(
        "Delete all existing sub-segments in the selected parent range before running",
        value=False,
        key="batch_agent_delete_session_subsegments",
        help=(
            "Removes saved child sub-segments belonging to parents in the selected "
            "range before batch discovery starts. Parent segments and children "
            "outside the range are kept."
        ),
    )

    session_id = st.session_state.get("last_session_id")

    if "batch_agent_stop" not in st.session_state:
        st.session_state["batch_agent_stop"] = False
    if "batch_agent_logs" not in st.session_state:
        st.session_state["batch_agent_logs"] = []

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "▶ Run Batch Sub-Segment Discovery",
            key="batch_agent_run", type="primary",
        )
    with col_clear:
        if st.button("Clear log", key="batch_agent_clear_log"):
            st.session_state["batch_agent_logs"] = []
            st.rerun()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_area = st.empty()

    def _flush_log():
        if st.session_state["batch_agent_logs"]:
            log_area.code(
                "\n".join(st.session_state["batch_agent_logs"]),
                language="text", line_numbers=True,
            )

    _flush_log()

    if not run_clicked:
        return
    if config is None:
        st.error("Deterministic annotation configuration is unavailable.")
        return

    try:
        from app.local_annotation_agent.workflow import run_annotation
    except ImportError as e:
        st.error(
            f"Missing dependency: {e}\n\n"
            "Install the AI service requirements before running calculations."
        )
        return

    main_label_set = set(LABEL_CATEGORIES.get("Main Labels", []))
    st.session_state["batch_agent_stop"] = False
    logs = st.session_state["batch_agent_logs"]
    total = len(process_indices)
    success_parents = 0
    total_children = 0
    error_parents = 0

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] {msg}")
        if len(logs) > 1000:
            del logs[: len(logs) - 1000]
        _flush_log()

    log(f"Starting deterministic sub-segment discovery: {total} parent(s)")
    if delete_existing_subsegments:
        deleted = _delete_selected_parent_subsegments(
            session_id,
            selected_annotation_key,
            selected_parent_ids,
        )
        log(f"Deleted {deleted} existing sub-segment(s) from the selected parent range.")
        _render_subsegment_coverage_bar(
            coverage_slot,
            selected_parent_spans,
            chart_key="batch_agent_subsegment_coverage_after_delete",
        )

    annotations = st.session_state.get("current_annotations", [])
    children_by_parent: dict[str, list[dict]] = {}
    for ann in annotations:
        pid = getattr(ann, "parent_id", None)
        if not pid:
            continue
        children_by_parent.setdefault(pid, []).append({
            "start_index": ann.start_index,
            "end_index": ann.end_index,
            "labels": list(ann.labels),
        })

    for i, idx in enumerate(process_indices):
        if st.session_state["batch_agent_stop"]:
            log("Stopped by user.")
            break

        if idx < 0 or idx >= len(parent_annotations):
            log(f"Skipping invalid index {idx}.")
            continue
        parent = parent_annotations[idx]
        if not getattr(parent, "id", None):
            log(f"Parent #{idx}: skipped (missing parent id).")
            progress_bar.progress((i + 1) / total)
            continue

        parent_main_labels = [l for l in parent.labels if l in main_label_set]
        p_start = int(parent.start_index) if parent.start_index is not None else 0
        p_end = int(parent.end_index) if parent.end_index is not None else len(df)

        status_text.markdown(
            f"**Parent #{idx}** _({i + 1}/{total})_ — [{p_start}, {p_end}), "
            "calculating labels..."
        )
        log(f"Parent #{idx}: running [{p_start}, {p_end}) "
            f"main_labels={parent_main_labels or '∅'}")

        try:
            result = run_annotation(
                flow="detailed",
                df=df,
                start_index=p_start,
                end_index=p_end,
                session_id=session_id,
                parent_main_labels=parent_main_labels,
                parent_selected_labels=list(parent.labels),
                existing_children=children_by_parent.get(parent.id, []),
                config=config,
            )
        except Exception as e:
            error_parents += 1
            log(f"Parent #{idx}: ERROR — {e}")
            log(traceback.format_exc().splitlines()[-1])
            progress_bar.progress((i + 1) / total)
            continue

        try:
            n_children = _persist_children_for_parent(
                parent, result, session_id, selected_annotation_key, df,
            )
        except Exception as e:
            error_parents += 1
            log(f"Parent #{idx}: persistence ERROR — {e}")
            progress_bar.progress((i + 1) / total)
            continue

        if n_children > 0:
            success_parents += 1
            total_children += n_children
            log(f"Parent #{idx}: saved {n_children} child sub-segment(s).")
        else:
            log(f"Parent #{idx}: pipeline produced no usable proposals.")

        progress_bar.progress((i + 1) / total)

    progress_bar.progress(1.0)
    status_text.markdown(
        f"**Done.** Parents updated: {success_parents}, "
        f"new children: {total_children}, errors: {error_parents}."
    )
    log(f"Finished. {success_parents}/{total} parents updated, "
        f"{total_children} children created, {error_parents} error(s).")
    _render_subsegment_coverage_bar(
        coverage_slot,
        selected_parent_spans,
        chart_key="batch_agent_subsegment_coverage_final",
    )

    st.session_state.pop("agent_annot_result", None)
    st.session_state.pop("agent_annot_followup_ctx", None)
    st.session_state.pop("agent_annot_followup_chat", None)


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


def _selected_parent_spans(
    annotations: list,
    process_indices: list[int],
    df_len: int,
) -> list[dict]:
    spans = []
    for idx in process_indices:
        if idx < 0 or idx >= len(annotations):
            continue
        ann = annotations[idx]
        start_index = getattr(ann, "start_index", None)
        end_index = getattr(ann, "end_index", None)
        s = 0 if start_index is None else int(start_index)
        e = df_len if end_index is None else int(end_index)
        s = max(0, min(s, df_len))
        e = max(0, min(e, df_len))
        if e <= s:
            continue
        spans.append({
            "id": getattr(ann, "id", None),
            "index": idx,
            "start_index": s,
            "end_index": e,
        })
    return spans


def _render_subsegment_coverage_bar(
    slot,
    parent_spans: list[dict],
    *,
    chart_key: str,
) -> None:
    parent_ids = {p.get("id") for p in parent_spans if p.get("id")}
    if not parent_ids:
        slot.empty()
        return

    target_ranges = [
        (int(p["start_index"]), int(p["end_index"]))
        for p in parent_spans
    ]
    child_ranges = []
    for ann in st.session_state.get("current_annotations", []) or []:
        if getattr(ann, "parent_id", None) not in parent_ids:
            continue
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        child_ranges.append((s, e))

    _render_coverage_bar(
        slot,
        target_ranges,
        child_ranges,
        title="Sub-segment coverage",
        legend_note="(🟩 child sub-segments · 🟥 uncovered parent span)",
        chart_key=chart_key,
    )


def render_batch_subseg(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_segment_input(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_batch_auto_annotation(df, selected_annotation_key)
