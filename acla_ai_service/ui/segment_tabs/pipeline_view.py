"""Pipeline graph view.

Four columns:

    Data Preparation → Annotation Components → Output Datasets → Model Components

Each annotation card owns:
- a **kind** dropdown (lap / detailed / batch_* / llm / …) that decides
  which tab the single Open button routes to;
- a **mode** picker — *source copy* (copy from an upstream dataset),
  *secondary worker* (read a target's output, write its own output), or
  *coworker* (share both input and output with a target node);
- a source picker — any cache_key in the store (source), or a sibling
  annotation id (any mode);
- the output dataset status;
- one Open button that routes to the tab whose ``ui_route`` matches the
  selected kind.

Plus "+ Add annotation" / "+ Add training" buttons at the bottom of
each column so the user can grow the pipeline incrementally.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import streamlit as st

from app.pipelines.manifest import node_kinds
from app.pipelines.manifest.models import (
    MODE_COWORKER,
    MODE_SOURCE,
    MODE_SECONDARY_WORKER,
    AnnotationNode,
    Pipeline,
    TrainingNode,
)
from app.pipelines.manifest.dataset_structure import (
    has_cached_data,
    is_segment_record,
    payload_records,
)
from app.pipelines.manifest.registry import save as save_pipeline, slugify
from app.pipelines.manifest.migrate_labels import migrate_dataset_labels
from app.pipelines.manifest.protection import collect_protected_session_ids
from app.pipelines.manifest.segment_refresh import refresh_node_segments
from app.pipelines.manifest.source_copy import (
    CHUNK_ID_PATH,
    DICTIONARY_KEY_PATH,
    source_copy_key,
    sync_source_copy,
)
from app.pipelines.training.config import TrainingPipelineConfig
from segment_tabs._training_runner import render_card, spawn
from segment_tabs.components.annotation_key_reference import (
    render_annotation_key_reference,
)

MODE_LABELS = {
    MODE_SOURCE: "Copy from source",
    MODE_SECONDARY_WORKER: "Secondary worker (target output → own output)",
    MODE_COWORKER: "Coworker (shares target's input + output)",
}
MODE_ORDER = [MODE_SOURCE, MODE_SECONDARY_WORKER, MODE_COWORKER]
MODE_DESCRIPTIONS = {
    MODE_SOURCE: (
        "Copy the selected source dataset into this annotation's private "
        "input. Later updates refresh only rows/sessions not touched by "
        "this annotation output."
    ),
    MODE_SECONDARY_WORKER: (
        "Read the target's *output* dataset and write this node's own "
        "output. Use when this node derives a second-stage result from "
        "another annotation."
    ),
    MODE_COWORKER: (
        "Read the target's *input* and write to the target's *output*. "
        "Use for parallel assistance — e.g. an AI agent helping the user "
        "annotate the same dataset side-by-side."
    ),
}

_SESSION_CHUNK_PRIMARY = {"label": "Session / chunk id", "path": CHUNK_ID_PATH}
_DICTIONARY_PRIMARY = {"label": "Dictionary key", "path": DICTIONARY_KEY_PATH}
_INPUT_PRIMARY_KEY_BY_KIND = {
    "lap": _SESSION_CHUNK_PRIMARY,
    "detailed": _SESSION_CHUNK_PRIMARY,
    "batch_bulk_label": _SESSION_CHUNK_PRIMARY,
    "batch_rule_based": _SESSION_CHUNK_PRIMARY,
    "batch_classifier": _SESSION_CHUNK_PRIMARY,
    "batch_subseg": _SESSION_CHUNK_PRIMARY,
    "batch_lap": _SESSION_CHUNK_PRIMARY,
    "parent_labels": _SESSION_CHUNK_PRIMARY,
}


_CARD_CSS = """
<style>
.pipe-col-header { font-weight: 600; font-size: 0.9rem;
  text-transform: uppercase; letter-spacing: 0.05em;
  color: #666; margin-bottom: 0.5rem; }
.pipe-card { border: 1px solid #d0d7de; border-radius: 8px;
  padding: 0.6rem 0.7rem; margin-bottom: 0.5rem; background: #fafbfc; }
.pipe-card.has-data  { border-left: 3px solid #2da44e; }
.pipe-card.empty     { border-left: 3px solid #d0d7de; }
.pipe-card.training  { border-left: 3px solid #0969da; }
.pipe-card.behind    { border-left: 3px solid #d29922; }
.pipe-card.secondary { border-left: 3px solid #8250df; }
.pipe-card.coworker  { border-left: 3px solid #1f9d9d; }
.pipe-card .title    { font-weight: 600; margin-bottom: 0.15rem; }
.pipe-card .sub      { font-size: 0.78rem; color: #57606a; word-break: break-all; }
.pipe-card .meta     { font-size: 0.72rem; color: #6e7781; margin-top: 0.2rem; }
.pipe-chip { display: inline-block; padding: 0.05rem 0.45rem;
  border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
.pipe-chip.green  { background: #dafbe1; color: #1a7f37; }
.pipe-chip.grey   { background: #eaeef2; color: #57606a; }
.pipe-chip.amber  { background: #fff3d4; color: #9a6700; }
.pipe-chip.purple { background: #ede2ff; color: #6639ba; }
.pipe-chip.teal   { background: #d4f4f4; color: #0a6a6a; }

/* Legacy mode-picker card styles. Kept scoped in case older session
   elements are still mounted during Streamlit reruns. */
div[class*="st-key-mode_pick_"] .stButton > button {
  text-align: left;
  justify-content: flex-start;
  white-space: normal;
  border: 1px solid #d0d7de;
  background: #fff;
  color: inherit;
  padding: 0.5rem 0.7rem;
  font-weight: normal;
  transition: border-color .15s, background .15s;
}
div[class*="st-key-mode_pick_"] .stButton > button:hover:not(:disabled) {
  border-color: #0969da;
  background: #f6f8fa;
}
div[class*="st-key-mode_pick_"] .stButton > button:disabled {
  border-color: #2da44e;
  background: #f0fdf4;
  opacity: 1;
  cursor: default;
}
div[class*="st-key-mode_pick_"] .stButton > button p {
  margin: 0;
  line-height: 1.35;
}
div[class*="st-key-mode_pick_"] .stButton > button p:first-child {
  font-weight: 600;
  font-size: 0.85rem;
}
div[class*="st-key-mode_pick_"] .stButton > button p:not(:first-child) {
  max-height: 0;
  opacity: 0;
  overflow: hidden;
  transition: max-height .25s ease, opacity .2s ease, margin-top .25s ease;
  font-size: 0.78rem;
  color: #57606a;
}
div[class*="st-key-mode_pick_"] .stButton > button:hover p:not(:first-child),
div[class*="st-key-mode_pick_"] .stButton > button:disabled p:not(:first-child) {
  max-height: 12rem;
  opacity: 1;
  margin-top: 0.35rem;
}
</style>
"""


def _card(html: str, kind_class: str = "") -> None:
    klass = f"pipe-card {kind_class}".strip()
    st.markdown(f'<div class="{klass}">{html}</div>', unsafe_allow_html=True)


def _render_mode_picker(current: str, key_prefix: str) -> Optional[str]:
    """Return a newly selected mode, or ``None`` when unchanged."""
    try:
        mode_idx = MODE_ORDER.index(current)
    except ValueError:
        mode_idx = 0
    picked = st.selectbox(
        "Mode",
        options=MODE_ORDER,
        index=mode_idx,
        format_func=lambda mode: MODE_LABELS[mode],
        key=f"mode_pick_{key_prefix}",
        help=MODE_DESCRIPTIONS.get(current),
    )
    return picked if picked != current else None


def _route(view: str, *, annotation_key: Optional[str] = None,
           session_key: Optional[str] = None, training_node: Optional[str] = None,
           node_id: Optional[str] = None) -> None:
    st.session_state["pipeline_routed_view"] = view
    # Always update the active node — None clears it so the popup logic
    # in segment_annotation_app can tell a stale route from a fresh one.
    st.session_state["pipeline_active_node_id"] = node_id
    # annotation_key may be empty when output isn't configured yet — let
    # the popup pick a value and the next rerun fill this in.
    st.session_state["pipeline_annotation_key"] = annotation_key or ""
    if session_key is not None:
        st.session_state["pipeline_session_key"] = session_key
    if training_node is not None:
        st.session_state["pipeline_training_node"] = training_node
    st.rerun()


def _source_options(pipeline: Pipeline, store: Any, self_id: str,
                    siblings_only: bool) -> list[str]:
    """Candidate sources for one annotation's input.

    Source mode: every cache_key in the store + every other annotation's
        id.
    Secondary worker / coworker: only sibling node ids (these modes
        target a node, not an external Lance dataset).
    """
    sibling_refs = [n.id for n in pipeline.annotations if n.id != self_id]
    if siblings_only:
        candidates = sibling_refs
    else:
        try:
            store_keys = sorted(store.list_cache_keys())
        except Exception:
            store_keys = []
        candidates = sibling_refs + store_keys
    return list(dict.fromkeys(candidates))


def _output_dataset_options(store: Any) -> list[str]:
    try:
        return sorted(store.list_cache_keys())
    except Exception:
        return []


def _with_current_option(options: list[str], current: Optional[str]) -> list[str]:
    if current and current not in options:
        return [current] + options
    return options


def _select_index(options: list[str], current: Optional[str]) -> int:
    if current in options:
        return options.index(current)
    return 0


def _valid_source_options_for_mode(
    pipeline: Pipeline,
    store: Any,
    node: AnnotationNode,
    mode: str,
) -> list[str]:
    return _source_options(
        pipeline,
        store,
        self_id=node.id,
        siblings_only=mode != MODE_SOURCE,
    )


def _annotation_input_status(
    pipeline: Pipeline, node: AnnotationNode, store: Any,
) -> tuple[str, str, str]:
    """Return (chip_html, detail_html, kind_class) for the input."""
    if not node.source_ref:
        return ('<span class="pipe-chip grey">no source</span>',
                "Pick a source below.", "empty")

    target_label = "Target" if node.mode != MODE_SOURCE else "Source"
    source_line = f"{target_label}: <code>{node.source_ref}</code>"

    if node.mode == MODE_SECONDARY_WORKER:
        read_key = pipeline.effective_input_key(node)   # = target's output
        if not read_key:
            return ('<span class="pipe-chip amber">target missing</span>',
                    f"{source_line}<br/>(secondary worker — target no longer resolves)",
                    "behind")
        try:
            exists = store.has_cached_data(read_key)
            meta = store.get_cache_metadata(read_key) if exists else None
            n = meta.total_records if meta else 0
        except Exception:
            exists, n = False, 0
        detail = (
            f"{source_line}<br/>"
            f"Input: target's output <code>{read_key}</code>"
            + (f" · {n:,} rec." if exists else " · target output empty.")
        )
        chip = '<span class="pipe-chip purple">secondary worker' + (
            " · empty</span>" if not exists else "</span>"
        )
        return (chip, detail, "secondary")

    if node.mode == MODE_COWORKER:
        read_key = pipeline.effective_input_key(node)    # = target's input
        write_key = pipeline.effective_output_key(node)  # = target's output
        if not read_key and not write_key:
            return ('<span class="pipe-chip amber">target missing</span>',
                    f"{source_line}<br/>(coworker — target no longer resolves)",
                    "behind")
        try:
            exists = bool(read_key) and store.has_cached_data(read_key)
            meta = store.get_cache_metadata(read_key) if exists else None
            n = meta.total_records if meta else 0
        except Exception:
            exists, n = False, 0
        detail = (
            f"{source_line}<br/>"
            f"Input: target's input <code>{read_key or '—'}</code>"
            + (f" · {n:,} rec." if exists else " · empty/not ready yet.")
        )
        chip = '<span class="pipe-chip teal">coworker' + (
            " · empty</span>" if not exists else "</span>"
        )
        return (chip, detail, "coworker")

    # Default source-copy mode below.
    source_key = pipeline.resolve_source_key(node.source_ref)
    if source_key and source_key != node.source_ref:
        source_line += f" → <code>{source_key}</code>"
    if not source_key:
        return ('<span class="pipe-chip amber">source missing</span>',
                f"{source_line}<br/>"
                "Source no longer resolves.",
                "behind")

    try:
        exists = store.has_cached_data(source_key)
        meta = store.get_cache_metadata(source_key) if exists else None
        n = meta.total_records if meta else 0
    except Exception:
        exists, n = False, 0

    if not exists:
        return ('<span class="pipe-chip amber">source empty</span>',
                f"{source_line}<br/>"
                "Source has no data to copy yet.",
                "empty")
    copy_key = source_copy_key(node.output_key) if node.output_key else None
    copy_exists = bool(copy_key) and has_cached_data(store, copy_key)
    copy_detail = (
        f"<br/>Private input copy: <code>{copy_key}</code>"
        if copy_key else "<br/>Private input copy configured on first open."
    )
    copy_detail += " · ready" if copy_exists else " · not copied yet"
    return ('<span class="pipe-chip green">source ready</span>',
            f"{source_line}<br/>"
            f"Copies from source · {n:,} rec"
            f"{copy_detail}",
            "has-data")


def _output_status(store: Any, output_key: str) -> tuple[str, int, str]:
    # The store routes per-cache_key custom directories internally — see
    # LanceTelemetryStore.register_directory — so the default singleton
    # works here regardless of whether the output lives in a node-specific
    # directory picked via the first-time popup.
    try:
        if not store.has_cached_data(output_key):
            return ("⚪ empty", 0, "")
        meta = store.get_cache_metadata(output_key)
        n = meta.total_records if meta else 0
        ts = meta.updated_at[:19] if meta and meta.updated_at else ""
        return ("🟢 has data", n, ts)
    except Exception:
        return ("⚠️ unknown", 0, "")


def _type_label(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        if not value:
            return "list[empty]"
        return f"list[{_type_label(value[0])}]"
    return type(value).__name__


def _structure_rows_from_record(record: Any, *, limit: int = 12) -> list[dict[str, str]]:
    if not isinstance(record, dict):
        return [{"field": "value", "type": _type_label(record)}]

    rows = [
        {"field": str(field), "type": _type_label(value)}
        for field, value in list(record.items())[:limit]
    ]
    extra = len(record) - limit
    if extra > 0:
        rows.append({"field": f"... {extra} more", "type": ""})

    telemetry = record.get("telemetry_data")
    if isinstance(telemetry, list) and telemetry and isinstance(telemetry[0], dict):
        sample = telemetry[0]
        for field, value in list(sample.items())[:limit]:
            rows.append({
                "field": f"telemetry_data[].{field}",
                "type": _type_label(value),
            })
        extra = len(sample) - limit
        if extra > 0:
            rows.append({"field": f"telemetry_data[] ... {extra} more", "type": ""})

    return rows


def _structure_rows_from_dictionary_payload(
    payload: dict[Any, Any],
    *,
    include_value_fields: bool = False,
    limit: int = 12,
) -> list[dict[str, str]]:
    rows = [{
        "field": "Dictionary key",
        "type": "unknown" if not payload else _type_label(next(iter(payload.keys()))),
        "path": DICTIONARY_KEY_PATH,
    }]
    if not include_value_fields or not payload:
        return rows

    first_value = next(iter(payload.values()))
    if isinstance(first_value, dict):
        rows.extend(_structure_rows_from_record(first_value, limit=limit))
    return rows


def _dataset_structure(
    store: Any,
    key: Optional[str],
    *,
    declared_rows: Optional[list[dict[str, str]]] = None,
    include_dictionary_value_fields: bool = False,
) -> dict[str, Any]:
    if not key:
        return {"status": "not configured", "summary": "", "rows": declared_rows or []}

    try:
        exists = store.has_cached_data(key)
    except Exception as exc:
        return {"status": f"unavailable: {exc}", "summary": "", "rows": []}

    if not exists:
        rows = declared_rows or []
        summary = "declared schema" if declared_rows else ""
        return {"status": "empty", "summary": summary, "rows": rows}

    try:
        chunk_ids = store.list_chunk_ids(key)
    except Exception as exc:
        return {"status": f"could not list chunks: {exc}", "summary": "", "rows": []}
    if not chunk_ids:
        return {"status": "empty", "summary": "", "rows": declared_rows or []}

    sample_chunk = chunk_ids[0]
    try:
        payload = store.get_chunk(key, sample_chunk)
    except Exception as exc:
        return {"status": f"could not read sample: {exc}", "summary": "", "rows": []}

    chunk_row = {
        "field": "Session / chunk id",
        "type": "str",
        "path": CHUNK_ID_PATH,
    }
    if isinstance(payload, dict) and not isinstance(payload.get("data"), list):
        return {
            "status": "ready",
            "summary": f"dict[value] from `{sample_chunk}`",
            "rows": [chunk_row] + _structure_rows_from_dictionary_payload(
                payload,
                include_value_fields=include_dictionary_value_fields,
            ),
        }

    container, records = payload_records(payload)
    sample = next((record for record in records if record is not None), None)
    record_label = "segment" if is_segment_record(sample) else "row"
    summary = f"{container}[{record_label}] from `{sample_chunk}`"
    return {
        "status": "ready",
        "summary": summary,
        "rows": [chunk_row] + _structure_rows_from_record(sample),
    }


def _render_dataset_structure_panel(
    store: Any,
    label: str,
    key: Optional[str],
    *,
    declared_rows: Optional[list[dict[str, str]]] = None,
) -> None:
    st.markdown(f"**{label}**")
    if key:
        st.caption(f"`{key}`")
    info = _dataset_structure(store, key, declared_rows=declared_rows)
    status = info["status"]
    summary = info["summary"]
    st.caption(f"{status}" + (f" · {summary}" if summary else ""))
    rows = info["rows"]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No structure available yet.")


def _declared_output_rows(spec: node_kinds.NodeKindSpec) -> list[dict[str, str]]:
    return [
        {
            "field": field.field,
            "type": field.type,
        }
        for field in spec.output_fields
    ]


def _field_options(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for row in rows:
        field = str(row.get("field") or "").strip()
        if not field or field.startswith("..."):
            continue
        path = str(row.get("path") or field).strip()
        if not path:
            continue
        if str(row.get("type") or "").startswith("list["):
            path = f"{field}[]"
        options.append({
            "label": field,
            "path": path,
        })
    return options


def _input_key_options(
    input_structure: dict[str, Any],
    default_primary: dict[str, str],
) -> list[dict[str, str]]:
    options = _field_options(input_structure.get("rows", []))
    if default_primary["path"] not in {option["path"] for option in options}:
        options.insert(0, dict(default_primary))
    return options


def _input_primary_key_for_node(
    node: AnnotationNode,
    input_structure: dict[str, Any],
) -> Optional[dict[str, str]]:
    configured = _INPUT_PRIMARY_KEY_BY_KIND.get(node.kind)
    if configured:
        return dict(configured)

    paths = {
        str(row.get("path") or row.get("field") or "").strip()
        for row in input_structure.get("rows", [])
    }
    if DICTIONARY_KEY_PATH in paths:
        return dict(_DICTIONARY_PRIMARY)
    return dict(_SESSION_CHUNK_PRIMARY)


def _preview_primary_key_values(
    store: Any,
    key: Optional[str],
    path: str,
    *,
    limit: int = 5,
) -> list[str]:
    if not key:
        return []
    try:
        if not store.has_cached_data(key):
            return []
        chunk_ids = list(store.list_chunk_ids(key))
    except Exception:
        return []

    if path == CHUNK_ID_PATH:
        return [str(chunk_id) for chunk_id in chunk_ids[:limit]]

    if path == DICTIONARY_KEY_PATH:
        values: list[str] = []
        for chunk_id in chunk_ids:
            try:
                payload = store.get_chunk(key, chunk_id)
            except Exception:
                continue
            if isinstance(payload, dict) and not isinstance(payload.get("data"), list):
                values.extend(str(value) for value in list(payload.keys())[:limit])
            if len(values) >= limit:
                break
        return values[:limit]

    return []


def _format_primary_key_preview(values: list[str]) -> str:
    if not values:
        return "No sample values yet."
    return "Sample values: " + ", ".join(f"`{value}`" for value in values)


def _render_source_protection_selector(
    pipeline: Pipeline,
    node: AnnotationNode,
    store: Any,
    input_key: Optional[str],
    output_structure: dict[str, Any],
    input_structure: dict[str, Any],
) -> tuple[bool, Optional[dict[str, str]]]:
    if node.mode != MODE_SOURCE:
        return True, None

    output_options = _field_options(output_structure.get("rows", []))
    if not output_options:
        st.warning(
            "Source-copy protection needs an output key. Copy from source "
            "is disabled until the output dataset structure is available."
        )
        return False, None

    input_primary = _input_primary_key_for_node(node, input_structure)
    if not input_primary:
        st.warning(
            "Source-copy protection needs an input key field. Copy from source "
            "is disabled until the input dataset structure is available."
        )
        return False, None

    input_options = _input_key_options(input_structure, input_primary)
    if not input_options:
        st.warning(
            "Source-copy protection needs an input primary key. Copy from "
            "source is disabled until the input dataset structure is available."
        )
        return False, None

    def save_reference(reference: dict[str, str]) -> None:
        node.protection_reference = reference
        save_pipeline(pipeline)

    return render_annotation_key_reference(
        component_id=node.id,
        output_options=output_options,
        input_options=input_options,
        default_input=input_primary,
        saved_reference=node.protection_reference,
        input_preview=lambda path: _format_primary_key_preview(
            _preview_primary_key_values(store, input_key, path)
        ),
        save_reference=save_reference,
    )


def _source_copy_summary(summary) -> str:
    return (
        f"Copied {summary.chunks_copied} chunk(s), "
        f"updated {summary.chunks_updated} chunk(s), "
        f"left {summary.chunks_skipped_unchanged} unchanged, "
        f"preserved {summary.rows_preserved} protected row(s)"
    )


def _run_source_copy_update(
    pipeline: Pipeline,
    node: AnnotationNode,
    store: Any,
    protection_reference: Optional[dict[str, str]],
) -> None:
    source_key = pipeline.resolve_source_key(node.source_ref)
    copy_key = source_copy_key(node.output_key) if node.output_key else None
    if not source_key or not copy_key:
        st.error("Source copy is not configured yet.")
        return
    if protection_reference is None:
        st.error("Source copy protection is not configured with output and input fields.")
        return
    try:
        summary = sync_source_copy(
            store,
            source_key=source_key,
            copy_key=copy_key,
            output_key=pipeline.effective_output_key(node),
            protection_reference=protection_reference,
        )
    except Exception as exc:
        st.error(f"Source copy failed: {exc}")
        return
    st.toast(_source_copy_summary(summary), icon="✅")
    if summary.rows_preserved:
        st.info(f"Preserved {summary.rows_preserved} protected row(s).")
    if summary.read_failures or summary.write_failures:
        st.warning(
            "Some chunks failed. "
            f"Read failures: {len(summary.read_failures)}; "
            f"write failures: {len(summary.write_failures)}."
        )


def _render_maintenance_dropdown(
    pipeline: Pipeline,
    node: AnnotationNode,
    store: Any,
    *,
    refresh_disabled: bool,
    migrate_disabled: bool,
) -> None:
    with st.popover("Maintenance", width="stretch"):
        st.caption("Dataset maintenance")

        if st.button(
            "Refresh segments",
            key=f"refresh_segs_{node.id}",
            width="stretch",
            disabled=refresh_disabled,
            help=(
                "Re-slice telemetry_data on every saved segment from the "
                "current input. Run after data preparation refreshes existing "
                "source data to propagate new columns into saved segments."
            ),
        ):
            try:
                protected_session_ids = collect_protected_session_ids(store)
                input_key = pipeline.effective_input_key(node)
                summary = refresh_node_segments(
                    store,
                    node,
                    input_key=input_key,
                    protected_session_ids=protected_session_ids,
                )
            except ValueError as exc:
                st.error(f"Refresh failed: {exc}")
            else:
                st.toast(
                    f"Refreshed {summary.segments_refreshed} segment(s) "
                    f"across {summary.chunks_written} chunk(s); "
                    f"skipped {summary.chunks_skipped_protected} protected chunk(s).",
                    icon="✅",
                )
                if summary.missing_input_sessions:
                    st.warning(
                        "No input session for: "
                        + ", ".join(summary.missing_input_sessions)
                    )
            st.rerun()

        if st.button(
            "Migrate legacy labels",
            key=f"migrate_labels_{node.id}",
            width="stretch",
            disabled=migrate_disabled,
            help=(
                "Replace old annotation labels in this node's output "
                "dataset, including integer labels, MS→MSP/MSR, and "
                "defensive O sub-labels→OD."
            ),
        ):
            dataset_key = pipeline.effective_output_key(node)
            try:
                summary = migrate_dataset_labels(store, dataset_key or "")
            except ValueError as exc:
                st.error(f"Migration failed: {exc}")
            else:
                if summary.labels_replaced:
                    st.toast(
                        f"Migrated {summary.labels_replaced} label(s) "
                        f"across {summary.segments_updated} segment(s).",
                        icon="✅",
                    )
                else:
                    st.info("No legacy labels found in this output dataset.")
            st.rerun()


# ── Data preparation card ───────────────────────────────────────────────────
def _prepare_cache_rows(store: Any, cfg: TrainingPipelineConfig) -> list[tuple[str, str, str]]:
    rows = [
        ("Raw sessions", cfg.session_data_cache_key),
        ("Processed sessions", cfg.processed_session_data_cache_key),
        ("Top laps", cfg.top_laps_cache_key),
        ("Enriched sessions", cfg.enriched_sessions_cache_key),
    ]
    result = []
    for label, key in rows:
        try:
            if store.has_cached_data(key):
                meta = store.get_cache_metadata(key)
                n = meta.total_records if meta else 0
                ts = meta.updated_at[:19] if meta and meta.updated_at else "unknown"
                status = f"ready · {n:,} records · updated {ts}"
            else:
                status = "not populated"
        except Exception as exc:
            status = f"unknown ({exc})"
        result.append((label, key, status))
    return result


def _render_prepare_cache_status(store: Any, cfg: TrainingPipelineConfig) -> None:
    st.markdown("**Output caches**")
    for label, key, status in _prepare_cache_rows(store, cfg):
        st.caption(f"{label}: `{key}` · {status}")


def _render_data_preparation_card(store: Any, cfg: TrainingPipelineConfig) -> None:
    _render_prepare_cache_status(store, cfg)

    def _start_form() -> None:
        with st.form("prepare_data_form"):
            if st.form_submit_button("Start data preparation", width="stretch"):
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "app.pipelines.training.prepare_training_data",
                ]
                spawn("prepare_data", cmd)
                st.rerun()

    render_card(
        "prepare_data",
        title="Data Preparation",
        description=(
            "Downloads backend sessions, selects top laps, and enriches "
            "telemetry for annotation."
        ),
        render_start_form=_start_form,
    )


# ── Annotation card ──────────────────────────────────────────────────────────
def _render_annotation_card(
    pipeline: Pipeline, node: AnnotationNode, store: Any, cfg: TrainingPipelineConfig,
) -> None:
    ann_specs = node_kinds.list_by_category("annotation")
    kind_choices = [s.kind for s in ann_specs]
    kind_labels = {s.kind: s.display for s in ann_specs}

    chip, detail, kind_class = _annotation_input_status(pipeline, node, store)
    spec = node_kinds.get(node.kind)
    effective = pipeline.effective_input_key(node)
    effective_out = pipeline.effective_output_key(node)
    source_key = pipeline.resolve_source_key(node.source_ref)
    structure_input_key = effective
    if node.mode == MODE_SOURCE and not has_cached_data(store, effective):
        structure_input_key = source_key
    declared_output_rows = _declared_output_rows(spec)
    input_structure = _dataset_structure(store, structure_input_key)
    output_structure = _dataset_structure(
        store,
        effective_out,
        declared_rows=declared_output_rows,
        include_dictionary_value_fields=True,
    )
    out_label, out_n, out_ts = _output_status(store, effective_out) if effective_out else ("—", 0, "")

    if node.mode == MODE_SOURCE:
        if not node.output_key:
            out_line = (
                '<br/>Writes output: <i>not configured yet</i> — '
                'pick a directory &amp; filename on first open.'
            )
        else:
            dir_hint = (
                f'<br/><span class="meta">in <code>{node.output_dir}</code></span>'
                if node.output_dir else ""
            )
            out_line = (
                f'<br/>Writes output: <code>{node.output_key}</code> · {out_label}'
                f'{dir_hint}'
            )
    else:
        share_label = "secondary worker" if node.mode == MODE_SECONDARY_WORKER else "coworker"
        if node.mode == MODE_SECONDARY_WORKER:
            if node.output_key:
                dir_hint = (
                    f'<br/><span class="meta">in <code>{node.output_dir}</code></span>'
                    if node.output_dir else ""
                )
                out_line = (
                    f'<br/>Writes output: <code>{node.output_key}</code> · {out_label}'
                    f'{dir_hint}'
                )
            else:
                out_line = (
                    '<br/>Writes output: <i>not configured yet</i> — '
                    'pick a directory &amp; filename on first open.'
                )
        else:
            out_line = (
                f'<br/>Writes output: <i>shared with target</i> '
                f'<code>{effective_out or "—"}</code> ({share_label}) · {out_label}'
            )

    display_name = node.name or spec.display

    with st.container(border=True):
        _card(
            f'<div class="title">✏️ {display_name} '
            f'<span style="font-weight:400;color:#6e7781">· {node.id}</span></div>'
            f'<div class="sub">{chip}<br/>{detail}<br/>'
            f'{out_line}'
            + (f' · {out_n:,} rec' if out_n else '')
            + (f' · updated {out_ts}' if out_ts else '')
            + '</div>',
            kind_class=kind_class,
        )

        with st.expander("Dataset structure", expanded=False):
            struct_cols = st.columns(2)
            with struct_cols[0]:
                _render_dataset_structure_panel(store, "Input", structure_input_key)
            with struct_cols[1]:
                _render_dataset_structure_panel(
                    store,
                    "Output",
                    effective_out,
                    declared_rows=declared_output_rows,
                )

        # ── Kind dropdown (no warning — change freely) ───────────────────
        try:
            kind_idx = kind_choices.index(node.kind)
        except ValueError:
            kind_idx = 0
        new_kind = st.selectbox(
            "Kind", options=kind_choices, index=kind_idx,
            format_func=lambda k: kind_labels[k],
            key=f"ann_kind_{node.id}",
        )
        if new_kind != node.kind:
            node.kind = new_kind
            save_pipeline(pipeline)
            st.rerun()

        # ── Mode / input / output controls ──────────────────────────────
        picked_mode = _render_mode_picker(node.mode, key_prefix=node.id)
        if picked_mode is not None:
            valid_sources = _valid_source_options_for_mode(
                pipeline,
                store,
                node,
                picked_mode,
            )
            node.mode = picked_mode
            if node.source_ref not in valid_sources:
                node.source_ref = None
                st.session_state.pop(f"ann_src_{node.id}", None)
            if node.mode != MODE_SOURCE:
                node.protection_reference = None
            save_pipeline(pipeline)
            st.rerun()

        siblings_only = node.mode != MODE_SOURCE
        source_options = _valid_source_options_for_mode(
            pipeline,
            store,
            node,
            node.mode,
        )
        source_placeholder = "— pick an input dataset —"
        source_display_options = [source_placeholder] + _with_current_option(
            source_options,
            node.source_ref,
        )
        source_widget_key = f"ann_src_{node.id}"
        if st.session_state.get(source_widget_key) not in {None, *source_display_options}:
            st.session_state.pop(source_widget_key, None)
        chosen_source = st.selectbox(
            "Input dataset",
            options=source_display_options,
            index=_select_index(source_display_options, node.source_ref),
            key=source_widget_key,
            placeholder="Type to search input datasets",
            help=("Copy from source: pick an input dataset or sibling "
                  "annotation. Worker modes target the selected sibling."),
        )
        new_source_ref = (
            None if chosen_source == source_placeholder else chosen_source
        )
        if new_source_ref != node.source_ref:
            node.source_ref = new_source_ref
            if node.mode == MODE_SOURCE:
                node.protection_reference = None
            save_pipeline(pipeline)
            st.rerun()

        if node.mode in {MODE_SOURCE, MODE_SECONDARY_WORKER}:
            output_placeholder = "— configure on first open —"
            output_options = [output_placeholder] + _with_current_option(
                _output_dataset_options(store),
                node.output_key,
            )
            chosen_output = st.selectbox(
                "Output dataset",
                options=output_options,
                index=_select_index(output_options, node.output_key),
                key=f"ann_out_{node.id}",
                placeholder="Type to search output datasets",
                help=("Choose the output dataset this node writes. Leave "
                      "unconfigured to use the first-open output popup."),
            )
            new_output_key = (
                "" if chosen_output == output_placeholder else chosen_output
            )
            if new_output_key != node.output_key:
                node.output_key = new_output_key
                node.output_dir = None
                if node.mode == MODE_SOURCE:
                    node.protection_reference = None
                save_pipeline(pipeline)
                st.rerun()
        else:
            st.caption(
                f"Output dataset: shared with target "
                f"`{pipeline.effective_output_key(node) or '—'}`"
            )

        protection_ready, protection_reference = _render_source_protection_selector(
            pipeline,
            node,
            store,
            structure_input_key,
            output_structure,
            input_structure,
        )

        # ── Action buttons ───────────────────────────────────────────────
        btn_cols = st.columns([1.2, 1, 1, 0.4])

        open_disabled = not effective
        if node.mode == MODE_SOURCE and node.output_key:
            open_disabled = not has_cached_data(store, effective)

        with btn_cols[0]:
            if node.mode != MODE_SOURCE:
                # Show a disabled stub for layout.
                stub_label = ("Secondary worker"
                              if node.mode == MODE_SECONDARY_WORKER
                              else "Coworker")
                st.button(stub_label, key=f"mode_status_{node.id}",
                          width="stretch", disabled=True)
            else:
                copy_key = source_copy_key(node.output_key) if node.output_key else None
                source_ready = has_cached_data(store, source_key)
                copy_ready = has_cached_data(store, copy_key)
                label = "Update from source" if copy_ready else "Copy from source"
                if st.button(
                    label,
                    key=f"source_copy_{node.id}",
                    width="stretch",
                    disabled=not (node.output_key and source_ready and protection_ready),
                ):
                    _run_source_copy_update(
                        pipeline,
                        node,
                        store,
                        protection_reference,
                    )
                    st.rerun()
        with btn_cols[1]:
            out_key = pipeline.effective_output_key(node)
            has_input_data = has_cached_data(store, effective)
            has_output_data = has_cached_data(store, out_key)
            refresh_disabled = not (
                node.mode == MODE_SOURCE and has_input_data and has_output_data
            )
            _render_maintenance_dropdown(
                pipeline,
                node,
                store,
                refresh_disabled=refresh_disabled,
                migrate_disabled=not has_output_data,
            )
        with btn_cols[2]:
            if st.button(f"Open", key=f"open_ann_{node.id}",
                         type="primary",
                         width="stretch", disabled=open_disabled):
                _route(spec.ui_route,
                       annotation_key=pipeline.effective_output_key(node),
                       session_key=effective,
                       node_id=node.id)
        with btn_cols[3]:
            if st.button("🗑", key=f"del_ann_{node.id}",
                         width="stretch",
                         help="Delete this annotation node"):
                pipeline.annotations = [n for n in pipeline.annotations if n.id != node.id]
                save_pipeline(pipeline)
                st.rerun()


# ── Add-annotation form (inline expander at the bottom of the column) ────────
def _render_add_annotation(pipeline: Pipeline, store: Any, cfg: TrainingPipelineConfig) -> None:
    with st.expander("➕ Add annotation component", expanded=False):
        ann_specs = node_kinds.list_by_category("annotation")
        kind_choices = [s.kind for s in ann_specs]
        kind_labels = {s.kind: s.display for s in ann_specs}
        kind_placeholder = "— pick annotation component —"
        chosen_kind = st.selectbox(
            "Kind",
            options=[kind_placeholder] + kind_choices,
            format_func=lambda k: kind_labels.get(k, k),
            key="add_ann_kind",
        )
        if chosen_kind == kind_placeholder:
            chosen_kind = None
        name = st.text_input(
            "Name", value="", key="add_ann_name",
            placeholder="e.g. lap round 1",
            help="Display label for this node. Required, must be unique across "
                 "all nodes in this pipeline, and can't be changed after creation. "
                 "Also drives the node id (slugified). You'll pick the output "
                 "dataset's directory and filename the first time you open "
                 "this annotation page.",
        )
        name_clean = name.strip()
        name_slug = slugify(name_clean) if name_clean else ""

        mode_placeholder = "— pick mode —"
        pending_mode = st.session_state.get("add_ann_mode")
        mode_options = [mode_placeholder] + MODE_ORDER
        mode_idx = (
            mode_options.index(pending_mode)
            if pending_mode in MODE_ORDER else 0
        )
        picked_mode = st.selectbox(
            "Mode",
            options=mode_options,
            index=mode_idx,
            format_func=lambda mode: MODE_LABELS.get(mode, mode),
            key="mode_pick_add",
            help=MODE_DESCRIPTIONS.get(pending_mode),
        )
        new_pending_mode = (
            None if picked_mode == mode_placeholder else picked_mode
        )
        if new_pending_mode != pending_mode:
            if new_pending_mode is None:
                st.session_state.pop("add_ann_mode", None)
            else:
                st.session_state["add_ann_mode"] = new_pending_mode
            # Mode change invalidates the pending source (options differ).
            st.session_state.pop("add_ann_source", None)
            st.rerun()
        pending_mode = new_pending_mode

        # ── Source picker ────────────────────────────────────────────────
        siblings_only = bool(pending_mode and pending_mode != MODE_SOURCE)
        source_options = (
            _source_options(
                pipeline, store, self_id="", siblings_only=siblings_only,
            )
            if chosen_kind and pending_mode else []
        )
        placeholder = "— pick an input dataset —"
        display_options = [placeholder] + source_options
        pending_source = st.session_state.get("add_ann_source")
        if pending_source and pending_source not in display_options:
            pending_source = None
            st.session_state.pop("add_ann_source", None)
        if st.session_state.get("add_ann_source_select") not in {None, *display_options}:
            st.session_state.pop("add_ann_source_select", None)
        try:
            src_idx = display_options.index(pending_source) if pending_source else 0
        except ValueError:
            src_idx = 0
        chosen_src = st.selectbox(
            "Input dataset", options=display_options, index=src_idx,
            key="add_ann_source_select",
            help=("Source mode: any cache_key in the store, or a sibling "
                  "annotation. Secondary worker / coworker: pick the "
                  "sibling annotation to target."),
            disabled=not (chosen_kind and pending_mode),
        )
        st.session_state["add_ann_source"] = (
            None if chosen_src == placeholder else chosen_src
        )
        if chosen_kind and pending_mode and not source_options:
            if siblings_only:
                st.caption(":warning: No sibling datasets are available yet.")
            else:
                st.caption(":warning: No datasets are available yet.")

        if name_slug and pending_mode:
            if pending_mode == MODE_SOURCE:
                caption = (
                    f"Node id will be: `{name_slug}`. Output dataset location is "
                    "configured on the annotation page (first-time popup)."
                )
            elif pending_mode == MODE_SECONDARY_WORKER:
                caption = (
                    f"Node id will be: `{name_slug}`. Output dataset location is "
                    "configured on the annotation page (first-time popup)."
                )
            else:
                caption = (
                    f"Node id will be: `{name_slug}`. Writes to the target's "
                    "output dataset (no new file is created)."
                )
            st.caption(caption)

        source_ref = st.session_state.get("add_ann_source")
        can_create = bool(name_clean and chosen_kind and pending_mode and source_ref)
        if st.button("Create annotation node", type="primary",
                     width="stretch",
                     disabled=not can_create):
            existing_ids = {n.id for n in pipeline.annotations} | {n.id for n in pipeline.trainings}
            if name_slug in existing_ids:
                st.error(f"A node named `{name_clean}` (id `{name_slug}`) already exists.")
                return
            pipeline.annotations.append(AnnotationNode(
                id=name_slug,
                kind=chosen_kind,
                name=name_clean,
                mode=pending_mode,
                source_ref=source_ref,
            ))
            # Reset the picker defaults for the next add.
            st.session_state.pop("add_ann_mode", None)
            st.session_state.pop("add_ann_source", None)
            save_pipeline(pipeline)
            st.rerun()


# ── Training card / add ──────────────────────────────────────────────────────
def _render_training_card(pipeline: Pipeline, node: TrainingNode, store: Any) -> None:
    spec = node_kinds.get(node.kind)
    input_key = pipeline.resolve_source_key(node.input_ref)
    display_name = node.name or spec.display
    _card(
        f'<div class="title">🏋️ {display_name} '
        f'<span style="font-weight:400;color:#6e7781">· {node.id}</span></div>'
        f'<div class="sub">in: <code>{node.input_ref}</code></div>'
        f'<div class="meta">resolves to: <code>{input_key or "—"}</code></div>',
        kind_class="training",
    )

    # Input picker — every annotation's output is a candidate.
    ann_refs = [n.id for n in pipeline.annotations]
    placeholder = "— pick an annotation —"
    display_options = [placeholder] + ann_refs
    try:
        default_idx = display_options.index(node.input_ref) if node.input_ref else 0
    except ValueError:
        display_options = [node.input_ref] + display_options
        default_idx = 0
    training_widget_key = f"tr_src_{node.id}"
    if st.session_state.get(training_widget_key) not in {None, *display_options}:
        st.session_state.pop(training_widget_key, None)
    new_ref = st.selectbox(
        "Input", options=display_options, index=default_idx,
        key=training_widget_key, label_visibility="collapsed",
    )
    if new_ref != placeholder and new_ref != node.input_ref:
        node.input_ref = new_ref
        save_pipeline(pipeline)
        st.rerun()

    btn_cols = st.columns([3, 0.4])
    with btn_cols[0]:
        if st.button(f"Configure {spec.display}", key=f"open_tr_{node.id}",
                     width="stretch"):
            _route(spec.ui_route,
                   annotation_key=input_key or None,
                   training_node=node.id)
    with btn_cols[1]:
        if st.button("🗑", key=f"del_tr_{node.id}",
                     width="stretch",
                     help="Delete this training node"):
            pipeline.trainings = [n for n in pipeline.trainings if n.id != node.id]
            save_pipeline(pipeline)
            st.rerun()


def _render_add_training(pipeline: Pipeline) -> None:
    with st.expander("➕ Add training component", expanded=False):
        tr_specs = node_kinds.list_by_category("training")
        kind_choices = [s.kind for s in tr_specs]
        kind_labels = {s.kind: s.display for s in tr_specs}
        kind_placeholder = "— pick training component —"
        chosen_kind = st.selectbox(
            "Kind",
            options=[kind_placeholder] + kind_choices,
            format_func=lambda k: kind_labels.get(k, k),
            key="add_tr_kind",
        )
        if chosen_kind == kind_placeholder:
            chosen_kind = None
        name = st.text_input(
            "Name", value="", key="add_tr_name",
            placeholder="e.g. classifier round 1",
            help="Display label for this node. Required, must be unique across "
                 "all nodes in this pipeline, and can't be changed after creation. "
                 "Also drives the node id (slugified).",
        )
        name_clean = name.strip()
        name_slug = slugify(name_clean) if name_clean else ""

        if name_slug:
            st.caption(f"Node id will be: `{name_slug}`")

        if st.button("Create training node", type="primary",
                     width="stretch",
                     disabled=not (name_clean and chosen_kind)):
            existing = {n.id for n in pipeline.annotations} | {n.id for n in pipeline.trainings}
            if name_slug in existing:
                st.error(f"A node named `{name_clean}` (id `{name_slug}`) already exists.")
                return
            pipeline.trainings.append(TrainingNode(
                id=name_slug, kind=chosen_kind, name=name_clean, input_ref="",
            ))
            save_pipeline(pipeline)
            st.rerun()


# ── Top-level entrypoint ────────────────────────────────────────────────────
def render_pipeline_view(pipeline: Pipeline, store: Any) -> None:
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.subheader(f"Pipeline: `{pipeline.id}` · v{pipeline.version}")
    st.caption(
        f"Created {pipeline.created_at[:19]} · "
        f"{len(pipeline.annotations)} annotation · "
        f"{len(pipeline.trainings)} training nodes"
    )

    cfg = TrainingPipelineConfig()

    col_prepare, col_ann, col_out, col_tr = st.columns([1.2, 1.5, 1.2, 1.2])

    # ── Data preparation ────────────────────────────────────────────────
    with col_prepare:
        st.markdown('<div class="pipe-col-header">Data Preparation</div>',
                    unsafe_allow_html=True)
        _render_data_preparation_card(store, cfg)

    # ── Annotation nodes ────────────────────────────────────────────────
    with col_ann:
        st.markdown('<div class="pipe-col-header">Annotation Components</div>',
                    unsafe_allow_html=True)
        for node in pipeline.annotations:
            _render_annotation_card(pipeline, node, store, cfg)
        _render_add_annotation(pipeline, store, cfg)

    # ── Output datasets (derived from annotation nodes) ─────────────────
    with col_out:
        st.markdown('<div class="pipe-col-header">Output Datasets</div>',
                    unsafe_allow_html=True)
        # Group producers by effective output_key so coworkers show up
        # under the shared target dataset while secondary workers show
        # their own derived output.
        seen: dict[str, list[str]] = {}
        for node in pipeline.annotations:
            key = pipeline.effective_output_key(node)
            if not key:
                continue
            if node.mode == MODE_SECONDARY_WORKER:
                label = f"{node.id} (secondary worker)"
            elif node.mode == MODE_COWORKER:
                label = f"{node.id} (coworker)"
            else:
                label = node.id
            seen.setdefault(key, []).append(label)
        if not seen:
            st.caption("No annotation outputs yet.")
        for key, producers in seen.items():
            try:
                exists = store.has_cached_data(key)
                meta = store.get_cache_metadata(key) if exists else None
                n = meta.total_records if meta else 0
                ts = meta.updated_at[:19] if meta and meta.updated_at else "—"
            except Exception:
                exists, n, ts = False, 0, "—"
            chip = ('<span class="pipe-chip green">live</span>' if exists
                    else '<span class="pipe-chip grey">not written</span>')
            _card(
                f'<div class="title">📋 {chip}</div>'
                f'<div class="sub">{key}</div>'
                f'<div class="meta">produced by: {", ".join(producers)}'
                + (f' · {n:,} records · updated {ts}' if exists else '')
                + '</div>',
                kind_class="has-data" if exists else "empty",
            )

    # ── Training nodes ──────────────────────────────────────────────────
    with col_tr:
        st.markdown('<div class="pipe-col-header">Model Components</div>',
                    unsafe_allow_html=True)
        for node in pipeline.trainings:
            _render_training_card(pipeline, node, store)
        _render_add_training(pipeline)
