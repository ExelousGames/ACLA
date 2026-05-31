"""Pipeline graph view.

Three columns:

    Annotation Components → Output Datasets → Model Components

Each annotation card owns:
- a **kind** dropdown (lap / detailed / batch_* / llm / …) that decides
  which tab the single Open button routes to;
- a **mode** picker — *fork* (copy source into a per-node input_key),
  *secondary worker* (read a target's output, write back to it), or
  *coworker* (share both input and output with a target node);
- a source picker — any cache_key in the store (fork), or a sibling
  annotation's ``<id>.output`` (any mode);
- the fork's behind-source status with a Pull button (fork mode only);
- the output dataset status;
- one Open button that routes to the tab whose ``ui_route`` matches the
  selected kind.

Plus "+ Add annotation" / "+ Add training" buttons at the bottom of
each column so the user can grow the pipeline incrementally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import streamlit as st

from app.pipelines.manifest import node_kinds
from app.pipelines.manifest.forking import (
    compare_against_source,
    derive_annotation_input_key,
    fork_dataset,
)
from app.pipelines.manifest.models import (
    MODE_COWORKER,
    MODE_COPY,
    MODE_SECONDARY_WORKER,
    AnnotationNode,
    Pipeline,
    TrainingNode,
)
from app.pipelines.manifest.registry import save as save_pipeline, slugify
from app.pipelines.manifest.label_migration import migrate_dataset_labels
from app.pipelines.manifest.segment_refresh import refresh_node_segments
from app.pipelines.training.config import TrainingPipelineConfig


MODE_LABELS = {
    MODE_COPY: "Copy from source",
    MODE_SECONDARY_WORKER: "Secondary worker (adds to target's output)",
    MODE_COWORKER: "Coworker (shares target's input + output)",
}
MODE_ORDER = [MODE_COPY, MODE_SECONDARY_WORKER, MODE_COWORKER]
MODE_DESCRIPTIONS = {
    MODE_COPY: (
        "Copy the source dataset into this node's own input. The copy is "
        "independent — edits here don't touch the source, and you can pull "
        "in upstream changes later via the **Update from source** button."
    ),
    MODE_SECONDARY_WORKER: (
        "Read **and** write the target's *output* dataset. Use when this "
        "node adds new data on top of another node's results — e.g. "
        "detailed annotation adding child segments under an existing label."
    ),
    MODE_COWORKER: (
        "Read the target's *input* and write to the target's *output*. "
        "Use for parallel assistance — e.g. an AI agent helping the user "
        "annotate the same dataset side-by-side."
    ),
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

/* Mode picker: each option is an st.button styled as a card. The
   button's label is multi-line markdown (title + description); CSS
   hides the description until the button is hovered, and keeps it
   visible on the disabled (current) option. Scoped by the
   `st-key-mode_pick_*` class Streamlit adds when the button has key. */
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


def _now_iso() -> str:
    return datetime.now().isoformat()


def _card(html: str, kind_class: str = "") -> None:
    klass = f"pipe-card {kind_class}".strip()
    st.markdown(f'<div class="{klass}">{html}</div>', unsafe_allow_html=True)


def _render_mode_picker(current: str, key_prefix: str) -> Optional[str]:
    """Popover dropdown of mode options. Returns a new mode if the user
    clicked one this run, else ``None``. Option cards are styled
    ``st.button``\\s — see the ``st-key-mode_pick_*`` rules in ``_CARD_CSS``
    for the hover-expand behavior."""
    st.caption("Mode")
    with st.popover(MODE_LABELS[current], use_container_width=True):
        st.caption("Hover an option to see what it does, click to switch.")
        picked: Optional[str] = None
        for mode in MODE_ORDER:
            is_current = mode == current
            title = MODE_LABELS[mode] + ("  ·  *current*" if is_current else "")
            label = f"**{title}**\n\n{MODE_DESCRIPTIONS[mode]}"
            if st.button(
                label, key=f"mode_pick_{key_prefix}_{mode}",
                use_container_width=True, disabled=is_current,
            ):
                picked = mode
    return picked


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

    Fork mode: every cache_key in the store + every other annotation's
        ``<id>.output``.
    Secondary worker / coworker: only sibling outputs (these modes
        target a node, not an external Lance dataset — there's nobody
        to chain off).
    """
    sibling_outputs = [
        f"{n.id}.output" for n in pipeline.annotations if n.id != self_id
    ]
    if siblings_only:
        return sibling_outputs
    try:
        store_keys = sorted(store.list_cache_keys())
    except Exception:
        store_keys = []
    return sibling_outputs + store_keys


def _fork_for_annotation(pipeline: Pipeline, node: AnnotationNode, store: Any) -> None:
    """(Re-)fork ``node.source_ref`` into a fresh per-annotation input_key."""
    source_key = pipeline.resolve_source_key(node.source_ref)
    if not source_key:
        st.error(f"Could not resolve source `{node.source_ref}`.")
        return
    if not store.has_cached_data(source_key):
        st.error(f"Source `{source_key}` has no data in the store yet.")
        return

    pipeline.version += 1
    target_key = derive_annotation_input_key(
        pipeline_id=pipeline.id, node_id=node.id, version=pipeline.version,
    )
    progress = st.progress(0.0, text=f"Forking `{source_key}` → `{target_key}`…")

    def _tick(done: int, total: int) -> None:
        if total > 0:
            progress.progress(done / total, text=f"Forking ({done}/{total})")

    fork_dataset(source_key=source_key, target_key=target_key,
                 store=store, progress=_tick)

    src_meta = store.get_cache_metadata(source_key)
    node.input_key = target_key
    node.copied_at = _now_iso()
    node.source_updated_at_on_copy = src_meta.updated_at if src_meta else None
    save_pipeline(pipeline)
    progress.empty()


def _annotation_input_status(
    pipeline: Pipeline, node: AnnotationNode, store: Any,
) -> tuple[str, str, str]:
    """Return (chip_html, detail_html, kind_class) for the input."""
    if not node.source_ref:
        return ('<span class="pipe-chip grey">no source</span>',
                "Pick a source below.", "empty")

    target_label = "Target" if node.mode != MODE_COPY else "Source"
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
            f"{source_line} → <code>{read_key}</code><br/>"
            f"Read &amp; write target's output (no copy)"
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
            f"Reads target's input: <code>{read_key or '—'}</code>"
            + (f" · {n:,} rec." if exists else " · empty/not copied yet.")
            + f"<br/>Writes target's output: <code>{write_key or '—'}</code>"
        )
        chip = '<span class="pipe-chip teal">coworker' + (
            " · empty</span>" if not exists else "</span>"
        )
        return (chip, detail, "coworker")

    # Copy mode below.
    source_key = pipeline.resolve_source_key(node.source_ref)
    if source_key and source_key != node.source_ref:
        source_line += f" → <code>{source_key}</code>"
    if not node.input_key:
        return ('<span class="pipe-chip amber">no copy yet</span>',
                f"{source_line}<br/>"
                f"No local copy yet — click <b>Copy from source</b> to make one.",
                "empty")

    if not source_key:
        return ('<span class="pipe-chip amber">source missing</span>',
                f"{source_line}<br/>"
                f"Copy: <code>{node.input_key}</code> (source no longer resolves).",
                "behind")

    cmp = compare_against_source(store, source_key, node.input_key)
    if cmp.is_behind:
        delta = cmp.source_total_records - cmp.copy_total_records
        return ('<span class="pipe-chip amber">copy behind source</span>',
                f"{source_line}<br/>"
                f"Copy: <code>{node.input_key}</code> · {cmp.copy_total_records:,} rec"
                f" (source has {cmp.source_total_records:,}"
                + (f", +{delta:,}" if delta > 0 else "") + ") — "
                f"click <b>Update from source</b> to re-copy.",
                "behind")
    return ('<span class="pipe-chip green">copy up to date</span>',
            f"{source_line}<br/>"
            f"Copy: <code>{node.input_key}</code> · {cmp.copy_total_records:,} rec",
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


def _has_cached_data(store: Any, key: Optional[str]) -> bool:
    if not key:
        return False
    try:
        return store.has_cached_data(key)
    except Exception:
        return False


def _render_maintenance_dropdown(
    pipeline: Pipeline,
    node: AnnotationNode,
    store: Any,
    *,
    refresh_disabled: bool,
    migrate_disabled: bool,
) -> None:
    with st.popover("Maintenance", use_container_width=True):
        st.caption("Dataset maintenance")

        if st.button(
            "Refresh segments",
            key=f"refresh_segs_{node.id}",
            use_container_width=True,
            disabled=refresh_disabled,
            help=(
                "Re-slice telemetry_data on every saved segment from the "
                "current input. Run after 'Update from source' to propagate "
                "new columns into segments that were annotated against the "
                "old input."
            ),
        ):
            try:
                summary = refresh_node_segments(store, node)
            except ValueError as exc:
                st.error(f"Refresh failed: {exc}")
            else:
                st.toast(
                    f"Refreshed {summary.segments_refreshed} segment(s) "
                    f"across {summary.chunks_written} chunk(s).",
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
            use_container_width=True,
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


# ── Annotation card ──────────────────────────────────────────────────────────
def _render_annotation_card(
    pipeline: Pipeline, node: AnnotationNode, store: Any, cfg: TrainingPipelineConfig,
) -> None:
    ann_specs = node_kinds.list_by_category("annotation")
    kind_choices = [s.kind for s in ann_specs]
    kind_labels = {s.kind: s.display for s in ann_specs}

    chip, detail, kind_class = _annotation_input_status(pipeline, node, store)
    spec = node_kinds.get(node.kind)
    effective_out = pipeline.effective_output_key(node)
    out_label, out_n, out_ts = _output_status(store, effective_out) if effective_out else ("—", 0, "")

    if node.mode == MODE_COPY:
        if not node.output_key:
            out_line = (
                '<br/>Output: <i>not configured yet</i> — '
                'pick a directory &amp; filename on first open.'
            )
        else:
            dir_hint = (
                f'<br/><span class="meta">in <code>{node.output_dir}</code></span>'
                if node.output_dir else ""
            )
            out_line = (
                f'<br/>Output: <code>{node.output_key}</code> · {out_label}'
                f'{dir_hint}'
            )
    else:
        share_label = "secondary worker" if node.mode == MODE_SECONDARY_WORKER else "coworker"
        out_line = (
            f'<br/>Output: <i>shared with target</i> <code>{effective_out or "—"}</code>'
            f' ({share_label}) · {out_label}'
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

        # ── Mode (read-only — chosen at creation, locked thereafter) ─────
        st.caption(
            f"Mode: **{MODE_LABELS[node.mode]}** 🔒",
            help=MODE_DESCRIPTIONS[node.mode],
        )

        # ── Source (read-only — chosen at creation, locked thereafter) ───
        placeholder = "— not set —"
        st.selectbox(
            "Source",
            options=[node.source_ref or placeholder],
            index=0,
            key=f"ann_src_{node.id}",
            label_visibility="collapsed",
            disabled=True,
            help="Source is locked after creation. Delete and recreate this "
                 "node to change it.",
        )

        # ── Action buttons ───────────────────────────────────────────────
        btn_cols = st.columns([1, 1, 1, 0.4])

        effective = pipeline.effective_input_key(node)
        open_disabled = not effective

        with btn_cols[0]:
            if node.mode != MODE_COPY:
                # No copy in non-copy modes — show a disabled stub for layout.
                stub_label = ("Secondary worker (no copy)"
                              if node.mode == MODE_SECONDARY_WORKER
                              else "Coworker (no copy)")
                st.button(stub_label, key=f"copy_{node.id}",
                          use_container_width=True, disabled=True)
            else:
                behind = (kind_class == "behind")
                has_copy = bool(node.input_key)
                if has_copy:
                    copy_label = "Update from source"
                    btn_disabled = not behind or not node.source_ref
                else:
                    copy_label = "Copy from source"
                    btn_disabled = not node.source_ref
                if st.button(copy_label, key=f"copy_{node.id}",
                             use_container_width=True,
                             disabled=btn_disabled):
                    _fork_for_annotation(pipeline, node, store)
                    st.rerun()
        with btn_cols[1]:
            out_key = pipeline.effective_output_key(node)
            has_input_data = _has_cached_data(store, node.input_key)
            has_output_data = _has_cached_data(store, out_key)
            refresh_disabled = not (
                node.mode == MODE_COPY and has_input_data and has_output_data
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
                         use_container_width=True, disabled=open_disabled):
                _route(spec.ui_route,
                       annotation_key=pipeline.effective_output_key(node),
                       session_key=effective,
                       node_id=node.id)
        with btn_cols[3]:
            if st.button("🗑", key=f"del_ann_{node.id}",
                         use_container_width=True,
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
        chosen_kind = st.selectbox(
            "Kind",
            options=kind_choices,
            format_func=lambda k: kind_labels[k],
            key="add_ann_kind",
        )
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

        # Pending mode lives in session state so the popover-button
        # picker (which reruns on click) survives between renders.
        pending_mode = st.session_state.setdefault("add_ann_mode", MODE_COPY)
        picked = _render_mode_picker(pending_mode, key_prefix="add")
        if picked is not None and picked != pending_mode:
            st.session_state["add_ann_mode"] = picked
            # Mode change invalidates the pending source (options differ).
            st.session_state.pop("add_ann_source", None)
            st.rerun()

        # ── Source picker ────────────────────────────────────────────────
        siblings_only = pending_mode != MODE_COPY
        source_options = _source_options(
            pipeline, store, self_id="", siblings_only=siblings_only,
        )
        placeholder = "— pick a target —" if siblings_only else "— pick a source —"
        display_options = [placeholder] + source_options
        pending_source = st.session_state.get("add_ann_source")
        try:
            src_idx = display_options.index(pending_source) if pending_source else 0
        except ValueError:
            src_idx = 0
        chosen_src = st.selectbox(
            "Source", options=display_options, index=src_idx,
            key="add_ann_source_select",
            help=("Copy mode: any cache_key in the store, or a sibling "
                  "annotation's output. Secondary worker / coworker: pick "
                  "the target sibling annotation."),
        )
        st.session_state["add_ann_source"] = (
            None if chosen_src == placeholder else chosen_src
        )
        if siblings_only and not source_options:
            st.caption(":warning: No sibling annotations to target — "
                       "create one in copy mode first.")

        if name_slug:
            if pending_mode == MODE_COPY:
                st.caption(
                    f"Node id will be: `{name_slug}`. Output dataset location is "
                    "configured on the annotation page (first-time popup)."
                )
            else:
                st.caption(
                    f"Node id will be: `{name_slug}`. Writes to the target's "
                    "output dataset (no new file is created)."
                )

        source_ref = st.session_state.get("add_ann_source")
        can_create = bool(name_clean) and bool(source_ref)
        if st.button("Create annotation node", type="primary",
                     use_container_width=True,
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
    ann_outputs = [f"{n.id}.output" for n in pipeline.annotations]
    placeholder = "— pick an annotation output —"
    display_options = [placeholder] + ann_outputs
    try:
        default_idx = display_options.index(node.input_ref) if node.input_ref else 0
    except ValueError:
        display_options = [node.input_ref] + display_options
        default_idx = 0
    new_ref = st.selectbox(
        "Input", options=display_options, index=default_idx,
        key=f"tr_src_{node.id}", label_visibility="collapsed",
    )
    if new_ref != placeholder and new_ref != node.input_ref:
        node.input_ref = new_ref
        save_pipeline(pipeline)
        st.rerun()

    btn_cols = st.columns([3, 0.4])
    with btn_cols[0]:
        if st.button(f"Configure {spec.display}", key=f"open_tr_{node.id}",
                     use_container_width=True):
            _route(spec.ui_route,
                   annotation_key=input_key or None,
                   training_node=node.id)
    with btn_cols[1]:
        if st.button("🗑", key=f"del_tr_{node.id}",
                     use_container_width=True,
                     help="Delete this training node"):
            pipeline.trainings = [n for n in pipeline.trainings if n.id != node.id]
            save_pipeline(pipeline)
            st.rerun()


def _render_add_training(pipeline: Pipeline) -> None:
    with st.expander("➕ Add training component", expanded=False):
        tr_specs = node_kinds.list_by_category("training")
        kind_choices = [s.kind for s in tr_specs]
        kind_labels = {s.kind: s.display for s in tr_specs}
        chosen_kind = st.selectbox(
            "Kind",
            options=kind_choices,
            format_func=lambda k: kind_labels[k],
            key="add_tr_kind",
        )
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
                     use_container_width=True,
                     disabled=not name_clean):
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

    col_ann, col_out, col_tr = st.columns([1.5, 1.2, 1.2])

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
        # Secondary worker / coworker nodes have no output of their own —
        # they collaborate on the target's output. Group producers by their
        # effective output_key so collaborators show up under the shared dataset.
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
