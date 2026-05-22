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
    MODE_FORK,
    MODE_SECONDARY_WORKER,
    AnnotationNode,
    Pipeline,
    TrainingNode,
)
from app.pipelines.manifest.registry import derive_output_key, save as save_pipeline, slugify
from app.infra.config.pipeline import PipelineConfig


MODE_LABELS = {
    MODE_FORK: "Copy from source",
    MODE_SECONDARY_WORKER: "Secondary worker (adds to target's output)",
    MODE_COWORKER: "Coworker (shares target's input + output)",
}
MODE_ORDER = [MODE_FORK, MODE_SECONDARY_WORKER, MODE_COWORKER]


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
</style>
"""


def _now_iso() -> str:
    return datetime.now().isoformat()


def _card(html: str, kind_class: str = "") -> None:
    klass = f"pipe-card {kind_class}".strip()
    st.markdown(f'<div class="{klass}">{html}</div>', unsafe_allow_html=True)


def _route(view: str, *, annotation_key: Optional[str] = None,
           session_key: Optional[str] = None, training_node: Optional[str] = None) -> None:
    st.session_state["pipeline_routed_view"] = view
    if annotation_key is not None:
        st.session_state["pipeline_annotation_key"] = annotation_key
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

    target_label = "Target" if node.mode != MODE_FORK else "Source"
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
            f"Read &amp; write target's output (no fork)"
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
            + (f" · {n:,} rec." if exists else " · empty/not forked yet.")
            + f"<br/>Writes target's output: <code>{write_key or '—'}</code>"
        )
        chip = '<span class="pipe-chip teal">coworker' + (
            " · empty</span>" if not exists else "</span>"
        )
        return (chip, detail, "coworker")

    # Fork mode below.
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
    try:
        if not store.has_cached_data(output_key):
            return ("⚪ empty", 0, "")
        meta = store.get_cache_metadata(output_key)
        n = meta.total_records if meta else 0
        ts = meta.updated_at[:19] if meta and meta.updated_at else ""
        return ("🟢 has data", n, ts)
    except Exception:
        return ("⚠️ unknown", 0, "")


# ── Annotation card ──────────────────────────────────────────────────────────
def _render_annotation_card(
    pipeline: Pipeline, node: AnnotationNode, store: Any, cfg: PipelineConfig,
) -> None:
    ann_specs = node_kinds.list_by_category("annotation")
    kind_choices = [s.kind for s in ann_specs]
    kind_labels = {s.kind: s.display for s in ann_specs}

    chip, detail, kind_class = _annotation_input_status(pipeline, node, store)
    spec = node_kinds.get(node.kind)
    effective_out = pipeline.effective_output_key(node)
    out_label, out_n, out_ts = _output_status(store, effective_out) if effective_out else ("—", 0, "")

    if node.mode == MODE_FORK:
        out_line = f'<br/>Output: <code>{node.output_key}</code> · {out_label}'
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

        # ── Mode picker: Fork / Secondary worker / Coworker ──────────────
        try:
            mode_idx = MODE_ORDER.index(node.mode)
        except ValueError:
            mode_idx = 0
        new_mode = st.radio(
            "Mode",
            options=MODE_ORDER,
            index=mode_idx,
            format_func=lambda m: MODE_LABELS[m],
            key=f"ann_mode_{node.id}",
            horizontal=True,
            help=("Fork: copy the source into this node's own dataset. "
                  "Secondary worker: read AND write the target's output dataset "
                  "(adds new data to it — e.g. detailed annotation adding children). "
                  "Coworker: read the target's input and write to the target's "
                  "output (parallel assist — e.g. AI agent helping the user)."),
        )
        if new_mode != node.mode:
            node.mode = new_mode
            # Switching to a non-fork mode invalidates the fork pointer and
            # forces the source to a sibling reference.
            if new_mode != MODE_FORK:
                node.input_key = None
                node.copied_at = None
                node.source_updated_at_on_copy = None
                if node.source_ref and not node.source_ref.endswith(".output"):
                    node.source_ref = None
            save_pipeline(pipeline)
            st.rerun()

        # ── Source picker ────────────────────────────────────────────────
        options = _source_options(
            pipeline, store, node.id, siblings_only=(node.mode != MODE_FORK),
        )
        placeholder = "— pick a target —" if node.mode != MODE_FORK else "— pick a source —"
        display_options = [placeholder] + options
        try:
            default_idx = display_options.index(node.source_ref) if node.source_ref else 0
        except ValueError:
            # Stale ref no longer in the options — surface it but mark with the placeholder.
            display_options = [node.source_ref] + display_options
            default_idx = 0
        new_ref = st.selectbox(
            "Source", options=display_options, index=default_idx,
            key=f"ann_src_{node.id}", label_visibility="collapsed",
        )
        new_ref_clean = None if new_ref == placeholder else new_ref
        if new_ref_clean != node.source_ref:
            node.source_ref = new_ref_clean
            # Switching source invalidates the old fork.
            node.input_key = None
            node.copied_at = None
            node.source_updated_at_on_copy = None
            save_pipeline(pipeline)
            st.rerun()

        # ── Action buttons ───────────────────────────────────────────────
        btn_cols = st.columns([1, 1, 0.4])

        effective = pipeline.effective_input_key(node)
        open_disabled = not effective

        with btn_cols[0]:
            if node.mode != MODE_FORK:
                # No copy in non-fork modes — show a disabled stub for layout.
                stub_label = ("Secondary worker (no copy)"
                              if node.mode == MODE_SECONDARY_WORKER
                              else "Coworker (no copy)")
                st.button(stub_label, key=f"fork_{node.id}",
                          use_container_width=True, disabled=True)
            else:
                behind = (kind_class == "behind")
                has_copy = bool(node.input_key)
                if has_copy:
                    fork_label = "Update from source"
                    btn_disabled = not behind or not node.source_ref
                else:
                    fork_label = "Copy from source"
                    btn_disabled = not node.source_ref
                if st.button(fork_label, key=f"fork_{node.id}",
                             use_container_width=True,
                             disabled=btn_disabled):
                    _fork_for_annotation(pipeline, node, store)
                    st.rerun()
        with btn_cols[1]:
            if st.button(f"Open", key=f"open_ann_{node.id}",
                         type="primary",
                         use_container_width=True, disabled=open_disabled):
                _route(spec.ui_route,
                       annotation_key=pipeline.effective_output_key(node),
                       session_key=effective)
        with btn_cols[2]:
            if st.button("🗑", key=f"del_ann_{node.id}",
                         use_container_width=True,
                         help="Delete this annotation node"):
                pipeline.annotations = [n for n in pipeline.annotations if n.id != node.id]
                save_pipeline(pipeline)
                st.rerun()


# ── Add-annotation form (inline expander at the bottom of the column) ────────
def _render_add_annotation(pipeline: Pipeline, cfg: PipelineConfig) -> None:
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
                 "Also drives the node id (slugified) and output dataset key.",
        )
        name_clean = name.strip()
        name_slug = slugify(name_clean) if name_clean else ""

        if name_slug:
            output_key_preview = derive_output_key(
                cfg.annotation_cache_key, pipeline.id, name_slug,
            )
            st.caption(
                f"Node id will be: `{name_slug}` · "
                f"output dataset: `{output_key_preview}`"
            )

        if st.button("Create annotation node", type="primary",
                     use_container_width=True,
                     disabled=not name_clean):
            existing_ids = {n.id for n in pipeline.annotations} | {n.id for n in pipeline.trainings}
            if name_slug in existing_ids:
                st.error(f"A node named `{name_clean}` (id `{name_slug}`) already exists.")
                return
            output_key = derive_output_key(
                cfg.annotation_cache_key, pipeline.id, name_slug,
            )
            pipeline.annotations.append(AnnotationNode(
                id=name_slug,
                kind=chosen_kind,
                name=name_clean,
                output_key=output_key,
            ))
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

    cfg = PipelineConfig()

    col_ann, col_out, col_tr = st.columns([1.5, 1.2, 1.2])

    # ── Annotation nodes ────────────────────────────────────────────────
    with col_ann:
        st.markdown('<div class="pipe-col-header">Annotation Components</div>',
                    unsafe_allow_html=True)
        for node in pipeline.annotations:
            _render_annotation_card(pipeline, node, store, cfg)
        _render_add_annotation(pipeline, cfg)

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
