"""Streamlit tab: draft, edit, approve, and export critique/guide
completions for every training unit under the selected annotation key.

Drafts are persisted in the segment store under a sidecar cache key
(``<annotation_key>_llm_drafts``), so reloads, container restarts, and
parallel browser tabs all see the same review state. No external
annotation server, no subprocess launcher — the tab calls
``training_unit_store`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from app.pipelines.training.training_unit_store import (
    Draft,
    TrainingUnit,
    all_drafts,
    build_chat_dataset,
    collect_training_units,
    delete_draft,
    draft_unit,
    export_annotation_jsonl,
    load_draft,
    save_draft,
)


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]  # acla_ai_service/
_DEFAULT_OUTPUT = _AI_SERVICE_DIR / "models" / "llm_datasets" / "telemetry_descriptions_v1.jsonl"
_DEFAULT_CHAT = _AI_SERVICE_DIR / "models" / "llm_datasets" / "telemetry_descriptions_v1.chat.jsonl"

_CLAUDE_MODELS = ["claude-sonnet-4-6", "claude-opus-4-7"]

_STATE_KEY = "llm_pipeline_state"


def _state(annotation_key: str) -> dict:
    s = st.session_state.setdefault(_STATE_KEY, {})
    # Reset on annotation-key change so per-unit widget state doesn't bleed across datasets.
    if s.get("annotation_key") != annotation_key:
        s.clear()
        s["annotation_key"] = annotation_key
    s.setdefault("filter", [])
    s.setdefault("model", _CLAUDE_MODELS[0])
    s.setdefault("use_thinking", False)
    s.setdefault("selected_unit_id", None)
    return s


def _badge(status: str) -> str:
    return {"missing": "⚪", "draft": "🟡", "approved": "🟢"}.get(status, "•")


def _unit_status(annotation_key: str, unit: TrainingUnit, drafts: dict) -> str:
    d = drafts.get(unit.unit_id)
    if d is None or not d.has_both():
        return "missing"
    return "approved" if d.approved else "draft"


def render_llm_pipeline(selected_annotation_key: str) -> None:
    st.header("LLM Annotation & Training Pipeline")
    st.caption(
        "Draft critique/guide completions with Claude → review/edit inline → "
        "approve → export the training JSONL. Drafts persist in the segment "
        "store under `<annotation_key>_llm_drafts`."
    )

    if not selected_annotation_key:
        st.warning("Select an annotation dataset in the sidebar first.")
        return

    state = _state(selected_annotation_key)

    try:
        units = collect_training_units(selected_annotation_key)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if not units:
        st.info("No labelled training units found under this annotation key yet.")
        return

    drafts = all_drafts(selected_annotation_key)

    # ── Summary ────────────────────────────────────────────────────────────
    totals = {"missing": 0, "draft": 0, "approved": 0}
    for u in units:
        totals[_unit_status(selected_annotation_key, u, drafts)] += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total units", len(units))
    c2.metric("⚪ Missing", totals["missing"])
    c3.metric("🟡 Drafts", totals["draft"])
    c4.metric("🟢 Approved", totals["approved"])

    # ── Drafting options ───────────────────────────────────────────────────
    with st.expander("Claude drafting options", expanded=False):
        state["model"] = st.selectbox(
            "Model", _CLAUDE_MODELS,
            index=_CLAUDE_MODELS.index(state["model"]) if state["model"] in _CLAUDE_MODELS else 0,
        )
        state["use_thinking"] = st.checkbox(
            "Use extended thinking", value=state["use_thinking"],
        )

    # ── Bulk actions ───────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(
            f"Draft all missing ({totals['missing']})",
            disabled=totals["missing"] == 0,
            use_container_width=True,
        ):
            _draft_all_missing(selected_annotation_key, units, drafts, state)
            st.rerun()
    with b2:
        output_path = st.session_state.get("llm_export_path", str(_DEFAULT_OUTPUT))
        if st.button("Export approved → JSONL", use_container_width=True,
                     disabled=totals["approved"] == 0):
            n = export_annotation_jsonl(
                selected_annotation_key, Path(output_path), only_approved=True,
            )
            st.success(f"Wrote {n} approved unit(s) → {output_path}")
    with b3:
        chat_path = st.session_state.get("llm_chat_path", str(_DEFAULT_CHAT))
        if st.button("Build chat-format JSONL", use_container_width=True):
            if not Path(output_path).exists():
                st.error(f"Annotation JSONL not found at {output_path} — run Export first.")
            else:
                rows = build_chat_dataset(Path(output_path), Path(chat_path))
                st.success(f"Built {rows} chat row(s) → {chat_path}")

    with st.expander("Output paths", expanded=False):
        st.session_state["llm_export_path"] = st.text_input(
            "Annotation JSONL (dataset_builder input)",
            value=st.session_state.get("llm_export_path", str(_DEFAULT_OUTPUT)),
        )
        st.session_state["llm_chat_path"] = st.text_input(
            "Chat-format JSONL (trainer input)",
            value=st.session_state.get("llm_chat_path", str(_DEFAULT_CHAT)),
        )

    st.info(
        "After exporting + building the chat JSONL, switch to the "
        "**🏋️ Training** tab to fine-tune the LLM."
    )

    st.divider()

    # ── Unit list + filter ─────────────────────────────────────────────────
    # Build the suggestion set from what's actually in the dataset so the
    # multiselect autocompletes against real values.
    status_opts = ["missing", "draft", "approved"]
    label_opts: set[str] = set()
    session_opts: set[str] = set()
    for u in units:
        session_opts.add(u.session_id)
        for name in (u.parent_names(), u.children_names()):
            for part in name.split(","):
                part = part.strip()
                if part:
                    label_opts.add(part)
    suggestions = status_opts + sorted(label_opts) + sorted(session_opts)

    # Drop any previously-selected terms no longer in the suggestion list so
    # st.multiselect doesn't raise on default-not-in-options.
    state["filter"] = [t for t in state["filter"] if t in suggestions]

    state["filter"] = st.multiselect(
        "Show units",
        options=suggestions,
        default=state["filter"],
        placeholder="Filter by status, label name, or session id (AND across terms)",
    )

    terms = [t.lower() for t in state["filter"]]

    def _passes_filter(u: TrainingUnit) -> bool:
        if not terms:
            return True
        haystack = " ".join((
            _unit_status(selected_annotation_key, u, drafts),
            u.unit_id,
            u.session_id,
            u.parent_names(),
            u.children_names(),
        )).lower()
        return all(t in haystack for t in terms)

    visible = [u for u in units if _passes_filter(u)]
    if not visible:
        st.info("No units match the current filter.")
        return

    # Pick a default selection if none / stale
    visible_ids = [u.unit_id for u in visible]
    if state["selected_unit_id"] not in visible_ids:
        state["selected_unit_id"] = visible_ids[0]

    options = [
        f"{_badge(_unit_status(selected_annotation_key, u, drafts))} "
        f"{u.unit_id}  ({u.session_id})"
        for u in visible
    ]
    selected_idx = visible_ids.index(state["selected_unit_id"])
    pick = st.selectbox(
        f"Unit ({len(visible)} of {len(units)})",
        options=range(len(visible)),
        format_func=lambda i: options[i],
        index=selected_idx,
    )
    state["selected_unit_id"] = visible[pick].unit_id

    _render_unit_editor(
        selected_annotation_key, visible[pick], drafts.get(visible[pick].unit_id), state,
    )


# ---------------------------------------------------------------------------
# Per-unit editor
# ---------------------------------------------------------------------------

def _render_unit_editor(
    annotation_key: str,
    unit: TrainingUnit,
    draft: Draft | None,
    state: dict,
) -> None:
    st.subheader(f"Unit `{unit.unit_id}`")

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Kind:** {unit.kind}")
    meta_cols[1].markdown(f"**Session:** `{unit.session_id}`")
    meta_cols[2].markdown(
        f"**Iloc:** {unit.start_index} – {unit.end_index}"
        if unit.start_index is not None and unit.end_index is not None
        else "**Iloc:** —"
    )

    st.markdown("**Labels (read-only context)**")
    st.code(unit.labels_text(), language="text")
    with st.expander("Per-child label breakdown", expanded=False):
        st.markdown(f"**parent:** {unit.parent_names()}")
        if unit.children_names():
            st.markdown(f"**children:** {unit.children_names()}")
        else:
            st.markdown("_(no children — isolated unit)_")

    # State for the per-unit text areas. Reset when the unit changes so the
    # last edit on a different unit doesn't leak in.
    last_id = state.get("last_editor_unit_id")
    if last_id != unit.unit_id:
        state["critique_text"] = draft.completion_critique if draft else ""
        state["guide_text"] = draft.completion_guide if draft else ""
        state["last_editor_unit_id"] = unit.unit_id

    if draft and draft.model_version:
        st.caption(f"Last draft from: `{draft.model_version}` · approved: {draft.approved}")

    # ── Draft button ───────────────────────────────────────────────────────
    if st.button("🤖 Generate / regenerate with Claude", use_container_width=False):
        with st.spinner(f"Drafting with {state['model']}…"):
            try:
                new_draft = draft_unit(
                    unit,
                    model=state["model"],
                    use_thinking=state["use_thinking"],
                )
            except RuntimeError as exc:
                st.error(f"Claude drafting failed: {exc}")
                return
        state["critique_text"] = new_draft.completion_critique
        state["guide_text"] = new_draft.completion_guide
        save_draft(annotation_key, unit.unit_id, new_draft)
        st.success("Drafted and saved (not yet approved).")
        st.rerun()

    # ── Editable completions ───────────────────────────────────────────────
    state["critique_text"] = st.text_area(
        "Critique (retrospective, post-lap voice) — 1-3 sentences, ≤60 words.",
        value=state["critique_text"],
        height=120,
        key=f"critique_{unit.unit_id}",
    )
    state["guide_text"] = st.text_area(
        "Guide (prescriptive, imperative voice) — 1-3 sentences, ≤60 words.",
        value=state["guide_text"],
        height=120,
        key=f"guide_{unit.unit_id}",
    )

    # ── Save / approve / delete ────────────────────────────────────────────
    a1, a2, a3 = st.columns([1, 1, 1])
    with a1:
        if st.button("💾 Save draft", use_container_width=True):
            updated = Draft(
                completion_critique=state["critique_text"],
                completion_guide=state["guide_text"],
                approved=draft.approved if draft else False,
                model_version=draft.model_version if draft else "manual",
            )
            save_draft(annotation_key, unit.unit_id, updated)
            st.success("Saved.")
            st.rerun()
    with a2:
        approve_disabled = not (state["critique_text"].strip() and state["guide_text"].strip())
        label = "✅ Approve" if not (draft and draft.approved) else "↩ Un-approve"
        if st.button(label, use_container_width=True, disabled=approve_disabled):
            updated = Draft(
                completion_critique=state["critique_text"],
                completion_guide=state["guide_text"],
                approved=not (draft.approved if draft else False),
                model_version=draft.model_version if draft else "manual",
            )
            save_draft(annotation_key, unit.unit_id, updated)
            st.rerun()
    with a3:
        if draft is not None and st.button(
            "🗑 Delete draft", use_container_width=True,
            help="Remove this unit's saved critique/guide.",
        ):
            delete_draft(annotation_key, unit.unit_id)
            state["critique_text"] = ""
            state["guide_text"] = ""
            st.rerun()


# ---------------------------------------------------------------------------
# Bulk draft
# ---------------------------------------------------------------------------

def _draft_all_missing(
    annotation_key: str,
    units: List[TrainingUnit],
    drafts: dict,
    state: dict,
) -> None:
    targets = [
        u for u in units
        if (d := drafts.get(u.unit_id)) is None or not d.has_both()
    ]
    if not targets:
        return

    bar = st.progress(0.0, text="Starting…")
    failures: List[str] = []
    for i, unit in enumerate(targets, 1):
        bar.progress(
            (i - 1) / len(targets),
            text=f"[{i}/{len(targets)}] {unit.unit_id} — drafting…",
        )
        try:
            new_draft = draft_unit(
                unit,
                model=state["model"],
                use_thinking=state["use_thinking"],
            )
            save_draft(annotation_key, unit.unit_id, new_draft)
        except RuntimeError as exc:
            failures.append(f"{unit.unit_id}: {exc}")
    bar.progress(1.0, text="Done.")
    if failures:
        st.error("Some units failed:\n\n" + "\n".join(f"- {f}" for f in failures))
    else:
        st.success(f"Drafted {len(targets)} unit(s). Review and approve below.")
