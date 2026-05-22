"""Streamlit tab: author multi-turn templated training conversations for
every segment-derived training unit and export them as chat JSONL.

Each unit owns a list of **entries**. Each entry is a multi-turn chat
(system / user / assistant turns) whose text is a template with ``{var}``
placeholders. Variables resolve against the unit's labels at export
time — labels are embedded directly into the message text; there is no
implicit system message. The full ``UnitEntries`` payload is persisted
in the segment store under ``<annotation_key>_llm_entries`` so reloads
and parallel browser tabs all see the same edit state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st

from app.pipelines.training.training_unit_store import (
    TrainingEntry,
    TrainingUnit,
    Turn,
    UnitEntries,
    all_entries,
    collect_training_units,
    delete_entries,
    export_entries_chat_jsonl,
    save_entries,
    template_variables,
)
from ui.custom_components.llm_entry_editor import llm_entry_editor


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]  # acla_ai_service/
_DEFAULT_CHAT = _AI_SERVICE_DIR / "models" / "llm_datasets" / "telemetry_descriptions_v1.chat.jsonl"

_STATE_KEY = "llm_pipeline_state"


def _state(annotation_key: str) -> dict:
    s = st.session_state.setdefault(_STATE_KEY, {})
    if s.get("annotation_key") != annotation_key:
        s.clear()
        s["annotation_key"] = annotation_key
    s.setdefault("filter", [])
    s.setdefault("selected_unit_id", None)
    s.setdefault("editor", {})  # unit_id -> live UnitEntries dict
    return s


def _hydrate_editor(state: dict, unit_id: str, stored: UnitEntries | None) -> dict:
    """Load (or initialize) the in-memory editor copy for a unit."""
    editor = state["editor"]
    if unit_id not in editor:
        base = stored or UnitEntries(entries=[], approved=False)
        editor[unit_id] = {
            "entries": [
                {"entry_id": e.entry_id, "turns": [{"role": t.role, "template": t.template} for t in e.turns]}
                for e in base.entries
            ],
            "approved": base.approved,
        }
    return editor[unit_id]


def _to_unit_entries(editor_state: dict) -> UnitEntries:
    return UnitEntries(
        entries=[
            TrainingEntry(
                entry_id=e["entry_id"],
                turns=[Turn(role=t["role"], template=t["template"]) for t in e["turns"]],
            )
            for e in editor_state["entries"]
        ],
        approved=editor_state["approved"],
    )


def _unit_status(stored: UnitEntries | None) -> str:
    if stored is None or stored.is_empty():
        return "empty"
    return "approved" if stored.approved else "draft"


def _badge(status: str) -> str:
    return {"empty": "⚪", "draft": "🟡", "approved": "🟢"}.get(status, "•")


def render_llm_pipeline(selected_annotation_key: str) -> None:
    st.header("LLM Annotation & Training Pipeline")
    st.caption(
        "Author multi-turn training conversations per segment. Use "
        "`{parent}`, `{child_1}` etc. as placeholders — they resolve "
        "against the unit's labels at export."
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

    stored_all: Dict[str, UnitEntries] = all_entries(selected_annotation_key)

    # ── Summary ────────────────────────────────────────────────────────────
    totals = {"empty": 0, "draft": 0, "approved": 0}
    total_entries = 0
    for u in units:
        s = stored_all.get(u.unit_id)
        totals[_unit_status(s)] += 1
        if s is not None:
            total_entries += len(s.entries)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total units", len(units))
    c2.metric("⚪ Empty", totals["empty"])
    c3.metric("🟡 Drafts", totals["draft"])
    c4.metric("🟢 Approved", totals["approved"])
    c5.metric("Σ entries", total_entries)

    # ── Export ─────────────────────────────────────────────────────────────
    chat_path = st.session_state.get("llm_chat_path", str(_DEFAULT_CHAT))
    with st.expander("Output path", expanded=False):
        st.session_state["llm_chat_path"] = st.text_input(
            "Chat-format JSONL (trainer input)", value=chat_path,
        )
        chat_path = st.session_state["llm_chat_path"]

    e1, e2 = st.columns(2)
    with e1:
        if st.button(
            "Export approved → chat JSONL",
            use_container_width=True,
            disabled=totals["approved"] == 0,
        ):
            n = export_entries_chat_jsonl(
                selected_annotation_key, Path(chat_path), only_approved=True,
            )
            st.success(f"Wrote {n} chat row(s) → {chat_path}")
    with e2:
        if st.button(
            "Export all (incl. drafts) → chat JSONL",
            use_container_width=True,
            disabled=(totals["approved"] + totals["draft"]) == 0,
        ):
            n = export_entries_chat_jsonl(
                selected_annotation_key, Path(chat_path), only_approved=False,
            )
            st.success(f"Wrote {n} chat row(s) → {chat_path}")

    st.info(
        "After exporting the chat JSONL, switch to the **🏋️ Training** tab "
        "to fine-tune the LLM."
    )

    st.divider()

    # ── Unit list + filter ─────────────────────────────────────────────────
    status_opts = ["empty", "draft", "approved"]
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
            _unit_status(stored_all.get(u.unit_id)),
            u.unit_id, u.session_id, u.parent_names(), u.children_names(),
        )).lower()
        return all(t in haystack for t in terms)

    visible = [u for u in units if _passes_filter(u)]
    if not visible:
        st.info("No units match the current filter.")
        return

    visible_ids = [u.unit_id for u in visible]
    if state["selected_unit_id"] not in visible_ids:
        state["selected_unit_id"] = visible_ids[0]

    options = [
        f"{_badge(_unit_status(stored_all.get(u.unit_id)))} {u.unit_id}  ({u.session_id})"
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
        selected_annotation_key,
        visible[pick],
        stored_all.get(visible[pick].unit_id),
        state,
    )


# ---------------------------------------------------------------------------
# Per-unit editor
# ---------------------------------------------------------------------------

def _render_unit_editor(
    annotation_key: str,
    unit: TrainingUnit,
    stored: UnitEntries | None,
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

    editor = _hydrate_editor(state, unit.unit_id, stored)
    vars_ = template_variables(unit)

    # Custom component owns all editing UI. Returns a payload on each
    # Save / Approve click in the iframe; ``nonce`` lets us ignore stale
    # payloads from Streamlit reruns that aren't tied to a new click.
    result = llm_entry_editor(
        entries=editor["entries"],
        variables=vars_,
        approved=editor["approved"],
        key=f"editor_{unit.unit_id}",
    )

    last_nonce = state.setdefault("last_nonce", {}).get(unit.unit_id, 0)
    if isinstance(result, dict) and int(result.get("nonce") or 0) > last_nonce:
        state["last_nonce"][unit.unit_id] = int(result["nonce"])
        # Mirror what came back from the component into our editor copy
        # so subsequent reruns keep showing the freshly-saved state.
        editor["entries"] = list(result.get("entries") or [])
        editor["approved"] = bool(result.get("approved", False))
        save_entries(annotation_key, unit.unit_id, _to_unit_entries(editor))
        action = result.get("action") or "save"
        st.toast("Approved." if action == "approve" else "Saved.", icon="✅")

    # Out-of-band side effects (component can't do these — they reload data).
    o1, o2 = st.columns(2)
    with o1:
        if st.button(
            "🔄 Reload from disk",
            use_container_width=True,
            key=f"reset_{unit.unit_id}",
            help="Drop in-memory edits and re-read the saved entries.",
        ):
            state["editor"].pop(unit.unit_id, None)
            state.setdefault("last_nonce", {}).pop(unit.unit_id, None)
            st.rerun()
    with o2:
        if stored is not None and st.button(
            "🗑 Delete all entries for this unit",
            use_container_width=True,
            key=f"del_{unit.unit_id}",
            help="Remove every saved entry for this unit.",
        ):
            delete_entries(annotation_key, unit.unit_id)
            state["editor"].pop(unit.unit_id, None)
            state.setdefault("last_nonce", {}).pop(unit.unit_id, None)
            st.rerun()
