"""Backend for the Streamlit LLM-pipeline tab.

A **training unit** is either a parent segment with its children, or a
lone top-level segment. Units are derived live from the segment store
under a given ``annotation_key``; the per-unit critique/guide drafts
the human reviewer edits are persisted in a sidecar cache key so they
survive Streamlit reruns and stay alongside the segments they describe.

Layout::

    <annotation_key>                 → segments (one chunk per session)
    <annotation_key>_llm_drafts      → drafts   (one chunk per unit_id)

The drafts cache key falls back to ``BlobStrategy`` automatically
(non-typed prefix), so each draft is stored as a small JSON blob.

Public surface (used by ``ui/segment_tabs/llm_pipeline.py``):

  * :func:`collect_training_units` — list every unit under an annotation key
  * :func:`load_draft` / :func:`save_draft` / :func:`delete_draft`
  * :func:`unit_states` — per-unit status (missing / draft / approved)
  * :func:`draft_unit` — call claude_sdk twice (critique + guide)
  * :func:`export_annotation_jsonl` — emit the dataset_builder input format
  * :func:`build_chat_dataset` — fan units into the trainer's chat JSONL
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.domain.labels import LABEL_MAPPING
from app.domain.segment import AnnotatedSegment
from app.pipelines.training.dataset_builder import (
    MODES,
    build_dataset,
    render_labels_text,
)
from app.skills.internal.annotation import skills


# ---------------------------------------------------------------------------
# Training units
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingUnit:
    """One unit as derived from the segment store."""
    unit_id: str
    kind: str  # "parent_with_children" | "isolated"
    parent_label_ids: List[str]
    children_label_ids: List[List[str]]
    session_id: str
    start_index: Optional[int]
    end_index: Optional[int]

    def labels_text(self) -> str:
        return render_labels_text(self.parent_label_ids, self.children_label_ids)

    def parent_names(self) -> str:
        return ", ".join(LABEL_MAPPING.get(l, l) for l in self.parent_label_ids) or "(none)"

    def children_names(self) -> str:
        if not self.children_label_ids:
            return ""
        return "; ".join(
            f"[{i + 1}] " + (", ".join(LABEL_MAPPING.get(l, l) for l in c) or "(none)")
            for i, c in enumerate(self.children_label_ids)
        )


def drafts_cache_key(annotation_key: str) -> str:
    return f"{annotation_key}_llm_drafts"


def collect_training_units(annotation_key: str) -> List[TrainingUnit]:
    """Walk every session under ``annotation_key`` and build the units."""
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    if not store.has_cached_data(annotation_key):
        raise RuntimeError(
            f"annotation_key {annotation_key!r} not found in segment store"
        )

    units: List[TrainingUnit] = []
    for session_id in store.list_chunk_ids(annotation_key):
        chunk = store.get_chunk(annotation_key, session_id)
        raw = chunk if isinstance(chunk, list) else (chunk or {}).get("data", [])
        segments = [AnnotatedSegment.from_dict(d) for d in raw]
        kids_by_parent: Dict[str, List[AnnotatedSegment]] = {}
        roots: List[AnnotatedSegment] = []
        for s in segments:
            if not s.id:
                continue
            if s.parent_id:
                kids_by_parent.setdefault(s.parent_id, []).append(s)
            else:
                roots.append(s)
        for root in roots:
            kids = sorted(
                kids_by_parent.get(root.id, []),
                key=lambda c: c.start_index if c.start_index is not None else 0,
            )
            parent_labels = [str(l) for l in (root.labels or [])]
            children_labels = [[str(l) for l in (c.labels or [])] for c in kids]
            if not parent_labels and not any(children_labels):
                continue
            units.append(TrainingUnit(
                unit_id=root.id,
                kind="parent_with_children" if kids else "isolated",
                parent_label_ids=parent_labels,
                children_label_ids=children_labels,
                session_id=session_id,
                start_index=root.start_index,
                end_index=root.end_index,
            ))
    return units


# ---------------------------------------------------------------------------
# Draft persistence (sidecar cache key, one chunk per unit_id)
# ---------------------------------------------------------------------------

@dataclass
class Draft:
    completion_critique: str = ""
    completion_guide: str = ""
    approved: bool = False
    model_version: str = ""
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Draft":
        return cls(
            completion_critique=str(data.get("completion_critique", "")),
            completion_guide=str(data.get("completion_guide", "")),
            approved=bool(data.get("approved", False)),
            model_version=str(data.get("model_version", "")),
            updated_at=float(data.get("updated_at", time.time())),
        )

    def has_both(self) -> bool:
        return bool(self.completion_critique.strip()) and bool(self.completion_guide.strip())


def load_draft(annotation_key: str, unit_id: str) -> Optional[Draft]:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    key = drafts_cache_key(annotation_key)
    if not store.has_cached_data(key):
        return None
    payload = store.get_chunk(key, unit_id)
    if not payload:
        return None
    return Draft.from_dict(payload)


def save_draft(annotation_key: str, unit_id: str, draft: Draft) -> None:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    draft.updated_at = time.time()
    store.save_chunk(drafts_cache_key(annotation_key), unit_id, draft.to_dict())


def delete_draft(annotation_key: str, unit_id: str) -> bool:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    return store.delete_chunk(drafts_cache_key(annotation_key), unit_id)


def all_drafts(annotation_key: str) -> Dict[str, Draft]:
    """Read every saved draft for the annotation key (keyed by unit_id)."""
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    key = drafts_cache_key(annotation_key)
    if not store.has_cached_data(key):
        return {}
    out: Dict[str, Draft] = {}
    for payload, unit_id in store.get_cached_data_chunks(key, include_ids=True):
        if isinstance(payload, dict):
            out[unit_id] = Draft.from_dict(payload)
    return out


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def unit_states(annotation_key: str, units: List[TrainingUnit]) -> Dict[str, str]:
    """Return per-unit status: 'missing', 'draft', or 'approved'."""
    drafts = all_drafts(annotation_key)
    out: Dict[str, str] = {}
    for u in units:
        d = drafts.get(u.unit_id)
        if d is None or not d.has_both():
            out[u.unit_id] = "missing"
        elif d.approved:
            out[u.unit_id] = "approved"
        else:
            out[u.unit_id] = "draft"
    return out


# ---------------------------------------------------------------------------
# Claude drafting
# ---------------------------------------------------------------------------

def _claude_prompt(mode: str, labels_text: str) -> str:
    template = skills.get("training_unit_annotation.claude_prompt_template")
    if not template:
        raise RuntimeError(
            "training_unit_annotation skill is missing — drop "
            "app/skills/internal/annotation/training_unit_annotation.json in place and restart."
        )
    mode_style = skills.get(f"training_unit_annotation.modes.{mode}.style") or ""
    mode_avoid_raw = skills.get(f"training_unit_annotation.modes.{mode}.avoid") or []
    mode_avoid = (
        "\n".join(f"  - {item}" for item in mode_avoid_raw)
        if isinstance(mode_avoid_raw, list)
        else str(mode_avoid_raw)
    )
    return (
        template
        .replace("{mode}", mode)
        .replace("{labels_text}", labels_text)
        .replace("{mode_style}", mode_style.strip())
        .replace("{mode_avoid}", mode_avoid)
    )


def draft_unit(
    unit: TrainingUnit,
    model: str = "claude-sonnet-4-6",
    use_thinking: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Draft:
    """Call claude_sdk twice (critique + guide) and return a Draft.

    Raises if either call returns empty text — VLM no-fallback policy.
    """
    from app.agents.backends.claude_sdk import (
        CLAUDE_VLM_MODELS,
        get_or_start_claude_backend,
    )
    if model not in CLAUDE_VLM_MODELS:
        raise RuntimeError(
            f"unknown claude model {model!r}; pick one of {list(CLAUDE_VLM_MODELS)}"
        )
    claude = get_or_start_claude_backend(model=model, use_thinking=use_thinking)
    labels_text = unit.labels_text()

    completions: Dict[str, str] = {}
    for mode in MODES:
        if on_progress is not None:
            on_progress(f"drafting {mode}…")
        text = claude.generate(prompt=_claude_prompt(mode, labels_text)).strip()
        if not text:
            raise RuntimeError(f"claude_sdk returned empty text for mode={mode!r}")
        completions[mode] = text

    return Draft(
        completion_critique=completions["critique"],
        completion_guide=completions["guide"],
        approved=False,
        model_version=f"claude-sdk:{model}" + (":thinking" if use_thinking else ""),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_annotation_jsonl(
    annotation_key: str,
    output_path: Path,
    only_approved: bool = True,
) -> int:
    """Write the dataset_builder annotation JSONL. Returns the row count."""
    units = collect_training_units(annotation_key)
    drafts = all_drafts(annotation_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for unit in units:
            draft = drafts.get(unit.unit_id)
            if draft is None or not draft.has_both():
                continue
            if only_approved and not draft.approved:
                continue
            record = {
                "unit_id": unit.unit_id,
                "kind": unit.kind,
                "parent_label_ids": unit.parent_label_ids,
                "children_label_ids": unit.children_label_ids,
                "completion_critique": draft.completion_critique.strip(),
                "completion_guide": draft.completion_guide.strip(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def build_chat_dataset(annotation_jsonl: Path, chat_jsonl: Path) -> int:
    """Fan the exported annotation JSONL into the trainer's chat JSONL."""
    return build_dataset(annotation_jsonl, chat_jsonl)


# ---------------------------------------------------------------------------
# Templated multi-turn entries (Foundry-style string-concatenation UI)
# ---------------------------------------------------------------------------
#
# A ``TrainingEntry`` is one full chat conversation authored by the human
# for a single training unit. Each ``Turn`` is one message (system / user /
# assistant) whose text is a template with ``{var}`` placeholders that get
# resolved against the unit's labels at export time. Labels are embedded
# directly into the message text — there is no implicit system message.

VALID_ROLES = ("system", "user", "assistant")


@dataclass
class Turn:
    role: str
    template: str

    def to_dict(self) -> dict:
        return {"role": self.role, "template": self.template}

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        role = str(data.get("role", "user"))
        if role not in VALID_ROLES:
            role = "user"
        return cls(role=role, template=str(data.get("template", "")))


@dataclass
class TrainingEntry:
    entry_id: str
    turns: List[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"entry_id": self.entry_id, "turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingEntry":
        return cls(
            entry_id=str(data.get("entry_id") or uuid.uuid4().hex[:8]),
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
        )

    @staticmethod
    def new() -> "TrainingEntry":
        return TrainingEntry(entry_id=uuid.uuid4().hex[:8], turns=[])


@dataclass
class UnitEntries:
    entries: List[TrainingEntry] = field(default_factory=list)
    approved: bool = False
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "approved": self.approved,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnitEntries":
        return cls(
            entries=[TrainingEntry.from_dict(e) for e in data.get("entries", [])],
            approved=bool(data.get("approved", False)),
            updated_at=float(data.get("updated_at", time.time())),
        )

    def is_empty(self) -> bool:
        return not any(t.template.strip() for e in self.entries for t in e.turns)


def entries_cache_key(annotation_key: str) -> str:
    return f"{annotation_key}_llm_entries"


def load_entries(annotation_key: str, unit_id: str) -> Optional[UnitEntries]:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    key = entries_cache_key(annotation_key)
    if not store.has_cached_data(key):
        return None
    payload = store.get_chunk(key, unit_id)
    if not payload:
        return None
    return UnitEntries.from_dict(payload)


def save_entries(annotation_key: str, unit_id: str, entries: UnitEntries) -> None:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    entries.updated_at = time.time()
    store.save_chunk(entries_cache_key(annotation_key), unit_id, entries.to_dict())


def delete_entries(annotation_key: str, unit_id: str) -> bool:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    return store.delete_chunk(entries_cache_key(annotation_key), unit_id)


def all_entries(annotation_key: str) -> Dict[str, UnitEntries]:
    from app.storage import get_shared_telemetry_store

    store = get_shared_telemetry_store()
    key = entries_cache_key(annotation_key)
    if not store.has_cached_data(key):
        return {}
    out: Dict[str, UnitEntries] = {}
    for payload, unit_id in store.get_cached_data_chunks(key, include_ids=True):
        if isinstance(payload, dict):
            out[unit_id] = UnitEntries.from_dict(payload)
    return out


def template_variables(unit: TrainingUnit) -> Dict[str, str]:
    """Resolved variables available inside an entry's templates.

    Keys are the placeholder names (without braces); values are the
    substituted text. Used both by the resolver and by the UI palette.
    """
    out: Dict[str, str] = {
        "parent": unit.parent_names(),
        "children": unit.children_names() or "(none)",
        "labels": unit.labels_text(),
        "session": unit.session_id,
        "unit_id": unit.unit_id,
    }
    for i, lid in enumerate(unit.parent_label_ids, 1):
        out[f"parent_{i}"] = LABEL_MAPPING.get(lid, lid)
    for i, cids in enumerate(unit.children_label_ids, 1):
        out[f"child_{i}"] = ", ".join(LABEL_MAPPING.get(l, l) for l in cids) or "(none)"
    return out


def resolve_template(template: str, unit: TrainingUnit) -> str:
    vars_ = template_variables(unit)
    out = template
    for k, v in vars_.items():
        out = out.replace("{" + k + "}", v)
    return out


def export_entries_chat_jsonl(
    annotation_key: str,
    output_path: Path,
    only_approved: bool = True,
) -> int:
    """Write one chat row per training entry, resolving all templates.

    Each entry becomes a single ``{"messages": [...]}`` line. Empty
    templates and entries with no non-empty turns are skipped. There is
    no implicit system message — turns are written verbatim in order.
    Returns the number of chat rows written.
    """
    units = {u.unit_id: u for u in collect_training_units(annotation_key)}
    stored = all_entries(annotation_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for unit_id, ue in stored.items():
            if only_approved and not ue.approved:
                continue
            unit = units.get(unit_id)
            if unit is None:
                continue
            for entry in ue.entries:
                messages = []
                for turn in entry.turns:
                    content = resolve_template(turn.template, unit).strip()
                    if not content:
                        continue
                    messages.append({"role": turn.role, "content": content})
                if not messages:
                    continue
                fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
    return written
