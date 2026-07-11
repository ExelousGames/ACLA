"""Domain helper: enriched label docs for annotation / agent code.

Two sources, each owning its own classification — nothing is re-derived
in Python:

* Main-label taxonomy lives in ``lap_annotation.json``.
  Sub-label and segment-type taxonomy stays in
  ``sub_label_annotation.json``.
* Circuit sections are deterministic geometry, owned by
  ``app.shared.circuit_sections``. We synthesize their docs from the
  section ranges (``type="circuit_section"``, ``parent=<circuit>``,
  ``normalized_position_range``), naming them from ``LABEL_MAPPING``.

Two verbs, mirroring the skill registry:

    get_label(label_id) -> Dict[str, Any] | None
    find_labels(**filters)     -> List[Dict[str, Any]]

Filter syntax is the same Mongo-style vocabulary as ``skills.find`` —
plain values for equality, ``{"$in": [...]}`` etc. for operators.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.shared.circuit_sections import CIRCUIT_SECTION_RANGES
from app.shared.labels import LABEL_MAPPING
from app.internal_knowledge_base import skills
from app.internal_knowledge_base._query import matches


def _circuit_section_docs() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for sid, rng in CIRCUIT_SECTION_RANGES.items():
        docs.append({
            "id": sid,
            "name": LABEL_MAPPING.get(sid, sid),
            "type": "circuit_section",
            "parent": sid.rstrip("0123456789"),
            "normalized_position_range": (
                (float(rng[0]), float(rng[1])) if rng is not None else None
            ),
        })
    return docs


def _label_docs() -> List[Dict[str, Any]]:
    lap_requirements = skills.get("lap_annotation.selection_requirements", {})
    sub_requirements = skills.get("sub_label_annotation.selection_requirements", {})
    docs: List[Dict[str, Any]] = []

    for doc in skills.iter("lap_annotation.labels"):
        next_doc = dict(doc)
        label_id = str(next_doc.get("id") or "")
        requirements = (
            lap_requirements.get(label_id)
            if isinstance(lap_requirements, dict)
            else None
        )
        if isinstance(requirements, dict):
            next_doc["selection_requirements"] = dict(requirements)
        docs.append(next_doc)

    for doc in skills.iter("sub_label_annotation.labels"):
        next_doc = dict(doc)
        label_id = str(next_doc.get("id") or "")
        requirements = (
            sub_requirements.get(label_id)
            if isinstance(sub_requirements, dict)
            else None
        )
        if isinstance(requirements, dict):
            next_doc["selection_requirements"] = dict(requirements)
        docs.append(next_doc)
    return docs


@lru_cache(maxsize=1)
def _label_index() -> Dict[str, Dict[str, Any]]:
    return {
        str(doc["id"]): doc
        for doc in [*_label_docs(), *_circuit_section_docs()]
    }


def _all_docs() -> List[Dict[str, Any]]:
    return [dict(doc) for doc in _label_index().values()]


def get_label(label_id: str) -> Optional[Dict[str, Any]]:
    doc = _label_index().get(label_id)
    return dict(doc) if doc is not None else None


def find_labels(**filters: Any) -> List[Dict[str, Any]]:
    docs = _all_docs()
    if not filters:
        return docs
    return [d for d in docs if matches(d, filters)]
