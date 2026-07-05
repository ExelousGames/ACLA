"""Domain helper: enriched label docs for annotation / agent code.

Two sources, each owning its own classification — nothing is re-derived
in Python:

* Prose label taxonomy (``main`` / ``sub`` / ``segment_type``) lives in
  ``sub_label_annotation.json``. Main-label descriptions are hydrated from
  ``lap_annotation.json``; sub-label and segment-type descriptions stay in
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


def _lap_main_descriptions() -> Dict[str, str]:
    labels = skills.get("lap_annotation.labels", {})
    if not isinstance(labels, dict):
        return {}
    return {
        str(label_id): str(doc.get("characteristics", "")).strip()
        for label_id, doc in labels.items()
        if isinstance(doc, dict) and str(doc.get("characteristics", "")).strip()
    }


def _label_docs() -> List[Dict[str, Any]]:
    lap_descriptions = _lap_main_descriptions()
    docs: List[Dict[str, Any]] = []
    for doc in skills.iter("sub_label_annotation.labels"):
        next_doc = dict(doc)
        if next_doc.get("type") == "main":
            description = lap_descriptions.get(str(next_doc.get("id") or ""))
            if description:
                next_doc["description"] = description
        docs.append(next_doc)
    return docs


def _all_docs() -> List[Dict[str, Any]]:
    return _label_docs() + _circuit_section_docs()


def get_label(label_id: str) -> Optional[Dict[str, Any]]:
    for doc in _all_docs():
        if doc.get("id") == label_id:
            return doc
    return None


def find_labels(**filters: Any) -> List[Dict[str, Any]]:
    docs = _all_docs()
    if not filters:
        return docs
    return [d for d in docs if matches(d, filters)]
