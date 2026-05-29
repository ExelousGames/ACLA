"""Domain helper: enriched label docs for annotation / agent code.

Two sources, each owning its own classification — nothing is re-derived
in Python:

* Prose labels (``main`` / ``sub`` / ``segment_type``) live in
  ``sub_label_annotation.json``. They already declare ``type`` and
  ``parent`` alongside their prose, so we read them straight through the
  skill query engine — the same data the hybrid ``search`` retriever
  indexes.
* Circuit sections are deterministic geometry, owned by
  ``app.domain.circuit_sections``. We synthesize their docs from the
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

from app.domain.circuit_sections import CIRCUIT_SECTION_RANGES
from app.domain.labels import LABEL_MAPPING
from app.skills.internal.annotation import skills
from app.skills.internal.annotation._query import matches


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


def _all_docs() -> List[Dict[str, Any]]:
    return skills.iter("sub_label_annotation.labels") + _circuit_section_docs()


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
