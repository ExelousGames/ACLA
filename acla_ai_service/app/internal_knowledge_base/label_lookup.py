"""Domain helper: enriched label docs for annotation / agent code.

* Main-label taxonomy lives in ``lap_annotation.json``.
* Circuit sections are backend map metadata, not classifier labels.

Two verbs, mirroring the skill registry:

    get_label(label_id) -> Dict[str, Any] | None
    find_labels(**filters)     -> List[Dict[str, Any]]

Filter syntax is the same Mongo-style vocabulary as ``skills.find`` —
plain values for equality, ``{"$in": [...]}`` etc. for operators.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.internal_knowledge_base import skills
from app.internal_knowledge_base._query import matches


def _label_docs() -> List[Dict[str, Any]]:
    lap_requirements = skills.get("lap_annotation.selection_requirements", {})
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
    return docs


@lru_cache(maxsize=1)
def _label_index() -> Dict[str, Dict[str, Any]]:
    return {
        str(doc["id"]): doc
        for doc in _label_docs()
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
