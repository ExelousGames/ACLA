"""Domain helper: merge sub_label_catalog yaml enrichment with the canonical
LABEL_MAPPING + LABEL_CATEGORIES from segment_models.

The skill folder under ``app/skills/sub_label_catalog/`` is pure yaml — it
carries the prose enrichment per label (description, annotation_guideline,
exclusive_with, normalized_position_range, ...). The classification
(``type``, ``parent``) is owned by ``segment_models.LABEL_CATEGORIES``.
This module composes the two so the annotation / agent code keeps the
same query shape it had when ``sub_label_catalog/data.py`` did the merge.

Two verbs, mirroring the skill registry:

    get_label(label_id) -> Dict[str, Any] | None
    find_labels(**filters)     -> List[Dict[str, Any]]

Filter syntax is the same Mongo-style vocabulary as ``skills.find`` —
plain values for equality, ``{"$in": [...]}`` etc. for operators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.models.segment_models import LABEL_CATEGORIES, LABEL_MAPPING
from app.skill_manager import skills
from app.skill_manager._query import matches

_CIRCUIT_PARENTS = {"brands_hatch", "silverstone"}
_MAIN_KEY = "Main Labels"
_SEGMENT_TYPE_KEY = "Segment Type"
_CANONICAL_FIELDS = {"id", "name", "type", "parent", "category"}


def _parse_position_range(raw: Any) -> Optional[Tuple[float, float]]:
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start, end = raw[0], raw[1]
    if start is None or end is None:
        return None
    try:
        return (float(start), float(end))
    except (TypeError, ValueError):
        return None


def _classify() -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}

    for lid in LABEL_CATEGORIES.get(_MAIN_KEY, []):
        out[lid] = {
            "type": "circuit" if lid in _CIRCUIT_PARENTS else "main",
            "parent": None,
            "category": _MAIN_KEY,
        }

    for lid in LABEL_CATEGORIES.get(_SEGMENT_TYPE_KEY, []):
        out[lid] = {
            "type": "segment_type",
            "parent": None,
            "category": _SEGMENT_TYPE_KEY,
        }

    for parent, children in LABEL_CATEGORIES.items():
        if parent in (_MAIN_KEY, _SEGMENT_TYPE_KEY):
            continue
        is_circuit = parent in _CIRCUIT_PARENTS
        for child_id in children:
            out[child_id] = {
                "type": "circuit_section" if is_circuit else "sub",
                "parent": parent,
                "category": parent,
            }
    return out


def _build_doc(lid: str, name: str, classified: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    info = classified.get(lid) or {"type": "unknown", "parent": None, "category": None}
    yaml_entry = skills.get(f"sub_label_catalog.labels.{lid}") or {}
    enrichment = {
        k: v for k, v in yaml_entry.items()
        if k not in _CANONICAL_FIELDS
    }
    if "normalized_position_range" in enrichment:
        enrichment["normalized_position_range"] = _parse_position_range(
            enrichment["normalized_position_range"]
        )
    return {
        "id": lid,
        "name": name,
        "type": info["type"],
        "parent": info["parent"],
        "category": info["category"],
        **enrichment,
    }


def get_label(label_id: str) -> Optional[Dict[str, Any]]:
    if label_id not in LABEL_MAPPING:
        return None
    return _build_doc(label_id, LABEL_MAPPING[label_id], _classify())


def find_labels(**filters: Any) -> List[Dict[str, Any]]:
    classified = _classify()
    docs = [_build_doc(lid, name, classified) for lid, name in LABEL_MAPPING.items()]
    if not filters:
        return docs
    return [d for d in docs if matches(d, filters)]
