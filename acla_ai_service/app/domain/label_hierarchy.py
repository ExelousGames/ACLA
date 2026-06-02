"""Helpers for preserving parent/sub-label grouping."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.domain.labels import LABEL_CATEGORIES, LABEL_MAPPING


def _parent_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for parent_id, child_ids in LABEL_CATEGORIES.items():
        if parent_id in {"Main Labels", "Segment Type"}:
            continue
        if parent_id not in LABEL_MAPPING:
            continue
        for child_id in child_ids:
            if child_id in LABEL_MAPPING:
                lookup[child_id] = parent_id
    return lookup


def normalize_grouped_label_ids(
    raw_label_ids: Any,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Return valid label IDs with required parent labels included.

    Sub-labels are valid on lap segments, but they should never stand alone:
    adding the parent main/circuit label keeps downstream grouping intact.
    """
    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    added_parents: List[str] = []
    parents = _parent_lookup()

    if not isinstance(raw_label_ids, list):
        rejected.append({
            "value": raw_label_ids, "reason": "label_ids was not a list",
        })
        return cleaned, rejected, added_parents

    for i, raw_lid in enumerate(raw_label_ids):
        if not isinstance(raw_lid, str):
            rejected.append({
                "index": i, "value": raw_lid, "reason": "must be string",
            })
            continue
        if raw_lid not in LABEL_MAPPING:
            rejected.append({
                "index": i, "value": raw_lid,
                "reason": f"unknown label_id '{raw_lid}'",
            })
            continue

        parent_id = parents.get(raw_lid)
        if parent_id and parent_id not in cleaned:
            cleaned.append(parent_id)
            if parent_id not in raw_label_ids and parent_id not in added_parents:
                added_parents.append(parent_id)

        if raw_lid not in cleaned:
            cleaned.append(raw_lid)

    return cleaned, rejected, added_parents
