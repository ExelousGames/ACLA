"""Data layer for sub_label_catalog.

LABEL_CATEGORIES + LABEL_MAPPING (from segment_models.py) are the source
of truth for which labels exist and how they're grouped. The YAML
enriches each label with prose: description, annotation_guideline,
exclusive_with, normalized_position_range, etc. YAML cannot override
the canonical fields (``id``, ``name``, ``type``, ``parent``) — those
come from segment_models.

Collections exposed via the query API:

  labels                  — id → merged document
  category_guidelines     — category name → guideline string (from YAML)

Each document in ``labels`` carries:
  id, name, type, parent, category, description, annotation_guideline,
  exclusive_with, normalized_position_range, plus any extra YAML fields.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.models.segment_models import LABEL_CATEGORIES, LABEL_MAPPING


def _parse_position_range(raw: Any) -> Optional[Tuple[float, float]]:
    """Normalise ``[start, end]`` to a (float, float) tuple; ``None`` if unset/invalid."""
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start, end = raw[0], raw[1]
    if start is None or end is None:
        return None
    try:
        return (float(start), float(end))
    except (TypeError, ValueError):
        return None

_CIRCUIT_PARENTS = {"brands_hatch", "silverstone"}
_MAIN_KEY = "Main Labels"
_SEGMENT_TYPE_KEY = "Segment Type"

_CANONICAL_FIELDS = {"id", "name", "type", "parent", "category"}


def _classify() -> Dict[str, Dict[str, Optional[str]]]:
    """Return ``{label_id: {type, parent, category}}`` from LABEL_CATEGORIES."""
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


def labels(raw_body: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Source of truth: LABEL_CATEGORIES + LABEL_MAPPING; enrichment: YAML."""
    yaml_labels: Dict[str, Any] = raw_body.get("labels") or {}
    classified = _classify()

    out: Dict[str, Dict[str, Any]] = {}
    for lid, name in LABEL_MAPPING.items():
        info = classified.get(lid)
        if info is None:
            # Label exists in LABEL_MAPPING but isn't in any LABEL_CATEGORIES
            # bucket — uncategorised, still listed.
            info = {"type": "unknown", "parent": None, "category": None}

        yaml_entry = yaml_labels.get(lid) or {}
        # YAML enrichment, but cannot override canonical fields
        enrichment = {
            k: v for k, v in yaml_entry.items()
            if k not in _CANONICAL_FIELDS
        }
        # Normalise position range to (float, float) tuple or None
        if "normalized_position_range" in enrichment:
            enrichment["normalized_position_range"] = _parse_position_range(
                enrichment["normalized_position_range"]
            )

        out[lid] = {
            "id": lid,
            "name": name,
            "type": info["type"],
            "parent": info["parent"],
            "category": info["category"],
            **enrichment,
        }
    return out


def category_guidelines(raw_body: Dict[str, Any]) -> Dict[str, str]:
    raw = raw_body.get("category_guidelines") or {}
    return {str(k): str(v).strip() for k, v in raw.items()}


COLLECTIONS = {
    "labels": labels,
    "category_guidelines": category_guidelines,
}
