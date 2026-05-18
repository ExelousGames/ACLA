"""Data layer for lap_annotation.

YAML's ``labels`` block is the source. Each entry becomes a document
with ``id`` injected from the key.
"""

from __future__ import annotations

from typing import Any, Dict


def labels(raw_body: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    yaml_labels: Dict[str, Any] = raw_body.get("labels") or {}
    return {
        str(lid): {"id": str(lid), **(entry or {})}
        for lid, entry in yaml_labels.items()
    }


COLLECTIONS = {
    "labels": labels,
}
