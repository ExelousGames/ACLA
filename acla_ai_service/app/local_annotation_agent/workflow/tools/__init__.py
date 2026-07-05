"""Annotation-domain helpers.

Generic telemetry tools (graph rendering, query dispatchers, expert-phase
detection, circuit-section locator) stay in ``agent/tools/`` because they are
agent capabilities, not annotation concerns.
"""

from __future__ import annotations

from typing import Any, Dict


def shape_label_doc_for_llm(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a label doc to the fields the agent needs to pick it."""
    row: Dict[str, Any] = {
        "id": doc["id"],
        "name": doc.get("name", doc["id"]),
        "type": doc.get("type"),
        "score": round(float(doc.get("score", 0.0)), 4),
    }
    for key, value in doc.items():
        if key in row or key in {"id", "name", "type", "description", "score"}:
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    if doc.get("parent"):
        row["parent"] = doc["parent"]
    desc = (doc.get("description") or "").strip()
    if desc:
        row["description"] = desc
    ex_with = doc.get("exclusive_with") or []
    if ex_with:
        row["exclusive_with"] = list(ex_with)
    return row

