"""Annotation-domain tools.

Tools that reach into the label catalogue or otherwise express annotation
intent. Generic telemetry tools (graph rendering, query dispatchers,
expert-phase detection, circuit-section locator) stay in
``agent/tools/`` because they are agent capabilities, not annotation
concerns.

``search_labels_handler`` is the Claude-side surface for the one hybrid
label retriever (``app.skills.internal.annotation.label_search.search_labels``) —
the agent discovers candidate labels by querying it, not from any
enumerated catalog.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _shape_for_llm(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a label doc to the fields the agent needs to pick it."""
    row: Dict[str, Any] = {
        "id": doc["id"],
        "name": doc.get("name", doc["id"]),
        "type": doc.get("type"),
        "score": round(float(doc.get("score", 0.0)), 4),
    }
    if doc.get("parent"):
        row["parent"] = doc["parent"]
    desc = (doc.get("description") or "").strip()
    if desc:
        row["description"] = desc
    ex_with = doc.get("exclusive_with") or []
    if ex_with:
        row["exclusive_with"] = list(ex_with)
    return row


def search_labels_handler(_surface, args: Dict[str, Any]) -> str:
    """Claude MCP handler — hybrid-search the label catalog.

    Params: ``query`` (required, plain-language telemetry description),
    optional ``types`` (a tier: ``"main"`` / ``"segment_type"`` /
    ``"sub"``) and ``parent_id`` (a main-label id to scope sub-labels).
    Returns the best-matching label docs, best-first.
    """
    from app.skills.internal.annotation.label_search import search

    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})
    filters: Dict[str, Any] = {}
    if str(args.get("types") or "").strip():
        filters["type"] = str(args["types"]).strip()
    if str(args.get("parent_id") or "").strip():
        filters["parent"] = str(args["parent_id"]).strip()

    results = search(query, filters=filters)
    return json.dumps([_shape_for_llm(d) for d in results], default=str)


CLAUDE_SEARCH_LABELS_TOOL: Dict[str, Any] = {
    "name": "search_labels",
    "description": (
        "Hybrid-search the annotation label catalog for the candidates "
        "matching your observations. `query` is a plain-language "
        "description of the telemetry you saw. Optional `types` scopes to "
        "one tier (\"main\", \"segment_type\", or \"sub\"); optional "
        "`parent_id` (a main-label id, e.g. \"MSP\") scopes to that "
        "label's sub-labels. Returns the best-matching labels with their "
        "descriptions, best-first. Re-query with different wording to "
        "broaden. This is the only way to discover labels — there is no "
        "full catalog listing. Circuit + circuit_section labels are not "
        "searchable here; pick them via the `get_circuit_id` / "
        "`locate_circuit_section` tools instead."
    ),
    "params_schema": {"query": str, "types": str, "parent_id": str},
    "handler": search_labels_handler,
}
