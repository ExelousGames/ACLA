"""Annotation-domain tools.

Tools that reach into the label catalogue or otherwise express annotation
intent. Generic telemetry tools (graph rendering, query dispatchers,
expert-phase detection, circuit-section locator) stay in
``agent/tools/`` because they are agent capabilities, not annotation
concerns.

``search_labels_handler`` is the tool-agent surface for the one hybrid
label retriever (``app.internal_knowledge_base.label_search.search_labels``) —
the agent discovers candidate labels by querying it, not from any
enumerated catalog.
"""

from __future__ import annotations

import json
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


def search_labels_handler(_surface, args: Dict[str, Any]) -> str:
    """Hybrid-search the label catalog.

    Params: ``query`` (required, plain-language telemetry description),
    optional ``types`` (a tier: ``"main"`` / ``"segment_type"`` /
    ``"sub"``) and ``parent_id`` (a main-label id to scope sub-labels).
    Returns the best-matching label docs, best-first.
    """
    from app.internal_knowledge_base.label_search import search

    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})
    filters: Dict[str, Any] = {}
    if str(args.get("types") or "").strip():
        filters["type"] = str(args["types"]).strip()
    if str(args.get("parent_id") or "").strip():
        filters["parent"] = str(args["parent_id"]).strip()

    request = getattr(_surface, "request", None)
    preflight_tags = _preflight_tags_from_request(request)
    semantic_query = " ".join([query, *preflight_tags]).strip()

    results = search(semantic_query, top_k=24, filters=filters)
    results = [
        doc for doc in results
        if _label_doc_allowed_for_surface(_surface, doc)
    ][:8]
    return json.dumps({
        "query": query,
        "semantic_query_tags": preflight_tags,
        "filters": filters,
        "candidates": [shape_label_doc_for_llm(d) for d in results],
    }, default=str)


def _preflight_tags_from_request(request: Any) -> list[str]:
    if request is None:
        return []
    event_terms: list[str] = []
    for attachment in getattr(request, "initial_attachments", []) or []:
        if getattr(attachment, "name", "") != "init.detailed_preflight_events":
            continue
        content = getattr(attachment, "content", None)
        if not isinstance(content, dict):
            continue
        events = content.get("events") or []
        if isinstance(events, list):
            event_terms.extend(
                str(event.get("event"))
                for event in events
                if isinstance(event, dict) and str(event.get("event") or "").strip()
            )
        event_text = str(content.get("event_text") or "").strip()
        if event_text:
            event_terms.extend(event_text.splitlines())
        if event_terms:
            return [term for term in event_terms if term.strip()][:80]
    for attachment in getattr(request, "initial_attachments", []) or []:
        if getattr(attachment, "name", "") != "init.annotation_preflight_context":
            continue
        content = getattr(attachment, "content", None)
        if not isinstance(content, dict):
            continue
        tags = content.get("tool_output_tags") or []
        return [str(tag) for tag in tags if str(tag).strip()][:80]
    return []


def _label_doc_allowed_for_surface(_surface, doc: Dict[str, Any]) -> bool:
    request = getattr(_surface, "request", None)
    if request is None:
        return True
    eligible = set(request.extra_state.get("eligible_behavior_label_ids") or [])
    if not eligible:
        return True

    required_parents = {"O", "OD", "PS", "RM", "MSP", "MSR"}
    label_id = str(doc.get("id") or "")
    parent_id = str(doc.get("parent") or "")
    if label_id in required_parents:
        return label_id in eligible
    if parent_id in required_parents:
        return parent_id in eligible
    return True


SEARCH_LABELS_TOOL: Dict[str, Any] = {
    "name": "search_labels",
    "description": (
        "Hybrid-search the annotation label catalog for the candidates "
        "matching your observations. `query` is a plain-language "
        "description of the telemetry you saw; include relevant "
        "`tool_output_tags` from the upfront preflight context when they "
        "help name deterministic outcomes. This tool also blends the "
        "current request's preflight tags into semantic retrieval. "
        "Optional `types` scopes to "
        "one tier (\"main\", \"segment_type\", or \"sub\"); optional "
        "`parent_id` (a main-label id, e.g. \"MSP\") scopes to that "
        "label's sub-labels. Returns the best-matching labels with their "
        "descriptions, best-first. A returned label is only a candidate; "
        "attach it only when its full definition fits the whole annotation "
        "range, not just a smaller slice. Re-query with different wording to "
        "broaden. This is the only way to discover labels — there is no "
        "full catalog listing. Circuit + circuit_section labels are not "
        "searchable here; pick them from splitter/preflight context or "
        "with `locate_circuit_section`."
    ),
    "params_schema": {"query": str, "types": str, "parent_id": str},
    "handler": search_labels_handler,
}
