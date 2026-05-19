"""Annotation-domain tools.

Tools that reach into the label catalogue or otherwise express annotation
intent. Generic telemetry tools (graph rendering, query dispatchers,
expert-phase detection, circuit-section locator) stay in
``agent/tools/`` because they are agent capabilities, not annotation
concerns.

Importing this module is enough; ``list_eligible_labels`` is consumed
directly by the annotation flows (detailed/lap) — there is no pipeline-
tool registration because no sub-agent invokes label listing as a
planned step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def list_eligible_labels(
    parent_id: Optional[str] = None,
    circuit_id: Optional[str] = None,
):
    """Return the eligible label IDs the annotation agent may attach.

    Two-mode tool:

    - ``parent_id=None`` (default) — returns the top tiers: circuit
      (gated by ``circuit_id`` when supplied), circuit_sections belonging
      to that circuit, segment types (ST1-ST6), and the main labels
      (EA / MS / RM / PS / OV / MD).
    - ``parent_id="MS"`` (or any main label) — returns the sub-labels
      whose ``parent`` matches, each with its ``description`` so the
      agent can decide whether the section's telemetry matches the
      sub-label's signature.

    Returned attachment ``eligible_labels`` has shape::

        {
            "parent_id": <str | None>,
            "circuit_id": <str | None>,
            "groups": [
                {"tier": "circuit" | "circuit_section" | "segment_type"
                         | "main" | "sub",
                 "entries": [{"id", "name", "description?", "range?",
                              "parent?", "exclusive_with?"}, ...]},
                ...
            ],
        }
    """
    from app.services.llm.agent.evaluators import PipelineAttachment
    from app.services.llm.label_catalog import find_labels, get_label

    def _attach(groups: List[Dict[str, Any]]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="eligible_labels",
            kind="structured",
            label="Eligible Labels",
            content={
                "parent_id": parent_id,
                "circuit_id": circuit_id,
                "groups": groups,
            },
        )

    if parent_id:
        sub_entries: List[Dict[str, Any]] = []
        for entry in find_labels(parent=parent_id):
            row: Dict[str, Any] = {
                "id": entry["id"],
                "name": entry["name"],
                "parent": entry.get("parent"),
            }
            desc = (entry.get("description") or "").strip()
            if desc:
                row["description"] = desc
            ex_with = entry.get("exclusive_with") or []
            if ex_with:
                row["exclusive_with"] = list(ex_with)
            sub_entries.append(row)
        return _attach([
            {"tier": "sub", "entries": sub_entries},
        ])

    groups: List[Dict[str, Any]] = []

    if circuit_id:
        circuit_entry = get_label(circuit_id)
        circuit_rows: List[Dict[str, Any]] = []
        if circuit_entry is not None:
            circuit_rows.append({
                "id": circuit_entry["id"],
                "name": circuit_entry["name"],
            })
        groups.append({"tier": "circuit", "entries": circuit_rows})

        section_rows: List[Dict[str, Any]] = []
        for entry in find_labels(type="circuit_section", parent=circuit_id):
            row: Dict[str, Any] = {"id": entry["id"], "name": entry["name"]}
            rng = entry.get("normalized_position_range")
            if rng is not None:
                row["normalized_position_range"] = [
                    float(rng[0]), float(rng[1]),
                ]
            section_rows.append(row)
        groups.append({"tier": "circuit_section", "entries": section_rows})

    st_rows: List[Dict[str, Any]] = []
    for entry in find_labels(type="segment_type"):
        row: Dict[str, Any] = {"id": entry["id"], "name": entry["name"]}
        desc = (entry.get("description") or "").strip()
        if desc:
            row["description"] = desc
        st_rows.append(row)
    groups.append({"tier": "segment_type", "entries": st_rows})

    main_rows: List[Dict[str, Any]] = []
    for entry in find_labels(type="main"):
        row = {"id": entry["id"], "name": entry["name"]}
        ex_with = entry.get("exclusive_with") or []
        if ex_with:
            row["exclusive_with"] = list(ex_with)
        main_rows.append(row)
    groups.append({"tier": "main", "entries": main_rows})

    return _attach(groups)
