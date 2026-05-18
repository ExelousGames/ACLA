"""Renderer for lap_annotation.

Renders the per-label characteristics block for the lap-flow planner /
synthesizer prompt. When ``circuit_id`` is given, the other circuit's
parent label is excluded.

Called via ``skills.render("lap_annotation", circuit_id=...)``.
"""

from __future__ import annotations

from typing import Any, List, Optional

_CIRCUIT_LABELS = {"brands_hatch", "silverstone"}


def render(skills, circuit_id: Optional[str] = None) -> str:
    labels = skills.iter("lap_annotation.labels")
    global_rules = skills.get("lap_annotation.global_rules", "")

    lines: List[str] = [
        "#### Lap Annotation Skill — Candidate Label Characteristics",
        "",
        "Each candidate parent label below lists the telemetry pattern "
        "that justifies attaching it. The `global_rules` block at the "
        "end is the per-section detection procedure.",
        "",
    ]

    for entry in labels:
        lid = entry["id"]
        if lid in _CIRCUIT_LABELS and circuit_id is not None and lid != circuit_id:
            continue
        name = entry.get("name", lid)
        applies_when = str(entry.get("applies_when", "")).strip()
        characteristics = str(entry.get("characteristics", "")).strip()

        lines.append(f"##### `{lid}` — {name}")
        if applies_when:
            lines.append(f"_Applies when:_ {applies_when}")
        if characteristics:
            lines.append(characteristics)
        lines.append("")

    if global_rules:
        lines.append("##### Global rules — how to find each label")
        for ln in str(global_rules).rstrip("\n").split("\n"):
            lines.append(ln)
        lines.append("")

    return "\n".join(lines)
