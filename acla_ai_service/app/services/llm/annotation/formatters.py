"""Annotation-side renderers for structured attachments.

Registered with the agent box's formatter registry at import time so the
default synth picker can render annotation-shaped attachments without
the box knowing the schema.
"""

from __future__ import annotations

from typing import Any

from app.services.llm.agent.evaluators import register_structured_formatter


def _format_parent_segment(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    parts: list[str] = []
    ps = content.get("parent_start")
    pe = content.get("parent_end")
    if ps is not None and pe is not None:
        parts.append(f"Range: [{ps}, {pe}] (length {pe - ps})")
    main_labels = content.get("main_labels") or []
    if main_labels:
        parts.append(f"Main labels: {', '.join(main_labels)}")
    children = content.get("existing_children") or []
    if children:
        child_lines = ["Existing children (avoid overlap):"]
        for c in children:
            cs = c.get("start_index")
            ce = c.get("end_index")
            cls = c.get("labels") or []
            child_lines.append(f"  - [{cs}, {ce}] labels={', '.join(cls)}")
        parts.append("\n".join(child_lines))
    return "\n".join(parts)


def _format_verified_labels(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    if not content:
        return "(no labels passed verification)"
    lines: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            lines.append(str(entry))
            continue
        lid = entry.get("label_id", "?")
        name = entry.get("name", "")
        sim = entry.get("similarity")
        desc = entry.get("description", "")
        sim_part = f" | sim={sim:.3f}" if isinstance(sim, (int, float)) else ""
        line = f"- {lid} | {name}{sim_part}"
        if desc:
            line = f"{line} — {desc}"
        lines.append(line)
    return "\n".join(lines)


def register_annotation_formatters() -> None:
    register_structured_formatter("parent_segment", _format_parent_segment)
    register_structured_formatter("verified_labels", _format_verified_labels)


register_annotation_formatters()
