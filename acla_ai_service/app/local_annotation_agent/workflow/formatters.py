"""Annotation-side renderers for structured attachments.

Registered with the agent box's formatter registry at import time so the
default synth picker can render annotation-shaped attachments without
the box knowing the schema.
"""

from __future__ import annotations

from typing import Any

from app.local_annotation_agent.evaluators import register_structured_formatter


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
        names = content.get("main_label_names") or []
        if names and len(names) == len(main_labels):
            joined = ", ".join(
                f"{label_id} ({name})"
                for label_id, name in zip(main_labels, names)
            )
            parts.append(f"Main labels: {joined}")
        else:
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


def _format_preflight_tool(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    lines: list[str] = []
    tool_id = content.get("tool_id")
    if tool_id:
        lines.append(f"Tool: {tool_id}")
    range_ = content.get("range")
    if range_:
        lines.append(f"Range: {range_}")
    tags = content.get("tags") or []
    if tags:
        lines.append("Tags: " + ", ".join(str(tag) for tag in tags[:30]))
    result = content.get("result")
    if result is not None:
        import json

        lines.append("Result:")
        lines.append(json.dumps(result, indent=2, sort_keys=True, default=str)[:3000])
    return "\n".join(lines)


def _format_preflight_labels(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    tags = content.get("tool_output_tags") or []
    candidates = content.get("candidates") or []
    lines: list[str] = []
    if tags:
        lines.append("Tool output tags: " + ", ".join(str(tag) for tag in tags[:40]))
    if not candidates:
        lines.append("Candidates: (none)")
        return "\n".join(lines)
    lines.append("Candidates:")
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        desc = str(entry.get("description") or "").strip()
        if len(desc) > 240:
            desc = desc[:237] + "..."
        lines.append(
            f"- {entry.get('id')} | {entry.get('name', '')} "
            f"| type={entry.get('type')} | score={entry.get('score')}"
            + (f" — {desc}" if desc else "")
        )
    return "\n".join(lines)


def _format_preflight_context(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    lines = [
        f"Flow: {content.get('flow')}",
        f"Range: {content.get('range')}",
        "Required tools: " + ", ".join(content.get("required_tools") or []),
    ]
    tags = content.get("tool_output_tags") or []
    if tags:
        lines.append("Tool output tags: " + ", ".join(str(tag) for tag in tags[:40]))
    candidate_ids = content.get("label_candidate_ids") or []
    if candidate_ids:
        lines.append("Semantic candidate IDs: " + ", ".join(str(x) for x in candidate_ids))
    return "\n".join(lines)


def register_annotation_formatters() -> None:
    register_structured_formatter("parent_segment", _format_parent_segment)
    register_structured_formatter("verified_labels", _format_verified_labels)
    register_structured_formatter("annotation_preflight_tool", _format_preflight_tool)
    register_structured_formatter("annotation_preflight_labels", _format_preflight_labels)
    register_structured_formatter("annotation_preflight_context", _format_preflight_context)


register_annotation_formatters()
