"""Renderer for graph_analysis.

Renders the per-graph description block for the describe_graphs VLM
prompt. Walks each graph's YAML record with a uniform formatter,
appends cross-graph guidelines whose triggers match the graph
combination, and (when trajectory graphs are present) appends the
trajectory shape vocabulary.

Called via ``skills.render("graph_analysis", graph_ids=[...])``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

_GUIDELINE_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "brake_and_speed":            {"required": {"brake", "speed"},                     "any_of": []},
    "throttle_and_speed":         {"required": {"throttle", "speed"},                  "any_of": []},
    "time_delta_and_features":    {"required": {"time_delta"},                         "any_of": [
        {"brake", "throttle", "speed", "speed_delta", "push_limit", "trajectory_balance"},
    ]},
    "trajectory_and_features":    {"required": set(),                                  "any_of": [
        {"trajectory_detailed", "trajectory_gas_brake", "trajectory_offset"},
        {"throttle", "brake", "speed", "speed_delta", "push_limit", "trajectory_balance"},
    ]},
    "balance_and_push_limit":     {"required": {"trajectory_balance", "push_limit"},   "any_of": []},
    "brake_and_throttle_overlap": {"required": {"brake", "throttle"},                  "any_of": []},
}

_TRAJECTORY_IDS = {"trajectory_detailed", "trajectory_gas_brake", "trajectory_offset"}


def _render_section(key: str, value: Any, indent: str = "  ") -> List[str]:
    if not value:
        return []
    out: List[str] = [key.replace("_", " ") + ":"]

    if isinstance(value, str):
        for ln in value.rstrip("\n").split("\n"):
            out.append(f"{indent}{ln}" if ln else "")
    elif isinstance(value, list):
        for item in value:
            out.append(f"{indent}- {item}")
    elif isinstance(value, dict):
        for k, v in value.items():
            v_str = "" if v is None else str(v).rstrip("\n")
            v_lines = v_str.split("\n")
            first = v_lines[0]
            out.append(f"{indent}- {k}: {first}" if first else f"{indent}- {k}:")
            cont_indent = indent + "    "
            for cont in v_lines[1:]:
                out.append(f"{cont_indent}{cont}" if cont else "")
    else:
        out.append(f"{indent}{value}")

    return out


def render(skills, graph_ids: Sequence[str]) -> str:
    requested = list(graph_ids)
    entries = [
        skills.get(f"graph_analysis.graphs.{gid}")
        for gid in requested
    ]
    entries = [e for e in entries if e]
    if not entries:
        return ""

    lines: List[str] = [
        "#### Graph Description Skill — How to Describe These Graphs",
        "",
    ]

    for entry in entries:
        gid = entry.get("id")
        title = entry.get("title", gid)
        lines.append(f"##### {title} (id: {gid})")
        lines.append("")
        for key, value in entry.items():
            if key in ("title", "id"):
                continue
            section = _render_section(key, value)
            if section:
                lines.extend(section)
                lines.append("")

    graph_id_set = set(requested)
    relevant: List[str] = []
    for gid, spec in _GUIDELINE_TRIGGERS.items():
        if not spec["required"].issubset(graph_id_set):
            continue
        if not all(any_set & graph_id_set for any_set in spec["any_of"]):
            continue
        text = skills.get(f"graph_analysis.cross_graph_guidelines.{gid}", "")
        if text:
            relevant.append(f"[{gid}] {str(text).strip()}")

    if relevant:
        lines.append("#### Cross-Graph Description Guidelines")
        for g in relevant:
            lines.append(g)
        lines.append("")

    if graph_id_set & _TRAJECTORY_IDS:
        traj_vocab = skills.get("graph_analysis.trajectory_shape_vocabulary", "")
        if traj_vocab:
            lines.append("#### Trajectory Shape Vocabulary")
            lines.append(str(traj_vocab).strip())
            lines.append("")

    return "\n".join(lines)
