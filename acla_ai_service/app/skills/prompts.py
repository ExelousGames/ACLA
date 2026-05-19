"""Consumer-side formatters that turn skill yaml data into prompt fragments.

Skill folders under ``app/skills/`` are pure data — they expose yaml via
the orchestrator's ``get`` / ``find`` / ``iter`` / ``search`` verbs and
nothing else. Anything that decides *which* parts of a skill to surface,
or shapes them into markdown for an LLM prompt, lives here, on the
caller side.

Two formatters are shared across callers:

    lap_annotation_prompt(circuit_id) -> str
    graph_analysis_prompt(graph_ids)  -> str

Each pulls structured data via ``skills`` and emits a markdown block.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.skills import skills

# ---------------------------------------------------------------------------
# lap_annotation
# ---------------------------------------------------------------------------

_LAP_CIRCUIT_LABELS = {"brands_hatch", "silverstone"}


def lap_annotation_prompt(circuit_id: Optional[str] = None) -> str:
    """Per-label `characteristics` block for the lap-flow planner / synthesizer.

    When ``circuit_id`` is given, the other circuit's parent label is
    excluded; ``None`` keeps both circuit entries.
    """
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
        if lid in _LAP_CIRCUIT_LABELS and circuit_id is not None and lid != circuit_id:
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


# ---------------------------------------------------------------------------
# graph_analysis
# ---------------------------------------------------------------------------

_GRAPH_GUIDELINE_TRIGGERS: Dict[str, Dict[str, Any]] = {
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


def _render_graph_section(key: str, value: Any, indent: str = "  ") -> List[str]:
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


def graph_analysis_prompt(graph_ids: Sequence[str]) -> str:
    """Per-graph description block for VLM prompts that read graph images.

    Walks each graph's yaml record with a uniform formatter, appends the
    cross-graph guidelines whose triggers match this graph combination,
    and (when trajectory graphs are present) appends the trajectory shape
    vocabulary.
    """
    requested = list(graph_ids)
    paired: List[tuple] = []
    for gid in requested:
        entry = skills.get(f"graph_analysis.graphs.{gid}")
        if entry:
            paired.append((gid, entry))
    if not paired:
        return ""

    lines: List[str] = [
        "#### Graph Description Skill — How to Describe These Graphs",
        "",
    ]

    for gid, entry in paired:
        title = entry.get("title", gid)
        lines.append(f"##### {title} (id: {gid})")
        lines.append("")
        for key, value in entry.items():
            if key in ("title", "id"):
                continue
            section = _render_graph_section(key, value)
            if section:
                lines.extend(section)
                lines.append("")

    graph_id_set = set(requested)
    relevant: List[str] = []
    for gid, spec in _GRAPH_GUIDELINE_TRIGGERS.items():
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
