"""
Graph Analysis Skill — loads and queries the YAML graph description knowledge base.

Provides structured instructions for the VLM on how to accurately *describe*
each telemetry graph type.  The VLM's only job is to produce detailed, precise
visual descriptions — it does NOT diagnose, label, or interpret.  Downstream
pipeline nodes use these descriptions to make decisions.

Used by the ``describe_graphs`` solver to inject graph-specific description
checklists into the VLM prompt when the current step includes graph images.

Usage::

    from app.models.graph_analysis_skill import get_graph_skill

    skill = get_graph_skill()
    prompt_text = skill.build_graph_prompt(["throttle", "brake"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

_SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "graph_analysis_skill.yaml"

# Module-level singleton
_skill_instance: Optional["GraphAnalysisSkill"] = None


# ---------------------------------------------------------------------------
# Data container for a single graph skill entry
# ---------------------------------------------------------------------------

class GraphSkillEntry:
    """One graph's YAML record (raw dict + its id)."""

    __slots__ = ("id", "raw")

    def __init__(self, graph_id: str, raw: Dict[str, Any]) -> None:
        self.id: str = graph_id
        self.raw: Dict[str, Any] = raw

    @property
    def title(self) -> str:
        return self.raw.get("title", self.id)


# ---------------------------------------------------------------------------
# Procedural section renderer
# ---------------------------------------------------------------------------
#
# Walks a YAML key → value pair and emits prompt lines without any
# per-key special casing.  String → indented block, list → bullets,
# dict → "- key: value" pairs.  Multi-line strings (YAML `>` with blank
# lines) keep their internal line breaks so paragraph structure survives.

def _render_section(
    key: str, value: Any, indent: str = "  ",
) -> List[str]:
    if not value:
        return []
    header = key.replace("_", " ") + ":"
    out: List[str] = [header]

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


# ---------------------------------------------------------------------------
# GraphAnalysisSkill
# ---------------------------------------------------------------------------

class GraphAnalysisSkill:
    """Queryable catalogue of graph description instructions for the VLM."""

    def __init__(
        self,
        entries: Dict[str, GraphSkillEntry],
        cross_graph_guidelines: Dict[str, str],
        trajectory_shape_vocabulary: str,
    ) -> None:
        self._entries = entries
        self.cross_graph_guidelines = cross_graph_guidelines
        self.trajectory_shape_vocabulary = trajectory_shape_vocabulary

    # -- bulk queries -------------------------------------------------------

    @property
    def all_ids(self) -> List[str]:
        return list(self._entries.keys())

    def get_entries_for_graphs(self, graph_ids: List[str]) -> List[GraphSkillEntry]:
        """Return skill entries for the given graph IDs (in order)."""
        return [self._entries[gid] for gid in graph_ids if gid in self._entries]

    # -- prompt building ----------------------------------------------------

    def build_graph_prompt(self, graph_ids: List[str]) -> str:
        """Build a VLM prompt section for describing the given graphs.

        Walks each graph's YAML record in declaration order and renders
        every section with the same procedural formatter — no per-key
        special casing.  Followed by gated cross-graph guidelines and the
        trajectory shape vocabulary.
        """
        entries = self.get_entries_for_graphs(graph_ids)
        if not entries:
            return ""

        lines: List[str] = [
            "#### Graph Description Skill — How to Describe These Graphs",
            "",
        ]

        for entry in entries:
            lines.append(f"##### {entry.title} (id: {entry.id})")
            lines.append("")
            for key, value in entry.raw.items():
                if key == "title":
                    continue  # already used as the graph header
                section = _render_section(key, value)
                if section:
                    lines.extend(section)
                    lines.append("")

        # Cross-graph guidelines (only include relevant ones)
        graph_id_set = set(graph_ids)
        relevant_guidelines: List[str] = []

        # Each trigger:
        #   required: every graph in this set must be present (subset check).
        #   any_of:   list of sets; each set must intersect the present
        #             graphs (i.e. at least one member of each must be there).
        guideline_triggers = {
            "brake_and_speed":            {"required": {"brake", "speed"},                     "any_of": []},
            "throttle_and_speed":         {"required": {"throttle", "speed"},                  "any_of": []},
            "time_delta_and_features":    {"required": {"time_delta"},                         "any_of": [
                {"brake", "throttle", "speed", "speed_delta", "push_limit"},
            ]},
            "trajectory_and_features":    {"required": set(),                                  "any_of": [
                {"trajectory_detailed", "trajectory_gas_brake", "trajectory_balance"},
                {"throttle", "brake", "speed", "speed_delta", "push_limit"},
            ]},
            "balance_and_push_limit":     {"required": {"trajectory_balance", "push_limit"},   "any_of": []},
            "brake_and_throttle_overlap": {"required": {"brake", "throttle"},                  "any_of": []},
        }

        for guideline_id, spec in guideline_triggers.items():
            if not spec["required"].issubset(graph_id_set):
                continue
            if not all(any_set & graph_id_set for any_set in spec["any_of"]):
                continue
            text = self.cross_graph_guidelines.get(guideline_id, "")
            if text:
                relevant_guidelines.append(f"[{guideline_id}] {text}")

        if relevant_guidelines:
            lines.append("#### Cross-Graph Description Guidelines")
            for g in relevant_guidelines:
                lines.append(g)
            lines.append("")

        # Trajectory shape vocabulary if trajectory is present
        has_trajectory = graph_id_set & {
            "trajectory_detailed", "trajectory_gas_brake", "trajectory_balance",
        }
        if has_trajectory and self.trajectory_shape_vocabulary:
            lines.append("#### Trajectory Shape Vocabulary")
            lines.append(self.trajectory_shape_vocabulary)
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_graph_skill(path: Optional[Path] = None) -> GraphAnalysisSkill:
    """Load the YAML graph analysis skill and return a :class:`GraphAnalysisSkill`."""
    skill_path = path or _SKILL_PATH
    with open(skill_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raw_graphs: Dict[str, Any] = raw.get("graphs", {})
    entries: Dict[str, GraphSkillEntry] = {}
    for graph_id, graph_data in raw_graphs.items():
        entries[graph_id] = GraphSkillEntry(graph_id, graph_data)

    cross_guidelines: Dict[str, str] = {
        str(k): str(v).strip()
        for k, v in (raw.get("cross_graph_guidelines") or {}).items()
    }

    trajectory_shape_vocab = str(raw.get("trajectory_shape_vocabulary", "")).strip()

    LOGGER.info(
        "Loaded graph analysis skill with %d graph entries.",
        len(entries),
    )

    return GraphAnalysisSkill(entries, cross_guidelines, trajectory_shape_vocab)


def get_graph_skill() -> GraphAnalysisSkill:
    """Return the module-level singleton, loading on first call."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = load_graph_skill()
    return _skill_instance
