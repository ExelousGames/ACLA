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
    """Analysis instructions and canonical phrases for one graph type."""

    __slots__ = (
        "id", "title", "graph_type", "axes", "visual_elements",
        "how_to_analyze", "glossary",
        "sentence_format_guide",
        "common_description_errors",
    )

    def __init__(self, graph_id: str, raw: Dict[str, Any]) -> None:
        self.id: str = graph_id
        self.title: str = raw.get("title", graph_id)
        self.graph_type: str = raw.get("graph_type", "unknown")
        self.axes: Dict[str, str] = raw.get("axes", {})
        self.visual_elements: List[str] = raw.get("visual_elements", [])
        self.how_to_analyze: str = (raw.get("how_to_analyze") or "").strip()
        # glossary is a term -> definition map for racing/visualization jargon
        # the VLM may not understand on its own (e.g. "trail braking",
        # "over-limit spike", "2x understeer amplification").  Plain-English
        # comparators ("earlier than", "wider than") do NOT belong here.
        raw_glossary = raw.get("glossary") or {}
        self.glossary: Dict[str, str] = {
            str(k): (str(v).strip() if v is not None else "")
            for k, v in raw_glossary.items()
        }
        self.sentence_format_guide: str = (raw.get("sentence_format_guide") or "").strip()
        self.common_description_errors: List[str] = raw.get("common_description_errors", [])


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

    # -- single graph -------------------------------------------------------

    def get_graph(self, graph_id: str) -> Optional[GraphSkillEntry]:
        """Return the skill entry for *graph_id*, or ``None``."""
        return self._entries.get(graph_id)

    def get_analysis_instructions(self, graph_id: str) -> Optional[str]:
        """Return the how_to_analyze procedure for *graph_id*, or ``None``."""
        entry = self._entries.get(graph_id)
        return entry.how_to_analyze if entry else None

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

        Returns a multi-section text block containing:
        - Per-graph description checklists, axes, visual elements, glossary
        - Relevant cross-graph description guidelines
        - Trajectory shape vocabulary (if trajectory graph present)
        """
        entries = self.get_entries_for_graphs(graph_ids)
        if not entries:
            return ""

        lines: List[str] = [
            "=== Graph Description Skill — How to Describe These Graphs ===",
            "",
        ]

        for entry in entries:
            lines.append(f"--- {entry.title} (id: {entry.id}) ---")
            lines.append(f"Graph type: {entry.graph_type}")

            # Axes
            if entry.axes:
                lines.append("Axes:")
                for axis_name, axis_desc in entry.axes.items():
                    lines.append(f"  {axis_name}: {axis_desc}")

            # Visual elements
            if entry.visual_elements:
                lines.append("Visual elements to identify:")
                for elem in entry.visual_elements:
                    lines.append(f"  - {elem}")

            # How to analyze — clear instructions on how to read the graph
            if entry.how_to_analyze:
                lines.append("How to analyze this graph:")
                lines.append(f"  {entry.how_to_analyze}")

            # Glossary — definitions of racing / visualization jargon used in
            # the analysis steps and sentence guide above.  Emitted as
            # "term: definition" pairs so the VLM can resolve any unfamiliar
            # phrase it encounters in the prompt body.
            if entry.glossary:
                lines.append("Glossary — definitions of terms used in this section:")
                for phrase, definition in entry.glossary.items():
                    if definition:
                        lines.append(f'  - "{phrase}": {definition}')
                    else:
                        lines.append(f'  - "{phrase}"')

            # Sentence format — skeleton + 1-2 example sentences showing how
            # to glue phrases with numerical values into prose.
            if entry.sentence_format_guide:
                lines.append("Sentence format:")
                lines.append(f"  {entry.sentence_format_guide}")

            # Common errors
            if entry.common_description_errors:
                lines.append("Common description errors to AVOID:")
                for err in entry.common_description_errors:
                    lines.append(f"  * {err}")

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
            lines.append("=== Cross-Graph Description Guidelines ===")
            for g in relevant_guidelines:
                lines.append(g)
            lines.append("")

        # Trajectory shape vocabulary if trajectory is present
        has_trajectory = graph_id_set & {
            "trajectory_detailed", "trajectory_gas_brake", "trajectory_balance",
        }
        if has_trajectory and self.trajectory_shape_vocabulary:
            lines.append("=== Trajectory Shape Vocabulary ===")
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
