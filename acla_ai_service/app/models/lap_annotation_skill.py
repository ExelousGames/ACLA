"""
Lap Annotation Skill — loads the YAML knowledge base used by the
lap-to-segment excerpter (manual.py).

The skill is label-centric: for each of the 8 candidate parent labels the
agent may attach to one circuit_section, it describes the telemetry
characteristic that justifies attaching it, and a `global_rules` block
prescribes the detection procedure.

Circuit_section IDs (brands_hatch1, silverstone3, ...) and ST1–ST6
descriptions are NOT duplicated here — those come from `label_catalog.yaml`
and the `locate_circuit_section` tool.

Usage::

    from app.models.lap_annotation_skill import get_lap_skill

    skill = get_lap_skill()
    prompt = skill.build_prompt(circuit_id="brands_hatch")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

_SKILL_PATH = (
    Path(__file__).resolve().parent.parent / "skills" / "lap_annotation_skill.yaml"
)

_skill_instance: Optional["LapAnnotationSkill"] = None

# Label IDs that are gated by which circuit the lap was driven on. When
# `build_prompt(circuit_id=...)` is called we keep only the matching circuit
# label and drop the other one from the rendered prompt.
_CIRCUIT_LABELS = {"brands_hatch", "silverstone"}


class LapLabelEntry:
    """One label's YAML record."""

    __slots__ = ("id", "raw")

    def __init__(self, label_id: str, raw: Dict[str, Any]) -> None:
        self.id: str = label_id
        self.raw: Dict[str, Any] = raw

    @property
    def name(self) -> str:
        return self.raw.get("name", self.id)

    @property
    def applies_when(self) -> str:
        return str(self.raw.get("applies_when", "")).strip()

    @property
    def characteristics(self) -> str:
        return str(self.raw.get("characteristics", "")).strip()


class LapAnnotationSkill:
    """Queryable lap-annotation skill — one entry per candidate label."""

    def __init__(
        self,
        labels: Dict[str, LapLabelEntry],
        global_rules: str,
    ) -> None:
        self._labels = labels
        self.global_rules = global_rules

    @property
    def all_label_ids(self) -> List[str]:
        return list(self._labels.keys())

    def get_label(self, label_id: str) -> Optional[LapLabelEntry]:
        return self._labels.get(label_id)

    def build_prompt(self, circuit_id: Optional[str] = None) -> str:
        """Render the skill block for the agent's system prompt.

        When ``circuit_id`` is given (the lap's circuit), the other
        circuit's label is filtered out so the agent only sees the
        relevant one. Main labels (EA / MS / RM / PS / OV / MD) are
        always included.
        """
        lines: List[str] = [
            "#### Lap Annotation Skill — Candidate Label Characteristics",
            "",
            "Each candidate parent label below lists the telemetry pattern "
            "that justifies attaching it. The `global_rules` block at the "
            "end is the per-section detection procedure.",
            "",
        ]

        for label_id, entry in self._labels.items():
            if (
                label_id in _CIRCUIT_LABELS
                and circuit_id is not None
                and label_id != circuit_id
            ):
                continue
            lines.append(f"##### `{label_id}` — {entry.name}")
            if entry.applies_when:
                lines.append(f"_Applies when:_ {entry.applies_when}")
            if entry.characteristics:
                lines.append(entry.characteristics)
            lines.append("")

        if self.global_rules:
            lines.append("##### Global rules — how to find each label")
            for ln in self.global_rules.rstrip("\n").split("\n"):
                lines.append(ln)
            lines.append("")

        return "\n".join(lines)


def load_lap_skill(path: Optional[Path] = None) -> LapAnnotationSkill:
    """Load the YAML lap-annotation skill."""
    skill_path = path or _SKILL_PATH
    with open(skill_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    labels_raw: Dict[str, Any] = raw.get("labels", {}) or {}
    labels: Dict[str, LapLabelEntry] = {
        str(lid): LapLabelEntry(str(lid), ldata or {})
        for lid, ldata in labels_raw.items()
    }

    global_rules = str(raw.get("global_rules", "")).strip()

    LOGGER.info(
        "Loaded lap annotation skill with %d label(s).", len(labels),
    )

    return LapAnnotationSkill(labels, global_rules)


def get_lap_skill() -> LapAnnotationSkill:
    """Return the module-level singleton, loading on first call."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = load_lap_skill()
    return _skill_instance
