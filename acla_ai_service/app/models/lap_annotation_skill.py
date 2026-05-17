"""
Lap Annotation Skill — loads and queries the YAML knowledge base used by
the lap-to-segment excerpter (manual.py).

Teaches the agent how to:
  - Map a `Graphics_normalized_car_position`-shaped lap range to per-section
    sub-ranges.
  - Decide when to `revise_segment_range` (shrink/extend) the rough split.
  - Pick the right parent labels (circuit + segment_type + section).

The companion deterministic tool `split_lap_by_circuit_sections` in
`annotation_agent_tools.py` pulls live `normalized_position_range` values
from `label_catalog.yaml` — the skill prompt echoes those ranges inline
so the VLM sees lo/hi without paying a tool call.

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


class LapCircuitEntry:
    """One circuit's YAML record."""

    __slots__ = ("id", "raw")

    def __init__(self, circuit_id: str, raw: Dict[str, Any]) -> None:
        self.id: str = circuit_id
        self.raw: Dict[str, Any] = raw

    @property
    def name(self) -> str:
        return self.raw.get("name", self.id)

    @property
    def section_order(self) -> List[str]:
        return list(self.raw.get("section_order") or [])

    @property
    def default_parent_labels(self) -> List[str]:
        return list(self.raw.get("default_parent_labels") or [])


class LapAnnotationSkill:
    """Queryable lap-annotation skill — one entry per circuit + global rules."""

    def __init__(
        self,
        circuits: Dict[str, LapCircuitEntry],
        global_rules: str,
    ) -> None:
        self._circuits = circuits
        self.global_rules = global_rules

    @property
    def all_circuit_ids(self) -> List[str]:
        return list(self._circuits.keys())

    def get_circuit(self, circuit_id: str) -> Optional[LapCircuitEntry]:
        return self._circuits.get(circuit_id)

    def build_prompt(self, circuit_id: Optional[str] = None) -> str:
        """Render the skill block for the given circuit (or all circuits).

        The returned text is injected verbatim into the agent's system
        prompt. Per-section `normalized_position_range` values are pulled
        live from the label catalog so the skill stays in sync with the
        single source of truth.
        """
        from app.models.label_catalog import get_label_catalog

        catalog = get_label_catalog()
        lines: List[str] = [
            "#### Lap Annotation Skill — How to Label One Circuit Section",
            "",
        ]

        circuits = (
            [self._circuits[circuit_id]]
            if circuit_id is not None and circuit_id in self._circuits
            else list(self._circuits.values())
        )

        for circuit in circuits:
            lines.append(f"##### {circuit.name} (id: {circuit.id})")
            lines.append("")
            lines.append(f"- default_parent_labels: {circuit.default_parent_labels}")
            lines.append("")

            section_order = circuit.section_order
            if section_order:
                lines.append("Ordered sections (lap-progress order; ranges from label_catalog.yaml):")
                for sid in section_order:
                    entry = catalog.get_label(sid)
                    if entry is None:
                        lines.append(f"  - `{sid}` (missing from catalog)")
                        continue
                    rng = entry.normalized_position_range
                    rng_str = (
                        f"[{rng[0]:.3f}, {rng[1]:.3f}]"
                        if rng is not None else "[null, null] — measure me"
                    )
                    lines.append(
                        f"  - `{sid}` ({entry.name}) — normalized_position_range {rng_str}"
                    )
                lines.append("")

            shrink_rules = circuit.raw.get("shrink_extend_rules") or []
            if shrink_rules:
                lines.append("Shrink / extend rules (call revise_segment_range when one fires):")
                for rule in shrink_rules:
                    lines.append(f"  - {rule}")
                lines.append("")

            st_hints = circuit.raw.get("segment_type_hints")
            if st_hints:
                lines.append("Segment type hints (pick exactly one ST1–ST6):")
                for ln in str(st_hints).rstrip("\n").split("\n"):
                    lines.append(f"  {ln}" if ln else "")
                lines.append("")

        if self.global_rules:
            lines.append("##### Global rules (every circuit)")
            for ln in self.global_rules.rstrip("\n").split("\n"):
                lines.append(ln)
            lines.append("")

        return "\n".join(lines)


def load_lap_skill(path: Optional[Path] = None) -> LapAnnotationSkill:
    """Load the YAML lap-annotation skill."""
    skill_path = path or _SKILL_PATH
    with open(skill_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    circuits_raw: Dict[str, Any] = raw.get("circuits", {}) or {}
    circuits: Dict[str, LapCircuitEntry] = {
        str(cid): LapCircuitEntry(str(cid), cdata or {})
        for cid, cdata in circuits_raw.items()
    }

    global_rules = str(raw.get("global_rules", "")).strip()

    LOGGER.info(
        "Loaded lap annotation skill with %d circuit(s).", len(circuits),
    )

    return LapAnnotationSkill(circuits, global_rules)


def get_lap_skill() -> LapAnnotationSkill:
    """Return the module-level singleton, loading on first call."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = load_lap_skill()
    return _skill_instance
