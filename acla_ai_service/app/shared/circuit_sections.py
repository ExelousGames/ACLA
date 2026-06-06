"""Per-section ``normalized_car_position`` ranges for each circuit_section
label. Owned by the domain layer; consumed by the lap splitter and the
``locate_circuit_section`` tool to project telemetry samples onto named
sections.

This lives outside ``app/skills/internal/annotation/sub_label_annotation.json``
on purpose: the skill JSON is the LLM-facing surface (RAG hybrid index
over descriptions + annotation_guideline), and geometric ranges are
neither read by the LLM nor part of label prose.

Each entry is ``(lo, hi)`` on the normalized lap position. ``hi < lo``
means the section wraps across the start/finish line (e.g. Brands Hatch
Paddock Hill Bend at [1.0, 0.09]). Sections whose range is not yet
measured (most of Silverstone) are simply absent from this map;
``locate_circuit_section`` skips them.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

CIRCUIT_SECTION_RANGES: Dict[str, Tuple[float, float]] = {
    # Brands Hatch
    "brands_hatch1": (0.94, 1.0),
    "brands_hatch2": (1.0, 0.09),
    "brands_hatch3": (0.11, 0.18),
    "brands_hatch4": (0.19, 0.25),
    "brands_hatch5": (0.25, 0.28),
    "brands_hatch6": (0.28, 0.35),
    "brands_hatch7": (0.35, 0.47),
    "brands_hatch9": (0.47, 0.55),
    "brands_hatch10": (0.55, 0.56),
    "brands_hatch11": (0.56, 0.63),
    "brands_hatch12": (0.63, 0.67),
    "brands_hatch13": (0.67, 0.72),
    "brands_hatch14": (0.72, 0.79),
    "brands_hatch15": (0.79, 0.84),
    "brands_hatch16": (0.84, 0.94),
    "brands_hatch17": (0.94, 1.0),
    "brands_hatch18": (0.09, 0.11),
    "brands_hatch19": (0.18, 0.19),
    # Silverstone — ranges TBD; entries omitted until measured.
}


def get_range(label_id: str) -> Optional[Tuple[float, float]]:
    return CIRCUIT_SECTION_RANGES.get(label_id)
