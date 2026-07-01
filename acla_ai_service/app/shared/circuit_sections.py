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
    #Moza
    "moza1": (0.93, 0.08),
    "moza2": (0.08, 0.164242),
    "moza3": (0.164242, 0.19),
    "moza4": (0.19, 0.285),
    "moza5": (0.285, 0.39),
    "moza6": (0.39, 0.467),
    "moza7": (0.467, 0.51),
    "moza8": (0.51, 0.623),
    "moza9": (0.623, 0.69046),
    "moza10": (0.69046, 0.709655),
    "moza11": (0.709655, 0.736),
    "moza12": (0.736, 0.84),
    "moza13": (0.84, 0.93),

    #Cota
    "cota1":(0.012884,0.0607),
    "cota2":(0.0607,0.148075),
    "cota3":(0.148075,0.189831),
    "cota4":(0.189831,0.22169),
    "cota5":(0.22169,0.243331),
    "cota6":(0.243331,0.262579),
    "cota7":(0.262579,0.301795),
    "cota8":(0.301795,0.328603),
    "cota9":(0.328603,0.353767),
    "cota10":(0.353767,0.373938),
    "cota11":(0.373938,0.417829),
    "cota12":(0.417829,0.503359),
    "cota13":(0.503359,0.626809),
    "cota14":(0.626809,0.702382),
    "cota15":(0.702382,0.756491),
    "cota16":(0.756491,0.796197),
    "cota17":(0.796197,0.879796),
    "cota18":(0.879796,0.938694),
    "cota19":(0.938694,0.992249),
    "cota20":(0.992249,0.012884),

    #indianapolis
    "indianapolis1":(0.081036,0.171591),
    "indianapolis2":(0.171591,0.204029),
    "indianapolis3":(0.204029,0.284349),
    "indianapolis4":(0.284349,0.323545),
    "indianapolis5":(0.323545,0.392204),
    "indianapolis6":(0.392204,0.439344),
    "indianapolis7":(0.439344,0.514357),
    "indianapolis8":(0.514357,0.580427),
    "indianapolis9":(0.580427,0.609054),
    "indianapolis10":(0.609054,0.630686),
    "indianapolis11":(0.630686,0.666688),
    "indianapolis12":(0.666688,0.744129),
    "indianapolis13":(0.744129,0.82138),
    "indianapolis14":(0.82138,0.866126),
    "indianapolis15":(0.866126,0.952653),
    "indianapolis16":(0.952653,0.081036),
}


def get_range(label_id: str) -> Optional[Tuple[float, float]]:
    return CIRCUIT_SECTION_RANGES.get(label_id)
