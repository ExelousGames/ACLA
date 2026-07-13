"""Pure range resolution for deterministic label evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Range = Tuple[int, int]


@dataclass(frozen=True)
class LabelEvidence:
    """Required evidence and the optional-support-expanded annotation range."""

    required_range: Range
    annotation_range: Range
    supporting_reasons: Tuple[str, ...] = ()

    def required_covers(self, start: int, end: int) -> bool:
        return self.required_range == (int(start), int(end))


def _envelope(ranges: Sequence[Range]) -> Range:
    return (
        min(start for start, _ in ranges),
        max(end for _, end in ranges),
    )


def _intersect_envelope(value: Range, allowed: Sequence[Range]) -> Optional[Range]:
    intersections = [
        (max(value[0], start), min(value[1], end))
        for start, end in allowed
        if max(value[0], start) <= min(value[1], end)
    ]
    return _envelope(intersections) if intersections else None


def resolve_label_evidence(
    *,
    required_ranges: Sequence[Range],
    parent_range: Range,
    allowed_phase_ranges: Optional[Sequence[Range]] = None,
    supporting_ranges: Sequence[Range] = (),
    supporting_reasons: Sequence[str] = (),
) -> Optional[LabelEvidence]:
    """Resolve evidence without deciding whether a workflow should accept it."""
    if not required_ranges:
        return None

    required_range = _envelope(required_ranges)
    if allowed_phase_ranges is not None:
        required_range = _intersect_envelope(required_range, allowed_phase_ranges)
        if required_range is None:
            return None

    annotation_range = _envelope([required_range, *supporting_ranges])
    parent_start, parent_end = (int(value) for value in parent_range)
    if not (
        parent_start
        <= annotation_range[0]
        <= annotation_range[1]
        <= parent_end
    ):
        return None

    return LabelEvidence(
        required_range=required_range,
        annotation_range=annotation_range,
        supporting_reasons=tuple(supporting_reasons),
    )
