"""Pure range resolution for deterministic label evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Range = Tuple[int, int]


@dataclass(frozen=True)
class LabelEvidence:
    """Required evidence and its resolved annotation range."""

    required_range: Range
    annotation_range: Range

    def required_covers(self, start: int, end: int) -> bool:
        return self.required_range == (int(start), int(end))

    def required_contains(self, start: int, end: int) -> bool:
        required_start, required_end = self.required_range
        return required_start <= int(start) and int(end) <= required_end


def _envelope(ranges: Sequence[Range]) -> Range:
    return (
        min(start for start, _ in ranges),
        max(end for _, end in ranges),
    )


def resolve_label_evidence(
    *,
    required_ranges: Sequence[Range],
    parent_range: Range,
) -> Optional[LabelEvidence]:
    """Resolve evidence without deciding whether a workflow should accept it."""
    if not required_ranges:
        return None

    required_range = _envelope(required_ranges)
    annotation_range = required_range
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
    )
