"""Label head definitions for the segment classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING


@dataclass(frozen=True)
class LabelHeadSpec:
    name: str
    label_ids: Tuple[str, ...]
    active_label_ids: Optional[Tuple[str, ...]] = None


def _known_labels(label_ids: Iterable[str]) -> Tuple[str, ...]:
    return tuple(label_id for label_id in label_ids if label_id in LABEL_MAPPING)


def build_label_head_specs() -> List[LabelHeadSpec]:
    specs: List[LabelHeadSpec] = [
        LabelHeadSpec("behavior_main", _known_labels(BEHAVIOR_LABELS)),
        LabelHeadSpec("segment_type", _known_labels(LABEL_CATEGORIES.get("Segment Type", []))),
    ]

    for parent_id in tuple(BEHAVIOR_LABELS):
        child_ids = _known_labels(LABEL_CATEGORIES.get(parent_id, []))
        if not child_ids:
            continue
        active_label_ids = tuple(dict.fromkeys((parent_id, *child_ids)))
        specs.append(LabelHeadSpec(f"sub:{parent_id}", child_ids, active_label_ids))

    return [spec for spec in specs if spec.label_ids]


def labels_for_head(label_ids: Sequence[str], spec: LabelHeadSpec) -> List[str]:
    allowed = set(spec.label_ids)
    return [label_id for label_id in label_ids if label_id in allowed]


def head_is_active(label_ids: Sequence[str], spec: LabelHeadSpec) -> bool:
    if spec.active_label_ids is None:
        return True
    active_labels = set(spec.active_label_ids)
    return any(label_id in active_labels for label_id in label_ids)


__all__ = [
    "LabelHeadSpec",
    "build_label_head_specs",
    "head_is_active",
    "labels_for_head",
]
