"""Annotation and classifier labels, excluding circuit identities and sections."""

from typing import Any

from app.shared.labels import (
    LABEL_CATEGORIES as _LABEL_CATEGORIES,
    LABEL_MAPPING as _LABEL_MAPPING,
    normalize_label_ids,
)


# Legacy dataset IDs are retained only for removal, not as classifier labels.
_LEGACY_TRACK_SECTION_COUNTS = {
    "brands_hatch": 19,
    "silverstone": 22,
    "moza": 14,
    "kyalami": 16,
    "suzuka": 19,
    "laguna_seca": 14,
    "mount_panorama": 22,
    "watkins_glen": 18,
    "indianapolis": 16,
    "cota": 20,
}
TRACK_LABELS = tuple(_LEGACY_TRACK_SECTION_COUNTS)
TRACK_LABEL_IDS = frozenset([
    *TRACK_LABELS,
    *(f"{track}{index}" for track, count in _LEGACY_TRACK_SECTION_COUNTS.items()
      for index in range(1, count + 1)),
    "sliverstone6",
])

LABEL_MAPPING = {
    label: name for label, name in _LABEL_MAPPING.items()
    if label not in TRACK_LABEL_IDS
}
LABEL_NAME_TO_ID = {name: label for label, name in LABEL_MAPPING.items()}
LABEL_CATEGORIES = {
    category: [label for label in labels if label not in TRACK_LABEL_IDS]
    for category, labels in _LABEL_CATEGORIES.items()
    if category != "Track" and category not in TRACK_LABELS
}


def classifier_label_ids(labels: Any) -> list[str]:
    """Discard track labels without moving them into annotation metadata."""
    return [
        label for label in normalize_label_ids(labels)
        if label not in TRACK_LABEL_IDS
    ]
