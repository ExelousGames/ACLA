"""Helpers for preserving parent/sub-label grouping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.shared.circuit_sections import CIRCUIT_SECTION_RANGES
from app.shared.labels import LABEL_CATEGORIES, LABEL_MAPPING, LABEL_NAME_TO_ID


BEHAVIOR_PARENT_LABEL_IDS = tuple(
    label_id for label_id in ("O", "OD", "EA", "PS", "RM", "MSP", "MSR")
    if label_id in LABEL_MAPPING
)
NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"
TRACK_PARENT_LABEL_IDS = tuple(
    label_id for label_id in LABEL_CATEGORIES
    if label_id in LABEL_MAPPING
    and label_id not in BEHAVIOR_PARENT_LABEL_IDS
)


def _parent_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for parent_id, child_ids in LABEL_CATEGORIES.items():
        if parent_id not in LABEL_MAPPING:
            continue
        for child_id in child_ids:
            if child_id in LABEL_MAPPING:
                lookup[child_id] = parent_id
    return lookup


def normalize_grouped_label_ids(
    raw_label_ids: Any,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Return valid label IDs with required parent labels included.

    Sub-labels are valid on lap segments, but they should never stand alone:
    adding the parent main/circuit label keeps downstream grouping intact.
    """
    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    added_parents: List[str] = []
    parents = _parent_lookup()

    if not isinstance(raw_label_ids, list):
        rejected.append({
            "value": raw_label_ids, "reason": "label_ids was not a list",
        })
        return cleaned, rejected, added_parents

    for i, raw_lid in enumerate(raw_label_ids):
        if not isinstance(raw_lid, str):
            rejected.append({
                "index": i, "value": raw_lid, "reason": "must be string",
            })
            continue
        if raw_lid not in LABEL_MAPPING:
            rejected.append({
                "index": i, "value": raw_lid,
                "reason": f"unknown label_id '{raw_lid}'",
            })
            continue

        parent_id = parents.get(raw_lid)
        if parent_id and parent_id not in cleaned:
            cleaned.append(parent_id)
            if parent_id not in raw_label_ids and parent_id not in added_parents:
                added_parents.append(parent_id)

        if raw_lid not in cleaned:
            cleaned.append(raw_lid)

    return cleaned, rejected, added_parents


def _dedupe_label_ids(label_ids: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for label_id in label_ids:
        if label_id in seen:
            continue
        seen.add(label_id)
        deduped.append(label_id)
    return deduped


def _parent_labels(label_ids: List[str]) -> List[str]:
    return [
        label_id for label_id in label_ids
        if label_id in BEHAVIOR_PARENT_LABEL_IDS
    ]


def _sub_label_ids(label_ids: List[str], parent_labels: List[str]) -> List[str]:
    return [label_id for label_id in label_ids if label_id not in parent_labels]


def _track_id(raw: Any) -> str:
    value = str(raw or "").strip()
    if value in LABEL_MAPPING:
        return value
    mapped = LABEL_NAME_TO_ID.get(value)
    if mapped:
        return mapped
    return value.lower().replace(" ", "_").replace("-", "_")


def _position_in_range(position: float, section_range: Tuple[float, float]) -> bool:
    lo, hi = section_range
    position = position % 1.0
    if hi >= lo:
        return lo <= position <= hi
    return position >= lo or position <= hi


def _section_candidates(track_id: str) -> List[Tuple[str, Tuple[float, float]]]:
    return [
        (section_id, section_range)
        for section_id, section_range in CIRCUIT_SECTION_RANGES.items()
        if section_id.startswith(track_id)
    ]


def _section_for_position(position: Any, track_id: str) -> Optional[str]:
    try:
        normalized_position = float(position) % 1.0
    except (TypeError, ValueError):
        return None

    for section_id, section_range in _section_candidates(track_id):
        if _position_in_range(normalized_position, section_range):
            return section_id
    return None


def _analysis_label_ids(label_ids: List[str]) -> List[str]:
    return [
        label_id for label_id in label_ids
        if label_id not in TRACK_PARENT_LABEL_IDS
        and label_id not in CIRCUIT_SECTION_RANGES
    ]


def _track_area_windows(
    telemetry_data: Sequence[Dict[str, Any]],
    track_id: str,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    current_section_id: Optional[str] = None
    current_start: Optional[int] = None

    for index, row in enumerate(telemetry_data):
        section_id = _section_for_position(row.get(NORMALIZED_POSITION_COLUMN), track_id)
        if section_id == current_section_id:
            continue

        if current_section_id is not None and current_start is not None:
            windows.append({
                "section_id": current_section_id,
                "start_index": current_start,
                "end_index": index,
            })

        current_section_id = section_id
        current_start = index if section_id is not None else None

    if current_section_id is not None and current_start is not None:
        windows.append({
            "section_id": current_section_id,
            "start_index": current_start,
            "end_index": len(telemetry_data),
        })

    return windows


def build_track_area_segments(
    raw_segments: List[Dict[str, Any]],
    telemetry_data: Sequence[Dict[str, Any]],
    track_name: Optional[str],
    include_empty_sections: bool = False,
) -> List[Dict[str, Any]]:
    """Build parent track-area segments with child analysis segments."""
    track_id = _track_id(track_name)
    if not track_id or not telemetry_data or not _section_candidates(track_id):
        return build_parent_label_segments(raw_segments)

    parent_windows = _track_area_windows(telemetry_data, track_id)
    if not parent_windows:
        return build_parent_label_segments(raw_segments)

    segments: List[Dict[str, Any]] = []

    for parent_window in parent_windows:
        parent_start = parent_window["start_index"]
        parent_end = parent_window["end_index"]
        parent_labels = [parent_window["section_id"]]
        child_segments: List[Dict[str, Any]] = []
        child_label_ids: List[str] = []

        for raw_segment in raw_segments:
            raw_start = raw_segment.get("start_index")
            raw_end = raw_segment.get("end_index")
            if raw_start is None or raw_end is None:
                continue

            child_start = max(parent_start, int(raw_start))
            child_end = min(parent_end, int(raw_end))
            if child_end <= child_start:
                continue

            cleaned_labels, _, _ = normalize_grouped_label_ids(raw_segment.get("labels", []))
            analysis_labels = _analysis_label_ids(cleaned_labels)
            if not analysis_labels:
                continue

            child_label_ids.extend(analysis_labels)
            child_segments.append({
                "start_index": int(raw_start),
                "end_index": int(raw_end),
                "labels": _dedupe_label_ids(analysis_labels),
            })

        if not child_segments and not include_empty_sections:
            continue

        segment_labels = _dedupe_label_ids(parent_labels + child_label_ids)
        segments.append({
            "id": f"{parent_labels[0]}:{parent_start}-{parent_end}",
            "labels": segment_labels,
            "parent_labels": parent_labels,
            "start_index": parent_start,
            "end_index": parent_end,
            "sub_labels": _dedupe_label_ids(child_label_ids),
            "sub_segments": child_segments,
            "child_segments": child_segments,
        })

    return segments


def build_parent_label_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge classifier windows into parent-label-first display segments."""
    segments: List[Dict[str, Any]] = []

    for raw_segment in raw_segments:
        cleaned_labels, _, _ = normalize_grouped_label_ids(raw_segment.get("labels", []))
        parent_labels = _parent_labels(cleaned_labels)
        if not parent_labels:
            continue

        start_index = raw_segment.get("start_index")
        end_index = raw_segment.get("end_index")
        if start_index is None or end_index is None:
            continue

        sub_label_ids = _sub_label_ids(cleaned_labels, parent_labels)
        sub_segment = {
            "start_index": start_index,
            "end_index": end_index,
            "labels": _dedupe_label_ids(sub_label_ids),
        }

        previous = segments[-1] if segments else None
        if (
            previous
            and previous["parent_labels"] == parent_labels
            and previous["end_index"] == start_index
        ):
            previous["end_index"] = end_index
            previous["labels"] = _dedupe_label_ids(previous["labels"] + cleaned_labels)
            previous["sub_labels"] = _dedupe_label_ids(_sub_label_ids(previous["labels"], parent_labels))
            previous["sub_segments"].append(sub_segment)
            continue

        segment_labels = _dedupe_label_ids(cleaned_labels)
        segments.append({
            "id": raw_segment.get("id"),
            "labels": segment_labels,
            "parent_labels": parent_labels,
            "start_index": start_index,
            "end_index": end_index,
            "sub_labels": _dedupe_label_ids(sub_label_ids),
            "sub_segments": [sub_segment],
        })

    return segments
