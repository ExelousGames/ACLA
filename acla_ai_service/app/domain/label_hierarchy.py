"""Helpers for preserving parent/sub-label grouping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.domain.labels import LABEL_CATEGORIES, LABEL_MAPPING


BEHAVIOR_MAIN_LABEL_IDS = tuple(
    label_id for label_id in LABEL_CATEGORIES["Main Labels"]
    if label_id in {"O", "OD", "MD", "EA", "PS", "RM", "MSP", "MSR"}
)


def _parent_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for parent_id, child_ids in LABEL_CATEGORIES.items():
        if parent_id in {"Main Labels", "Segment Type"}:
            continue
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


def _label_display(label_id: str) -> Dict[str, str]:
    return {
        "label_id": label_id,
        "label_name": LABEL_MAPPING.get(label_id, label_id),
    }


def _dedupe_label_ids(label_ids: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for label_id in label_ids:
        if label_id in seen:
            continue
        seen.add(label_id)
        deduped.append(label_id)
    return deduped


def _main_label_id(label_ids: List[str]) -> Optional[str]:
    for main_label_id in BEHAVIOR_MAIN_LABEL_IDS:
        if main_label_id in label_ids:
            return main_label_id
    return None


def _sub_label_ids(label_ids: List[str], main_label_id: str) -> List[str]:
    return [label_id for label_id in label_ids if label_id != main_label_id]


def build_main_label_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge classifier windows into main-label-first display segments."""
    segments: List[Dict[str, Any]] = []

    for raw_segment in raw_segments:
        cleaned_labels, _, _ = normalize_grouped_label_ids(raw_segment.get("labels", []))
        main_label_id = _main_label_id(cleaned_labels)
        if not main_label_id:
            continue

        start_index = raw_segment.get("start_index")
        end_index = raw_segment.get("end_index")
        if start_index is None or end_index is None:
            continue

        sub_label_ids = _sub_label_ids(cleaned_labels, main_label_id)
        sub_segment = {
            "start_index": start_index,
            "end_index": end_index,
            "labels": [_label_display(label_id) for label_id in sub_label_ids],
        }

        previous = segments[-1] if segments else None
        if (
            previous
            and previous["main_label_id"] == main_label_id
            and previous["end_index"] == start_index
        ):
            previous["end_index"] = end_index
            previous["labels"] = _dedupe_label_ids(previous["labels"] + cleaned_labels)
            previous["sub_labels"] = [
                _label_display(label_id)
                for label_id in _sub_label_ids(previous["labels"], main_label_id)
            ]
            previous["sub_segments"].append(sub_segment)
            continue

        segment_labels = _dedupe_label_ids(cleaned_labels)
        segments.append({
            "id": raw_segment.get("id"),
            "labels": segment_labels,
            "main_label_id": main_label_id,
            "main_label_name": LABEL_MAPPING.get(main_label_id, main_label_id),
            "start_index": start_index,
            "end_index": end_index,
            "sub_labels": [_label_display(label_id) for label_id in sub_label_ids],
            "sub_segments": [sub_segment],
        })

    return segments
