"""Tests for label parent/sub-label grouping helpers."""

from __future__ import annotations

from app.domain.label_hierarchy import build_main_label_segments, normalize_grouped_label_ids


def test_normalize_grouped_label_ids_adds_missing_parent() -> None:
    cleaned, rejected, added = normalize_grouped_label_ids(["MSP1", "ST3"])

    assert cleaned == ["MSP", "MSP1", "ST3"]
    assert rejected == []
    assert added == ["MSP"]


def test_normalize_grouped_label_ids_keeps_existing_parent_order() -> None:
    cleaned, rejected, added = normalize_grouped_label_ids(["MSP", "ST3", "MSP1"])

    assert cleaned == ["MSP", "ST3", "MSP1"]
    assert rejected == []
    assert added == []


def test_build_main_label_segments_resolves_child_parent() -> None:
    segments = build_main_label_segments([
        {
            "id": "segment-1",
            "labels": ["MSP1", "ST3"],
            "start_index": 0,
            "end_index": 3,
        }
    ])

    assert segments[0]["main_label_id"] == "MSP"
    assert segments[0]["sub_labels"] == [
        {"label_id": "MSP1", "label_name": "Initiate brake too late"},
        {"label_id": "ST3", "label_name": "Approach to corner"},
    ]


def test_build_main_label_segments_keeps_segment_type_as_context() -> None:
    segments = build_main_label_segments([
        {
            "id": "segment-1",
            "labels": ["ST3"],
            "start_index": 0,
            "end_index": 3,
        }
    ])

    assert segments == []
