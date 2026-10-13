import copy

import pytest

from training.pipelines.manifest.remove_track_labels import remove_dataset_track_labels
from training.shared.labels import LABEL_CATEGORIES, LABEL_MAPPING, classifier_label_ids
from training.storage.lance import LanceTelemetryStore


def test_annotation_catalog_keeps_behavior_and_segment_types_only():
    assert classifier_label_ids([
        "silverstone", "silverstone1", "sliverstone6", "brands_hatch8",
        "MSP", "MSP1", "ST1", "ST20", "custom-label", "MSP",
    ]) == ["MSP", "MSP1", "ST1", "ST20", "custom-label"]
    assert LABEL_CATEGORIES["Main Labels"] == LABEL_CATEGORIES["Behavior"]
    assert "Track" not in LABEL_CATEGORIES
    assert "silverstone" not in LABEL_CATEGORIES
    assert "silverstone1" not in LABEL_MAPPING
    assert LABEL_MAPPING["ST1"] == "In the corner"


@pytest.mark.parametrize("wrapped", [False, True])
def test_migration_preserves_data_and_is_idempotent(tmp_path, wrapped):
    store = LanceTelemetryStore(str(tmp_path))
    key = "manual_segment_annotations_test"
    segments = [
        {
            "id": "parent", "labels": ["silverstone", "silverstone1", "MSP", "ST1"],
            "start_index": 0, "end_index": 2, "notes": "Original reasoning",
            "telemetry_data": [{"Static_track": "silverstone", "speed": 40.0}],
        },
        {
            "id": "child", "parent_id": "parent", "start_index": 1, "end_index": 2,
            "labels": ["MSP1", "sliverstone6", "brands_hatch8", "ST9"],
        },
        {"id": "track-only", "labels": ["silverstone"]},
    ]
    payload = {"data": segments, "description": "unchanged"} if wrapped else segments
    original = copy.deepcopy(payload)
    assert store.save_chunk(key, "session", payload)
    expected_summary = dict(
        sessions_processed=1, sessions_updated=1, segments_updated=3, labels_removed=5,
    )
    assert remove_dataset_track_labels(store, key, dry_run=True) == expected_summary
    assert store.get_chunk(key, "session") == original
    assert remove_dataset_track_labels(store, key) == expected_summary
    expected = copy.deepcopy(original)
    records = expected["data"] if wrapped else expected
    records[0]["labels"] = ["MSP", "ST1"]
    records[1]["labels"] = ["MSP1", "ST9"]
    records[2]["labels"] = []
    assert store.get_chunk(key, "session") == expected
    assert remove_dataset_track_labels(store, key) == dict(
        sessions_processed=1, sessions_updated=0, segments_updated=0, labels_removed=0,
    )


def test_migration_reports_failed_write(tmp_path, monkeypatch):
    store = LanceTelemetryStore(str(tmp_path))
    key = "manual_segment_annotations_test"
    assert store.save_chunk(key, "session", [{"labels": ["silverstone"]}])
    monkeypatch.setattr(store, "save_chunk", lambda *_args: False)
    with pytest.raises(RuntimeError, match="Failed to save"):
        remove_dataset_track_labels(store, key)


def test_auto_annotation_can_still_split_named_track_sections(monkeypatch):
    import pandas as pd

    from training.shared import annotation_agent_tools
    from training.shared.annotation_agent_tools import split_lap_by_circuit_sections

    monkeypatch.setattr(annotation_agent_tools, "get_circuit_sections", lambda *_args: [
        {"id": "backend-a", "name": "First corner", "normalized_position_range": (0.1, 0.18)},
        {"id": "backend-b", "name": "Second corner", "normalized_position_range": (0.19, 0.25)},
    ])
    result = split_lap_by_circuit_sections(
        pd.DataFrame({"Graphics_normalized_car_position": [0.12, 0.15, 0.20, 0.23]}),
        0, 4, circuit_id="brands_hatch", include_interaction_windows=False,
    )
    assert [
        (segment["start_index"], segment["end_index"], segment["circuit_section_id"])
        for segment in result.content["segments"]
    ] == [(0, 2, "backend-a"), (2, 4, "backend-b")]
