from __future__ import annotations

import pandas as pd

from app.ml.segment_classifier.service import SegmentClassifierService
from app.shared.segment import PredictedSegment


def _detection(label, score, start, end, child_label):
    return PredictedSegment(
        id=f"parent-{label}-{start}",
        label=label,
        score=score,
        start_index=start,
        end_index=end,
        subsegments=[PredictedSegment(
            id=f"child-{child_label}",
            label=child_label,
            score=0.7,
            start_index=start,
            end_index=end,
        )],
    )


def test_classify_ranges_retains_every_detection_and_remaps_indices(tmp_path, monkeypatch):
    service = SegmentClassifierService(models_directory=tmp_path)
    service.behavior_label_ids = ["MSP", "RM"]
    calls = []
    monkeypatch.setattr(
        service,
        "detect_segments",
        lambda dataframe: calls.append(len(dataframe)) or [
            _detection("RM", 0.8, 0, 2, "RM1"),
            _detection("MSP", 0.9, 1, 4, "MSP1"),
        ],
    )

    result = service.classify_ranges(
        pd.DataFrame({"row": range(10)}),
        [{"start_index": 2, "end_index": 7}],
    )

    assert calls == [5]
    assert [item.label for item in result] == ["RM", "MSP"]
    assert [
        (item.start_index, item.end_index)
        for item in result
    ] == [(2, 4), (3, 6)]
    assert result[1].telemetry_data == [{"row": value} for value in range(3, 6)]
    assert [child.label for child in result[1].subsegments] == ["MSP1"]
    assert (
        result[1].subsegments[0].start_index,
        result[1].subsegments[0].end_index,
    ) == (3, 6)


def test_classify_ranges_preserves_detection_order_for_equal_scores(tmp_path, monkeypatch):
    service = SegmentClassifierService(models_directory=tmp_path)
    service.behavior_label_ids = ["MSP", "RM"]
    monkeypatch.setattr(
        service,
        "detect_segments",
        lambda dataframe: [
            _detection("RM", 0.9, 0, 1, "RM1"),
            _detection("MSP", 0.9, 2, 3, "MSP1"),
            _detection("MSP", 0.9, 1, 2, "MSP2"),
        ],
    )

    result = service.classify_ranges(
        pd.DataFrame({"row": range(4)}),
        [{"start_index": 0, "end_index": 4}],
    )

    assert [item.label for item in result] == ["RM", "MSP", "MSP"]
    assert [child.label for child in result[0].subsegments] == ["RM1"]
    assert [child.label for child in result[1].subsegments] == ["MSP1"]
    assert [child.label for child in result[2].subsegments] == ["MSP2"]


def test_classify_ranges_skips_invalid_ranges_without_detection_calls(tmp_path, monkeypatch):
    service = SegmentClassifierService(models_directory=tmp_path)
    calls = []
    monkeypatch.setattr(
        service,
        "detect_segments",
        lambda dataframe: calls.append(len(dataframe)) or [],
    )

    result = service.classify_ranges(
        pd.DataFrame({"row": range(4)}),
        [
            {},
            {"start_index": -1, "end_index": 2},
            {"start_index": 2, "end_index": 2},
            {"start_index": 0, "end_index": 5},
            {"start_index": 1, "end_index": 3},
        ],
    )

    assert result == []
    assert calls == [2]


def test_classify_ranges_calls_each_valid_empty_range_once(tmp_path, monkeypatch):
    service = SegmentClassifierService(models_directory=tmp_path)
    calls = []
    monkeypatch.setattr(
        service,
        "detect_segments",
        lambda dataframe: calls.append(len(dataframe)) or [],
    )

    result = service.classify_ranges(
        pd.DataFrame({"row": range(4)}),
        [
            {"start_index": 0, "end_index": 2},
            {"start_index": 2, "end_index": 4},
        ],
    )

    assert result == []
    assert calls == [2, 2]
