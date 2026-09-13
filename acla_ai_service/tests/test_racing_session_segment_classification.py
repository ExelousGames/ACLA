from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api import racing_session
from app.pipelines.inference.preprocessing import InferenceTelemetryBatch
from app.shared.segment import PredictedSegment


def _label(name: str, start: int, end: int):
    return {"label_name": name, "start_index": start, "end_index": end}


def _predicted_segment(*child_labels: str) -> PredictedSegment:
    return PredictedSegment(
        id="parent-segment",
        label="MSP",
        score=0.9,
        start_index=0,
        end_index=4,
        subsegments=[
            PredictedSegment(
                id=f"child-{index}",
                label=label,
                score=0.8,
                start_index=index,
                end_index=index + 1,
            )
            for index, label in enumerate(child_labels)
        ],
    )


def _classify(monkeypatch, segment, telemetry_data=None, track_name=None):
    monkeypatch.setattr(
        racing_session,
        "split_runtime_segments",
        lambda dataframe, circuit_id: {
            "circuit_id": circuit_id or "test_circuit",
            "segments": [{"start_index": 0, "end_index": len(dataframe)}],
        },
    )
    monkeypatch.setattr(
        racing_session,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: [segment],
        ),
    )
    return racing_session._classify_telemetry_segments(
        telemetry_data if telemetry_data is not None else [{}, {}, {}, {}],
        track_name,
    )


def _expert_row(index: int):
    return {
        "Graphics_normalized_car_position": index / 10,
        "expert_time_difference": float(index * 10),
        "expert_optimal_time": float(90_000 + index * 250),
        "expert_optimal_player_pos_x": float(index),
        "expert_optimal_player_pos_y": float(index + 1),
        "expert_optimal_player_pos_z": float(index + 2),
        "expert_optimal_throttle": 0.8,
        "expert_optimal_brake": 0.1,
        "expert_optimal_gear": 4.0,
    }


def _configure_endpoint_services(monkeypatch, segment):
    segments = segment if isinstance(segment, list) else [segment]
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: InferenceTelemetryBatch(
            records=[dict(row) for row in records],
            raw_indices=list(range(len(records))),
        ),
    )
    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: SimpleNamespace(
            enrich=lambda records, track=None, car=None: records,
        ),
    )

    class TireGripService:
        async def enrich(self, records):
            return records

    monkeypatch.setattr(
        racing_session,
        "get_tire_grip_analysis",
        TireGripService,
    )
    monkeypatch.setattr(
        racing_session,
        "split_runtime_segments",
        lambda dataframe, circuit_id: {
            "circuit_id": circuit_id or "test_circuit",
            "segments": [{"start_index": 0, "end_index": len(dataframe)}],
        },
    )
    monkeypatch.setattr(
        racing_session,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: segments,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["segment-classification", "live-baseline-analysis"],
)
async def test_classifier_endpoints_return_flat_labels_with_independent_ranges(
    endpoint,
    monkeypatch,
):
    segment = _predicted_segment("MSP1", "MSP2")
    records = [_expert_row(index) for index in range(4)]
    _configure_endpoint_services(monkeypatch, segment)

    if endpoint == "segment-classification":
        result = await racing_session.classify_session_segments(
            racing_session.SegmentClassificationRequest(
                session_id="session-1",
                telemetry_data=records,
            )
        )
    else:
        result = await racing_session.analyze_live_baseline(
            racing_session.LiveBaselineAnalysisRequest(records=records)
        )

    assert result["parent_segment_count"] == 1
    assert len(result["segments"]) == 1
    assert result["segments"][0]["id"] == "parent-segment"
    assert result["segments"][0]["labels"] == [
        _label("MSP", 0, 4), _label("MSP1", 0, 1), _label("MSP2", 1, 2),
    ]
    assert result["segments"][0]["time_gap"] == {
        "start_ms": 0.0, "end_ms": 30.0, "delta_ms": 30.0,
    }
    assert result["segments"][0]["start_index"] == 0
    assert result["segments"][0]["end_index"] == 4
    assert [
        row["expert_optimal_time"]
        for row in result["segments"][0]["expert_reference_data"]
    ] == [90_000.0, 90_250.0, 90_500.0, 90_750.0]
    assert "expert_reference_data" not in result


@pytest.mark.asyncio
async def test_live_baseline_returns_one_segment_with_all_labels_for_track_section(
    monkeypatch,
):
    predictions = [
        _predicted_segment("MSP1"),
        PredictedSegment(
            id="recovery-segment",
            label="RM",
            score=0.8,
            start_index=0,
            end_index=4,
            subsegments=[PredictedSegment(
                label="RM1",
                score=0.7,
                start_index=1,
                end_index=3,
            )],
        ),
        PredictedSegment(
            id="segment-type",
            label="ST1",
            score=0.7,
            start_index=2,
            end_index=4,
        ),
    ]
    records = [
        {
            **_expert_row(index),
            "Graphics_normalized_car_position": position,
        }
        for index, position in enumerate((0.12, 0.14, 0.16, 0.17))
    ]
    _configure_endpoint_services(monkeypatch, predictions)

    result = await racing_session.analyze_live_baseline(
        racing_session.LiveBaselineAnalysisRequest(
            track="brands_hatch",
            records=records,
        )
    )

    assert result["parent_segment_count"] == 1
    assert len(result["segments"]) == 1
    assert result["segments"][0]["id"] == "brands_hatch3:0-4"
    assert result["segments"][0]["labels"] == [
        _label("MSP", 0, 4), _label("MSP1", 0, 1),
        _label("RM", 0, 4), _label("RM1", 1, 3), _label("ST1", 2, 4),
    ]
    assert result["segments"][0]["track_section"] == "brands_hatch3"
    assert result["segments"][0]["start_index"] == 0
    assert result["segments"][0]["end_index"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["segment-classification", "live-baseline-analysis"],
)
async def test_classifier_endpoints_scope_sparse_expert_rows_to_each_segment(
    endpoint,
    monkeypatch,
):
    cleaned = [_expert_row(index) for index in range(4)]
    raw_indices = [1, 4, 9, 12]
    source = [{"source_index": index} for index in range(13)]
    _configure_endpoint_services(monkeypatch, _predicted_segment())
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: InferenceTelemetryBatch(
            records=cleaned,
            raw_indices=raw_indices,
        ),
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        lambda *args, **kwargs: [
            {
                "id": "segment-1",
                "labels": [_label("EA", 0, 2)],
                "start_index": 0,
                "end_index": 2,
            },
            {
                "id": "segment-2",
                "labels": [_label("MSP", 2, 4)],
                "start_index": 2,
                "end_index": 4,
            },
            {
                "id": "segment-without-rows",
                "labels": [_label("MSR", 6, 8)],
                "start_index": 6,
                "end_index": 8,
            },
        ],
    )

    if endpoint == "segment-classification":
        result = await racing_session.classify_session_segments(
            racing_session.SegmentClassificationRequest(
                session_id="session-1",
                telemetry_data=source,
            )
        )
    else:
        result = await racing_session.analyze_live_baseline(
            racing_session.LiveBaselineAnalysisRequest(records=source)
        )

    assert result["samples_analyzed"] == 13
    assert [
        (segment["start_index"], segment["end_index"])
        for segment in result["segments"][:2]
    ] == [(1, 5), (9, 13)]
    assert [segment["labels"] for segment in result["segments"][:2]] == [
        [_label("EA", 1, 5)], [_label("MSP", 9, 13)],
    ]
    assert [
        [row["raw_index"] for row in segment["expert_reference_data"]]
        for segment in result["segments"]
    ] == [[1, 4], [9, 12], []]
    assert "expert_reference_data" not in result
    if endpoint == "live-baseline-analysis":
        assert result["expert_time_available"] is True


def test_main_label_without_subsegments_remains_a_single_label(monkeypatch):
    segments = _classify(monkeypatch, _predicted_segment())

    assert len(segments) == 1
    assert segments[0]["labels"] == [_label("MSP", 0, 4)]


def test_non_behavior_and_custom_classifier_labels_are_preserved(monkeypatch):
    segment = PredictedSegment(
        id="segment-type",
        label="ST1",
        score=0.9,
        start_index=0,
        end_index=4,
        subsegments=[PredictedSegment(
            label="custom-label",
            score=0.8,
            start_index=0,
            end_index=4,
        )],
    )

    segments = _classify(monkeypatch, segment)

    assert len(segments) == 1
    assert segments[0]["labels"] == [
        _label("ST1", 0, 4), _label("custom-label", 0, 4),
    ]


def test_repeated_labels_keep_distinct_ranges_in_service_order(monkeypatch):
    segments = _classify(
        monkeypatch,
        _predicted_segment("MSP2", "MSP1", "MSP2", "MSP1"),
    )

    assert len(segments) == 1
    assert segments[0]["labels"] == [
        _label("MSP", 0, 4), _label("MSP2", 0, 1), _label("MSP1", 1, 2),
        _label("MSP2", 2, 3), _label("MSP1", 3, 4),
    ]


def test_track_section_splitting_preserves_every_classifier_label(monkeypatch):
    telemetry_data = [
        {"Graphics_normalized_car_position": 0.12},
        {"Graphics_normalized_car_position": 0.15},
        {"Graphics_normalized_car_position": 0.20},
        {"Graphics_normalized_car_position": 0.23},
    ]

    segments = _classify(
        monkeypatch,
        _predicted_segment("MSP1", "MSP2"),
        telemetry_data=telemetry_data,
        track_name="brands_hatch",
    )

    assert len(segments) == 2
    assert [segment["track_section"] for segment in segments] == [
        "brands_hatch3",
        "brands_hatch4",
    ]
    assert [segment["labels"] for segment in segments] == [
        [_label("MSP", 0, 2), _label("MSP1", 0, 1), _label("MSP2", 1, 2)],
        [_label("MSP", 2, 4)],
    ]


def test_standalone_label_does_not_add_a_taxonomy_parent(monkeypatch):
    segment = PredictedSegment(label="MSP1", score=0.9, start_index=0, end_index=4)
    segments = _classify(monkeypatch, segment)
    assert segments[0]["labels"] == [_label("MSP1", 0, 4)]


def test_exact_duplicate_label_ranges_are_deduplicated(monkeypatch):
    segment = _predicted_segment("MSP1")
    segment.subsegments.append(segment.subsegments[0])
    segments = _classify(monkeypatch, segment)
    assert segments[0]["labels"] == [_label("MSP", 0, 4), _label("MSP1", 0, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["segment-classification", "live-baseline-analysis"])
async def test_section_and_label_ranges_use_original_telemetry_indices(endpoint, monkeypatch):
    segment = _predicted_segment("MSP1")
    segment.subsegments[0].start_index = 1
    segment.subsegments[0].end_index = 3
    records = [
        {**_expert_row(index), "Graphics_normalized_car_position": position}
        for index, position in enumerate((0.12, 0.15, 0.20, 0.23))
    ]
    _configure_endpoint_services(monkeypatch, segment)
    monkeypatch.setattr(
        racing_session, "preprocess_inference_telemetry",
        lambda source: InferenceTelemetryBatch(records=records, raw_indices=[1, 4, 9, 12]),
    )
    source = [{} for _ in range(13)]
    if endpoint == "segment-classification":
        result = await racing_session.classify_session_segments(
            racing_session.SegmentClassificationRequest(
                track_name="brands_hatch", telemetry_data=source,
            )
        )
    else:
        result = await racing_session.analyze_live_baseline(
            racing_session.LiveBaselineAnalysisRequest(track="brands_hatch", records=source)
        )
    first, second = result["segments"]
    assert first["id"] == "brands_hatch3:1-5"
    assert second["id"] == "brands_hatch4:9-13"
    assert first["labels"] == [_label("MSP", 1, 5), _label("MSP1", 4, 5)]
    assert second["labels"] == [_label("MSP", 9, 13), _label("MSP1", 9, 10)]
