from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api import racing_session
from app.pipelines.inference.preprocessing import InferenceTelemetryBatch
from app.shared.segment import PredictedSegment


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
            classify_ranges=lambda dataframe, ranges: [segment],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["segment-classification", "live-baseline-analysis"],
)
async def test_classifier_endpoints_return_one_parent_range_with_all_sub_labels(
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
    assert result["segments"][0]["labels"] == ["MSP", "MSP1", "MSP2"]
    assert result["segments"][0]["start_index"] == 0
    assert result["segments"][0]["end_index"] == 4
    assert [
        row["expert_optimal_time"]
        for row in result["segments"][0]["expert_reference_data"]
    ] == [90_000.0, 90_250.0, 90_500.0, 90_750.0]
    assert "expert_reference_data" not in result


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
                "labels": ["EA"],
                "start_index": 0,
                "end_index": 2,
            },
            {
                "id": "segment-2",
                "labels": ["MSP"],
                "start_index": 2,
                "end_index": 4,
            },
            {
                "id": "segment-without-rows",
                "labels": ["MSR"],
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
    assert segments[0]["labels"] == ["MSP"]


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
    assert segments[0]["labels"] == ["ST1", "custom-label"]


def test_repeated_child_labels_are_deduplicated_in_service_order(monkeypatch):
    segments = _classify(
        monkeypatch,
        _predicted_segment("MSP2", "MSP1", "MSP2", "MSP1"),
    )

    assert len(segments) == 1
    assert segments[0]["labels"] == ["MSP", "MSP2", "MSP1"]


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
        ["MSP", "MSP1", "MSP2"],
        ["MSP", "MSP1", "MSP2"],
    ]
