from app.api import racing_session

import pytest


@pytest.mark.asyncio
async def test_live_baseline_analysis_returns_circuit_sections_without_classifier_children(monkeypatch):
    class EmptySegmentClassifier:
        def scan_telemetry_data(self, dataframe):
            return []

    class MissingExpertService:
        def extract_expert_state_for_telemetry(self, telemetry_data):
            raise ValueError("No stored fastest laps")

    monkeypatch.setattr(
        racing_session,
        "get_segment_classifier",
        lambda: EmptySegmentClassifier(),
    )
    monkeypatch.setattr(
        racing_session,
        "get_expert_imitation_learning",
        lambda: MissingExpertService(),
    )

    result = await racing_session.analyze_live_baseline(
        racing_session.LiveBaselineAnalysisRequest(
            track="brands_hatch",
            baseline_lap=4,
            records=[
                {"Graphics_normalized_car_position": 0.02},
                {"Graphics_normalized_car_position": 0.04},
                {"Graphics_normalized_car_position": 0.06},
                {"Graphics_normalized_car_position": 0.08},
                {"Graphics_normalized_car_position": 0.12},
                {"Graphics_normalized_car_position": 0.14},
            ],
        ),
    )

    assert result["session_id"] == "live-baseline-lap-4"
    assert result["segment_count"] == 2
    assert [segment["parent_labels"] for segment in result["segments"]] == [
        ["brands_hatch2"],
        ["brands_hatch3"],
    ]
    assert result["segments"][0]["child_segments"] == []
    assert result["expert_time_available"] is False


def test_annotate_segments_uses_parent_label_range_for_time_gap():
    segments = [
        {
            "parent_labels": ["brands_hatch1"],
            "start_index": 0,
            "end_index": 5,
            "child_segments": [
                {
                    "start_index": 1,
                    "end_index": 5,
                    "labels": ["MSP1"],
                },
            ],
            "sub_segments": [
                {
                    "start_index": 1,
                    "end_index": 5,
                    "labels": ["MSP1"],
                },
            ],
        },
        {
            "parent_labels": ["brands_hatch2"],
            "start_index": 3,
            "end_index": 5,
            "child_segments": [],
        },
        {
            "parent_labels": ["MSP"],
            "start_index": 3,
            "end_index": 5,
        },
    ]
    expert_rows = [
        {"expert_time_difference": 0},
        {"expert_time_difference": 80},
        {"expert_time_difference": 200},
        {"expert_time_difference": 220},
        {"expert_time_difference": 180},
    ]

    result = racing_session._annotate_segments_with_time_gaps(segments, expert_rows)

    assert result[0]["time_gap"] == {
        "start_ms": 0.0,
        "end_ms": 180.0,
        "delta_ms": 180.0,
    }
    assert "time_gap" not in result[0]["child_segments"][0]
    assert "time_gap" not in result[0]["sub_segments"][0]
    assert result[1]["time_gap"] == {
        "start_ms": 220.0,
        "end_ms": 180.0,
        "delta_ms": -40.0,
    }
    assert result[2]["time_gap"] == {
        "start_ms": 220.0,
        "end_ms": 180.0,
        "delta_ms": -40.0,
    }


def test_extract_expert_rows_reports_unavailable_when_expert_time_missing(monkeypatch):
    class MissingExpertService:
        def extract_expert_state_for_telemetry(self, telemetry_data):
            raise ValueError("No stored fastest laps")

    monkeypatch.setattr(
        racing_session,
        "get_expert_imitation_learning",
        lambda: MissingExpertService(),
    )

    expert_rows, expert_time_available = racing_session._extract_expert_rows(
        [{"Graphics_current_time": 1000}],
    )

    assert expert_rows == []
    assert expert_time_available is False
