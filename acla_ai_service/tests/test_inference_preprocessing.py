from __future__ import annotations

import pytest

from app.pipelines.inference.preprocessing import preprocess_inference_telemetry
from app.shared.telemetry import TelemetryFeatures


def _telemetry_row(current_time: float, **overrides):
    row = {
        feature: 0.0
        for feature in TelemetryFeatures.get_features_for_top_lap_reference()
    }
    row.update(
        {
            "Graphics_current_time": current_time,
            "Graphics_completed_lap": 0,
            "Graphics_track_grip_status": 2,
            "Physics_gear": 3,
        }
    )
    row.update(overrides)
    return row


def test_preprocess_inference_telemetry_matches_training_shape_and_tracks_raw_rows():
    source = [
        _telemetry_row(
            0,
            Graphics_player_pos_x=0.0,
            Graphics_player_pos_y=10.0,
            Graphics_player_pos_z=20.0,
            Physics_velocity_y=1.0,
            Physics_velocity_z=2.0,
            unexpected_feature=99,
        ),
        _telemetry_row(
            100,
            Graphics_player_pos_x=1.0,
            Graphics_player_pos_y=11.0,
            Graphics_player_pos_z=21.0,
            Physics_velocity_y=3.0,
            Physics_velocity_z=4.0,
            unexpected_feature=99,
        ),
        _telemetry_row(
            499,
            Graphics_player_pos_x=2.0,
            Graphics_player_pos_y=12.0,
            Graphics_player_pos_z=22.0,
            Physics_velocity_y=5.0,
            Physics_velocity_z=6.0,
            unexpected_feature=99,
        ),
        _telemetry_row(
            500,
            Graphics_player_pos_x=3.0,
            Graphics_player_pos_y=13.0,
            Graphics_player_pos_z=23.0,
            Physics_velocity_y=7.0,
            Physics_velocity_z=8.0,
            unexpected_feature=99,
        ),
        _telemetry_row(
            999,
            Graphics_player_pos_x=4.0,
            Graphics_player_pos_y=14.0,
            Graphics_player_pos_z=24.0,
            Physics_velocity_y=9.0,
            Physics_velocity_z=10.0,
            unexpected_feature=99,
        ),
    ]

    result = preprocess_inference_telemetry(source)

    expected_features = TelemetryFeatures.get_features_for_top_lap_reference()
    assert result.raw_indices == [2, 4]
    assert [row["Graphics_current_time"] for row in result.records] == [499, 999]
    assert list(result.records[0]) == expected_features
    assert "unexpected_feature" not in result.records[0]
    assert result.records[0]["Graphics_player_pos_y"] == pytest.approx(21.0)
    assert result.records[0]["Graphics_player_pos_z"] == pytest.approx(11.0)
    assert result.records[0]["Physics_velocity_y"] == pytest.approx(4.0)
    assert result.records[0]["Physics_velocity_z"] == pytest.approx(3.0)
    assert source[0]["Graphics_player_pos_y"] == 10.0
    assert source[0]["Graphics_player_pos_z"] == 20.0


def test_preprocess_inference_telemetry_removes_player_position_anomalies():
    source = [
        _telemetry_row(0, Graphics_player_pos_x=0.0),
        _telemetry_row(500, Graphics_player_pos_x=1_000.0),
        _telemetry_row(1_000, Graphics_player_pos_x=2.0),
    ]

    result = preprocess_inference_telemetry(source)

    assert result.raw_indices == [0, 2]
    assert [
        row["Graphics_player_pos_x"]
        for row in result.records
    ] == [0.0, 2.0]


def test_preprocess_inference_telemetry_rechecks_anomalies_after_downsampling():
    source = [
        _telemetry_row(0, Graphics_player_pos_x=0.0),
        _telemetry_row(499, Graphics_player_pos_x=124.75),
        _telemetry_row(500, Graphics_player_pos_x=125.0),
        _telemetry_row(501, Graphics_player_pos_x=124.75),
        _telemetry_row(1_000, Graphics_player_pos_x=0.0),
    ]

    result = preprocess_inference_telemetry(source)

    assert result.raw_indices == [1, 4]
    assert [
        row["Graphics_player_pos_x"]
        for row in result.records
    ] == [pytest.approx(62.375), 0.0]


def test_preprocess_inference_telemetry_requires_the_complete_feature_contract():
    with pytest.raises(ValueError, match="Missing features"):
        preprocess_inference_telemetry(
            [{"Graphics_current_time": 0, "Graphics_player_pos_x": 0.0}]
        )

    with pytest.raises(ValueError, match="Missing features"):
        preprocess_inference_telemetry([{"unexpected_feature": 1}])
