from __future__ import annotations

import pandas as pd
import pytest

from app.pipelines.inference.preprocessing import (
    RAW_ROW_INDEX_COLUMN,
    preprocess_inference_telemetry,
)
from app.shared.telemetry import FeatureProcessor, TelemetryFeatures


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
    assert result.raw_indices == [0, 3, 4]
    assert [row["Graphics_current_time"] for row in result.records] == [0, 500, 999]
    assert list(result.records[0]) == expected_features
    assert "unexpected_feature" not in result.records[0]
    assert result.records[0]["Graphics_player_pos_y"] == pytest.approx(20.0)
    assert result.records[0]["Graphics_player_pos_z"] == pytest.approx(10.0)
    assert result.records[0]["Physics_velocity_y"] == pytest.approx(2.0)
    assert result.records[0]["Physics_velocity_z"] == pytest.approx(1.0)
    assert result.records[1]["Graphics_player_pos_y"] == pytest.approx(23.0)
    assert result.records[1]["Graphics_player_pos_z"] == pytest.approx(13.0)
    assert source[0]["Graphics_player_pos_y"] == 10.0
    assert source[0]["Graphics_player_pos_z"] == 20.0


def test_preprocess_inference_telemetry_removes_player_position_anomalies():
    source = [
        _telemetry_row(0, Graphics_player_pos_x=0.0),
        _telemetry_row(500, Graphics_player_pos_x=1_000.0),
        _telemetry_row(1_000, Graphics_player_pos_x=2.0),
    ]

    result = preprocess_inference_telemetry(source)

    assert result.raw_indices == [0, 0, 2]
    assert [
        row["Graphics_player_pos_x"]
        for row in result.records
    ] == [0.0, 1.0, 2.0]


def test_preprocess_inference_telemetry_rechecks_anomalies_after_downsampling():
    source = [
        _telemetry_row(0, Graphics_player_pos_x=0.0),
        _telemetry_row(499, Graphics_player_pos_x=125.75),
        _telemetry_row(500, Graphics_player_pos_x=126.0),
        _telemetry_row(501, Graphics_player_pos_x=125.75),
        _telemetry_row(1_000, Graphics_player_pos_x=0.0),
    ]

    result = preprocess_inference_telemetry(source)

    assert result.raw_indices == [0, 4]
    assert [
        row["Graphics_player_pos_x"]
        for row in result.records
    ] == [0.0, 0.0]


def test_downsampling_interpolates_one_coherent_irregular_timeline():
    times = [125, 425, 725, 1_125, 1_325]
    normalized_positions = [(time - 125) / 1_200 for time in times]
    source = pd.DataFrame(
        {
            "Graphics_current_time": times,
            "Graphics_normalized_car_position": normalized_positions,
            "Graphics_player_pos_x": [position * 12 for position in normalized_positions],
            "Graphics_player_pos_y": [position * -6 for position in normalized_positions],
            "Physics_speed_kmh": [position * 240 for position in normalized_positions],
            "Physics_gear": [1, 2, 3, 4, 5],
            "Graphics_is_valid_lap": [True, True, False, False, True],
            "Static_track": ["spa", "spa", "spa", "spa", "spa"],
            RAW_ROW_INDEX_COLUMN: ["0", "1", "2", "3", "4"],
        }
    )

    result = FeatureProcessor(source).strip_dataframe_by_time_gap(source, 500)

    assert list(result.columns) == list(source.columns)
    assert result["Graphics_current_time"].tolist() == [125, 625, 1_125, 1_325]
    assert result["Graphics_normalized_car_position"].tolist() == pytest.approx(
        [0.0, 5 / 12, 10 / 12, 1.0]
    )
    assert result["Graphics_player_pos_x"].tolist() == pytest.approx(
        [0.0, 5.0, 10.0, 12.0]
    )
    assert result["Graphics_player_pos_y"].tolist() == pytest.approx(
        [0.0, -2.5, -5.0, -6.0]
    )
    assert result["Physics_speed_kmh"].tolist() == pytest.approx(
        [0.0, 100.0, 200.0, 240.0]
    )
    assert result["Physics_gear"].tolist() == [1, 2, 4, 5]
    assert result["Graphics_is_valid_lap"].tolist() == [True, True, False, True]
    assert result["Static_track"].tolist() == ["spa"] * 4
    assert result[RAW_ROW_INDEX_COLUMN].tolist() == ["0", "2", "3", "4"]
    assert not any(column.startswith("__temp") for column in result.columns)


def test_downsampling_keeps_curved_coordinates_aligned_with_sample_time():
    source = pd.DataFrame(
        {
            "Graphics_current_time": [0, 750, 1_250, 2_000],
            "Graphics_normalized_car_position": [0.0, 0.375, 0.625, 1.0],
            "Graphics_player_pos_x": [0.0, 7.5, 12.5, 20.0],
            "Graphics_player_pos_y": [0.0, 6.0, 6.0, 0.0],
        }
    )

    result = FeatureProcessor(source).strip_dataframe_by_time_gap(source, 500)

    assert result["Graphics_current_time"].tolist() == [0, 500, 1_000, 1_500, 2_000]
    assert result["Graphics_normalized_car_position"].tolist() == pytest.approx(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    assert result["Graphics_player_pos_x"].tolist() == pytest.approx(
        [0.0, 5.0, 10.0, 15.0, 20.0]
    )
    assert result["Graphics_player_pos_y"].tolist() == pytest.approx(
        [0.0, 4.0, 6.0, 4.0, 0.0]
    )


def test_downsampling_handles_duplicate_timestamps_and_sparse_gaps():
    source = pd.DataFrame(
        {
            "Graphics_current_time": [0, 0, 1_600],
            "Graphics_normalized_car_position": [0.0, 0.0, 1.0],
            "Graphics_player_pos_x": [999.0, 0.0, 16.0],
            RAW_ROW_INDEX_COLUMN: ["0", "1", "2"],
        }
    )

    result = FeatureProcessor(source).strip_dataframe_by_time_gap(source, 500)

    assert result["Graphics_current_time"].tolist() == [0, 500, 1_000, 1_500, 1_600]
    assert result["Graphics_player_pos_x"].tolist() == pytest.approx(
        [0.0, 5.0, 10.0, 15.0, 16.0]
    )
    assert result["Graphics_normalized_car_position"].tolist() == pytest.approx(
        [0.0, 0.3125, 0.625, 0.9375, 1.0]
    )
    assert result[RAW_ROW_INDEX_COLUMN].tolist() == ["1", "1", "2", "2", "2"]


@pytest.mark.parametrize(
    ("source", "expected_times"),
    [
        (
            pd.DataFrame(
                {
                    "Graphics_current_time": [100, 900, 1_100, 1_800],
                    "Graphics_completed_lap": [0, 0, 1, 1],
                }
            ),
            [100, 600, 900, 1_100, 1_600, 1_800],
        ),
        (
            pd.DataFrame(
                {
                    "Graphics_current_time": [100, 700, 50, 650],
                    "Graphics_completed_lap": [0, 0, 0, 0],
                }
            ),
            [100, 600, 700, 50, 550, 650],
        ),
        (
            pd.DataFrame(
                {
                    "Graphics_current_time": [0, 400, 500, 1_100],
                    "Graphics_completed_lap": [0, 0, 0, 0],
                    "Graphics_normalized_car_position": [0.95, 0.99, 0.01, 0.2],
                }
            ),
            [0, 400, 500, 1_000, 1_100],
        ),
    ],
)
def test_downsampling_starts_a_new_grid_at_each_lap_boundary(source, expected_times):
    result = FeatureProcessor(source).strip_dataframe_by_time_gap(source, 500)

    assert result["Graphics_current_time"].tolist() == expected_times


def test_preprocess_inference_telemetry_requires_the_complete_feature_contract():
    with pytest.raises(ValueError, match="Missing features"):
        preprocess_inference_telemetry(
            [{"Graphics_current_time": 0, "Graphics_player_pos_x": 0.0}]
        )

    with pytest.raises(ValueError, match="Missing features"):
        preprocess_inference_telemetry([{"unexpected_feature": 1}])
