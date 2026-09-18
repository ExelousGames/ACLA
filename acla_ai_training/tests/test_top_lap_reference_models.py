from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.shared.expert_features import ExpertFeatureCatalog
from app.shared.telemetry import FeatureProcessor
from app.top_laps.runtime import RuntimeTopLapReferenceModel
from training.top_laps.service import TopLapReferenceModelService


def _top_lap(track: str = "spa", car: str = "car-a", grip: int = 2):
    return [
        {
            "Static_track": track,
            "Static_car_model": car,
            "Graphics_track_grip_status": grip,
            "Graphics_normalized_car_position": 0.0,
            "Graphics_player_pos_x": 0.0,
            "Graphics_player_pos_y": 1.0,
            "Graphics_player_pos_z": 2.0,
            "Physics_velocity_x": 10.0,
            "Physics_velocity_y": 0.0,
            "Physics_velocity_z": 0.0,
            "Physics_speed_kmh": 100.0,
            "Graphics_current_time": 0.0,
            "Physics_steer_angle": 0.1,
            "Physics_gas": 0.5,
            "Physics_brake": 0.0,
            "Physics_gear": 2,
        },
        {
            "Static_track": track,
            "Static_car_model": car,
            "Graphics_track_grip_status": grip,
            "Graphics_normalized_car_position": 1.0,
            "Graphics_player_pos_x": 10.0,
            "Graphics_player_pos_y": 3.0,
            "Graphics_player_pos_z": 4.0,
            "Physics_velocity_x": 20.0,
            "Physics_velocity_y": 0.0,
            "Physics_velocity_z": 0.0,
            "Physics_speed_kmh": 200.0,
            "Graphics_current_time": 1000.0,
            "Physics_steer_angle": 0.2,
            "Physics_gas": 1.0,
            "Physics_brake": 0.1,
            "Physics_gear": 4,
        },
    ]


def _training_payload(track: str = "spa", car: str = "car-a"):
    training = TopLapReferenceModelService()
    training.top_lap_store.record_lap(_top_lap(track=track, car=car))
    return training, training.serialize_reference_model()


def _runtime_record():
    return {
        "Graphics_track_grip_status": 5,
        "Graphics_normalized_car_position": 0.5,
        "Graphics_player_pos_x": 4.0,
        "Graphics_player_pos_y": 2.0,
        "Graphics_player_pos_z": 3.0,
        "Physics_velocity_x": 14.0,
        "Physics_velocity_y": 0.0,
        "Physics_velocity_z": 0.0,
        "Physics_speed_kmh": 140.0,
        "Graphics_current_time": 600.0,
    }


@pytest.mark.asyncio
async def test_service_builds_from_cached_top_laps():
    class TelemetryStore:
        def has_cached_data(self, cache_key):
            return cache_key == "top-laps"

        def get_cached_data_chunks(self, cache_key, include_ids=False):
            assert cache_key == "top-laps"
            assert include_ids is True
            return iter([([_top_lap()], "chunk-1")])

    service = TopLapReferenceModelService()
    result = await service.build_from_cached_top_laps(
        "top-laps", telemetry_store=TelemetryStore(),
    )

    assert result["reference_summary"]["reference_built"] == [
        "top_lap_store"
    ]
    assert result["metadata"]["total_training_samples"] == 2
    assert service.serialize_reference_model()["top_lap_store"]


def test_serialized_payload_preserves_all_buckets_through_backend_json():
    training = TopLapReferenceModelService()
    buckets = [("spa", "car-a", 2), ("spa", "car-a", 5), ("monza", "car-b", 3)]
    for track, car, grip in buckets:
        training.top_lap_store.record_lap(_top_lap(track, car, grip))

    payload = json.loads(json.dumps(training.serialize_reference_model()))

    assert set(payload) == {"top_lap_store"}
    assert set(payload["top_lap_store"]) == {
        f"{track}|{car}|grip{grip}" for track, car, grip in buckets
    }
    assert all(isinstance(value, str) for value in payload["top_lap_store"].values())
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(payload)

    assert set(runtime.top_lap_store.entries) == set(buckets)
    for key, expected in training.top_lap_store.entries.items():
        actual = runtime.top_lap_store.entries[key]
        assert actual.target_features == expected.target_features
        np.testing.assert_array_equal(actual.x, expected.x)
        np.testing.assert_array_equal(actual.y, expected.y)


def test_serializing_empty_reference_store_is_rejected():
    with pytest.raises(ValueError, match="No stored top laps to serialize"):
        TopLapReferenceModelService().serialize_reference_model()


def test_nearest_grip_runtime_values_match_pipeline_features():
    training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel()
    source = _runtime_record()

    runtime.install_backend_payload(payload)
    enriched = runtime.enrich([source], track="spa", car="car-a")

    training_input = {
        **source,
        "Static_track": "spa",
        "Static_car_model": "car-a",
    }
    training_features = training.extract_reference_features(
        [training_input]
    )[0]

    assert runtime.is_ready()
    assert source == _runtime_record()
    assert enriched[0]["Static_track"] == "spa"
    assert enriched[0]["Static_car_model"] == "car-a"
    assert {
        feature.value for feature in ExpertFeatureCatalog.ExpertFeatures
    }.issubset(enriched[0])
    for key, value in training_features.items():
        assert enriched[0][key] == pytest.approx(value)


def test_reference_service_loads_and_samples_serialized_model():
    _training, payload = _training_payload()
    loaded = TopLapReferenceModelService().load_reference_model(payload)

    result = loaded.sample_reference_actions(
        pd.DataFrame(
            [{
                **_runtime_record(),
                "Static_track": "spa",
                "Static_car_model": "car-a",
            }]
        )
    )

    assert result["optimal_actions"]["expert_optimal_speed"] == (
        pytest.approx(150.0)
    )


def test_resampled_lap_reference_aligns_start_middle_and_end():
    source = pd.DataFrame(
        {
            "Static_track": ["spa"] * 4,
            "Static_car_model": ["car-a"] * 4,
            "Graphics_track_grip_status": [2] * 4,
            "Graphics_completed_lap": [0] * 4,
            "Graphics_current_time": [0.0, 400.0, 800.0, 1_200.0],
            "Graphics_normalized_car_position": [0.0, 0.4, 0.8, 1.0],
            "Graphics_player_pos_x": [0.0, 4.0, 8.0, 10.0],
            "Graphics_player_pos_y": [0.0, 4.0, 2.0, 0.0],
            "Graphics_player_pos_z": [0.0] * 4,
            "Physics_velocity_x": [10.0] * 4,
            "Physics_velocity_y": [0.0] * 4,
            "Physics_velocity_z": [0.0] * 4,
            "Physics_speed_kmh": [100.0, 140.0, 180.0, 200.0],
            "Physics_steer_angle": [0.0, 0.2, -0.1, 0.0],
            "Physics_gas": [1.0] * 4,
            "Physics_brake": [0.0] * 4,
            "Physics_gear": [2, 3, 4, 5],
        }
    )
    resampled = FeatureProcessor(source).strip_dataframe_by_time_gap(source, 500)
    selected = resampled.iloc[[0, 1, -1]].to_dict("records")

    service = TopLapReferenceModelService()
    service.top_lap_store.record_lap(resampled.to_dict("records"))
    references = service.extract_reference_features(selected)

    assert [row["Graphics_current_time"] for row in selected] == [0.0, 500.0, 1_200.0]
    assert [row["Graphics_normalized_car_position"] for row in selected] == pytest.approx(
        [0.0, 0.5, 1.0]
    )
    assert [row["Graphics_player_pos_x"] for row in selected] == pytest.approx(
        [0.0, 5.0, 10.0]
    )
    assert [row["Graphics_player_pos_y"] for row in selected] == pytest.approx(
        [0.0, 3.5, 0.0]
    )
    assert [row["expert_optimal_time"] for row in references] == pytest.approx(
        [0.0, 500.0, 1_200.0]
    )
    assert [row["expert_optimal_player_pos_x"] for row in references] == pytest.approx(
        [0.0, 5.0, 10.0]
    )
    assert [row["expert_optimal_player_pos_y"] for row in references] == pytest.approx(
        [0.0, 3.5, 0.0]
    )
    assert [row["distance_to_expert_line"] for row in references] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
