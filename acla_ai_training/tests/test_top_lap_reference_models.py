from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pandas as pd
import pytest

from app.shared.expert_features import ExpertFeatureCatalog
from app.shared.telemetry import FeatureProcessor
from app.top_laps.runtime import RuntimeTopLapReferenceModel
from training.top_laps.service import TopLapReferenceModelService
from training.pipelines.training.pipeline.cleaning import _update_top_lap, process_single_chunk


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
    buckets = [("acc", "spa", "car-a", 2), ("acc", "spa", "car-a", 5), ("iracing_live", "spa", "car-a", 2), ("acc", "monza", "car-b", 3)]
    for game, track, car, grip in buckets:
        training.top_lap_store.record_lap(_top_lap(track, car, grip), game=game)

    payload = json.loads(json.dumps(training.serialize_reference_model()))

    assert set(payload) == {"top_lap_store"}
    assert set(payload["top_lap_store"]) == {
        f"{game}|{track}|{car}|grip{grip}" for game, track, car, grip in buckets
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


@pytest.mark.asyncio
async def test_cached_reference_laps_stay_separate_by_session_game():
    laps = []
    for game, speed in [("acc", 150.0), ("iracing_recorded", 250.0)]:
        laps.append([
            {**row, "game_recorded_from": game, "Physics_speed_kmh": speed}
            for row in _top_lap()
        ])

    class TelemetryStore:
        def has_cached_data(self, cache_key):
            return True

        def get_cached_data_chunks(self, **kwargs):
            return iter([(laps, "chunk-1")])

    service = TopLapReferenceModelService()
    result = await service.build_from_cached_top_laps(
        "top-laps", telemetry_store=TelemetryStore(),
    )
    assert len(result["metadata"]["buckets_recorded"]) == 2
    assert len(service.top_lap_store.entries) == 2
    assert result["reference_summary"]["position_summary"]["games_recorded"] == 2
    assert {bucket["game"] for bucket in result["metadata"]["buckets_recorded"]} == {
        "acc", "iracing_recorded",
    }
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(json.loads(json.dumps(service.serialize_reference_model())))

    for game, speed in [("acc", 150.0), ("iracing_recorded", 250.0)]:
        enriched = runtime.enrich([_runtime_record()], track="spa", car="car-a", game=game)
        assert enriched[0]["expert_optimal_speed"] == pytest.approx(speed)
        sampled = service.sample_reference_actions(pd.DataFrame(_top_lap()), game=game)
        assert sampled["optimal_actions"]["expert_optimal_speed"] == pytest.approx(speed)
        features = service.extract_reference_features(_top_lap(), game=game)
        assert features[0]["expert_optimal_speed"] == pytest.approx(speed)


@pytest.mark.asyncio
async def test_session_metadata_game_survives_download_cleaning_and_top_lap_selection(monkeypatch):
    from app.integrations.backend import client as backend_client

    metadata = [
        {"sessionId": "session-acc", "game_recorded_from": "acc", "chunkCount": 1, "dataPoints": 2},
        {"sessionId": "session-iracing", "game_recorded_from": "iracing_recorded", "chunkCount": 1, "dataPoints": 2},
    ]
    source = [
        {**row, "Graphics_completed_lap": 1, "game_recorded_from": "incorrect-row-game"}
        for row in _top_lap()
    ]
    backend = backend_client.BackendService()
    monkeypatch.setattr(backend, "call_backend_function", AsyncMock(side_effect=[
        {"downloadId": "download-1", "totalSessions": 2, "sessionMetadata": metadata},
        {},
    ]))
    monkeypatch.setattr(backend, "get_auth_headers", lambda: {})
    http_client = httpx.AsyncClient
    monkeypatch.setattr(backend_client.httpx, "AsyncClient", lambda **kwargs: http_client(
        **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=source)),
    ))
    cached_chunks = []

    class TelemetryStore:
        def has_cached_data(self, cache_key):
            return False

        async def cache_chunks_streaming(self, *, cache_key, chunks_iterator):
            async for chunk in chunks_iterator:
                cached_chunks.append(chunk)
            return True

    result = await backend.get_all_racing_sessions_streaming("sessions", data_cache=TelemetryStore())
    assert result["success"] is True
    assert len(cached_chunks) == 2
    top_laps = {}
    features = [name for name in source[0] if name != "game_recorded_from"]
    for (records, chunk_id), session in zip(cached_chunks, metadata):
        assert all(row["game_recorded_from"] == session["game_recorded_from"] for row in records)
        _, count, candidates, cleaned, error, _ = process_single_chunk(records, chunk_id, features, 500)
        assert error is None
        assert count == 1
        assert all(row["game_recorded_from"] == session["game_recorded_from"] for row in cleaned)
        for candidate in candidates:
            _update_top_lap(top_laps, candidate)
    assert set(top_laps) == {
        ("acc", "spa", "car-a", 2), ("iracing_recorded", "spa", "car-a", 2),
    }
    original_acc = top_laps[("acc", "spa", "car-a", 2)]
    faster_acc = {**original_acc, "lap_time_ms": original_acc["lap_time_ms"] - 1}
    _update_top_lap(top_laps, faster_acc)
    assert top_laps[("acc", "spa", "car-a", 2)] is faster_acc
    assert len(top_laps) == 2


def test_nearest_grip_does_not_use_another_games_reference():
    from app.top_laps.runtime import TopLapReferenceModelError

    _, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(payload)
    with pytest.raises(TopLapReferenceModelError, match="iracing_live"):
        runtime.enrich([_runtime_record()], track="spa", car="car-a", game="iracing_live")


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
