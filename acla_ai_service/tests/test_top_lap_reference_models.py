from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.top_laps.runtime import (
    RuntimeTopLapReferenceModel,
    TopLapReferenceModelError,
)
from app.top_laps.service import TopLapReferenceModelService
from app.ml import model_hub
from app.shared.expert_features import ExpertFeatureCatalog


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
async def test_service_builds_from_cached_top_laps(monkeypatch):
    class TelemetryStore:
        def has_cached_data(self, cache_key):
            return cache_key == "top-laps"

        def get_cached_data_chunks(self, cache_key, include_ids=False):
            assert cache_key == "top-laps"
            assert include_ids is True
            return iter([([_top_lap()], "chunk-1")])

    service = TopLapReferenceModelService()
    monkeypatch.setattr(
        service,
        "get_shared_data_cache",
        lambda: TelemetryStore(),
    )

    result = await service.build_from_cached_top_laps("top-laps")

    assert result["reference_summary"]["reference_built"] == [
        "top_lap_store"
    ]
    assert result["metadata"]["total_training_samples"] == 2
    assert service.serialize_reference_model()["top_lap_store"]


def test_nearest_grip_runtime_values_match_pipeline_features(tmp_path):
    training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
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
    assert json.loads(runtime.artifact_path.read_text()) == payload


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


def test_existing_static_identity_takes_precedence_over_request_fallback(tmp_path):
    _training, payload = _training_payload(track="telemetry-track", car="telemetry-car")
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
    runtime.install_backend_payload(payload)
    source = {
        **_runtime_record(),
        "Static_track": "telemetry-track",
        "Static_car_model": "telemetry-car",
    }

    enriched = runtime.enrich(
        [source],
        track="request-track",
        car="request-car",
    )

    assert enriched[0]["Static_track"] == "telemetry-track"
    assert enriched[0]["Static_car_model"] == "telemetry-car"


def test_unmatched_track_or_car_is_rejected(tmp_path):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
    runtime.install_backend_payload(payload)

    with pytest.raises(TopLapReferenceModelError):
        runtime.enrich(
            [_runtime_record()],
            track="unknown-track",
            car="car-a",
        )


def test_malformed_payload_does_not_replace_installed_artifact(tmp_path):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
    runtime.install_backend_payload(payload)
    installed_bytes = runtime.artifact_path.read_bytes()

    with pytest.raises(ValueError):
        runtime.install_backend_payload(
            {"top_lap_store": {"broken": "not-base64"}}
        )

    assert runtime.artifact_path.read_bytes() == installed_bytes
    assert runtime.is_ready()
    assert runtime.enrich([_runtime_record()], track="spa", car="car-a")


def test_old_payload_key_is_not_accepted(tmp_path):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")

    with pytest.raises(ValueError, match="top_lap_store"):
        runtime.install_backend_payload(
            {"fastest_lap_store": payload["top_lap_store"]}
        )

    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_startup_download_failure_never_loads_existing_artifact(
    tmp_path,
    monkeypatch,
):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
    runtime.install_backend_payload(payload)

    class FailingBackend:
        async def getCompleteActiveModelData(self, modelType):
            raise RuntimeError("backend unavailable")

    reference_spec = next(
        spec for spec in model_hub._MODEL_SPECS
        if spec.name == "top_lap_reference"
    )
    monkeypatch.setattr(model_hub, "_MODEL_SPECS", (reference_spec,))
    monkeypatch.setattr(
        model_hub,
        "get_top_lap_reference_model",
        lambda: runtime,
    )

    result = await model_hub.hydrate_chatbot_models(FailingBackend())

    assert result == {"top_lap_reference": False}
    assert runtime.artifact_path.exists()
    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_invalid_startup_payload_preserves_file_but_leaves_not_ready(
    tmp_path,
    monkeypatch,
):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")
    runtime.install_backend_payload(payload)
    installed_bytes = runtime.artifact_path.read_bytes()

    class Backend:
        async def getCompleteActiveModelData(self, modelType):
            return SimpleNamespace(
                modelData={"top_lap_store": {"broken": "not-base64"}}
            )

    reference_spec = next(
        spec for spec in model_hub._MODEL_SPECS
        if spec.name == "top_lap_reference"
    )
    monkeypatch.setattr(model_hub, "_MODEL_SPECS", (reference_spec,))
    monkeypatch.setattr(
        model_hub,
        "get_top_lap_reference_model",
        lambda: runtime,
    )

    result = await model_hub.hydrate_chatbot_models(Backend())

    assert result == {"top_lap_reference": False}
    assert runtime.artifact_path.read_bytes() == installed_bytes
    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_startup_installs_active_backend_payload(tmp_path, monkeypatch):
    _training, payload = _training_payload()
    runtime = RuntimeTopLapReferenceModel(tmp_path / "top_lap_store.json")

    class Backend:
        async def getCompleteActiveModelData(self, modelType):
            assert modelType == "top_lap_reference"
            return SimpleNamespace(modelData=payload)

    reference_spec = next(
        spec for spec in model_hub._MODEL_SPECS
        if spec.name == "top_lap_reference"
    )
    monkeypatch.setattr(model_hub, "_MODEL_SPECS", (reference_spec,))
    monkeypatch.setattr(
        model_hub,
        "get_top_lap_reference_model",
        lambda: runtime,
    )

    result = await model_hub.hydrate_chatbot_models(Backend())

    assert result == {"top_lap_reference": True}
    assert runtime.is_ready()
