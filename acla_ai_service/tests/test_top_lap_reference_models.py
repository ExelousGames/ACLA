from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ml import model_hub
from app.top_laps.model import TopLapStore
from app.top_laps.runtime import (
    RuntimeTopLapReferenceModel,
    TopLapReferenceModelError,
)
from app.top_laps.shared import serialize_top_lap_store


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


def _backend_payload(track: str = "spa", car: str = "car-a"):
    store = TopLapStore()
    store.record_lap(_top_lap(track=track, car=car))
    return serialize_top_lap_store(store)


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


def test_runtime_installs_and_enriches_without_filesystem_access(monkeypatch):
    payload = _backend_payload()

    def reject_filesystem_access(*args, **kwargs):
        raise AssertionError("Runtime top-lap inference must stay in memory")

    with monkeypatch.context() as filesystem:
        for target in ("builtins.open", "io.open", "os.open", "os.mkdir"):
            filesystem.setattr(target, reject_filesystem_access)
        runtime = RuntimeTopLapReferenceModel()
        runtime.install_backend_payload(payload)
        enriched = runtime.enrich([_runtime_record()], track="spa", car="car-a")

    assert runtime.is_ready()
    assert enriched[0]["expert_optimal_speed"] == pytest.approx(150.0)


def test_existing_static_identity_takes_precedence_over_request_fallback():
    payload = _backend_payload(track="telemetry-track", car="telemetry-car")
    runtime = RuntimeTopLapReferenceModel()
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


def test_unmatched_track_or_car_is_rejected():
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(payload)

    with pytest.raises(TopLapReferenceModelError):
        runtime.enrich(
            [_runtime_record()],
            track="unknown-track",
            car="car-a",
        )


def test_malformed_payload_does_not_replace_installed_reference():
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(payload)
    installed_store = runtime.top_lap_store

    with pytest.raises(ValueError):
        runtime.install_backend_payload(
            {"top_lap_store": {**payload["top_lap_store"], "broken": "not-base64"}}
        )

    assert runtime.top_lap_store is installed_store
    assert runtime.is_ready()
    enriched = runtime.enrich([_runtime_record()], track="spa", car="car-a")
    assert enriched[0]["expert_optimal_speed"] == pytest.approx(150.0)


def test_old_payload_key_is_not_accepted():
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()

    with pytest.raises(ValueError, match="top_lap_store"):
        runtime.install_backend_payload(
            {"fastest_lap_store": payload["top_lap_store"]}
        )

    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_startup_download_failure_clears_previous_reference(
    monkeypatch,
):
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()
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
    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_invalid_startup_payload_leaves_not_ready(
    monkeypatch,
):
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()
    runtime.install_backend_payload(payload)

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
    assert not runtime.is_ready()


@pytest.mark.asyncio
async def test_startup_installs_active_backend_payload(monkeypatch):
    payload = _backend_payload()
    runtime = RuntimeTopLapReferenceModel()

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
