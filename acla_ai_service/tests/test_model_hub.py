from types import SimpleNamespace
from pathlib import Path

import pytest

from app.ml import model_hub


class FakeBackend:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = set(failures or [])

    async def getCompleteActiveModelData(self, modelType: str):
        self.calls.append(modelType)
        if modelType in self.failures:
            raise RuntimeError(f"{modelType} unavailable")
        return SimpleNamespace(modelData={"modelType": modelType})


class FakeStore:
    def __init__(self):
        self.entries = {}


class FakeImitationLearning:
    def __init__(self):
        self.fastest_lap_store = FakeStore()

    def deserialize_imitation_model(self, payload):
        self.fastest_lap_store.entries[("track", "car", 2)] = payload
        return self


class FakeSegmentClassifier:
    def __init__(self):
        self.model = None
        self.mlb = None
        self.scaler = None

    def deserialize_artifacts(self, payload):
        self.payload = payload

    def load_model(self):
        self.model = object()
        self.mlb = object()
        self.scaler = object()
        return True


class FakeOpportunityForecaster:
    def __init__(self):
        self.model = None
        self.scaler = None

    def deserialize_artifacts(self, payload):
        self.payload = payload

    def load_model(self):
        self.model = object()
        self.scaler = object()
        return True


class FakeTireGrip:
    def deserialize_tire_grip_model(self, payload):
        self.payload = payload
        return self


@pytest.fixture(autouse=True)
def reset_model_hub(monkeypatch):
    model_hub._hydration_status.clear()

    services = SimpleNamespace(
        segment=FakeSegmentClassifier(),
        opportunity=FakeOpportunityForecaster(),
        imitation=FakeImitationLearning(),
        tire=FakeTireGrip(),
    )
    monkeypatch.setattr(model_hub, "get_segment_classifier", lambda: services.segment)
    monkeypatch.setattr(model_hub, "get_opportunity_forecaster", lambda: services.opportunity)
    monkeypatch.setattr(model_hub, "get_expert_imitation_learning", lambda: services.imitation)
    monkeypatch.setattr(model_hub, "get_tire_grip_analysis", lambda: services.tire)

    yield services

    model_hub._hydration_status.clear()


def _patch_successful_hydrators(monkeypatch, services, calls):
    def hydrate_segment(payload):
        calls.append(("segment_classifier", payload["modelType"]))

    def load_segment():
        services.segment.model = object()
        services.segment.mlb = object()
        services.segment.scaler = object()
        return True

    def hydrate_opportunity(payload):
        calls.append(("opportunity_forecaster", payload["modelType"]))

    def load_opportunity():
        services.opportunity.model = object()
        services.opportunity.scaler = object()
        return True

    def hydrate_imitation(payload):
        calls.append(("imitation_learning", payload["modelType"]))
        services.imitation.fastest_lap_store.entries[("track", "car", 2)] = object()
        return services.imitation

    def hydrate_tire(payload):
        calls.append(("tire_grip_analysis", payload["modelType"]))
        return services.tire

    monkeypatch.setattr(services.segment, "deserialize_artifacts", hydrate_segment)
    monkeypatch.setattr(services.segment, "load_model", load_segment)
    monkeypatch.setattr(services.opportunity, "deserialize_artifacts", hydrate_opportunity)
    monkeypatch.setattr(services.opportunity, "load_model", load_opportunity)
    monkeypatch.setattr(services.imitation, "deserialize_imitation_model", hydrate_imitation)
    monkeypatch.setattr(services.tire, "deserialize_tire_grip_model", hydrate_tire)


@pytest.mark.asyncio
async def test_hydrate_chatbot_models_downloads_and_hydrates_each_model(monkeypatch, reset_model_hub):
    hydrate_calls = []
    _patch_successful_hydrators(monkeypatch, reset_model_hub, hydrate_calls)

    backend = FakeBackend()
    result = await model_hub.hydrate_chatbot_models(backend=backend)

    assert result == {
        "segment_classifier": True,
        "opportunity_forecaster": True,
        "imitation_learning": True,
        "tire_grip_analysis": True,
    }
    assert backend.calls == [
        "segment_classifier",
        "opportunity_forecaster",
        "imitation_learning",
        "tire_grip_analysis",
    ]
    assert hydrate_calls == [
        ("segment_classifier", "segment_classifier"),
        ("opportunity_forecaster", "opportunity_forecaster"),
        ("imitation_learning", "imitation_learning"),
        ("tire_grip_analysis", "tire_grip_analysis"),
    ]


@pytest.mark.asyncio
async def test_hydrate_chatbot_models_keeps_startup_resilient_per_model(monkeypatch, reset_model_hub):
    hydrate_calls = []
    _patch_successful_hydrators(monkeypatch, reset_model_hub, hydrate_calls)

    backend = FakeBackend(failures={"imitation_learning"})
    result = await model_hub.hydrate_chatbot_models(backend=backend)

    assert result == {
        "segment_classifier": True,
        "opportunity_forecaster": True,
        "imitation_learning": False,
        "tire_grip_analysis": True,
    }
    assert backend.calls == [
        "segment_classifier",
        "opportunity_forecaster",
        "imitation_learning",
        "tire_grip_analysis",
    ]
    assert ("imitation_learning", "imitation_learning") not in hydrate_calls


@pytest.mark.asyncio
async def test_hydrate_chatbot_models_skips_backend_for_ready_models(reset_model_hub):
    services = reset_model_hub
    services.segment.model = object()
    services.segment.mlb = object()
    services.segment.scaler = object()
    services.opportunity.model = object()
    services.opportunity.scaler = object()
    services.imitation.fastest_lap_store.entries[("track", "car", 2)] = object()
    model_hub._hydration_status["tire_grip_analysis"] = True

    backend = FakeBackend(failures={
        "segment_classifier",
        "opportunity_forecaster",
        "imitation_learning",
        "tire_grip_analysis",
    })
    result = await model_hub.hydrate_chatbot_models(backend=backend)

    assert result == {
        "segment_classifier": True,
        "opportunity_forecaster": True,
        "imitation_learning": True,
        "tire_grip_analysis": True,
    }
    assert backend.calls == []


def test_fastapi_runtime_model_usage_goes_through_hub():
    runtime_files = [
        "app/api/racing_session.py",
        "app/racing_engineer/expert_actions.py",
        "app/racing_engineer/service.py",
        "app/services/user_session_analysis.py",
    ]
    forbidden_patterns = [
        "from app.ml.segment_classifier.service import segment_classifier",
        "from app.ml.opportunity_forecaster import opportunity_forecaster",
        "model_cache_service.get_model_or_fetch",
        ".get_model_or_fetch(",
        "ExpertImitateLearningService(",
        "TireGripAnalysisService(",
    ]

    for file_name in runtime_files:
        source = Path(file_name).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in source, f"{file_name} must use app.ml.model_hub, found {pattern}"
