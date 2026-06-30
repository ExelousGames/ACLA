from types import SimpleNamespace
import importlib
import sys
import types

import pytest
from fastapi import APIRouter


@pytest.fixture
def startup_app(monkeypatch):
    fake_api = types.ModuleType("app.api")
    fake_api.annotation_router = APIRouter()
    fake_api.health_router = APIRouter()
    fake_api.racing_session_router = APIRouter()

    fake_voice = types.ModuleType("app.api.voice")
    fake_voice.router = APIRouter()

    monkeypatch.setitem(sys.modules, "app.api", fake_api)
    monkeypatch.setitem(sys.modules, "app.api.voice", fake_voice)
    sys.modules.pop("app.startup.app", None)

    module = importlib.import_module("app.startup.app")
    yield module

    sys.modules.pop("app.startup.app", None)


@pytest.fixture(autouse=True)
def quiet_startup(monkeypatch, startup_app):
    monkeypatch.setattr(startup_app, "start_chat_sidecar", lambda: None)

    async def fake_llama_health():
        return SimpleNamespace(
            reachable=False,
            base_url="http://llama",
            error="disabled in test",
            models=[],
            latency_ms=0.0,
        )

    monkeypatch.setattr(startup_app, "check_llama_server", fake_llama_health)


@pytest.mark.asyncio
async def test_lifespan_hydrates_chatbot_models_after_backend_connect(monkeypatch, startup_app):
    calls = []

    async def establish_connection():
        calls.append("backend")
        return True

    async def hydrate_chatbot_models():
        calls.append("hydrate")
        return {"segment_classifier": True}

    monkeypatch.setattr(startup_app.backend_service, "establish_connection", establish_connection)
    monkeypatch.setattr(startup_app, "hydrate_chatbot_models", hydrate_chatbot_models)

    async with startup_app.lifespan(SimpleNamespace()):
        pass

    assert calls == ["backend", "hydrate"]


@pytest.mark.asyncio
async def test_lifespan_skips_chatbot_model_hydration_when_backend_is_down(monkeypatch, startup_app):
    calls = []

    async def establish_connection():
        calls.append("backend")
        return False

    async def hydrate_chatbot_models():
        calls.append("hydrate")
        return {"segment_classifier": True}

    monkeypatch.setattr(startup_app.backend_service, "establish_connection", establish_connection)
    monkeypatch.setattr(startup_app, "hydrate_chatbot_models", hydrate_chatbot_models)

    async with startup_app.lifespan(SimpleNamespace()):
        pass

    assert calls == ["backend"]
