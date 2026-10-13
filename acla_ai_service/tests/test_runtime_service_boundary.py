"""The live API must import without the local training workspace installed."""

import ast
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import httpx
import pytest


def test_runtime_source_does_not_import_training_modules():
    service_root = Path(__file__).resolve().parents[1]
    forbidden = (
        "training",
        "ui",
        "app.pipelines",
        "app.storage",
        "app.local_annotation_agent",
        "app.annotation_providers",
        "app.llama",
        "app.local_llm",
    )
    violations = []
    for path in (service_root / "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                continue
            for module in modules:
                if any(module == name or module.startswith(name + ".") for name in forbidden):
                    violations.append(f"{path.relative_to(service_root)}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)
    assert not (service_root / "app" / "storage").exists()
    assert not (service_root / "app" / "pipelines").exists()


def test_shared_model_adapters_import_without_training(tmp_path):
    service_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.util
import sys
from app.integrations.backend.client import BackendService
from app.ml.opportunity_forecaster.service import OpportunityForecasterService
from app.ml.segment_cropper.service import SegmentCropperService
from app.ml.transformer.model import ExpertActionTransformer
from app.top_laps.runtime import RuntimeTopLapReferenceModel

assert importlib.util.find_spec('training') is None
assert not hasattr(BackendService, 'save_ai_model')
assert not hasattr(OpportunityForecasterService, 'train')
assert not hasattr(RuntimeTopLapReferenceModel, 'build_from_cached_top_laps')
assert not hasattr(RuntimeTopLapReferenceModel, 'serialize_reference_model')
assert not any(name.startswith(('training', 'streamlit')) for name in sys.modules)
assert 'app.ml.transformer.scaler' in sys.modules
assert not any(name.startswith(('app.storage', 'app.pipelines')) for name in sys.modules)
"""],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(service_root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_live_api_and_classifier_do_not_load_training(tmp_path):
    service_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(service_root),
        "TELEMETRY_STORE_DIR": str(tmp_path / "telemetry"),
    }
    subprocess.run(
        [sys.executable, "-c", """
import importlib.util
import sys
from app.startup.app import app
from app.ml.segment_classifier.service import SegmentClassifierService

assert importlib.util.find_spec('training') is None
paths = {route.path for route in app.routes}
assert {'/health', '/voice/stream', '/racing-session/segment-classification'} <= paths
assert not any(path.startswith('/annotation') for path in paths)
assert not any(name.startswith(('training', 'streamlit', 'app.storage',
                                'app.pipelines')) for name in sys.modules)
"""],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (tmp_path / "telemetry").exists()


@pytest.mark.asyncio
async def test_cloud_startup_does_not_probe_or_spawn_local_llm(monkeypatch):
    import importlib

    startup = importlib.import_module("app.startup.app")
    monkeypatch.setattr(startup.settings, "chat_llm_model", "hosted:test-model")
    monkeypatch.setattr(startup.settings, "hosted_llm_base_url", "https://hosted.example/v1")
    monkeypatch.setattr(startup.settings, "hosted_llm_api_key", "test-cloud-key")
    connect = AsyncMock(return_value=True)
    hydrate = AsyncMock(return_value={"segment_classifier": True})
    close = AsyncMock()
    monkeypatch.setattr(startup.backend_service, "establish_connection", connect)
    monkeypatch.setattr(startup, "hydrate_chatbot_models", hydrate)
    monkeypatch.setattr(startup, "close_speech_core", close)
    get = AsyncMock(side_effect=AssertionError("Unexpected LLM health probe"))
    spawn = Mock(side_effect=AssertionError("Unexpected local LLM process"))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr(subprocess, "Popen", spawn)

    async with startup.lifespan(startup.app):
        connect.assert_awaited_once()
        hydrate.assert_awaited_once()

    get.assert_not_awaited()
    spawn.assert_not_called()
    close.assert_awaited_once()
