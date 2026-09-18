"""Guidance uses the same remote providers as chat, without local fallback."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from openai import AsyncOpenAI
import pandas as pd
import pytest

from app.infra.config import settings
from app.racing_engineer import top_lap_reference_guidance as guidance


@pytest.fixture(params=["openai", "hosted"])
def cloud_provider(monkeypatch, request):
    monkeypatch.setattr(settings, "chat_llm_model", f"{request.param}:guidance-model")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "ACLA_TEST_CLOUD_API_KEY")
    monkeypatch.setenv("ACLA_TEST_CLOUD_API_KEY", "test-cloud-key")
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://hosted.example/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "test-cloud-key")
    return "api.openai.com" if request.param == "openai" else "hosted.example"


@pytest.fixture
def telemetry(monkeypatch):
    record = {"Physics_speed_kmh": 120.0}
    dataframe = pd.DataFrame([record])
    processor = Mock()
    processor.general_cleaning_for_analysis.return_value = dataframe
    processor.filter_features_by_list.return_value = dataframe
    monkeypatch.setattr(guidance, "FeatureProcessor", Mock(return_value=processor))
    monkeypatch.setattr(guidance, "get_top_lap_reference_model", lambda: SimpleNamespace(
        enrich=lambda records, **_kwargs: records,
    ))
    monkeypatch.setattr(guidance, "get_tire_grip_analysis", lambda: SimpleNamespace(
        extract_tire_grip_features=AsyncMock(return_value=[]),
    ))
    return record


def _mock_cloud(monkeypatch, *, status=200, content="Release the brake smoothly."):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "guidance-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
        })

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    monkeypatch.setattr(guidance, "AsyncOpenAI", lambda **kwargs: AsyncOpenAI(
        **kwargs, http_client=http_client, max_retries=0,
    ))
    return requests, http_client


@pytest.mark.asyncio
async def test_guidance_calls_selected_cloud_provider(monkeypatch, cloud_provider, telemetry):
    requests, client = _mock_cloud(monkeypatch)

    result = await guidance.generate_top_lap_reference_guidance(telemetry)

    assert result["status"] == "success"
    assert result["llm"]["raw_output"] == "Release the brake smoothly."
    assert len(requests) == 1
    assert requests[0].url.host == cloud_provider
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-cloud-key"
    assert json.loads(requests[0].content) == {
        "model": "guidance-model",
        "messages": [{"role": "user", "content": result["llm"]["user"]}],
    }
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status,content", [(503, "Unavailable"), (200, None)])
async def test_cloud_failure_returns_error_without_fallback(
    monkeypatch, cloud_provider, telemetry, status, content,
):
    requests, client = _mock_cloud(monkeypatch, status=status, content=content)

    result = await guidance.generate_top_lap_reference_guidance(telemetry)

    assert result["status"] == "error"
    assert "LLM generation failed" in result["error_message"]
    assert len(requests) == 1
    assert requests[0].url.host == cloud_provider
    assert client.is_closed


@pytest.mark.asyncio
async def test_missing_cloud_credentials_do_not_create_client(
    monkeypatch, cloud_provider, telemetry,
):
    monkeypatch.delenv("ACLA_TEST_CLOUD_API_KEY")
    monkeypatch.setattr(settings, "hosted_llm_api_key", None)
    create_client = Mock(side_effect=AssertionError("Must validate credentials first"))
    monkeypatch.setattr(guidance, "AsyncOpenAI", create_client)

    result = await guidance.generate_top_lap_reference_guidance(telemetry)

    assert result["status"] == "error"
    assert "requires" in result["error_message"]
    create_client.assert_not_called()
