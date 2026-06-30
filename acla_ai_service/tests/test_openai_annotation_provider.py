import json
import os
import sys
from types import ModuleType
from unittest.mock import patch

from app.annotation_providers.openai_runner import run_openai_compatible
from app.annotation_providers.registry import (
    clear_provider_cache,
    list_annotation_providers,
)
from app.infra.config import settings
from app.shared.contracts import AgentRequest, NoopCallbacks, ProviderConfig


class _FakeMessage:
    content = ""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeFunction:
    name = "submit_result"
    arguments = '{"payload_json": "{\\"items\\": []}", "summary": "done"}'


class _FakeToolCall:
    id = "call_1"
    function = _FakeFunction()


class _FakeCompletions:
    def create(self, **kwargs):
        return _FakeResponse(_FakeMessage([_FakeToolCall()]))


class _FakeChat:
    completions = _FakeCompletions()


class _FakeOpenAI:
    chat = _FakeChat()

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeDirectMessage:
    content = (
        '{"label_ids": ["brands_hatch", "brands_hatch1", "MSP"], '
        '"reasoning": "The annotated range fits Brabham Straight and MSP."}'
    )
    tool_calls = []

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class _FakeDirectCompletions:
    def create(self, **kwargs):
        return _FakeResponse(_FakeDirectMessage())


class _FakeDirectChat:
    completions = _FakeDirectCompletions()


class _FakeDirectOpenAI:
    chat = _FakeDirectChat()

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_openai_provider_is_visible_without_api_key():
    clear_provider_cache()
    try:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            settings, "annotation_enabled_providers", None
        ):
            providers = list_annotation_providers()
    finally:
        clear_provider_cache()

    openai = next((provider for provider in providers if provider.id == "openai"), None)
    assert openai is not None
    assert openai.configured is False


def test_openai_runner_captures_submit_result():
    fake_openai_module = ModuleType("openai")
    fake_openai_module.OpenAI = _FakeOpenAI

    request = AgentRequest(
        provider_id="openai",
        config=ProviderConfig(
            provider_id="openai",
            model="gpt-4o",
            max_new_tokens=100,
            temperature=0.0,
        ),
        planner_prompt="Return an empty annotation result.",
        synth_prompt=lambda _state: ("", ""),
        df_ref=[],
        parent_start=0,
        parent_end=10,
        callbacks=NoopCallbacks(),
    )

    with patch.dict(sys.modules, {"openai": fake_openai_module}), patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}
    ):
        response = run_openai_compatible(request)

    assert response.verdict == "submitted"
    assert response.raw_response == '{"items": []}'
    assert response.attachments["synthesizer.summary"].content == "done"


def test_openai_runner_accepts_direct_json_result():
    fake_openai_module = ModuleType("openai")
    fake_openai_module.OpenAI = _FakeDirectOpenAI

    request = AgentRequest(
        provider_id="openai",
        config=ProviderConfig(
            provider_id="openai",
            model="gpt-4o",
            max_new_tokens=100,
            temperature=0.0,
        ),
        planner_prompt="Return a lap annotation result.",
        synth_prompt=lambda _state: ("", ""),
        df_ref=[],
        parent_start=0,
        parent_end=10,
        callbacks=NoopCallbacks(),
    )

    with patch.dict(sys.modules, {"openai": fake_openai_module}), patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}
    ):
        response = run_openai_compatible(request)

    assert response.verdict == "submitted"
    assert json.loads(response.raw_response) == {
        "label_ids": ["brands_hatch", "brands_hatch1", "MSP"],
        "reasoning": "The annotated range fits Brabham Straight and MSP.",
    }
    assert (
        response.attachments["synthesizer.summary"].content
        == "The annotated range fits Brabham Straight and MSP."
    )
