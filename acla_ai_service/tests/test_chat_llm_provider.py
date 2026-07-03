import pytest

from app.chat_llm import (
    ChatLLMConfig,
    parse_chat_llm_model_selector,
    resolve_chat_llm_config,
)
from app.infra.config import settings
from app.racing_engineer import service as racing_service
from app.voice import pipecat_pipeline


def test_openai_selector_reads_default_api_key_env(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_model", "openai:gpt-5.5")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = resolve_chat_llm_config()

    assert config.provider == "openai"
    assert config.api_key == "test-key"
    assert config.model == "gpt-5.5"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.openai_client_kwargs() == {
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
    }


def test_openai_selector_supports_custom_api_key_env(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_model", "openai:gpt-5.5")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "ACLA_CHAT_OPENAI_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ACLA_CHAT_OPENAI_KEY", "custom-key")

    config = resolve_chat_llm_config()

    assert config.api_key == "custom-key"


def test_openai_selector_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_model", "openai:gpt-5.5")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "MISSING_OPENAI_KEY")
    monkeypatch.delenv("MISSING_OPENAI_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_OPENAI_KEY"):
        resolve_chat_llm_config()


def test_hosted_selector_uses_hosted_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "hosted-key")

    config = resolve_chat_llm_config("hosted:qwen/qwen3-32b")

    assert config == ChatLLMConfig(
        provider="hosted",
        base_url="https://example.test/v1",
        api_key="hosted-key",
        model="qwen/qwen3-32b",
    )


def test_model_selector_override_does_not_change_default_settings(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_model", "openai:default-model")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = resolve_chat_llm_config(" openai:selected-model ")

    assert config.provider == "openai"
    assert config.model == "selected-model"


def test_hosted_selector_requires_hosted_settings(monkeypatch):
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "")

    with pytest.raises(RuntimeError, match="HOSTED_LLM_API_KEY"):
        resolve_chat_llm_config("hosted:selected-hosted-model")


@pytest.mark.parametrize("selector", ["local:model", "bogus:model"])
def test_invalid_chat_selector_provider_is_rejected(selector):
    with pytest.raises(RuntimeError, match="openai, hosted"):
        parse_chat_llm_model_selector(selector)


@pytest.mark.parametrize("selector", ["", "gpt-5.5", "openai:", ":gpt-5.5"])
def test_invalid_chat_selector_shape_is_rejected(selector):
    with pytest.raises(RuntimeError, match="CHAT_LLM_MODEL"):
        parse_chat_llm_model_selector(selector)


def test_racing_engineer_service_uses_chat_llm_resolver(monkeypatch):
    captured = {}
    resolver_calls = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_resolver(model=None):
        resolver_calls.append(model)
        return ChatLLMConfig(
            provider="hosted",
            base_url="https://example.test/v1",
            api_key="resolved-key",
            model="resolved-model",
        )

    monkeypatch.setattr(
        racing_service,
        "resolve_chat_llm_config",
        fake_resolver,
    )
    monkeypatch.setattr(racing_service, "AsyncOpenAI", FakeAsyncOpenAI)

    service = racing_service.AIService(
        chat_llm_model="hosted:resolved-override",
    )

    assert resolver_calls == ["hosted:resolved-override"]
    assert captured == {
        "api_key": "resolved-key",
        "base_url": "https://example.test/v1",
    }
    assert service.chat_model == "resolved-model"


def test_voice_llm_service_uses_chat_llm_resolver(monkeypatch):
    class FakeOpenAILLMService:
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        pipecat_pipeline,
        "resolve_chat_llm_config",
        lambda model=None: ChatLLMConfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="voice-key",
            model="voice-model",
        ),
    )

    llm = pipecat_pipeline._build_openai_llm_service(FakeOpenAILLMService)

    assert llm.kwargs["base_url"] == "https://api.openai.com/v1"
    assert llm.kwargs["api_key"] == "voice-key"
    assert llm.kwargs["settings"].kwargs == {
        "model": "voice-model",
        "max_completion_tokens": 1000,
    }


def test_voice_llm_service_passes_session_model_selector(monkeypatch):
    captured = {}

    class FakeOpenAILLMService:
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_resolver(model=None):
        captured["model"] = model
        return ChatLLMConfig(
            provider="hosted",
            base_url="https://example.test/v1",
            api_key="voice-key",
            model="voice-model",
        )

    monkeypatch.setattr(pipecat_pipeline, "resolve_chat_llm_config", fake_resolver)

    pipecat_pipeline._build_openai_llm_service(
        FakeOpenAILLMService,
        "hosted:voice-model-override",
    )

    assert captured == {"model": "hosted:voice-model-override"}
