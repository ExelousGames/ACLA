import pytest

from app.chat_llm import ChatLLMConfig, resolve_chat_llm_config
from app.infra.config import settings
from app.racing_engineer import service as racing_service
from app.voice import pipecat_pipeline


def test_openai_provider_reads_default_api_key_env(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "openai")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setattr(settings, "chat_openai_model", "gpt-5.5")
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


def test_openai_provider_supports_custom_api_key_env(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "openai")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "ACLA_CHAT_OPENAI_KEY")
    monkeypatch.setattr(settings, "chat_openai_model", "gpt-5.5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ACLA_CHAT_OPENAI_KEY", "custom-key")

    config = resolve_chat_llm_config()

    assert config.api_key == "custom-key"


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "openai")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "MISSING_OPENAI_KEY")
    monkeypatch.setattr(settings, "chat_openai_model", "gpt-5.5")
    monkeypatch.delenv("MISSING_OPENAI_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_OPENAI_KEY"):
        resolve_chat_llm_config()


def test_openai_provider_requires_model(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "openai")
    monkeypatch.setattr(settings, "chat_openai_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setattr(settings, "chat_openai_model", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="CHAT_OPENAI_MODEL"):
        resolve_chat_llm_config()


def test_hosted_provider_validates_existing_hosted_settings(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "hosted")
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "hosted-key")
    monkeypatch.setattr(settings, "hosted_llm_model", "hosted-model")

    config = resolve_chat_llm_config()

    assert config == ChatLLMConfig(
        provider="hosted",
        base_url="https://example.test/v1",
        api_key="hosted-key",
        model="hosted-model",
    )


def test_provider_override_selects_hosted_without_changing_default(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "openai")
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "hosted-key")
    monkeypatch.setattr(settings, "hosted_llm_model", "hosted-model")

    config = resolve_chat_llm_config("hosted")

    assert config.provider == "hosted"
    assert config.model == "hosted-model"


def test_hosted_provider_requires_all_hosted_settings(monkeypatch):
    monkeypatch.setattr(settings, "chat_llm_provider", "hosted")
    monkeypatch.setattr(settings, "hosted_llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "hosted_llm_api_key", "")
    monkeypatch.setattr(settings, "hosted_llm_model", None)

    with pytest.raises(RuntimeError, match="HOSTED_LLM_API_KEY, HOSTED_LLM_MODEL"):
        resolve_chat_llm_config()


@pytest.mark.parametrize("provider", ["local", "bogus", ""])
def test_invalid_chat_provider_is_rejected(monkeypatch, provider):
    monkeypatch.setattr(settings, "chat_llm_provider", provider)

    with pytest.raises(RuntimeError, match="openai, hosted"):
        resolve_chat_llm_config()


def test_racing_engineer_service_uses_chat_llm_resolver(monkeypatch):
    captured = {}
    provider_calls = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_resolver(provider=None):
        provider_calls.append(provider)
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

    service = racing_service.AIService(chat_llm_provider="hosted")

    assert provider_calls == ["hosted"]
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
        lambda provider=None: ChatLLMConfig(
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
        "temperature": 0.3,
        "max_tokens": 1000,
    }


def test_voice_llm_service_passes_session_provider_override(monkeypatch):
    captured = {}

    class FakeOpenAILLMService:
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_resolver(provider=None):
        captured["provider"] = provider
        return ChatLLMConfig(
            provider="hosted",
            base_url="https://example.test/v1",
            api_key="voice-key",
            model="voice-model",
        )

    monkeypatch.setattr(pipecat_pipeline, "resolve_chat_llm_config", fake_resolver)

    pipecat_pipeline._build_openai_llm_service(FakeOpenAILLMService, "hosted")

    assert captured == {"provider": "hosted"}
