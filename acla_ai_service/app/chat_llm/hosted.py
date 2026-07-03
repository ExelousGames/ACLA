"""Hosted OpenAI-compatible chat LLM provider."""

from __future__ import annotations

from app.chat_llm.resolver import ChatLLMConfig
from app.infra.config import settings


def resolve_hosted_chat_llm_config(model: str) -> ChatLLMConfig:
    selected_model = str(model or "").strip()
    missing = [
        name for name, val in (
            ("HOSTED_LLM_BASE_URL", settings.hosted_llm_base_url),
            ("HOSTED_LLM_API_KEY", settings.hosted_llm_api_key),
            ("CHAT_LLM_MODEL hosted model", selected_model),
        ) if not val
    ]
    if missing:
        raise RuntimeError(
            "CHAT_LLM_MODEL provider 'hosted' requires "
            f"{', '.join(missing)}"
        )

    return ChatLLMConfig(
        provider="hosted",
        base_url=str(settings.hosted_llm_base_url),
        api_key=str(settings.hosted_llm_api_key),
        model=selected_model,
    )
