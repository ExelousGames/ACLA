"""OpenAI chat LLM provider."""

from __future__ import annotations

import os
from app.chat_llm.resolver import ChatLLMConfig
from app.infra.config import settings

_OPENAI_BASE_URL = "https://api.openai.com/v1"


def resolve_openai_chat_llm_config(model: str) -> ChatLLMConfig:
    api_key_env = str(settings.chat_openai_api_key_env or "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"CHAT_LLM_MODEL provider 'openai' requires API key env var {api_key_env}."
        )

    selected_model = str(model or "").strip()
    if not selected_model:
        raise RuntimeError("CHAT_LLM_MODEL must include an OpenAI model name.")

    return ChatLLMConfig(
        provider="openai",
        base_url=_OPENAI_BASE_URL,
        api_key=api_key,
        model=selected_model,
    )
