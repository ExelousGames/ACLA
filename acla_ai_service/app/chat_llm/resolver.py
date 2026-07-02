"""Resolve the configured remote chat LLM provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.infra.config import settings


@dataclass(frozen=True)
class ChatLLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None

    def openai_client_kwargs(self) -> Dict[str, str]:
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


VALID_CHAT_LLM_PROVIDERS = frozenset(("openai", "hosted"))


def normalize_chat_llm_provider(provider: Optional[str]) -> Optional[str]:
    if provider is None:
        return None
    normalized = str(provider).strip().lower()
    return normalized or None


def resolve_chat_llm_config(provider: Optional[str] = None) -> ChatLLMConfig:
    provider = normalize_chat_llm_provider(provider) or normalize_chat_llm_provider(
        settings.chat_llm_provider,
    )
    if provider == "openai":
        from app.chat_llm.openai import resolve_openai_chat_llm_config

        return resolve_openai_chat_llm_config()
    if provider == "hosted":
        from app.chat_llm.hosted import resolve_hosted_chat_llm_config

        return resolve_hosted_chat_llm_config()
    raise RuntimeError(
        "CHAT_LLM_PROVIDER must be one of: openai, hosted "
        f"(got {provider!r})"
    )
