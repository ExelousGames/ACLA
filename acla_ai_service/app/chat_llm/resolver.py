"""Resolve the selected remote chat LLM model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


VALID_CHAT_LLM_MODEL_PROVIDERS = frozenset(("openai", "hosted"))


def normalize_chat_llm_model(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None
    normalized = str(model).strip()
    return normalized or None


def parse_chat_llm_model_selector(selector: Optional[str]) -> Tuple[str, str]:
    selected = normalize_chat_llm_model(selector)
    if not selected:
        raise RuntimeError(
            "CHAT_LLM_MODEL is required and must use '<provider>:<model>'."
        )

    provider, sep, model = selected.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not sep or not provider or not model:
        raise RuntimeError(
            "CHAT_LLM_MODEL must use '<provider>:<model>' "
            "(for example 'openai:gpt-5.5')."
        )
    if provider not in VALID_CHAT_LLM_MODEL_PROVIDERS:
        raise RuntimeError(
            "CHAT_LLM_MODEL provider must be one of: openai, hosted "
            f"(got {provider!r})"
        )
    return provider, model


def resolve_chat_llm_config(
    model: Optional[str] = None,
) -> ChatLLMConfig:
    provider, selected_model = parse_chat_llm_model_selector(
        model or settings.chat_llm_model,
    )
    if provider == "openai":
        from app.chat_llm.openai import resolve_openai_chat_llm_config

        return resolve_openai_chat_llm_config(selected_model)
    if provider == "hosted":
        from app.chat_llm.hosted import resolve_hosted_chat_llm_config

        return resolve_hosted_chat_llm_config(selected_model)
    raise AssertionError(f"Unhandled chat LLM provider: {provider!r}")
