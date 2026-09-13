"""Annotation-only AI provider registry."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

from app.infra.config import settings
from app.annotation_providers.models import (
    AnnotationProvider,
    ProviderModel,
    ProviderOption,
)


class ProviderConfigurationError(RuntimeError):
    """Raised when the selected annotation provider cannot run."""


def _csv(value: str | None) -> List[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _configured_env(name: str | None) -> bool:
    return bool(name and os.getenv(name))


def _claude_provider() -> AnnotationProvider:
    from app.annotation_providers.claude_backend import CLAUDE_VLM_MODELS

    models = [
        ProviderModel(id=model_id, label=str(spec.get("label") or model_id))
        for model_id, spec in CLAUDE_VLM_MODELS.items()
    ]
    return AnnotationProvider(
        id="claude_cli",
        label="Claude CLI",
        runner="claude_cli",
        models=models,
        default_model=models[0].id if models else "",
        description="Claude Code / claude-agent-sdk session using local CLI auth.",
        options=[
            ProviderOption("use_thinking", "Use extended thinking", "checkbox", default=False),
        ],
    )


def _openai_provider() -> AnnotationProvider:
    models = [
        ProviderModel(id=model_id, label=model_id)
        for model_id in _csv(settings.annotation_openai_models)
    ]
    return AnnotationProvider(
        id="openai",
        label="OpenAI / ChatGPT",
        runner="openai_compatible",
        models=models,
        default_model=settings.annotation_openai_default_model or (models[0].id if models else ""),
        description="OpenAI ChatGPT models using the annotation tool-agent contract.",
        required_settings=[settings.annotation_openai_api_key_env],
        configured=_configured_env(settings.annotation_openai_api_key_env),
        options=[
            ProviderOption("api_key_env", "API key env var", "text",
                           default=settings.annotation_openai_api_key_env,
                           help="Environment variable containing the OpenAI API key."),
        ],
    )


def _openai_compatible_provider() -> AnnotationProvider:
    models = [
        ProviderModel(id=model_id, label=model_id)
        for model_id in _csv(settings.annotation_openai_compatible_models)
    ]
    default_model = (
        settings.annotation_openai_compatible_default_model
        or (models[0].id if models else "")
    )
    return AnnotationProvider(
        id="openai_compatible",
        label="OpenAI-Compatible",
        runner="openai_compatible",
        models=models,
        default_model=default_model,
        description="Any OpenAI-compatible annotation endpoint.",
        required_settings=[
            "ANNOTATION_OPENAI_COMPATIBLE_BASE_URL",
            settings.annotation_openai_compatible_api_key_env,
            "ANNOTATION_OPENAI_COMPATIBLE_MODEL",
        ],
        configured=bool(
            settings.annotation_openai_compatible_base_url
            and _configured_env(settings.annotation_openai_compatible_api_key_env)
            and default_model
        ),
        options=[
            ProviderOption("base_url", "Base URL", "text",
                           default=settings.annotation_openai_compatible_base_url or ""),
            ProviderOption("api_key_env", "API key env var", "text",
                           default=settings.annotation_openai_compatible_api_key_env),
        ],
    )


def _local_vlm_provider() -> AnnotationProvider:
    from app.local_annotation_agent.backend import QWEN25_VL_MODELS

    models = [
        ProviderModel(
            id=model_id,
            label=str(spec.get("label") or model_id),
            max_context=spec.get("max_context"),
            max_new_tokens=spec.get("max_new_tokens"),
        )
        for model_id, spec in QWEN25_VL_MODELS.items()
    ]
    return AnnotationProvider(
        id="local_vlm",
        label="Local VLM",
        runner="local_pipeline",
        models=models,
        default_model=models[0].id if models else "",
        description="Local llama.cpp-backed VLM using the annotation tool-agent contract.",
        options=[
            ProviderOption("gguf_path", "GGUF path", "text", default="", advanced=True),
            ProviderOption("mmproj_path", "MMProj path", "text", default="", advanced=True),
            ProviderOption("context_size", "Context size", "number", default=32768, advanced=True),
            ProviderOption("n_gpu_layers", "GPU layers", "number", default=-1, advanced=True),
            ProviderOption("quantization_type", "Quantization", "text", default="Q4_K_M", advanced=True),
        ],
    )


def _all_providers() -> List[AnnotationProvider]:
    return [
        _local_vlm_provider(),
        _claude_provider(),
        _openai_provider(),
        _openai_compatible_provider(),
    ]


def _enabled_ids(all_ids: Iterable[str]) -> List[str]:
    all_id_set = set(all_ids)
    explicit = _csv(settings.annotation_enabled_providers)
    if explicit:
        return [provider_id for provider_id in explicit if provider_id in all_id_set]
    ids = ["local_vlm", "claude_cli", "openai"]
    for provider in _all_providers():
        if provider.id in ids:
            continue
        if provider.configured:
            ids.append(provider.id)
    return [provider_id for provider_id in ids if provider_id in all_id_set]


@lru_cache(maxsize=1)
def list_annotation_providers() -> List[AnnotationProvider]:
    providers = _all_providers()
    allowed = set(_enabled_ids(provider.id for provider in providers))
    return [provider for provider in providers if provider.id in allowed]


def get_annotation_provider(provider_id: str) -> AnnotationProvider:
    provider_id = (provider_id or "").strip()
    for provider in list_annotation_providers():
        if provider.id == provider_id:
            return provider
    available = ", ".join(p.id for p in list_annotation_providers()) or "(none)"
    raise ProviderConfigurationError(
        f"Unknown annotation provider {provider_id!r}. Available providers: {available}"
    )


def validate_provider_ready(provider: AnnotationProvider) -> None:
    if provider.configured:
        return
    missing = ", ".join(provider.required_settings) or "provider configuration"
    raise ProviderConfigurationError(
        f"Annotation provider {provider.id!r} is enabled but not configured. "
        f"Missing: {missing}"
    )


def clear_provider_cache() -> None:
    list_annotation_providers.cache_clear()
