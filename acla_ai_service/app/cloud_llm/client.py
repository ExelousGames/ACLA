"""Hosted (cloud) LLM client — Groq / any OpenAI-compatible endpoint.

Single place that builds the ``AsyncOpenAI`` client from ``settings.hosted_llm_*``.
The racing-engineer chatbot and the voice pipeline call ``make_hosted_client()``
when a hosted LLM is configured; otherwise they fall back to the local
llama-server sidecar.
"""

from __future__ import annotations

from typing import Optional, Tuple

from openai import AsyncOpenAI

from app.infra.config import settings


def make_hosted_client() -> Optional[Tuple[AsyncOpenAI, str]]:
    """Return ``(client, model)`` for the configured hosted LLM, or ``None``.

    Returns ``None`` when ``HOSTED_LLM_BASE_URL`` is unset (caller should fall
    back to the local llama-server). Raises ``RuntimeError`` when the base URL
    is set but the api key / model are missing — same fail-loud behaviour the
    chatbot relied on previously.
    """
    if not settings.hosted_llm_base_url:
        return None

    missing = [
        name for name, val in (
            ("HOSTED_LLM_API_KEY", settings.hosted_llm_api_key),
            ("HOSTED_LLM_MODEL", settings.hosted_llm_model),
        ) if not val
    ]
    if missing:
        raise RuntimeError(
            f"HOSTED_LLM_BASE_URL is set; also requires {', '.join(missing)}"
        )

    client = AsyncOpenAI(
        base_url=settings.hosted_llm_base_url,
        api_key=settings.hosted_llm_api_key,
    )
    return client, settings.hosted_llm_model
