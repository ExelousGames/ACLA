"""Lightweight reachability check for the llama-server sidecar.

The chat path (Phase 1) and voice path (Phase 3) both depend on llama-server
being up. This module hides the HTTP details so callers — health endpoints,
voice pipeline startup — can just call `await is_llama_server_healthy()`
and react to the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx

from app.core import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class LlamaHealth:
    """Status snapshot of the llama-server sidecar."""

    reachable: bool
    base_url: str
    models: List[str]
    latency_ms: Optional[float]
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "reachable": self.reachable,
            "base_url": self.base_url,
            "models": self.models,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def _models_endpoint() -> str:
    """Return `<llama_server_url>/models`, tolerating a trailing slash."""
    base = settings.llama_server_url.rstrip("/")
    return f"{base}/models"


async def check_llama_server() -> LlamaHealth:
    """Hit llama-server's `/v1/models` and report status.

    Never raises — failures are returned as `LlamaHealth(reachable=False, ...)`.
    Timeout is governed by `settings.llama_health_timeout_seconds`.
    """
    url = _models_endpoint()
    timeout = settings.llama_health_timeout_seconds

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)

        latency_ms = response.elapsed.total_seconds() * 1000.0
        if response.status_code != 200:
            return LlamaHealth(
                reachable=False,
                base_url=settings.llama_server_url,
                models=[],
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        payload = response.json()
        # OpenAI-compatible response shape: {"data": [{"id": "..."}, ...]}
        models = [
            entry.get("id", "")
            for entry in payload.get("data", [])
            if entry.get("id")
        ]
        return LlamaHealth(
            reachable=True,
            base_url=settings.llama_server_url,
            models=models,
            latency_ms=latency_ms,
            error=None,
        )

    except httpx.TimeoutException:
        return LlamaHealth(
            reachable=False,
            base_url=settings.llama_server_url,
            models=[],
            latency_ms=None,
            error=f"timeout after {timeout}s",
        )
    except Exception as exc:  # noqa: BLE001 — health checks must never raise
        LOGGER.debug("llama-server health check failed", exc_info=exc)
        return LlamaHealth(
            reachable=False,
            base_url=settings.llama_server_url,
            models=[],
            latency_ms=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def is_llama_server_healthy() -> bool:
    """Convenience boolean wrapper for places that don't need the details."""
    health = await check_llama_server()
    return health.reachable
