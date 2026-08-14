"""Retrieve and validate the tool catalog for one voice session."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Protocol, TypedDict

import httpx


class SessionAIToolDescriptor(TypedDict):
    """Pipecat-independent tool shape shared with the backend registry."""

    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str]


class SessionToolCatalogError(RuntimeError):
    """Raised when the backend session-tool catalog cannot be used safely."""


class _BackendClient(Protocol):
    async def call_backend_function(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 60.0,
    ) -> Any: ...

    async def establish_connection(self, max_retries: int = 3) -> bool: ...


class _UnauthorizedCatalogResponse(Exception):
    pass


_SESSION_MODES = frozenset({"front_desk", "live", "recorded", "user_summary"})
_AGENT_MODES = frozenset({"track_guide", "overtake", "live_performance_analyst"})
_DESCRIPTOR_FIELDS = frozenset({"name", "description", "properties", "required"})
_BACKEND_TIMEOUT_SECONDS = 5.0

_AI_SERVICE_TOOLS: tuple[SessionAIToolDescriptor, ...] = (
    {
        "name": "explain_label",
        "description": (
            "Look up an ACLA racing label or label code and return its "
            "plain-English definition and, when available, a coaching fix."
        ),
        "properties": {
            "label_id": {
                "type": "string",
                "description": (
                    "The label code or human-readable label name, for example "
                    "'MSP44' or 'Oversteering at entry'."
                ),
            },
        },
        "required": ["label_id"],
    },
    {
        "name": "get_track_knowledge",
        "description": (
            "Fetch keyed ACLA track notes for a known circuit, optionally "
            "focused on a specific corner."
        ),
        "properties": {
            "track": {
                "type": "string",
                "description": (
                    "The lowercase track id from the ACLA track corpus, such as "
                    "'spa'."
                ),
            },
            "corner": {
                "type": "string",
                "description": (
                    "Optional corner name or section to focus on, such as "
                    "'Eau Rouge'."
                ),
            },
        },
        "required": ["track"],
    },
    {
        "name": "search_racing_knowledge",
        "description": (
            "Search the ACLA racing knowledge corpus for free-text questions, "
            "driving theory, setup advice, track guidance, or knowledge that "
            "does not have an exact label or track id."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The natural-language racing question or topic to search for."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Optional maximum number of knowledge chunks to return.",
            },
        },
        "required": ["query"],
    },
)


class SessionAIToolService:
    """Own AI tools and fetch the backend-filtered browser-relayed tools."""

    def __init__(self, backend_client: Optional[_BackendClient] = None) -> None:
        self._backend_client = backend_client

    def get_ai_tools(self) -> List[SessionAIToolDescriptor]:
        """Return independent copies of the three AI-owned knowledge tools."""
        return [deepcopy(tool) for tool in _AI_SERVICE_TOOLS]

    async def get_session_tools(
        self,
        session_context: Dict[str, Any],
    ) -> List[SessionAIToolDescriptor]:
        """Fetch and validate the browser-relayed tools for one voice session."""
        canonical_context = self._validate_session_context(session_context)

        for attempt in range(2):
            try:
                response = await self._request_catalog(canonical_context)
            except _UnauthorizedCatalogResponse as exc:
                if attempt == 1:
                    raise SessionToolCatalogError(
                        "Backend rejected the refreshed AI service token",
                    ) from exc
                if not await self._refresh_authentication():
                    raise SessionToolCatalogError(
                        "Could not refresh the AI service backend token",
                    ) from exc
                continue

            return self._validate_catalog(response)

        raise SessionToolCatalogError("Backend session-tool lookup failed")

    async def _request_catalog(self, session_context: Dict[str, Any]) -> Any:
        backend_client = self._get_backend_client()
        try:
            response = await asyncio.wait_for(
                backend_client.call_backend_function(
                    "session-tools",
                    "POST",
                    {"session_context": deepcopy(session_context)},
                    timeout_seconds=_BACKEND_TIMEOUT_SECONDS,
                ),
                timeout=_BACKEND_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            raise SessionToolCatalogError(
                "Backend session-tool lookup timed out",
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise _UnauthorizedCatalogResponse() from exc
            raise SessionToolCatalogError(
                f"Backend session-tool lookup failed with HTTP {exc.response.status_code}",
            ) from exc
        except SessionToolCatalogError:
            raise
        except Exception as exc:
            raise SessionToolCatalogError(
                f"Backend session-tool lookup failed: {exc}",
            ) from exc

        if isinstance(response, dict) and "error" in response:
            error = str(response.get("error") or "unknown backend error")
            if error.lstrip().startswith("HTTP 401"):
                raise _UnauthorizedCatalogResponse()
            raise SessionToolCatalogError(
                f"Backend session-tool lookup failed: {error}",
            )
        return response

    async def _refresh_authentication(self) -> bool:
        backend_client = self._get_backend_client()
        try:
            return bool(await asyncio.wait_for(
                backend_client.establish_connection(max_retries=1),
                timeout=_BACKEND_TIMEOUT_SECONDS,
            ))
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            raise SessionToolCatalogError(
                "AI service backend token refresh timed out",
            ) from exc
        except Exception as exc:
            raise SessionToolCatalogError(
                f"AI service backend token refresh failed: {exc}",
            ) from exc

    def _get_backend_client(self) -> _BackendClient:
        if self._backend_client is None:
            from app.integrations.backend.client import backend_service

            self._backend_client = backend_service
        return self._backend_client

    @staticmethod
    def _validate_session_context(session_context: Any) -> Dict[str, Any]:
        if not isinstance(session_context, dict):
            raise SessionToolCatalogError("session_context must be an object")

        session_mode = session_context.get("session_mode")
        if not isinstance(session_mode, str) or session_mode not in _SESSION_MODES:
            raise SessionToolCatalogError("session_context.session_mode is invalid")

        agent_mode = session_context.get("agent_mode")
        if (
            agent_mode is not None
            and (not isinstance(agent_mode, str) or agent_mode not in _AGENT_MODES)
        ):
            raise SessionToolCatalogError("session_context.agent_mode is invalid")

        return {
            "session_mode": session_mode,
            **({"agent_mode": agent_mode} if agent_mode is not None else {}),
        }

    def _validate_catalog(self, response: Any) -> List[SessionAIToolDescriptor]:
        if not isinstance(response, list):
            raise SessionToolCatalogError("Backend session-tool response must be an array")

        names = {tool["name"] for tool in _AI_SERVICE_TOOLS}
        tools: List[SessionAIToolDescriptor] = []
        for index, raw_tool in enumerate(response):
            if not isinstance(raw_tool, dict) or set(raw_tool) != _DESCRIPTOR_FIELDS:
                raise SessionToolCatalogError(
                    f"Session tool at index {index} must contain exactly "
                    "name, description, properties, and required",
                )

            name = raw_tool.get("name")
            description = raw_tool.get("description")
            properties = raw_tool.get("properties")
            required = raw_tool.get("required")
            if not isinstance(name, str) or not name.strip():
                raise SessionToolCatalogError(
                    f"Session tool at index {index} has an invalid name",
                )
            name = name.strip()
            if name in names:
                raise SessionToolCatalogError(f"Duplicate tool name: {name}")
            if not isinstance(description, str):
                raise SessionToolCatalogError(
                    f"Session tool {name} has an invalid description",
                )
            if not isinstance(properties, dict):
                raise SessionToolCatalogError(
                    f"Session tool {name} has invalid properties",
                )
            if (
                not isinstance(required, list)
                or not all(isinstance(field, str) and field for field in required)
                or len(set(required)) != len(required)
                or any(field not in properties for field in required)
            ):
                raise SessionToolCatalogError(
                    f"Session tool {name} has an invalid required list",
                )

            names.add(name)
            tools.append({
                "name": name,
                "description": description,
                "properties": deepcopy(properties),
                "required": list(required),
            })

        return tools


__all__ = [
    "SessionAIToolDescriptor",
    "SessionAIToolService",
    "SessionToolCatalogError",
]
