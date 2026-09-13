"""Reusable support for isolated, one-request LLM side chats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Type, TypeVar


T = TypeVar("T")


class SideAIChatError(RuntimeError):
    """Raised when an isolated side-chat request cannot produce a result."""


class SideAIChat(ABC, Generic[T]):
    """Run an asynchronous LLM request without sharing parent-chat messages."""

    error_type: Type[SideAIChatError] = SideAIChatError

    def __init__(self, llm_client: Any, model: str) -> None:
        self._llm_client = llm_client
        self._model = model

    async def run(self, request: Any) -> T:
        """Send a fresh isolated request and parse the subclass result."""
        messages = [{"role": "system", "content": self.task_prompt(request)}]
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._model,
                messages=messages,
                **self.request_options(request),
            )
            return self.parse_result(response, request)
        except self.error_type:
            raise
        except Exception as exc:
            raise self.error_type(
                f"Side-chat provider request failed: {exc}",
            ) from exc

    @abstractmethod
    def task_prompt(self, request: Any) -> str:
        """Build the complete task prompt for one isolated request."""

    def request_options(self, request: Any) -> Dict[str, Any]:
        """Return provider request fields beyond model and messages."""
        return {}

    @abstractmethod
    def parse_result(self, response: Any, request: Any) -> T:
        """Parse and validate the provider response."""


__all__ = ["SideAIChat", "SideAIChatError"]
