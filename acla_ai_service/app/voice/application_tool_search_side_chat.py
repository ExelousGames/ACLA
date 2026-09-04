"""Isolated LLM selector for parent-requested application tool calls."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List, TypedDict

from app.chat_llm import resolve_chat_llm_config
from app.side_ai_chat import SideAIChat, SideAIChatError
from app.voice.session_ai_tool_service import SessionAIToolDescriptor


APPLICATION_TOOL_SEARCH_NAME = "search_application_tool"


class ApplicationToolSearchRequest(TypedDict):
    parent_messages: List[Dict[str, Any]]
    session_context: Dict[str, Any]
    allowed_tools: List[SessionAIToolDescriptor]


class SelectedToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]


class ApplicationToolSearchError(SideAIChatError):
    """Raised when application tool selection or validation fails."""


def _value(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


class ApplicationToolSearchSideChat(SideAIChat[SelectedToolCall]):
    """Select exactly one fully populated call from an allowed tool catalog."""

    error_type = ApplicationToolSearchError

    @classmethod
    def from_chat_llm_model(
        cls,
        chat_llm_model: str | None = None,
    ) -> "ApplicationToolSearchSideChat":
        from openai import AsyncOpenAI

        llm_config = resolve_chat_llm_config(chat_llm_model)
        return cls(
            AsyncOpenAI(**llm_config.openai_client_kwargs()),
            llm_config.model,
        )

    def task_prompt(self, request: ApplicationToolSearchRequest) -> str:
        parent_messages = request.get("parent_messages")
        if (
            not isinstance(parent_messages, list)
            or not parent_messages
            or not all(isinstance(message, dict) for message in parent_messages)
        ):
            raise ApplicationToolSearchError(
                "parent_messages must be a non-empty list of messages",
            )

        allowed_tools = request.get("allowed_tools")
        if not isinstance(allowed_tools, list) or not allowed_tools:
            raise ApplicationToolSearchError("No application tools are available")

        catalog = json.dumps(
            deepcopy(allowed_tools),
            ensure_ascii=True,
            sort_keys=True,
        )
        session_context = json.dumps(
            deepcopy(request.get("session_context") or {}),
            ensure_ascii=True,
            sort_keys=True,
        )
        parent_session = json.dumps(
            deepcopy(parent_messages),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        return (
            "You select application tools for an isolated parent chat. "
            "Choose exactly one tool from the allowed catalog that best fulfills "
            "the parent's request. Fill every required argument from the complete "
            "parent session and session context. Do not invent missing values, "
            "choose an unlisted tool, answer conversationally, or emit more than "
            "one tool call.\n\n"
            f"Complete allowed-tool catalog:\n{catalog}\n\n"
            f"Current normalized parent session context:\n{session_context}\n\n"
            f"Complete parent session messages:\n{parent_session}\n\n"
            "Selector request:\nChoose and call the one allowed application "
            "tool that best fulfills the latest request in the parent session."
        )

    def request_options(
        self,
        request: ApplicationToolSearchRequest,
    ) -> Dict[str, Any]:
        tools = []
        for descriptor in request["allowed_tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": descriptor["name"],
                    "description": descriptor["description"],
                    "parameters": {
                        "type": "object",
                        "properties": deepcopy(descriptor["properties"]),
                        "required": list(descriptor["required"]),
                    },
                },
            })
        return {
            "tools": tools,
            "tool_choice": "required",
        }

    def parse_result(
        self,
        response: Any,
        request: ApplicationToolSearchRequest,
    ) -> SelectedToolCall:
        choices = _value(response, "choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ApplicationToolSearchError(
                "Application tool selector must return exactly one choice",
            )

        message = _value(choices[0], "message")
        tool_calls = _value(message, "tool_calls", [])
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ApplicationToolSearchError(
                "Application tool selector must return exactly one tool call",
            )

        function = _value(tool_calls[0], "function")
        name = _value(function, "name")
        raw_arguments = _value(function, "arguments")
        if not isinstance(name, str) or not name:
            raise ApplicationToolSearchError(
                "Application tool selector returned an invalid tool name",
            )

        catalog = {
            descriptor["name"]: descriptor
            for descriptor in request["allowed_tools"]
        }
        descriptor = catalog.get(name)
        if descriptor is None:
            raise ApplicationToolSearchError(
                f"Application tool selector returned unknown tool: {name}",
            )

        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ApplicationToolSearchError(
                    f"Application tool selector returned malformed arguments for {name}",
                ) from exc
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise ApplicationToolSearchError(
                f"Application tool selector arguments for {name} must be a JSON object",
            )

        missing = [
            field for field in descriptor["required"]
            if field not in arguments
        ]
        if missing:
            raise ApplicationToolSearchError(
                f"Application tool selector omitted required arguments for {name}: "
                f"{', '.join(missing)}",
            )

        return {
            "name": name,
            "arguments": deepcopy(arguments),
        }


__all__ = [
    "APPLICATION_TOOL_SEARCH_NAME",
    "ApplicationToolSearchError",
    "ApplicationToolSearchRequest",
    "ApplicationToolSearchSideChat",
    "SelectedToolCall",
]
