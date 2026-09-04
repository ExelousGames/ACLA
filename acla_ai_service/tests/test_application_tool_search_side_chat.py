from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.voice.application_tool_search_side_chat import (
    ApplicationToolSearchError,
    ApplicationToolSearchSideChat,
)


def _tool(name="show_map", *, required=None):
    required = ["map_id"] if required is None else required
    return {
        "name": name,
        "description": "Display a circuit map.",
        "properties": {
            "map_id": {
                "type": "string",
                "description": "Circuit map identifier.",
            },
            "zoom": {"type": "integer"},
        },
        "required": required,
    }


def _response(name="show_map", arguments='{"map_id":"spa"}'):
    return {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }],
            },
        }],
    }


def _side_chat(response=None):
    create = AsyncMock(return_value=response or _response())
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    return ApplicationToolSearchSideChat(client, "selector-model"), create


def _request(tools=None):
    return {
        "parent_messages": [
            {"role": "system", "content": "You are a race engineer."},
            {"role": "user", "content": "Show the Spa map at the default zoom."},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "name": "search_application_tool",
                        "arguments": "{}",
                    },
                }],
            },
        ],
        "session_context": {
            "session_mode": "recorded",
            "agent_mode": "track_guide",
        },
        "allowed_tools": tools or [_tool()],
    }


@pytest.mark.asyncio
async def test_side_chat_sends_only_isolated_selection_messages_and_full_catalog():
    side_chat, create = _side_chat()
    request = _request()

    selected = await side_chat.run(request)

    assert selected == {
        "name": "show_map",
        "arguments": {"map_id": "spa"},
    }
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "selector-model"
    assert kwargs["tool_choice"] == "required"
    assert len(kwargs["messages"]) == 1
    prompt = kwargs["messages"][0]["content"]
    serialized_parent = json.dumps(
        request["parent_messages"],
        ensure_ascii=True,
        sort_keys=True,
    )
    assert serialized_parent in prompt
    assert prompt.index(serialized_parent) < prompt.index("Selector request:")
    assert json.dumps(
        request["session_context"],
        ensure_ascii=True,
        sort_keys=True,
    ) in prompt
    assert json.dumps(
        request["allowed_tools"],
        ensure_ascii=True,
        sort_keys=True,
    ) in prompt
    assert kwargs["tools"] == [{
        "type": "function",
        "function": {
            "name": "show_map",
            "description": "Display a circuit map.",
            "parameters": {
                "type": "object",
                "properties": request["allowed_tools"][0]["properties"],
                "required": ["map_id"],
            },
        },
    }]


@pytest.mark.asyncio
async def test_side_chat_copies_selected_arguments():
    arguments = {"map_id": "spa"}
    side_chat, _ = _side_chat(_response(arguments=arguments))

    selected = await side_chat.run(_request())
    arguments["map_id"] = "monza"

    assert selected["arguments"] == {"map_id": "spa"}


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_messages", [None, [], ["not-a-message"]])
async def test_side_chat_requires_parent_messages_without_calling_provider(
    parent_messages,
):
    side_chat, create = _side_chat()
    request = _request()
    request["parent_messages"] = parent_messages

    with pytest.raises(ApplicationToolSearchError, match="parent_messages"):
        await side_chat.run(request)

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_side_chat_rejects_empty_catalog_without_calling_provider():
    side_chat, create = _side_chat()
    request = _request()
    request["allowed_tools"] = []

    with pytest.raises(ApplicationToolSearchError, match="No application tools"):
        await side_chat.run(request)

    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"choices": []}, "exactly one choice"),
        ({"choices": [{"message": {"tool_calls": []}}]}, "exactly one tool call"),
        (
            {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "show_map", "arguments": "{}"}},
                {"function": {"name": "show_map", "arguments": "{}"}},
            ]}}]},
            "exactly one tool call",
        ),
        (_response(name="unknown"), "unknown tool"),
        (_response(arguments="not-json"), "malformed arguments"),
        (_response(arguments="[]"), "JSON object"),
        (_response(arguments="{}"), "omitted required arguments"),
    ],
)
async def test_side_chat_rejects_invalid_selections(response, message):
    side_chat, _ = _side_chat(response)

    with pytest.raises(ApplicationToolSearchError, match=message):
        await side_chat.run(_request())


@pytest.mark.asyncio
async def test_side_chat_wraps_provider_failures():
    side_chat, create = _side_chat()
    create.side_effect = RuntimeError("provider offline")

    with pytest.raises(
        ApplicationToolSearchError,
        match="Side-chat provider request failed: provider offline",
    ):
        await side_chat.run(_request())


@pytest.mark.asyncio
async def test_side_chat_does_not_mutate_request_catalog():
    side_chat, _ = _side_chat()
    request = _request()
    original = deepcopy(request)

    await side_chat.run(request)

    assert request == original
