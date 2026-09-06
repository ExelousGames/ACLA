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
    assert "tool_choice" not in kwargs
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


@pytest.mark.asyncio
@pytest.mark.parametrize("serialized", [False, True], ids=["object", "json"])
async def test_side_chat_preserves_workflow_schema_and_arguments(
    user_workflow_case,
    serialized,
):
    descriptor = user_workflow_case["descriptor"]
    arguments = user_workflow_case["arguments"]
    original = deepcopy(user_workflow_case)
    side_chat, create = _side_chat(_response(
        name=descriptor["name"],
        arguments=json.dumps(arguments) if serialized else arguments,
    ))
    request = _request([descriptor])

    selected = await side_chat.run(request)

    assert selected == {"name": descriptor["name"], "arguments": arguments}
    function = create.await_args.kwargs["tools"][0]["function"]
    assert function == {
        "name": descriptor["name"],
        "description": descriptor["description"],
        "parameters": {
            "type": "object",
            "properties": descriptor["properties"],
            "required": descriptor["required"],
        },
    }
    prompt = create.await_args.kwargs["messages"][0]["content"]
    assert json.dumps([descriptor], ensure_ascii=True, sort_keys=True) in prompt
    assert user_workflow_case == original

    name = descriptor["name"]
    selected["arguments"][name]["tools"][0].clear()
    function["parameters"]["properties"][name]["properties"].clear()
    assert user_workflow_case == original


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_list", [False, True], ids=["unwrapped", "legacy"])
async def test_side_chat_rejects_creation_arguments_without_repeated_name(
    user_workflow_case,
    legacy_list,
):
    descriptor = user_workflow_case["descriptor"]
    name = descriptor["name"]
    arguments = deepcopy(user_workflow_case["arguments"][name])
    if legacy_list:
        children = arguments.pop("tools")
        if name == "set_procedure_plan":
            arguments["requests"] = [
                {"name": tool_name, "title": child["title"],
                 "payload": {"arguments": child["arguments"]}}
                for entry in children for tool_name, child in entry.items()
            ]
        elif name == "create_repeatable_plan":
            arguments["steps"] = [
                {"name": tool_name, **child}
                for entry in children for tool_name, child in entry.items()
            ]
            tool_name, child = next(iter(arguments["stop_when"]["tool"].items()))
            arguments["stop_when"]["tool"] = {"name": tool_name, **child}
        else:
            arguments["events"] = [
                {"event": child["event"],
                 "tool": {"name": tool_name, "arguments": child["arguments"]}}
                for entry in children for tool_name, child in entry.items()
            ]
    side_chat, _ = _side_chat(_response(name=name, arguments=json.dumps(arguments)))

    with pytest.raises(ApplicationToolSearchError, match=(
        f"omitted required arguments for {name}: {name}"
    )):
        await side_chat.run(_request([descriptor]))


def test_selector_prompt_uses_catalog_guidance(user_workflow_case):
    side_chat, _ = _side_chat()
    descriptor = user_workflow_case["descriptor"]
    descriptor["description"] = "Use the workflow instructions supplied by this catalog."

    prompt = side_chat.task_prompt(_request([descriptor]))

    assert "Follow the selected tool's catalog description and argument schema" in prompt
    assert descriptor["description"] in prompt
    assert json.dumps([descriptor], ensure_ascii=True, sort_keys=True) in prompt
    assert "User workflow creation uses a strict tool-only input protocol" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("name, arguments", [
    ("advance_plan_step", {"reason": "The current request is complete."}),
    ("clear_procedure_plan", {"reason": "The driver cancelled."}),
    ("retry_repeatable_plan_task", {}),
    ("get_live_range_todo_list", {}),
])
async def test_side_chat_keeps_control_and_read_arguments_unwrapped(name, arguments):
    descriptor = {
        "name": name,
        "description": "Control or read an existing workflow.",
        "properties": {"reason": {"type": "string"}} if arguments else {},
        "required": [],
    }
    side_chat, _ = _side_chat(_response(name=name, arguments=json.dumps(arguments)))

    assert await side_chat.run(_request([descriptor])) == {
        "name": name,
        "arguments": arguments,
    }
