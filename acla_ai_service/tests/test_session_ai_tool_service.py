import asyncio
from unittest.mock import AsyncMock

import pytest

from app.voice.session_ai_tool_service import (
    SessionAIToolService,
    SessionToolCatalogError,
)


def _tool(name="show_map"):
    return {
        "name": name,
        "description": "Display a circuit map.",
        "properties": {"map_id": {"type": "string"}},
        "required": ["map_id"],
    }


def _backend(response):
    backend = AsyncMock()
    backend.call_backend_function.return_value = response
    backend.establish_connection.return_value = True
    return backend


@pytest.mark.asyncio
async def test_get_session_tools_fetches_once_with_five_second_timeout():
    backend = _backend([_tool()])
    service = SessionAIToolService(backend)

    tools = await service.get_session_tools({
        "session_mode": "recorded",
        "agent_mode": "track_guide",
        "ignored": "value",
    })

    assert tools == [_tool()]
    backend.call_backend_function.assert_awaited_once_with(
        "session-tools",
        "POST",
        {
            "session_context": {
                "session_mode": "recorded",
                "agent_mode": "track_guide",
            },
        },
        timeout_seconds=5.0,
    )
    backend.establish_connection.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_session_tools_refreshes_authentication_and_retries_once():
    backend = _backend(None)
    backend.call_backend_function.side_effect = [
        {"error": "HTTP 401: Unauthorized"},
        [_tool()],
    ]

    tools = await SessionAIToolService(backend).get_session_tools({
        "session_mode": "live",
    })

    assert tools == [_tool()]
    assert backend.call_backend_function.await_count == 2
    backend.establish_connection.assert_awaited_once_with(max_retries=1)


@pytest.mark.asyncio
async def test_get_session_tools_reports_timeout_without_fallback():
    backend = _backend(None)
    backend.call_backend_function.side_effect = asyncio.TimeoutError

    with pytest.raises(SessionToolCatalogError, match="timed out"):
        await SessionAIToolService(backend).get_session_tools({
            "session_mode": "front_desk",
        })

    backend.establish_connection.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"tools": [_tool()]},
    [None],
    [{**_tool(), "title": "Must not be present"}],
    [{**_tool(), "description": None}],
    [{**_tool(), "properties": []}],
    [{**_tool(), "required": ["missing"]}],
])
async def test_get_session_tools_rejects_malformed_responses(response):
    with pytest.raises(SessionToolCatalogError):
        await SessionAIToolService(_backend(response)).get_session_tools({
            "session_mode": "user_summary",
        })


@pytest.mark.asyncio
@pytest.mark.parametrize("response, duplicate_name", [
    ([_tool("same"), _tool("same")], "same"),
    ([_tool("explain_label")], "explain_label"),
])
async def test_get_session_tools_rejects_duplicate_and_ai_owned_names(
    response,
    duplicate_name,
):
    with pytest.raises(SessionToolCatalogError, match=duplicate_name):
        await SessionAIToolService(_backend(response)).get_session_tools({
            "session_mode": "live",
        })


def test_get_ai_tools_returns_independent_copies_of_three_knowledge_tools():
    service = SessionAIToolService(_backend([]))
    first = service.get_ai_tools()
    second = service.get_ai_tools()

    assert {tool["name"] for tool in first} == {
        "explain_label",
        "get_track_knowledge",
        "search_racing_knowledge",
    }
    assert all(set(tool) == {"name", "description", "properties", "required"}
               for tool in first)
    first[0]["properties"].clear()
    assert second[0]["properties"]


def test_get_side_chat_tools_exposes_only_search_application_tool():
    service = SessionAIToolService(_backend([]))
    first = service.get_side_chat_tools()
    second = service.get_side_chat_tools()

    assert [tool["name"] for tool in first] == ["search_application_tool"]
    assert first[0]["required"] == []
    assert first[0]["properties"] == {}
    assert "parent conversation" in first[0]["description"]
    first[0]["description"] = "changed"
    assert second[0]["description"] != "changed"


@pytest.mark.asyncio
async def test_get_session_tools_rejects_reserved_side_chat_name():
    with pytest.raises(SessionToolCatalogError, match="search_application_tool"):
        await SessionAIToolService(
            _backend([_tool("search_application_tool")]),
        ).get_session_tools({"session_mode": "live"})
