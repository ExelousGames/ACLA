import json

import pytest

from app.voice.tool_relay import ToolRelay


@pytest.mark.asyncio
async def test_send_tool_call_sends_frame_and_returns_without_result():
    relay = ToolRelay()
    conn = object()
    sent = []

    async def send_text(payload: str) -> None:
        sent.append(json.loads(payload))

    relay.bind(conn, send_text, lambda data: None)

    call_id = await relay.send_tool_call(
        conn,
        "advance_plan_step",
        {"reason": "ready"},
    )

    assert call_id
    assert sent == [{
        "type": "tool_call",
        "id": call_id,
        "name": "advance_plan_step",
        "arguments": {"reason": "ready"},
    }]


@pytest.mark.asyncio
async def test_tool_result_is_forwarded_as_ai_visible_payload():
    relay = ToolRelay()
    conn = object()
    payloads = []

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(conn, send_text, payloads.append)

    relay.handle_text_frame(conn, {
        "type": "tool_result",
        "id": "call-1",
        "name": "frontend_classifier",
        "result": {
            "ok": True,
            "label": "understeering_at_entry",
            "metadata": {"confidence": 0.92},
        },
    })

    assert payloads == [{
        "type": "tool_result",
        "id": "call-1",
        "name": "frontend_classifier",
        "result": {
            "ok": True,
            "label": "understeering_at_entry",
            "metadata": {"confidence": 0.92},
        },
    }]


@pytest.mark.asyncio
async def test_unknown_frame_is_not_forwarded_to_ai_visible_payloads():
    relay = ToolRelay()
    conn = object()
    payloads = []

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(conn, send_text, payloads.append)

    relay.handle_text_frame(conn, {
        "type": "legacy_frame",
        "data": {"event": "legacy_frame"},
    })

    assert payloads == []


@pytest.mark.asyncio
async def test_tool_error_is_not_forwarded_to_ai_visible_payloads():
    relay = ToolRelay()
    conn = object()
    payloads = []

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(conn, send_text, payloads.append)

    relay.handle_text_frame(conn, {
        "type": "tool_error",
        "id": "call-1",
        "name": "frontend_classifier",
        "error": {"message": "failed"},
    })

    assert payloads == []
