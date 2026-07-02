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
async def test_legacy_tool_result_is_ignored():
    relay = ToolRelay()
    conn = object()
    observations = []

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(conn, send_text, observations.append)

    relay.handle_text_frame(conn, {
        "type": "tool_result",
        "id": "legacy",
        "result": {"ok": True},
    })
    relay.handle_text_frame(conn, {
        "type": "observation",
        "data": {"text": "frontend data"},
    })

    assert observations == [{"text": "frontend data"}]

