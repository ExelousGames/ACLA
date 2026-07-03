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
        "Advance plan",
    )

    assert call_id
    assert sent == [{
        "type": "tool_call",
        "id": call_id,
        "name": "advance_plan_step",
        "title": "Advance plan",
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

    assert len(payloads) == 1
    assert json.loads(payloads[0]) == {
        "type": "tool_result",
        "id": "call-1",
        "name": "frontend_classifier",
        "result": {
            "ok": True,
            "label": "understeering_at_entry",
            "metadata": {"confidence": 0.92},
        },
    }


@pytest.mark.asyncio
async def test_oversized_tool_result_is_not_forwarded_to_ai_visible_payload():
    relay = ToolRelay(max_ai_visible_tool_payload_chars=300)
    conn = object()
    payloads = []

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(conn, send_text, payloads.append)

    relay.handle_text_frame(conn, {
        "type": "tool_result",
        "id": "call-1",
        "name": "live_range_tracker",
        "result": {
            "telemetry_rows": [{"speed": 120, "brake": 0.1}] * 100,
        },
    })

    assert len(payloads) == 1
    forwarded = json.loads(payloads[0])
    assert forwarded == {
        "ai_visible_payload_truncated": True,
        "id": "call-1",
        "max_payload_chars": 300,
        "message": (
            "Tool payload omitted because it exceeded the AI-visible size cap. "
            "Use a compact tool result or a server-side classifier path."
        ),
        "name": "live_range_tracker",
        "original_payload_chars": forwarded["original_payload_chars"],
        "result": {
            "message": (
                "Tool payload omitted because it exceeded the AI-visible size cap. "
                "Use a compact tool result or a server-side classifier path."
            ),
            "status": "omitted",
        },
        "type": "tool_result",
    }
    assert forwarded["original_payload_chars"] > 300
    assert "telemetry_rows" not in payloads[0]


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
