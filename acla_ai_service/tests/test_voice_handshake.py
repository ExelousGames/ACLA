import importlib.util
import json
from pathlib import Path

import pytest


def _load_voice_api():
    voice_path = Path(__file__).resolve().parents[1] / "app" / "api" / "voice.py"
    spec = importlib.util.spec_from_file_location("voice_api_under_test", voice_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_voice_api = _load_voice_api()
_await_frontend_info = _voice_api._await_frontend_info
_HandshakeError = _voice_api._HandshakeError
_TextFilteringWebSocket = _voice_api._TextFilteringWebSocket


class FakeWebSocket:
    def __init__(self, payload):
        self.payload = payload

    async def receive(self):
        return {"text": json.dumps(self.payload)}


@pytest.mark.asyncio
async def test_frontend_info_accepts_user_summary_session_mode_and_legacy_result_handling():
    (
        tools,
        tool_metadata,
        query_scope_schema,
        tool_result_handling,
        session_context,
    ) = await _await_frontend_info(
        FakeWebSocket({
            "type": "frontend_info",
            "tools": [],
            "tool_metadata": {
                "get_next_corner": {
                    "title": "Looking up next corner",
                    "description": "Return the next corner.",
                    "parameters": {},
                },
            },
            "query_scope_schema": None,
            "tool_result_handling": (
                "Treat complete or ok=true as a successful result. "
                "Treat running as not ready yet."
            ),
            "session_context": {"session_mode": "user_summary"},
        }),
        timeout=1.0,
    )

    assert tools == []
    assert tool_metadata["get_next_corner"]["title"] == "Looking up next corner"
    assert query_scope_schema is None
    assert tool_result_handling == (
        "Treat complete or ok=true as a successful result. "
        "Treat running as not ready yet."
    )
    assert session_context == {"session_mode": "user_summary"}


@pytest.mark.asyncio
async def test_frontend_info_accepts_front_desk_session_mode_display_label():
    (
        tools,
        tool_metadata,
        query_scope_schema,
        tool_result_handling,
        session_context,
    ) = await _await_frontend_info(
        FakeWebSocket({
            "type": "frontend_info",
            "tools": [],
            "session_context": {"session_mode": "Front desk"},
        }),
        timeout=1.0,
    )

    assert tools == []
    assert tool_metadata == {}
    assert query_scope_schema is None
    assert tool_result_handling is None
    assert session_context == {"session_mode": "front_desk"}


@pytest.mark.asyncio
async def test_frontend_info_rejects_list_tool_result_handling():
    with pytest.raises(
        _HandshakeError,
        match="'tool_result_handling' must be a string or null",
    ):
        await _await_frontend_info(
            FakeWebSocket({
                "type": "frontend_info",
                "tools": [],
                "tool_result_handling": [
                    "Treat complete or ok=true as a successful result.",
                    "Treat running as not ready yet.",
                ],
                "session_context": {"session_mode": "user_summary"},
            }),
            timeout=1.0,
        )


class StreamingFakeWebSocket:
    def __init__(self):
        import asyncio

        self.messages = asyncio.Queue()

    async def receive(self):
        return await self.messages.get()


@pytest.mark.asyncio
async def test_text_filtering_websocket_routes_tool_result_without_audio_read():
    import asyncio

    from app.voice.tool_relay import get_relay

    raw_ws = StreamingFakeWebSocket()
    proxy = _TextFilteringWebSocket(raw_ws)
    relay = get_relay()
    payloads = asyncio.Queue()

    async def send_text(payload: str) -> None:
        _ = payload

    relay.bind(
        proxy,
        send_text,
        lambda text: None,
        tool_result_sink=payloads.put_nowait,
    )
    proxy.start_text_control_pump()
    try:
        await raw_ws.messages.put({
            "text": json.dumps({
                "type": "tool_result",
                "id": "call-1",
                "result": {"ok": True},
            }),
        })

        routed_text = await asyncio.wait_for(payloads.get(), timeout=1.0)
        assert json.loads(routed_text) == {
            "type": "tool_result",
            "id": "call-1",
            "result": {"ok": True},
        }

        await raw_ws.messages.put({"bytes": b"pcm"})
        assert await asyncio.wait_for(proxy.receive(), timeout=1.0) == {"bytes": b"pcm"}
    finally:
        await proxy.stop_text_control_pump()
        relay.unbind(proxy)
