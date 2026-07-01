import importlib.util
import json
from pathlib import Path

import pytest


def _load_await_frontend_info():
    voice_path = Path(__file__).resolve().parents[1] / "app" / "api" / "voice.py"
    spec = importlib.util.spec_from_file_location("voice_api_under_test", voice_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._await_frontend_info


_await_frontend_info = _load_await_frontend_info()


class FakeWebSocket:
    def __init__(self, payload):
        self.payload = payload

    async def receive(self):
        return {"text": json.dumps(self.payload)}


@pytest.mark.asyncio
async def test_frontend_info_accepts_user_summary_session_mode():
    tools, tool_metadata, query_scope_schema, session_context = await _await_frontend_info(
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
            "session_context": {"session_mode": "user_summary"},
        }),
        timeout=1.0,
    )

    assert tools == []
    assert tool_metadata["get_next_corner"]["title"] == "Looking up next corner"
    assert query_scope_schema is None
    assert session_context == {"session_mode": "user_summary"}
