import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest

from app.voice import pipecat_pipeline
from app.voice.tool_relay import get_relay


class FakeConn:
    def __init__(self) -> None:
        self.sent = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class FakeParams:
    def __init__(self, function_name: str, arguments: dict) -> None:
        self.function_name = function_name
        self.arguments = arguments
        self.callback_payloads = []

    async def result_callback(self, payload) -> None:
        self.callback_payloads.append(payload)


@pytest.mark.asyncio
async def test_frontend_tool_call_is_fire_and_forget():
    conn = FakeConn()
    relay = get_relay()
    relay.bind(conn, conn.send_text, lambda data: None)

    async def tool_executor(function_name, arguments, context):
        raise AssertionError("frontend tool should not call server executor")

    handler, _, _ = pipecat_pipeline._make_tool_handler(
        tool_executor,
        SimpleNamespace(session_id="s1", session_context={}, user_id="u1"),
        conn,
        frontend_tool_names=frozenset(["advance_plan_step"]),
        tool_titles={"advance_plan_step": "Advance plan"},
    )

    try:
        params = FakeParams("advance_plan_step", {"reason": "ready"})
        await asyncio.wait_for(handler(params), timeout=0.2)
    finally:
        relay.unbind(conn)

    assert params.callback_payloads == []
    tool_calls = [frame for frame in conn.sent if frame.get("type") == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "advance_plan_step"
    assert tool_calls[0]["arguments"] == {"reason": "ready"}
    assert {frame.get("status") for frame in conn.sent if frame.get("type") == "tool_event"} == {
        "started",
        "dispatched",
    }


@pytest.mark.asyncio
async def test_server_tool_still_returns_result_callback():
    conn = FakeConn()
    calls = []

    async def tool_executor(function_name, arguments, context):
        calls.append((function_name, arguments, context))
        return {"ok": True}

    handler, _, _ = pipecat_pipeline._make_tool_handler(
        tool_executor,
        SimpleNamespace(session_id="s1", session_context={"mode": "test"}, user_id="u1"),
        conn,
        frontend_tool_names=frozenset(["advance_plan_step"]),
        tool_titles={"explain_label": "Explain label"},
    )

    params = FakeParams("explain_label", {"label_id": "MSP44"})
    await handler(params)

    assert params.callback_payloads == [{"ok": True}]
    assert calls[0][0] == "explain_label"
    assert calls[0][1] == {"label_id": "MSP44"}
    assert calls[0][2]["session_id"] == "s1"
    assert calls[0][2]["session_context"] == {"mode": "test"}
    assert calls[0][2]["user_id"] == "u1"
    assert calls[0][2]["_conn"] is conn


def test_user_text_sink_message_uses_frontend_supplied_native_messages():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "content": json.dumps({"type": "tool_result", "id": "tool-1"}),
        },
    ]
    text = json.dumps({
        "type": "tool_result",
        "id": "tool-1",
        "messages": messages,
    })

    assert pipecat_pipeline._llm_context_messages_from_user_text(text) == messages


def test_user_text_sink_message_keeps_plain_text_as_user_role():
    assert pipecat_pipeline._llm_context_messages_from_user_text("Box this lap") == [{
        "role": "user",
        "content": "Box this lap",
    }]


@pytest.mark.asyncio
async def test_frontend_function_tag_recovery_does_not_inject_tool_result(monkeypatch):
    class Frame:
        pass

    class LLMFullResponseStartFrame(Frame):
        pass

    class LLMFullResponseEndFrame(Frame):
        pass

    class TextFrame(Frame):
        def __init__(self, text: str) -> None:
            self.text = text

    class LLMRunFrame(Frame):
        pass

    class FrameProcessor:
        def __init__(self) -> None:
            self.pushed = []

        async def process_frame(self, frame, direction) -> None:
            _ = frame, direction

        async def push_frame(self, frame, direction) -> None:
            self.pushed.append((frame, direction))

    frames_module = types.ModuleType("pipecat.frames.frames")
    frames_module.Frame = Frame
    frames_module.LLMFullResponseStartFrame = LLMFullResponseStartFrame
    frames_module.LLMFullResponseEndFrame = LLMFullResponseEndFrame
    frames_module.TextFrame = TextFrame
    frames_module.LLMRunFrame = LLMRunFrame
    monkeypatch.setitem(sys.modules, "pipecat.frames.frames", frames_module)

    processor_module = types.ModuleType("pipecat.processors.frame_processor")
    processor_module.FrameDirection = object
    processor_module.FrameProcessor = FrameProcessor
    monkeypatch.setitem(sys.modules, "pipecat.processors.frame_processor", processor_module)

    sent = []

    async def send_frontend_tool(name, args):
        sent.append((name, args))
        return "call_id"

    async def dispatch_server_tool(name, args):
        raise AssertionError("frontend recovery should not dispatch server tool")

    class Context:
        def __init__(self) -> None:
            self.messages = []

        def add_message(self, message) -> None:
            self.messages.append(message)

    class Task:
        def __init__(self) -> None:
            self.frames = []

        async def queue_frame(self, frame) -> None:
            self.frames.append(frame)

    context = Context()
    task = Task()
    Recovery = pipecat_pipeline._build_function_tag_recovery()
    processor = Recovery(
        send_frontend_tool,
        dispatch_server_tool,
        frozenset(["advance_plan_step"]),
        context,
        lambda: task,
    )

    await processor.process_frame(LLMFullResponseStartFrame(), None)
    await processor.process_frame(
        TextFrame('<function=advance_plan_step>{"reason":"ready"}</function>'),
        None,
    )
    await processor.process_frame(LLMFullResponseEndFrame(), None)
    await asyncio.sleep(0)

    assert sent == [("advance_plan_step", {"reason": "ready"})]
    assert context.messages == []
    assert task.frames == []
