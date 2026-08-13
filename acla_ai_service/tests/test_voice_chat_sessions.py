from __future__ import annotations

from collections import deque
import json
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest

from app.api import voice
from app.voice import chat_sessions, pipecat_pipeline, tool_relay
from app.voice.chat_sessions import ChatSessionRegistry
from app.voice.tool_relay import ToolRelay


class _FakeWebSocket:
    def __init__(self, incoming=None):
        self.incoming = deque(incoming or [])
        self.accepted = False
        self.sent_json = []
        self.sent_text = []
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.incoming:
            return self.incoming.popleft()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def send_text(self, payload):
        self.sent_text.append(payload)

    async def close(self, code=1000, reason=None):
        self.closed = (code, reason)


def _frontend_info(session_context=None):
    return {
        "type": "websocket.receive",
        "text": json.dumps({
            "type": "frontend_info",
            "tools": [],
            "session_context": session_context or {},
        }),
    }


@pytest.mark.asyncio
async def test_handshake_sanitizes_legacy_mode_aliases():
    websocket = _FakeWebSocket([_frontend_info({
        "session_mode": "live",
        "agent_mode": "track_guide",
        "context_kind": "recorded",
        "active_agent_session": {"agent_mode": "overtake"},
        "agent_session": {"agent_mode": "live_performance_analyst"},
        "agent_modes": {"active": ["overtake"]},
        "active_screen": {
            "assistant_mode": "recorded",
            "label": "Live Session",
        },
    })])

    *_, session_context = await voice._await_frontend_info(
        websocket,
        timeout=1.0,
    )

    assert session_context == {
        "session_mode": "live",
        "agent_mode": "track_guide",
        "active_screen": {"label": "Live Session"},
    }


def test_startup_routing_ignores_conflicting_legacy_modes():
    assert pipecat_pipeline._startup_agent_behavior_name({}) == "front_desk"
    assert pipecat_pipeline._startup_agent_behavior_name({
        "session_mode": "live",
        "agent_mode": "track_guide",
        "context_kind": "recorded",
        "agent_session": {"agent_mode": "live_performance_analyst"},
    }) == "track_guide"
    assert pipecat_pipeline._startup_agent_behavior_name({
        "session_mode": "live",
        "active_agent_session": {"agent_mode": "overtake"},
        "active_screen": {"assistant_mode": "recorded"},
    }) == "live"


def test_prompt_context_excludes_legacy_mode_aliases():
    prompt = pipecat_pipeline._format_session_context_for_prompt({
        "session_mode": "recorded",
        "context_kind": "live",
        "active_agent_session": {"agent_mode": "overtake"},
        "agent_session": {"agent_mode": "track_guide"},
        "agent_modes": {"active": ["overtake"]},
        "active_screen": {"assistant_mode": "live", "label": "Analysis"},
    })

    assert '"session_mode": "recorded"' in prompt
    assert '"label": "Analysis"' in prompt
    assert "context_kind" not in prompt
    assert "active_agent_session" not in prompt
    assert "agent_session" not in prompt
    assert "agent_modes" not in prompt
    assert "assistant_mode" not in prompt


@pytest.fixture
def registry(monkeypatch):
    value = ChatSessionRegistry()
    monkeypatch.setattr(chat_sessions, "_CHAT_SESSION_REGISTRY", value)
    return value


def _install_fake_ai_service(monkeypatch):
    module = ModuleType("app.racing_engineer")

    class FakeAIService:
        def __init__(self, chat_llm_model=None):
            self.chat_llm_model = chat_llm_model

        async def _execute_function(self, function_name, arguments, context):
            return None

    module.AIService = FakeAIService
    monkeypatch.setitem(sys.modules, "app.racing_engineer", module)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "chat_session_id", "user_id", "error_type"),
    [
        (None, None, "user-1", "ChatSessionActionRequired"),
        ("legacy", None, "user-1", "InvalidChatSessionAction"),
        ("create", "client-id", "user-1", "ChatSessionIdNotAllowed"),
        ("resume", None, "user-1", "ChatSessionIdRequired"),
        ("resume", "", "user-1", "ChatSessionIdRequired"),
        ("create", None, None, "UserIdRequired"),
        ("create", None, "  ", "UserIdRequired"),
    ],
)
async def test_voice_stream_rejects_legacy_and_malformed_contracts(
    action,
    chat_session_id,
    user_id,
    error_type,
):
    websocket = _FakeWebSocket()

    await voice.voice_stream(
        websocket,
        session_id="telemetry-1",
        user_id=user_id,
        chat_llm_model=None,
        chat_session_action=action,
        chat_session_id=chat_session_id,
    )

    assert websocket.accepted is True
    assert websocket.sent_json[-1]["error_type"] == error_type
    assert websocket.closed[0] == 1008


@pytest.mark.asyncio
async def test_create_does_not_allocate_id_before_successful_handshake(
    monkeypatch,
    registry,
):
    generated_id = uuid.UUID("00000000-0000-4000-8000-000000000001")
    monkeypatch.setattr(chat_sessions.uuid, "uuid4", lambda: generated_id)
    websocket = _FakeWebSocket([{
        "type": "websocket.receive",
        "text": json.dumps({"type": "not_frontend_info"}),
    }])

    await voice.voice_stream(
        websocket,
        session_id=None,
        user_id="user-1",
        chat_llm_model=None,
        chat_session_action="create",
        chat_session_id=None,
    )

    assert websocket.sent_json[-1]["error_type"] == "HandshakeError"
    assert websocket.closed[0] == 1002
    assert registry.get(str(generated_id)) is None


@pytest.mark.asyncio
async def test_create_returns_server_uuid_and_is_active_while_pipeline_runs(
    monkeypatch,
    registry,
):
    _install_fake_ai_service(monkeypatch)
    active_states = []

    async def fake_run(websocket, config, tool_executor, **kwargs):
        active_states.append(registry.get(config.chat_session_id).active)

    monkeypatch.setattr(pipecat_pipeline, "run_voice_session", fake_run)

    identifiers = []
    for _ in range(2):
        websocket = _FakeWebSocket([_frontend_info()])
        await voice.voice_stream(
            websocket,
            session_id="telemetry-1",
            user_id="user-1",
            chat_llm_model=None,
            chat_session_action="create",
            chat_session_id=None,
        )
        ready = websocket.sent_json[0]
        assert ready["type"] == "chat_session_ready"
        assert ready["resumed"] is False
        uuid.UUID(ready["chat_session_id"])
        identifiers.append(ready["chat_session_id"])
        assert registry.get(ready["chat_session_id"]).active is False

    assert identifiers[0] != identifiers[1]
    assert active_states == [True, True]


@pytest.mark.asyncio
async def test_resume_uses_same_session_and_passes_stored_history(
    monkeypatch,
    registry,
):
    _install_fake_ai_service(monkeypatch)
    created = registry.create_attached("user-1")
    history = [
        {"role": "user", "content": "How was turn one?"},
        {"role": "assistant", "content": "You released the brake early."},
    ]
    registry.detach(created.chat_session_id, history)
    captured = []

    async def fake_run(websocket, config, tool_executor, **kwargs):
        captured.append(config)

    monkeypatch.setattr(pipecat_pipeline, "run_voice_session", fake_run)
    websocket = _FakeWebSocket([
        _frontend_info({"track_name": "spa", "connection": "latest"}),
    ])

    await voice.voice_stream(
        websocket,
        session_id="telemetry-2",
        user_id="user-1",
        chat_llm_model=None,
        chat_session_action="resume",
        chat_session_id=created.chat_session_id,
    )

    assert websocket.sent_json[0] == {
        "type": "chat_session_ready",
        "chat_session_id": created.chat_session_id,
        "resumed": True,
    }
    assert captured[0].committed_history == history
    assert captured[0].session_id == "telemetry-2"
    assert captured[0].chat_session_id == created.chat_session_id
    assert captured[0].session_context["connection"] == "latest"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "error_type"),
    [
        ("unknown", "ChatSessionNotFound"),
        ("cross_user", "ChatSessionOwnerMismatch"),
        ("active", "ChatSessionAlreadyActive"),
    ],
)
async def test_invalid_resumes_fail_without_replacing_sessions(
    registry,
    case,
    error_type,
):
    existing = registry.create_attached("owner")
    if case != "active":
        registry.detach(existing.chat_session_id)
    requested_id = (
        str(uuid.uuid4()) if case == "unknown" else existing.chat_session_id
    )
    requesting_user = "intruder" if case == "cross_user" else "owner"
    websocket = _FakeWebSocket()

    await voice.voice_stream(
        websocket,
        session_id=None,
        user_id=requesting_user,
        chat_llm_model=None,
        chat_session_action="resume",
        chat_session_id=requested_id,
    )

    assert websocket.sent_json[-1]["error_type"] == error_type
    assert websocket.closed[0] == 1008
    assert not any(
        frame.get("type") == "chat_session_ready"
        for frame in websocket.sent_json
    )
    if case == "unknown":
        assert registry.get(requested_id) is None
    assert registry.get(existing.chat_session_id).user_id == "owner"


def test_resumed_context_has_fresh_root_before_stored_history(monkeypatch):
    monkeypatch.setattr(
        pipecat_pipeline,
        "_build_system_prompt",
        lambda session_context, handling: f"fresh:{session_context['connection']}",
    )
    history = [
        {"role": "user", "content": "Prior question"},
        {"role": "assistant", "content": "Prior answer"},
    ]
    config = pipecat_pipeline.VoiceSessionConfig(
        chat_session_id="chat-1",
        committed_history=history,
        session_context={"connection": "new"},
        user_id="user-1",
    )

    messages, history_length = pipecat_pipeline._build_initial_context_messages(
        config,
        None,
    )

    assert messages == [
        {"role": "system", "content": "fresh:new"},
        *history,
    ]
    assert history_length == len(history)


def test_startup_prompt_keeps_neutral_procedure_plan_guidance():
    prompt = pipecat_pipeline._build_system_prompt({"session_mode": "front_desk"})

    assert "Procedure plan mode:" in prompt
    assert (
        "The application owns visible plan state and subscribed request execution."
        in prompt
    )
    assert "Tool calls are fire-and-forget." in prompt
    assert "advance_plan_step" not in prompt


def test_active_plan_prompt_does_not_direct_result_triggered_advancement():
    prompt = pipecat_pipeline._format_procedure_plan_for_prompt({
        "goal": "Review braking",
        "current_request": 0,
        "active_request": {"type": "tool_call", "name": "analyze_braking"},
        "requests": [{"type": "tool_call", "name": "analyze_braking"}],
    })

    assert "Procedure plan mode is active." in prompt
    assert '"goal": "Review braking"' in prompt
    assert "the application owns visible plan state" in prompt
    assert "Tool calls are fire-and-forget" in prompt
    assert "advance_plan_step" not in prompt


@pytest.mark.asyncio
async def test_tool_relay_routes_by_chat_session_after_rebinding():
    relay = ToolRelay()
    old_sent = []
    new_sent = []
    old_user_text = []
    new_user_text = []
    new_tool_results = []

    async def send_old(payload):
        old_sent.append(json.loads(payload))

    async def send_new(payload):
        new_sent.append(json.loads(payload))

    relay.bind("chat-1", send_old, old_user_text.append)
    relay.handle_text_frame("chat-1", {"type": "user_text", "text": "old"})
    await relay.send_tool_call("chat-1", "old_tool")
    relay.unbind("chat-1")

    relay.bind(
        "chat-1",
        send_new,
        new_user_text.append,
        tool_result_sink=new_tool_results.append,
    )
    relay.handle_text_frame("chat-1", {"type": "user_text", "text": "new"})
    relay.handle_text_frame(
        "chat-1",
        {"type": "tool_result", "id": "call-1", "result": {"ok": True}},
    )
    await relay.send_tool_call("chat-1", "new_tool")

    assert old_user_text == ["old"]
    assert new_user_text == ["new"]
    assert [frame["name"] for frame in old_sent] == ["old_tool"]
    assert [frame["name"] for frame in new_sent] == ["new_tool"]
    assert json.loads(new_tool_results[0])["id"] == "call-1"


def test_tool_relay_sanitizes_context_updates_and_typed_message_context():
    relay = ToolRelay()
    contexts = []
    user_text = []

    async def send_text(payload):
        return None

    relay.bind(
        "chat-1",
        send_text,
        user_text.append,
        session_context_sink=contexts.append,
    )
    relay.handle_text_frame("chat-1", {
        "type": "session_context",
        "session_context": {
            "session_mode": "live",
            "context_kind": "recorded",
            "active_screen": {"assistant_mode": "recorded", "label": "Live"},
        },
    })
    relay.handle_text_frame("chat-1", {
        "type": "user_text",
        "text": "How was that lap?",
        "session_context": {
            "session_mode": "live",
            "agent_mode": "live_performance_analyst",
            "agent_session": {"agent_mode": "track_guide"},
        },
    })

    assert contexts == [
        {
            "session_mode": "live",
            "active_screen": {"label": "Live"},
        },
        {
            "session_mode": "live",
            "agent_mode": "live_performance_analyst",
        },
    ]
    assert user_text == ["How was that lap?"]


@pytest.mark.asyncio
async def test_text_filter_routes_control_frames_with_chat_session_id(monkeypatch):
    relay = ToolRelay()
    monkeypatch.setattr(tool_relay, "_RELAY", relay)
    received = []

    async def send_text(payload):
        return None

    relay.bind("chat-1", send_text, received.append)
    websocket = _FakeWebSocket([
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "user_text", "text": "new transport"}),
        },
        {"type": "websocket.disconnect", "code": 1000},
    ])
    filtered = voice._TextFilteringWebSocket(websocket, "chat-1")

    disconnect = await filtered.receive()

    assert received == ["new transport"]
    assert disconnect["type"] == "websocket.disconnect"
    await filtered.stop_text_control_pump()


@pytest.mark.asyncio
async def test_pipeline_failure_persists_complete_turn_and_detaches(
    monkeypatch,
    registry,
):
    initial_history = [
        {"role": "user", "content": "Prior question"},
        {"role": "assistant", "content": "Prior answer"},
    ]
    created = registry.create_attached("user-1")
    registry.detach(created.chat_session_id, initial_history)
    registry.resume_attached(created.chat_session_id, "user-1")

    context = SimpleNamespace(messages=[
        {"role": "system", "content": "connection root"},
        *initial_history,
        {"role": "user", "content": "Completed question"},
        {"role": "assistant", "content": "Completed answer"},
        {"role": "user", "content": "Interrupted question"},
    ])
    task = SimpleNamespace(
        _acla_llm_context=context,
        _acla_initial_history_length=len(initial_history),
    )

    async def fake_build(*args, **kwargs):
        return task

    class FailingRunner:
        async def run(self, task):
            raise RuntimeError("pipeline failed")

    pipecat_module = ModuleType("pipecat")
    pipeline_package = ModuleType("pipecat.pipeline")
    runner_module = ModuleType("pipecat.pipeline.runner")
    runner_module.PipelineRunner = FailingRunner
    pipecat_module.pipeline = pipeline_package
    pipeline_package.runner = runner_module
    monkeypatch.setitem(sys.modules, "pipecat", pipecat_module)
    monkeypatch.setitem(sys.modules, "pipecat.pipeline", pipeline_package)
    monkeypatch.setitem(sys.modules, "pipecat.pipeline.runner", runner_module)
    monkeypatch.setattr(pipecat_pipeline, "build_voice_pipeline_task", fake_build)

    relay = ToolRelay()
    monkeypatch.setattr(tool_relay, "_RELAY", relay)

    async def send_text(payload):
        return None

    relay.bind(created.chat_session_id, send_text, lambda text: None)

    class FakeTransport:
        stopped = False

        async def stop_text_control_pump(self):
            self.stopped = True

    transport = FakeTransport()
    config = pipecat_pipeline.VoiceSessionConfig(
        chat_session_id=created.chat_session_id,
        committed_history=initial_history,
        user_id="user-1",
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await pipecat_pipeline.run_voice_session(
            transport,
            config,
            tool_executor=None,
        )

    stored = registry.get(created.chat_session_id)
    assert stored.active is False
    assert stored.committed_history == [
        *initial_history,
        {"role": "user", "content": "Completed question"},
        {"role": "assistant", "content": "Completed answer"},
    ]
    assert transport.stopped is True
    assert await relay.send_tool_call(created.chat_session_id, "after_failure") is None
