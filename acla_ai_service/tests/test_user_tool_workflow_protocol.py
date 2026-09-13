from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.voice import pipecat_pipeline, tool_relay
from app.voice.application_tool_search_side_chat import ApplicationToolSearchSideChat
from app.voice.model_command_protocol_service import ModelCommandProtocolService
from app.voice.tool_relay import ToolRelay


@pytest.mark.asyncio
async def test_workflow_catalog_to_selector_to_relay_preserves_protocol(
    monkeypatch,
    user_workflow_case,
):
    descriptor = user_workflow_case["descriptor"]
    arguments = user_workflow_case["arguments"]
    original = deepcopy(user_workflow_case)
    backend = AsyncMock()
    backend.call_backend_function.return_value = [descriptor]
    session_context = {
        "session_mode": "live",
        "agent_mode": "live_performance_analyst",
    }
    model_commands = await ModelCommandProtocolService(backend).get_model_commands(
        session_context,
    )
    parent_tools, allowed_tools, model_command_names = (
        pipecat_pipeline._build_voice_tool_surfaces(model_commands)
    )
    assert [tool["name"] for tool in parent_tools] == ["search_application_tool"]
    assert allowed_tools[0] == descriptor

    create = AsyncMock(return_value={"choices": [{"message": {"tool_calls": [{
        "function": {
            "name": descriptor["name"],
            "arguments": json.dumps(arguments),
        },
    }]}}]})
    selector = ApplicationToolSearchSideChat(SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ), "selector-model")
    relay = ToolRelay()
    monkeypatch.setattr(tool_relay, "_RELAY", relay)
    frames = []

    async def send_text(payload):
        frames.append(json.loads(payload))

    relay.bind("workflow-chat", send_text, lambda text: None)
    callbacks = {}
    result_callback = AsyncMock()
    server_executor = AsyncMock()
    config = pipecat_pipeline.VoiceSessionConfig(
        chat_session_id="workflow-chat",
        session_context=session_context,
    )
    handler, _, _ = pipecat_pipeline._make_tool_handler(
        server_executor,
        config,
        "workflow-chat",
        model_command_names=model_command_names,
        allowed_tools=allowed_tools,
        application_tool_search=selector,
        parent_message_source=lambda: [{
            "role": "user",
            "content": "Create this workflow: " + json.dumps(arguments),
        }],
        pending_model_command_callbacks=callbacks,
    )

    await handler(SimpleNamespace(
        function_name="search_application_tool",
        arguments={},
        result_callback=result_callback,
    ))

    create.assert_awaited_once()
    function = create.await_args.kwargs["tools"][0]["function"]
    assert function["description"] == descriptor["description"]
    assert function["parameters"] == {
        "type": "object",
        "properties": descriptor["properties"],
        "required": descriptor["required"],
    }
    assert len(frames) == 1
    assert frames[0] == {
        "type": "tool_call",
        "id": frames[0]["id"],
        "name": descriptor["name"],
        "arguments": arguments,
    }
    server_executor.assert_not_awaited()
    result_callback.assert_not_awaited()
    assert callbacks == {frames[0]["id"]: result_callback}
    result = {"status": "ready", "data": {"goal": "Review", "requests": []}}
    await callbacks.pop(frames[0]["id"])(result)
    result_callback.assert_awaited_once_with(result)
    assert user_workflow_case == original


def test_startup_prompt_leaves_workflow_instructions_in_catalog(user_workflow_case):
    prompt = pipecat_pipeline._build_system_prompt({
        "session_mode": "live",
        "agent_mode": "live_performance_analyst",
    })

    assert "```json" not in prompt
    assert user_workflow_case["descriptor"]["name"] not in prompt
    assert "No legacy compatibility" not in prompt
    assert "Your only application-tool entry point" in prompt
