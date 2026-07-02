import sys
import types

import pytest

from app.external_knowledge_base import agent_behavior, behavior, reload
from app.voice import pipecat_pipeline


def test_live_performance_agent_behavior_knowledge_is_loaded():
    reload()

    assert "advance_plan_step" in behavior("procedure_plan")["_raw_body"]
    assert "frontend" not in behavior("procedure_plan")["_raw_body"].lower()
    assert "collecting_baseline" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "get_live_focus_section" not in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "live_analysis_plan_started" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "calling\n  `set_procedure_plan`" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "baseline_collection" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "live_recorded_analysis" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "Do not expect the\n  frontend to provide this startup plan" in agent_behavior("live_performance_analyst")["_raw_body"]
    assert "Do not fall back to live lap or section classification" in agent_behavior("live_performance_analyst")["_raw_body"]


def test_agent_behavior_knowledge_is_loaded():
    reload()

    for name in (
        "live",
        "recorded",
        "user_summary",
        "track_guide",
        "overtake",
        "live_performance_analyst",
    ):
        doc = agent_behavior(name)
        assert doc is not None
        assert doc["_raw_body"]
    assert agent_behavior("main_chatbot") is None


def test_system_prompt_defaults_to_live_session_knowledge():
    reload()

    prompt = pipecat_pipeline._build_system_prompt({})

    assert "Tool use:" in prompt
    assert "Procedure plan mode:" in prompt
    assert "Emotion signaling" in prompt
    assert "Transcript resilience" in prompt
    assert "Live chatbot session startup behavior:" in prompt
    assert "Recorded chatbot session startup behavior:" not in prompt
    assert "User summary chatbot session startup behavior:" not in prompt
    assert "Track Guide agent startup behavior:" not in prompt
    assert "Overtake agent startup behavior:" not in prompt
    assert "Live Performance Analyst startup behavior:" not in prompt


def test_system_prompt_ignores_legacy_frontend_tool_result_handling():
    reload()

    prompt = pipecat_pipeline._build_system_prompt(
        {},
        "Treat complete or ok=true as a successful result. "
        "Treat running as not ready yet. "
        "Treat failed, blocked, or skipped as unavailable.",
    )

    assert "Frontend tool result handling:" not in prompt
    assert "Treat complete or ok=true as a successful result." not in prompt
    assert "Treat running as not ready yet." not in prompt
    assert "Treat failed, blocked, or skipped as unavailable." not in prompt


@pytest.mark.parametrize(
    ("session_mode", "included", "excluded"),
    [
        (
            "live",
            "Live chatbot session startup behavior:",
            [
                "Recorded chatbot session startup behavior:",
                "User summary chatbot session startup behavior:",
            ],
        ),
        (
            "recorded",
            "Recorded chatbot session startup behavior:",
            [
                "Live chatbot session startup behavior:",
                "User summary chatbot session startup behavior:",
            ],
        ),
        (
            "user_summary",
            "User summary chatbot session startup behavior:",
            [
                "Live chatbot session startup behavior:",
                "Recorded chatbot session startup behavior:",
            ],
        ),
    ],
)
def test_system_prompt_uses_one_chatbot_session_mode_knowledge(session_mode, included, excluded):
    reload()

    prompt = pipecat_pipeline._build_system_prompt({
        "session_mode": session_mode,
    })

    assert included in prompt
    for text in excluded:
        assert text not in prompt


def test_system_prompt_uses_one_sub_agent_startup_knowledge():
    reload()

    prompt = pipecat_pipeline._build_system_prompt({
        "session_mode": "recorded",
        "agent_mode": "track_guide",
    })

    assert '"agent_mode": "track_guide"' in prompt
    assert "Track Guide agent startup behavior:" in prompt
    assert "Live chatbot session startup behavior:" not in prompt
    assert "Recorded chatbot session startup behavior:" not in prompt
    assert "User summary chatbot session startup behavior:" not in prompt
    assert "Overtake agent startup behavior:" not in prompt
    assert "Live Performance Analyst startup behavior:" not in prompt


def test_system_prompt_unknown_agent_mode_falls_back_to_session_mode(caplog):
    reload()

    prompt = pipecat_pipeline._build_system_prompt({
        "session_mode": "recorded",
        "agent_mode": "mystery_mode",
    })

    assert "Recorded chatbot session startup behavior:" in prompt
    assert "Live chatbot session startup behavior:" not in prompt
    assert "Track Guide agent startup behavior:" not in prompt
    assert "Unknown voice agent_mode" in caplog.text


def test_server_tool_schema_excludes_frontend_telemetry_tools(monkeypatch):
    class FakeFunctionSchema:
        def __init__(self, name, description, properties, required):
            self.name = name
            self.description = description
            self.properties = properties
            self.required = required

    fake_module = types.ModuleType("pipecat.adapters.schemas.function_schema")
    fake_module.FunctionSchema = FakeFunctionSchema
    monkeypatch.setitem(sys.modules, "pipecat.adapters.schemas.function_schema", fake_module)

    schemas = pipecat_pipeline._build_server_tool_schemas(
        None,
        {
            "analyze_telemetry": {
                "description": "Analyze telemetry from the frontend.",
            },
            "classify_live_section": {
                "description": "Classify the active Live Performance Analyst focus section.",
            },
            "explain_label": {
                "description": "Wrong frontend-era description.",
                "parameters": {
                    "label_id": {
                        "description": "Wrong frontend-era parameter doc.",
                    },
                },
            },
        },
    )

    names = {schema.name for schema in schemas}
    assert "analyze_telemetry" not in names
    assert "classify_live_section" not in names
    assert {"explain_label", "get_track_knowledge", "search_racing_knowledge"} <= names

    by_name = {schema.name: schema for schema in schemas}
    assert "ACLA racing label" in by_name["explain_label"].description
    assert "Wrong frontend-era description" not in by_name["explain_label"].description
    assert "ACLA track notes" in by_name["get_track_knowledge"].description
    assert "ACLA racing knowledge corpus" in by_name["search_racing_knowledge"].description
    assert "label code" in by_name["explain_label"].properties["label_id"]["description"]
    assert "Wrong frontend-era parameter doc" not in by_name["explain_label"].properties["label_id"]["description"]
    assert "track id" in by_name["get_track_knowledge"].properties["track"]["description"]
    assert "natural-language" in by_name["search_racing_knowledge"].properties["query"]["description"]


def test_frontend_tool_schema_exposes_advance_plan_step(monkeypatch):
    class FakeFunctionSchema:
        def __init__(self, name, description, properties, required):
            self.name = name
            self.description = description
            self.properties = properties
            self.required = required

    fake_module = types.ModuleType("pipecat.adapters.schemas.function_schema")
    fake_module.FunctionSchema = FakeFunctionSchema
    monkeypatch.setitem(sys.modules, "pipecat.adapters.schemas.function_schema", fake_module)

    schemas = pipecat_pipeline._build_frontend_tool_schemas([
        {
            "name": "advance_plan_step",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    ], {
        "advance_plan_step": {
            "description": (
                "Report that the current visible procedure plan request is complete "
                "so the frontend can execute the next tool_call."
            ),
            "parameters": {
                "reason": {
                    "description": "Optional short reason the current plan request is complete.",
                },
            },
        },
    })
    schema = schemas[0]

    assert schema.name == "advance_plan_step"
    assert "frontend" in schema.description
    assert "tool_call" in schema.description
    assert schema.properties["reason"]["description"]


def test_tool_payload_prompt_is_pure_tool_payload_json():
    prompt = pipecat_pipeline._format_tool_payload_for_prompt(
        {
            "event": "baseline_classifier_request_ready",
            "text": "baseline_classifier_request_ready.",
            "goal": "Collect a baseline and use recorded-session analysis to choose a focus.",
            "current_request": 1,
            "requests": [
                {
                    "type": "driver_action",
                    "subscriber": "driver",
                    "title": "Collect a clean baseline lap",
                    "status": "complete",
                },
                {
                    "type": "frontend_request",
                    "subscriber": "live_recorded_analysis",
                    "title": "Request recorded-session classifier",
                    "status": "pending",
                    "payload": {"force": False},
                },
            ],
        },
        {},
    )

    assert prompt.startswith("{")
    assert prompt.endswith("}")
    assert '"event": "baseline_classifier_request_ready"' in prompt
    assert '"text": "baseline_classifier_request_ready."' in prompt
    assert "live_recorded_analysis" in prompt
    assert "Request recorded-session classifier" in prompt
    assert "Procedure plan mode is active" not in prompt
    assert "Full tool payload JSON" not in prompt
    assert "classify_live_section" not in prompt


def test_tool_payload_prompt_includes_full_payload_fields():
    prompt = pipecat_pipeline._format_tool_payload_for_prompt(
        {
            "event": "tool_result",
            "text": "Recorded-session classifier finished.",
            "ok": True,
            "result": {
                "label": "understeering_at_entry",
                "confidence": 0.92,
                "evidence": ["late rotation", "front slip"],
            },
            "metadata": {
                "tool_call_id": "abc123",
                "source": "frontend",
            },
        },
        {},
    )

    assert prompt.startswith("{")
    assert prompt.endswith("}")
    assert "Full tool payload JSON" not in prompt
    assert '"event": "tool_result"' in prompt
    assert '"ok": true' in prompt
    assert '"label": "understeering_at_entry"' in prompt
    assert '"confidence": 0.92' in prompt
    assert '"evidence": ["late rotation", "front slip"]' in prompt
    assert '"tool_call_id": "abc123"' in prompt
    assert '"source": "frontend"' in prompt


