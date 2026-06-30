import sys
import types

import pytest

from app.external_knowledge_base import agent_behavior, behavior, reload, tool
from app.racing_engineer.service import AIService, _live_section_stats
from app.voice import pipecat_pipeline


def test_live_performance_tool_knowledge_is_loaded():
    reload()

    assert tool("start_agent_session")["title"] == "Starting agent mode"
    assert tool("stop_agent_session")["title"] == "Stopping agent mode"
    assert "live_performance_analyst" in tool("start_agent_session")["_raw_body"]
    assert tool("start_live_performance_analysis") is None
    assert tool("stop_per_turn_coaching") is None
    assert tool("set_procedure_plan")["title"] == "Setting procedure plan"
    assert tool("get_live_focus_section")["title"] == "Analyzing focus section"
    assert "show_map_arguments" in tool("get_live_focus_section")["_raw_body"]
    assert tool("classify_live_section")["title"] == "Classifying live section"
    assert "active Live Performance Analyst focus section" in tool("classify_live_section")["description"]
    assert "requests" in tool("set_procedure_plan")["_raw_body"]
    assert "focus_name" not in tool("set_procedure_plan")["_raw_body"]
    assert "request's `payload`" in tool("set_procedure_plan")["_raw_body"]
    assert "advance_plan_step" in behavior("procedure_plan")["_raw_body"]
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


def test_server_tool_schema_exposes_live_section_classifier(monkeypatch):
    class FakeFunctionSchema:
        def __init__(self, name, description, properties, required):
            self.name = name
            self.description = description
            self.properties = properties
            self.required = required

    fake_module = types.ModuleType("pipecat.adapters.schemas.function_schema")
    fake_module.FunctionSchema = FakeFunctionSchema
    monkeypatch.setitem(sys.modules, "pipecat.adapters.schemas.function_schema", fake_module)

    schemas = pipecat_pipeline._build_server_tool_schemas({"type": "object"})
    live_schema = next(schema for schema in schemas if schema.name == "classify_live_section")

    assert "Live Performance Analyst" in live_schema.description
    assert set(live_schema.properties) == {"section_id", "section_name", "lap"}


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

    reload()
    schemas = pipecat_pipeline._build_frontend_tool_schemas([
        {
            "name": "advance_plan_step",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    ])
    schema = schemas[0]

    assert schema.name == "advance_plan_step"
    assert "frontend" in schema.description
    assert "tool_call" in schema.description
    assert schema.properties["reason"]["description"]


def test_observation_prompt_includes_generic_plan_mode_contract():
    prompt = pipecat_pipeline._format_observation_for_prompt(
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

    assert "Procedure plan mode is active" in prompt
    assert "advance_plan_step" in prompt
    assert "live_recorded_analysis" in prompt
    assert "Request recorded-session classifier" in prompt
    assert "classify_live_section" not in prompt


@pytest.mark.asyncio
async def test_classify_live_section_uses_hidden_frontend_telemetry_tool(monkeypatch):
    service = object.__new__(AIService)
    captured = {}

    async def fake_composite(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(service, "_composite_analyze", fake_composite)

    result = await service._classify_live_section_impl(
        conn=object(),
        section_id="brands_hatch2",
        section_name=None,
        lap="last",
    )

    assert result == {"status": "ok"}
    assert captured["frontend_tool"] == "_get_live_section_telemetry"
    assert captured["frontend_args"] == {"lap": "last", "section_id": "brands_hatch2"}
    assert captured["record_live_classification"] is True


def test_live_section_stats_are_compact_and_numeric_only():
    stats = _live_section_stats([
        {"Physics_speed_kmh": 100, "Physics_brake": 0.2, "noise": "x"},
        {"Physics_speed_kmh": 120, "Physics_brake": 0.6, "noise": "y"},
    ])

    assert stats == {
        "speed": {"min": 100.0, "max": 120.0, "avg": 110.0},
        "brake": {"min": 0.2, "max": 0.6, "avg": 0.4},
    }
