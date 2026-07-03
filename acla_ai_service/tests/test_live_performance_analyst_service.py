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
        "front_desk",
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


def test_system_prompt_defaults_to_front_desk_session_knowledge():
    reload()

    prompt = pipecat_pipeline._build_system_prompt({})

    assert "Tool use:" in prompt
    assert "Procedure plan mode:" in prompt
    assert "Emotion signaling" in prompt
    assert "Transcript resilience" in prompt
    assert "Front desk chatbot session startup behavior:" in prompt
    assert "Live chatbot session startup behavior:" not in prompt
    assert "Recorded chatbot session startup behavior:" not in prompt
    assert "User summary chatbot session startup behavior:" not in prompt
    assert "Track Guide agent startup behavior:" not in prompt
    assert "Overtake agent startup behavior:" not in prompt
    assert "Live Performance Analyst startup behavior:" not in prompt


def test_system_prompt_uses_service_owned_tool_result_handling():
    reload()

    prompt = pipecat_pipeline._build_system_prompt(
        {},
        "LEGACY FRONTEND HANDLING SHOULD NOT BE COPIED.",
    )

    assert "Frontend tool result handling:" in prompt
    assert "Treat complete or ok=true as a successful result" in prompt
    assert "Treat running as not ready yet" in prompt
    assert "Treat failed, blocked, or skipped as unavailable" in prompt
    assert "LEGACY FRONTEND HANDLING SHOULD NOT BE COPIED." not in prompt


@pytest.mark.parametrize(
    ("session_mode", "included", "excluded"),
    [
        (
            "front_desk",
            "Front desk chatbot session startup behavior:",
            [
                "Live chatbot session startup behavior:",
                "Recorded chatbot session startup behavior:",
                "User summary chatbot session startup behavior:",
            ],
        ),
        (
            "live",
            "Live chatbot session startup behavior:",
            [
                "Front desk chatbot session startup behavior:",
                "Recorded chatbot session startup behavior:",
                "User summary chatbot session startup behavior:",
            ],
        ),
        (
            "recorded",
            "Recorded chatbot session startup behavior:",
            [
                "Front desk chatbot session startup behavior:",
                "Live chatbot session startup behavior:",
                "User summary chatbot session startup behavior:",
            ],
        ),
        (
            "user_summary",
            "User summary chatbot session startup behavior:",
            [
                "Front desk chatbot session startup behavior:",
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
    assert "Front desk chatbot session startup behavior:" not in prompt
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
