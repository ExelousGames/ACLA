import sys
import types

import pytest

from app.external_knowledge_base import behavior, reload, tool
from app.racing_engineer.service import AIService, _live_section_stats
from app.voice import pipecat_pipeline


def test_live_performance_tool_knowledge_is_loaded():
    reload()

    assert tool("start_live_performance_analysis")["title"] == "Starting live analyst"
    assert tool("set_procedure_plan")["title"] == "Setting procedure plan"
    assert tool("get_live_focus_section")["title"] == "Reading focus section"
    assert tool("classify_live_section")["title"] == "Classifying live section"
    assert "hidden frontend relay" in tool("classify_live_section")["description"]
    assert "requests" in tool("set_procedure_plan")["_raw_body"]
    assert "focus_name" not in tool("set_procedure_plan")["_raw_body"]
    assert "request's `payload`" in tool("set_procedure_plan")["_raw_body"]
    assert "collecting_baseline" in behavior("live_performance_analyst")["_raw_body"]
    assert "live_analysis_plan_started" in behavior("live_performance_analyst")["_raw_body"]
    assert "lap='last'" in behavior("live_performance_analyst")["_raw_body"]


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


def test_live_analyst_observations_drive_the_right_actions():
    plan_msg = pipecat_pipeline._format_observation_for_llm({
        "source": "live_performance_analyst",
        "agent_mode": "live_performance_analyst",
        "event": "recorded_analysis_plan_ready",
        "goal": "Improve Paddock Hill.",
        "snapshot": {"track": "brands_hatch", "live_session_type": "solo_practice"},
        "focus": {
            "section": {"id": "brands_hatch2", "name": "Paddock Hill", "from": 0.1, "to": 0.2},
            "baseline": {"childLabels": ["Initiate brake too late"]},
        },
    })
    baseline_msg = pipecat_pipeline._format_observation_for_llm({
        "source": "live_performance_analyst",
        "agent_mode": "live_performance_analyst",
        "event": "live_baseline_ready_for_classification",
        "completed_lap": 2,
        "snapshot": {"track": "brands_hatch", "live_session_type": "solo_practice"},
        "candidate_sections": [
            {"id": "brands_hatch2", "name": "Paddock Hill", "from": 0.1, "to": 0.2},
        ],
    })
    coaching_msg = pipecat_pipeline._format_observation_for_llm({
        "source": "live_performance_analyst",
        "agent_mode": "live_performance_analyst",
        "event": "live_analysis_window",
        "snapshot": {"live_session_type": "traffic_or_race"},
        "focus": {
            "section": {"id": "brands_hatch2", "name": "Paddock Hill", "from": 0.1, "to": 0.2},
            "baseline": {"mistakeCount": 2, "severity": 2, "childLabels": ["Initiate brake too late"]},
            "timing": {"secondsAhead": 9.5, "distanceAhead": 0.06},
        },
    })

    assert "set_procedure_plan" in plan_msg
    assert "requests array" in plan_msg
    assert "request payload" in plan_msg
    assert "plan-level focus" in plan_msg
    assert "plan=" not in plan_msg
    assert "classify_live_section" in baseline_msg
    assert "Do not expose raw telemetry" in baseline_msg
    assert "Call show_map" in coaching_msg
    assert "traffic_or_race" in coaching_msg


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
