import json

import pandas as pd
import pytest

from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
)
from app.local_annotation_agent.workflow.flows import lap as lap_flow
from app.shared.contracts import AgentRequest, AgentResponse, ProviderConfig


def _request(df, *, parent_start=10, parent_end=20):
    return AgentRequest(
        provider_id="test",
        config=ProviderConfig(provider_id="test"),
        planner_prompt="",
        synth_prompt=lambda _state: ("", ""),
        df_ref=df,
        parent_start=parent_start,
        parent_end=parent_end,
        extra_state={
            "tool_agent_revision_bounds": {
                "start": 8,
                "end": 25,
                "initial_start": parent_start,
                "initial_end": parent_end,
            },
        },
    )


def test_lap_tool_agent_request_starts_at_section_not_full_lap():
    df = pd.DataFrame({"metric": range(100)})

    request = lap_flow.build_request(
        provider_id="test",
        prompt_mode="tool_agent",
        df=df,
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        revision_start=8,
        revision_end=25,
        circuit_id="brands_hatch",
    )

    assert request.parent_start == 10
    assert request.parent_end == 20
    assert request.extra_state["tool_agent_revision_bounds"]["start"] == 8
    assert request.extra_state["tool_agent_revision_bounds"]["end"] == 25


def test_lap_parse_rejects_result_outside_revision_envelope():
    response = AgentResponse(
        raw_response=json.dumps({
            "revised_range": [0, 100],
            "label_ids": ["MSP"],
            "reasoning": "incorrectly used the full lap",
        }),
        verdict="pass",
    )

    with pytest.raises(RuntimeError, match="outside revision envelope"):
        lap_flow.parse(
            response,
            prompt_mode="local_pipeline",
            lap_start=0,
            lap_end=100,
            section_id="brands_hatch1",
            section_start=10,
            section_end=20,
            revision_start=8,
            revision_end=25,
            circuit_id="brands_hatch",
        )


def test_tool_surface_queries_clamp_to_current_working_range():
    df = pd.DataFrame({"metric": [float(i) for i in range(100)]})
    capture = ToolAgentCapture(cur_start=10, cur_end=20)
    surface = AnnotationToolSurface(_request(df), capture)

    result = json.loads(surface.query_telemetry(
        "read_values_at_indices",
        json.dumps({
            "range": [0, 99],
            "column": "metric",
            "indices": [9, 10, 19, 50],
        }),
    ))

    assert result["params"]["range"] == [10, 20]
    by_iloc = {sample["iloc"]: sample for sample in result["result"]["samples"]}
    assert by_iloc[10]["value"] == 10.0
    assert by_iloc[19]["value"] == 19.0
    assert by_iloc[9]["value"] is None
    assert by_iloc[50]["value"] is None


def test_tool_surface_revision_cannot_jump_to_full_lap():
    df = pd.DataFrame({"metric": [float(i) for i in range(100)]})
    capture = ToolAgentCapture(cur_start=10, cur_end=20)
    surface = AnnotationToolSurface(_request(df), capture)

    rejected = json.loads(surface.revise_range(0, 100))
    assert rejected["ok"] is False
    assert capture.cur_start == 10
    assert capture.cur_end == 20

    accepted = json.loads(surface.revise_range(8, 22))
    assert accepted["ok"] is True
    assert capture.cur_start == 8
    assert capture.cur_end == 22
