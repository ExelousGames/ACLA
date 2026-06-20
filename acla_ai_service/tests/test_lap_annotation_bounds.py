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
    )


def test_lap_tool_agent_request_is_fixed_to_section():
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
        circuit_id="brands_hatch",
    )

    assert request.parent_start == 10
    assert request.parent_end == 20
    assert "Splitter context" in request.planner_prompt
    assert "`brands_hatch1`" in request.planner_prompt
    assert (
        "resolve to one circuit_section id"
        in request.planner_prompt
    )
    assert set(request.extra_state) == {
        "tool_agent_extra_tools",
        "annotation_session_context",
        "eligible_behavior_label_ids",
    }


def test_lap_annotation_prompt_includes_segment_action_model():
    prompt = lap_flow.lap_annotation_prompt("practice")

    assert "Segment / Action Model" in prompt
    assert "parent segment is a group of one or more complete driving actions" in prompt
    assert "turning / trajectory, throttle input, and brake input" in prompt
    assert "Every behavior, segment-type, and sub-label" in prompt
    assert "describe the final annotation range as a whole" in prompt


def test_lap_annotation_prompt_does_not_forbid_pit_stop_with_recovery_merge():
    prompt = lap_flow.lap_annotation_prompt("practice")

    assert "PS is incompatible with MSP / RM" not in prompt
    assert "do NOT combine with EA / MSP / MSR / RM / O / OD" not in prompt
    assert "Do NOT attach for normal pit exit" not in prompt


def test_lap_parse_rejects_extra_range_fields():
    response = AgentResponse(
        raw_response=json.dumps({
            "start_index": 0,
            "label_ids": ["MSP"],
            "reasoning": "incorrectly used the full lap",
        }),
        verdict="pass",
    )

    with pytest.raises(RuntimeError, match="unsupported output field"):
        lap_flow.parse(
            response,
            prompt_mode="tool_agent",
            lap_start=0,
            lap_end=100,
            section_id="brands_hatch1",
            section_start=10,
            section_end=20,
            circuit_id="brands_hatch",
        )


def test_lap_parse_tool_agent_accepts_summary_field_as_reasoning_fallback():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["MSP"],
            "summary": "Observed steady maintenance speed through the full split.",
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        prompt_mode="tool_agent",
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.label_ids == ["MSP"]
    assert (
        result.reasoning
        == "Observed steady maintenance speed through the full split."
    )


def test_lap_parse_does_not_add_section_when_agent_omits_it():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["MSP"],
            "reasoning": (
                "Circuit section was ambiguous in locate_circuit_section, "
                "but the splitter assigned this range to Brabham Straight."
            ),
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        prompt_mode="tool_agent",
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.label_ids == ["MSP"]


def test_lap_parse_resolves_same_range_sections_to_racing_surface_without_ps():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["brands_hatch1", "brands_hatch17", "MSP"],
            "reasoning": (
                "locate_circuit_section was ambiguous between Brabham "
                "Straight and Pit, but there is no pit-lane procedure "
                "evidence."
            ),
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        prompt_mode="tool_agent",
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.label_ids == [
        "brands_hatch",
        "brands_hatch1",
        "MSP",
    ]
    assert result.rejected_proposals == [{
        "value": "brands_hatch17",
        "reason": (
            "same-range circuit_section ambiguity resolved to brands_hatch1"
        ),
    }]


def test_lap_parse_resolves_same_range_sections_to_pit_with_ps():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["brands_hatch1", "brands_hatch17", "PS", "RM"],
            "reasoning": (
                "locate_circuit_section was ambiguous between Brabham "
                "Straight and Pit, and pit-lane evidence supports PS with "
                "recovery / merge."
            ),
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        prompt_mode="tool_agent",
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.label_ids == [
        "brands_hatch",
        "brands_hatch17",
        "PS",
        "RM",
    ]
    assert result.rejected_proposals == [{
        "value": "brands_hatch1",
        "reason": (
            "same-range circuit_section ambiguity resolved to brands_hatch17"
        ),
    }]


def test_lap_parse_does_not_add_splitter_section_to_empty_drop():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": [],
            "reasoning": "No behavior parent label fit the whole range.",
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        prompt_mode="tool_agent",
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.label_ids == []


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
