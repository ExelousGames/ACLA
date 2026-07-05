import json
from types import SimpleNamespace

from app.annotation_providers.registry import (
    clear_provider_cache,
    list_annotation_providers,
)
from app.infra.config import settings
from app.local_annotation_agent.workflow import AnnotationPipelineConfig
from app.local_annotation_agent.workflow.flows import detailed as detailed_flow
from app.local_annotation_agent.workflow.flows import lap as lap_flow
from app.shared.contracts import DEFAULT_AGENT_MAX_TURNS, AgentResponse


def test_local_vlm_provider_is_visible(monkeypatch):
    clear_provider_cache()
    try:
        monkeypatch.setattr(settings, "annotation_enabled_providers", None)
        providers = list_annotation_providers()
    finally:
        clear_provider_cache()

    local = next((provider for provider in providers if provider.id == "local_vlm"), None)
    assert local is not None
    assert local.runner == "local_pipeline"


def test_annotation_pipeline_default_max_turns_is_lowered():
    provider_config = AnnotationPipelineConfig().to_provider_config()

    assert provider_config.provider_options["max_turns"] == DEFAULT_AGENT_MAX_TURNS
    assert DEFAULT_AGENT_MAX_TURNS == 5


def test_annotation_pipeline_preserves_explicit_max_turns():
    provider_config = AnnotationPipelineConfig(
        provider_options={"max_turns": 8}
    ).to_provider_config()

    assert provider_config.provider_options["max_turns"] == 8


def test_detailed_local_request_uses_shared_tool_agent_prompt(monkeypatch):
    monkeypatch.setattr(
        detailed_flow,
        "build_preflight_context",
        lambda **_kwargs: SimpleNamespace(prompt_block="FRONT", attachments=[]),
    )
    monkeypatch.setattr(
        detailed_flow,
        "_embedding_label_candidates",
        lambda **_kwargs: [],
    )

    request = detailed_flow.build_request(
        provider_id="local_vlm",
        df=[],
        parent_start=0,
        parent_end=20,
        parent_main_labels=["MSP"],
    )

    assert "root_agent" not in request.extra_state
    assert "tool_agent_extra_tools" in request.extra_state
    assert request.planner_prompt.startswith("FRONT")
    assert '"agent": "label_verifier"' not in request.planner_prompt
    intro, outro = request.synth_prompt({})
    assert (intro, outro) == ("", "")


def test_detailed_local_parse_accepts_valid_synth_json():
    response = AgentResponse(
        raw_response=json.dumps({
            "proposals": [{
                "label_id": "MSP",
                "start_index": 2,
                "end_index": 8,
                "reasoning": "The evidence supports this strict child range.",
            }]
        }),
        verdict="submitted",
    )

    result = detailed_flow.parse(
        response,
        parent_start=0,
        parent_end=10,
    )

    assert result.accepted is True
    assert result.final_labels == ["MSP"]
    assert result.sub_start == 2
    assert result.sub_end == 8


def test_lap_local_request_uses_shared_tool_agent_prompt(monkeypatch):
    monkeypatch.setattr(
        lap_flow,
        "build_preflight_context",
        lambda **_kwargs: SimpleNamespace(prompt_block="FRONT", attachments=[]),
    )

    request = lap_flow.build_request(
        provider_id="local_vlm",
        df=[],
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert "root_agent" not in request.extra_state
    assert "tool_agent_extra_tools" in request.extra_state
    assert request.planner_prompt.startswith("FRONT")
    assert '"agent": "label_verifier"' not in request.planner_prompt
    intro, outro = request.synth_prompt({})
    assert (intro, outro) == ("", "")


def test_lap_local_parse_marks_valid_json_submitted():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["brands_hatch", "brands_hatch1", "MSP"],
            "reasoning": "The whole range fits the selected section and MSP.",
        }),
        verdict="submitted",
    )

    result = lap_flow.parse(
        response,
        lap_start=0,
        lap_end=100,
        section_id="brands_hatch1",
        section_start=10,
        section_end=20,
        circuit_id="brands_hatch",
    )

    assert result.submitted is True
    assert result.label_ids == ["brands_hatch", "brands_hatch1", "MSP"]
