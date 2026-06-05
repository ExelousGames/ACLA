from __future__ import annotations

from app.local_annotation_agent import AgentResponse
from app.local_annotation_agent.workflow.flows import lap as lap_flow


def _interaction(section_id: str = "brands_hatch1") -> dict:
    return {
        "windows": [{"start_index": 10, "end_index": 40}],
        "section_context": [
            {
                "circuit_section_id": section_id,
                "circuit_section_name": "Brabham Straight",
                "range": [10, 40],
            }
        ],
    }


def test_lap_parse_adds_splitter_section_context_to_opponent_labels() -> None:
    response = AgentResponse(
        raw_response='{"label_ids": ["O"], "reasoning": "Completed pass."}',
    )

    result = lap_flow.parse(
        response,
        backend="local",
        lap_start=0,
        lap_end=100,
        section_id="interaction_window",
        section_start=10,
        section_end=40,
        circuit_id="brands_hatch",
        opponent_interaction=_interaction(),
    )

    assert result.label_ids == ["brands_hatch", "brands_hatch1", "O"]


def test_lap_parse_keeps_empty_opponent_drop_result_empty() -> None:
    response = AgentResponse(
        raw_response='{"label_ids": [], "reasoning": "Only close following."}',
    )

    result = lap_flow.parse(
        response,
        backend="local",
        lap_start=0,
        lap_end=100,
        section_id="interaction_window",
        section_start=10,
        section_end=40,
        circuit_id="brands_hatch",
        opponent_interaction=_interaction(),
    )

    assert result.label_ids == []
