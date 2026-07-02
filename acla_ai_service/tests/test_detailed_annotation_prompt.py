import json

from app.shared.contracts import AgentResponse
from app.local_annotation_agent.workflow.flows import detailed as detailed_flow


def test_detailed_prompt_requires_full_parent_scan():
    prompt = detailed_flow._tool_agent_task_prompt(
        parent_start=800,
        parent_end=825,
        parent_main_labels=["MSP"],
        existing_children=[],
    )

    assert "Audit the full parent range before submitting" in prompt
    assert "Do not stop after the first supported episode" in prompt
    assert "submit every strongly supported child proposal" in prompt


def test_detailed_prompt_does_not_let_existing_children_suppress_overlap():
    prompt = detailed_flow._tool_agent_task_prompt(
        parent_start=800,
        parent_end=825,
        parent_main_labels=["MSP"],
        existing_children=[
            {
                "start_index": 816,
                "end_index": 820,
                "labels": ["ST15"],
            }
        ],
    )

    assert "Already discovered sub-segments only block exact duplicate proposals" in prompt
    assert "They do not mark their ilocs as handled" in prompt
    assert "different labels that overlap or nest inside them" in prompt
    assert "One category does not explain away or cover another category" in prompt
    assert "Existing child proposals for duplicate checks" in prompt
    assert "do NOT re-propose" not in prompt


def test_detailed_prompt_local_event_ranges_ignore_unrelated_recovery():
    prompt = detailed_flow._tool_agent_task_prompt(
        parent_start=800,
        parent_end=825,
        parent_main_labels=["MSP"],
        existing_children=[],
    )

    assert "onset, release, peak, spike, dip, or short instability" in prompt
    assert "fit the child range to that event" in prompt
    assert "A later reversal or recovery is not disqualifying" in prompt


def test_detailed_prompt_includes_json_support_sign_rule():
    prompt = detailed_flow._tool_agent_task_prompt(
        parent_start=800,
        parent_end=825,
        parent_main_labels=["MSP"],
        existing_children=[],
    )

    assert "When the selected label description names supporting signs" in prompt
    assert "include that supporting sign as part of the child segment range" in prompt


def test_detailed_prompt_requires_rejected_ambiguous_option_reason():
    prompt = detailed_flow._tool_agent_task_prompt(
        parent_start=800,
        parent_end=825,
        parent_main_labels=["MSP"],
        existing_children=[],
    )

    assert "ambiguous between two plausible labels or ranges" in prompt
    assert "which other option was not selected and why" in prompt


def test_detailed_candidate_block_with_no_candidates_authorizes_empty_only():
    prompt = detailed_flow._embedding_candidates_prompt_block([])

    assert "submit an empty proposals list" in prompt
    assert "no label_id is authorized" in prompt


def test_detailed_parse_accepts_direct_label_ids_payload():
    response = AgentResponse(
        raw_response=json.dumps({
            "label_ids": ["RM", "ST2", "RM7"],
            "reasoning": "Recovery and merge evidence fits the submitted range.",
        }),
        verdict="submitted",
    )

    result = detailed_flow.parse(
        response,
        prompt_mode="tool_agent",
        parent_start=0,
        parent_end=20,
    )

    assert result.accepted is True
    assert result.final_labels == ["RM", "ST2", "RM7"]
    assert result.sub_start == 0
    assert result.sub_end == 20
    assert result.label_annotations == [
        {
            "label_id": "RM",
            "start_index": 0,
            "end_index": 20,
            "reasoning": "Recovery and merge evidence fits the submitted range.",
        },
        {
            "label_id": "ST2",
            "start_index": 0,
            "end_index": 20,
            "reasoning": "Recovery and merge evidence fits the submitted range.",
        },
        {
            "label_id": "RM7",
            "start_index": 0,
            "end_index": 20,
            "reasoning": "Recovery and merge evidence fits the submitted range.",
        },
    ]
