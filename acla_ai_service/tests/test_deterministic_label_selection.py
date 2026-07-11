from pathlib import Path

import pandas as pd

import app.local_annotation_agent.workflow as workflow
from app.local_annotation_agent.workflow import deterministic


def test_catalog_requirements_are_valid():
    assert deterministic.validate_catalog() == []


def test_ea_accepts_either_complete_requirement_branch():
    requirements = deterministic._requirements_for(
        "EA", deterministic.get_label("EA")
    )
    first = deterministic.evaluate_requirements(
        requirements,
        {
            "time_gap.total_change_abs_ms": 20,
            "time_gap.ending_direction": "falling",
        },
    )
    second = deterministic.evaluate_requirements(
        requirements,
        {
            "time_gap.has_spike": False,
            "brake.similarity": 1.0,
            "throttle.similarity": 1.0,
        },
    )
    assert first.matched and first.branch == 0
    assert second.matched and second.branch == 1


def test_missing_fact_fails_closed():
    result = deterministic.evaluate_requirements(
        {
            "enabled": True,
            "any_of": [{"all_of": [
                {"fact": "time_gap.direction", "operator": "eq", "value": "rising"}
            ]}],
        },
        {},
    )
    assert not result.matched


def test_disabled_label_never_matches():
    requirements = deterministic._requirements_for(
        "MSP19", deterministic.get_label("MSP19")
    )
    assert not deterministic.evaluate_requirements(requirements, {}).matched


def test_exclusive_matches_are_suppressed(monkeypatch):
    docs = {
        "A": {
            "id": "A", "exclusive_with": ["B"],
            "selection_requirements": {
                "enabled": True,
                "any_of": [{"all_of": [{"fact": "phase.entry", "operator": "eq", "value": True}]}],
            },
        },
        "B": {
            "id": "B", "exclusive_with": ["A"],
            "selection_requirements": {
                "enabled": True,
                "any_of": [{"all_of": [{"fact": "phase.entry", "operator": "eq", "value": True}]}],
            },
        },
    }
    monkeypatch.setattr(deterministic, "get_label", docs.get)
    result = deterministic.evaluate_labels(["A", "B"], {"phase.entry": True})
    assert result.labels == []
    assert result.conflicts == [("A", "B")]


def test_detailed_discovery_preserves_multiple_ranges_and_deduplicates(monkeypatch):
    def fake_facts(_df, start, end, **_kwargs):
        ranges = [(1, 4), (6, 9)] if (start, end) == (0, 10) else []
        return {"segment.shape_key": "straight"}, ranges

    monkeypatch.setattr(deterministic, "calculate_facts", fake_facts)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(11)),
        parent_start=0,
        parent_end=10,
        parent_main_labels=["EA"],
        existing_children=[{"start_index": 1, "end_index": 4, "labels": ["ST2"]}],
    )
    assert [(p["start_index"], p["end_index"], p["label_id"]) for p in result.label_annotations] == [
        (6, 9, "ST2")
    ]


def test_interaction_section_uses_unique_splitter_context():
    section = deterministic._resolve_circuit_section(
        pd.DataFrame(),
        "silverstone",
        "interaction_window",
        0,
        10,
        {"section_context": [{"circuit_section_id": "silverstone1"}]},
    )
    assert section == "silverstone1"


def test_public_pipeline_bypasses_provider_and_returns_lap_contract(monkeypatch):
    monkeypatch.setattr(
        deterministic,
        "calculate_facts",
        lambda *_args, **_kwargs: ({
            "time_gap.total_change_abs_ms": 10,
            "time_gap.ending_direction": "falling",
            "segment.shape_key": "straight",
        }, []),
    )
    result = workflow.run_annotation(
        flow="lap",
        df=pd.DataFrame(index=range(10)),
        lap_start=0,
        lap_end=10,
        section_id="silverstone1",
        section_start=0,
        section_end=9,
        circuit_id="silverstone",
    )
    assert result.label_ids == ["silverstone", "silverstone1", "EA", "ST2"]
    assert result.submitted


def test_lap_result_explains_failed_behavior_requirements(monkeypatch):
    monkeypatch.setattr(
        deterministic,
        "calculate_facts",
        lambda *_args, **_kwargs: ({"time_gap.has_spike": False}, []),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(10)),
        lap_start=0,
        lap_end=9,
        section_id="silverstone1",
        section_start=0,
        section_end=9,
        circuit_id="silverstone",
    )

    rejected = {item["value"]: item["reason"] for item in result.rejected_proposals}
    assert set(rejected) == {"EA", "PS", "RM", "MSP"}
    assert "time_gap.total_change_abs_ms" in rejected["EA"]
    assert "actual=None" in rejected["EA"]


def test_annotation_flows_do_not_import_removed_retrieval_code():
    root = Path(__file__).parents[1]
    flow_text = "\n".join(
        path.read_text()
        for path in (root / "app/local_annotation_agent/workflow/flows").glob("*.py")
    )
    assert "preflight" not in flow_text
    assert "label_search" not in flow_text
    assert (root / "app/external_knowledge_base/_embedder.py").exists()
