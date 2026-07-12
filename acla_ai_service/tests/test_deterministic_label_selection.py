import json
from pathlib import Path

import numpy as np
import pandas as pd

import app.local_annotation_agent.workflow as workflow
from app.local_annotation_agent.workflow import deterministic


def test_telemetry_series_is_smoothed_with_centered_three_sample_median():
    df = pd.DataFrame({"signal": [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]})

    values = deterministic._series(df, "signal")

    assert values.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_catalog_requirements_are_valid():
    assert deterministic.validate_catalog() == []


def test_main_labels_are_owned_by_lap_annotation_catalog():
    main_labels = list(deterministic.skills.iter("lap_annotation.labels"))
    sub_labels = list(deterministic.skills.iter("sub_label_annotation.labels"))

    assert {doc["id"] for doc in main_labels} == {
        "EA", "MSP", "MSR", "RM", "PS", "O", "OD",
    }
    assert all(doc["type"] == "main" for doc in main_labels)
    assert all(doc["type"] != "main" for doc in sub_labels)

    msp = deterministic.get_label("MSP")
    assert "characteristics" not in msp
    assert "description" not in msp
    assert msp["selection_requirements"]
    assert msp["selection_requirements_ref"] == (
        "lap_annotation.selection_requirements.MSP"
    )
    assert msp["exclusive_with"] == ["EA", "PS", "MSR"]
    assert "annotation_guideline" not in msp

    catalog_path = (
        Path(__file__).parents[1]
        / "app/internal_knowledge_base/sub_label_annotation.json"
    )
    catalog_text = catalog_path.read_text(encoding="utf-8")
    sub_catalog = json.loads(catalog_text)
    assert "annotation_guideline" not in catalog_text
    assert all(
        doc.get("type") != "main" for doc in sub_catalog["labels"].values()
    )


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
            "time_gap.total_change_abs_ms": 50,
            "time_gap.has_significant_rise": False,
            "brake.similarity": 1.0,
            "throttle.similarity": 1.0,
        },
    )
    assert first.matched and first.branch == 0
    assert second.matched and second.branch == 1


def test_slope_facts_distinguish_start_middle_and_end_rises(monkeypatch):
    def fake_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 50}, {"value": 100}],
            "extra": {
                "delta_value": 100,
                "total_change_direction": "rising",
                "total_change_is_label_significant": True,
                "slope_shape": "slope_steady_over_section",
                "previous_end_slope": 50,
                "end_slope": 50,
                "point_trend_runs": [
                    {
                        "start_iloc": 0, "end_iloc": 2, "direction": "rising",
                        "is_label_significant": True,
                    },
                    {
                        "start_iloc": 2, "end_iloc": 8, "direction": "falling",
                        "is_label_significant": True,
                    },
                    {
                        "start_iloc": 8, "end_iloc": 10, "direction": "rising",
                        "is_label_significant": True,
                    },
                ],
            },
        }, None)

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.run_pipeline_query", fake_query,
    )

    df = pd.DataFrame({
        "expert_time_difference": [0, 20, 40, 40, 30, 20, 10, 0, 10, 20, 30],
    })
    facts = deterministic._slope_facts(df, 0, 10)

    assert facts["time_gap.starting_direction"] == "rising"
    assert facts["time_gap.ending_direction"] == "rising"
    assert facts["time_gap.has_significant_rise"] is True
    assert facts["time_gap.middle_has_significant_rise"] is False
    assert facts["time_gap.middle_has_new_significant_rise"] is False
    assert facts["time_gap.flattening_at_end"] is False
    assert facts["time_gap.overall_gap"] == 100
    assert "time_gap.significant" not in facts


def test_slope_facts_smooth_ending_slope_windows():
    step_slopes = [20.0] * 6 + [10.0] * 3 + [5.0, 5.0, 100.0]
    df = pd.DataFrame({
        "expert_time_difference": np.cumsum([0.0, *step_slopes]),
    })

    facts = deterministic._slope_facts(df, 0, 12)

    assert facts["time_gap.flattening_at_end"] is True


def test_slope_facts_ignore_gentle_middle_rise():
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 5, 10, 15, 20, 20, 20, 20],
    })

    facts = deterministic._slope_facts(df, 0, 9)

    assert facts["time_gap.middle_has_significant_rise"] is False
    assert facts["time_gap.middle_significant_rise_ranges"] == []


def test_slope_facts_ignore_steady_middle_rise():
    for rate in (20, 200):
        df = pd.DataFrame({
            "expert_time_difference": np.arange(9) * rate,
        })

        facts = deterministic._slope_facts(df, 0, 8)

        assert facts["time_gap.middle_has_significant_rise"] is False
        assert facts["time_gap.middle_significant_rise_ranges"] == []


def test_slope_facts_identify_accelerating_middle_rise_then_flattening(monkeypatch):
    def fake_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 100}, {"value": 150}],
            "extra": {
                "delta_value": 100,
                "total_change_direction": "rising",
                "total_change_is_label_significant": True,
                "slope_shape": "slope_decreasing_over_section",
                "previous_end_slope": 100,
                "end_slope": 50,
                "point_trend_runs": [
                    {
                        "start_iloc": 0, "end_iloc": 7, "direction": "rising",
                        "is_label_significant": True,
                    },
                    {
                        "start_iloc": 7, "end_iloc": 10, "direction": "flat",
                        "is_label_significant": False,
                    },
                ],
            },
        }, None)

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.run_pipeline_query", fake_query,
    )

    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 10, 30, 70, 100, 110, 120, 125, 130],
    })
    facts = deterministic._slope_facts(df, 0, 10)

    assert facts["time_gap.middle_has_significant_rise"] is True
    assert facts["time_gap.middle_significant_rise_ranges"] == [[3, 6]]
    assert facts["time_gap.middle_has_new_significant_rise"] is False
    assert facts["time_gap.flattening_at_end"] is True

    evaluation = deterministic.evaluate_requirements(
        {"any_of": [{"all_of": [{
            "fact": "time_gap.middle_has_significant_rise",
            "operator": "eq",
            "value": True,
        }]}]},
        facts,
    )
    assert evaluation.passed == [
        "time_gap.middle_has_significant_rise: True (rising at iloc 3-6)",
    ]

    def fake_rise_fall_flat_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 100}, {"value": 50}],
            "extra": {
                "delta_value": 50,
                "total_change_direction": "rising",
                "total_change_is_label_significant": True,
                "slope_shape": "slope_decreasing_over_section",
                "previous_end_slope": 100,
                "end_slope": -50,
                "point_trend_runs": [
                    {
                        "start_iloc": 0, "end_iloc": 3, "direction": "rising",
                        "is_label_significant": True,
                    },
                    {
                        "start_iloc": 3, "end_iloc": 7, "direction": "falling",
                        "is_label_significant": True,
                    },
                    {
                        "start_iloc": 7, "end_iloc": 10, "direction": "flat",
                        "is_label_significant": False,
                    },
                ],
            },
        }, None)

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.run_pipeline_query",
        fake_rise_fall_flat_query,
    )

    facts = deterministic._slope_facts(pd.DataFrame(), 0, 10)

    assert facts["time_gap.flattening_at_end"] is True


def test_slope_facts_identify_significant_middle_spike():
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 0, 200, 200, 200, 200, 200, 200],
    })

    facts = deterministic._slope_facts(df, 0, 9)

    assert facts["time_gap.middle_has_significant_rise"] is True
    assert facts["time_gap.middle_significant_rise_ranges"] == [[4, 4]]


def test_slope_facts_ignore_single_sample_middle_noise():
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 0, 200, 0, 0, 0, 0, 0],
    })

    facts = deterministic._slope_facts(df, 0, 9)

    assert facts["time_gap.middle_has_significant_rise"] is False
    assert facts["time_gap.middle_significant_rise_ranges"] == []


def test_behavior_requirements_respect_slope_location():
    msp = deterministic._requirements_for("MSP", deterministic.get_label("MSP"))
    rm = deterministic._requirements_for("RM", deterministic.get_label("RM"))

    rising_only_at_start = {
        "time_gap.direction": "rising",
        "time_gap.overall_gap": 100,
        "time_gap.middle_has_significant_rise": False,
    }
    falling_with_middle_rise = {
        "time_gap.direction": "falling",
        "time_gap.overall_gap": 100,
        "time_gap.middle_has_significant_rise": True,
    }
    recovery_merge = {
        "time_gap.starting_direction": "rising",
        "time_gap.middle_has_new_significant_rise": False,
        "time_gap.flattening_at_end": True,
    }

    assert not deterministic.evaluate_requirements(msp, rising_only_at_start).matched
    assert not deterministic.evaluate_requirements(msp, {
        **rising_only_at_start,
        "time_gap.overall_gap": 50,
        "time_gap.middle_has_significant_rise": True,
    }).matched
    assert deterministic.evaluate_requirements(msp, {
        **rising_only_at_start,
        "time_gap.overall_gap": 50.1,
        "time_gap.middle_has_significant_rise": True,
    }).matched
    assert not deterministic.evaluate_requirements(rm, falling_with_middle_rise).matched
    assert deterministic.evaluate_requirements(rm, recovery_merge).matched
    assert not deterministic.evaluate_requirements(rm, {
        **recovery_merge, "time_gap.starting_direction": "falling",
    }).matched
    assert not deterministic.evaluate_requirements(rm, {
        **recovery_merge, "time_gap.middle_has_new_significant_rise": True,
    }).matched
    assert not deterministic.evaluate_requirements(rm, {
        **recovery_merge, "time_gap.flattening_at_end": False,
    }).matched


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
    assert result.passed == []
    assert result.failed == ["time_gap.direction: unavailable"]


def test_failed_requirement_reports_facts_from_closest_branch_only():
    result = deterministic.evaluate_requirements(
        {
            "enabled": True,
            "any_of": [
                {"all_of": [
                    {"fact": "time_gap.direction", "operator": "eq", "value": "rising"},
                    {"fact": "time_gap.overall_gap", "operator": "gt", "value": 50},
                ]},
                {"all_of": [
                    {"fact": "time_gap.direction", "operator": "eq", "value": "falling"},
                    {"fact": "time_gap.end_ms", "operator": "gt", "value": 0},
                    {"fact": "time_gap.ending_direction", "operator": "eq", "value": "rising"},
                ]},
            ],
        },
        {"time_gap.direction": "rising", "time_gap.overall_gap": 50},
    )

    assert not result.matched
    assert result.passed == ["time_gap.direction: 'rising'"]
    assert result.failed == ["time_gap.overall_gap: 50"]


def test_pit_stop_requires_pit_section_and_raw_telemetry():
    df = pd.DataFrame({
        "Graphics_player_pos_x": [0.0, 1.0, 2.0, 3.0],
        "Graphics_player_pos_y": [4.0, 4.0, 4.0, 4.0],
        "expert_optimal_player_pos_x": [0.0, 1.0, 2.0, 3.0],
        "expert_optimal_player_pos_y": [0.0, 0.0, 0.0, 0.0],
        "Physics_speed_kmh": [40.0, 40.0, 40.0, 40.0],
        "expert_optimal_speed": [100.0, 100.0, 100.0, 100.0],
        "speed_difference": [-999.0, -999.0, -999.0, -999.0],
        "trajectory_offset": [0.0, 0.0, 0.0, 0.0],
    })
    requirements = deterministic._requirements_for(
        "PS", deterministic.get_label("PS")
    )
    pit_facts, _ = deterministic.calculate_facts(
        df, 0, 3, section_id="silverstone22"
    )
    straight_facts, _ = deterministic.calculate_facts(
        df, 0, 3, section_id="silverstone13"
    )

    assert pit_facts["section.name"] == "Pit"
    assert pit_facts["section.overlap_names"] == ["Pit"]
    assert pit_facts["trajectory.peak_abs_offset_m"] == 4.0
    assert pit_facts["speed.gap_peak_abs_kmh"] == 60.0
    assert deterministic.evaluate_requirements(requirements, pit_facts).matched
    assert not deterministic.evaluate_requirements(requirements, straight_facts).matched


def test_pit_stop_checks_overlaps_when_splitter_selects_adjacent_straight(monkeypatch):
    class Attachment:
        content = {
            "best_match": None,
            "top_matches": [
                {"label_id": "brands_hatch1"},
                {"label_id": "brands_hatch17"},
            ],
        }

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.locate_circuit_section",
        lambda *_args, **_kwargs: Attachment(),
    )
    monkeypatch.setattr(
        deterministic,
        "calculate_facts",
        lambda *_args, **_kwargs: ({
            "trajectory.peak_abs_offset_m": 4.0,
            "speed.expert_faster": True,
            "speed.gap_peak_abs_kmh": 60.0,
        }, []),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(4)),
        lap_start=0,
        lap_end=3,
        section_id="brands_hatch1",
        section_start=0,
        section_end=3,
        circuit_id="brands_hatch",
    )

    assert result.section_id == "brands_hatch17"
    assert result.label_ids == ["brands_hatch", "brands_hatch17", "PS"]


def test_pit_stop_accepts_ten_metre_separation_without_speed_evidence():
    requirements = deterministic._requirements_for(
        "PS", deterministic.get_label("PS")
    )

    assert deterministic.evaluate_requirements(requirements, {
        "section.overlap_names": ["Pit"],
        "trajectory.peak_abs_offset_m": 10.0,
    }).matched
    assert not deterministic.evaluate_requirements(requirements, {
        "section.overlap_names": ["Pit"],
        "trajectory.peak_abs_offset_m": 9.9,
    }).matched
    assert not deterministic.evaluate_requirements(requirements, {
        "section.overlap_names": ["Straight"],
        "trajectory.peak_abs_offset_m": 10.0,
    }).matched


def test_far_driver_in_overlapping_pit_prefers_ps_over_rm(monkeypatch):
    monkeypatch.setattr(
        deterministic,
        "calculate_facts",
        lambda *_args, **_kwargs: ({
            "trajectory.peak_abs_offset_m": 10.0,
            "time_gap.direction": "falling",
            "time_gap.overall_gap": 100,
        }, []),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(4)),
        lap_start=0,
        lap_end=3,
        section_id="silverstone22",
        section_start=0,
        section_end=3,
        circuit_id="silverstone",
    )

    assert result.label_ids == ["silverstone", "silverstone22", "PS"]
    assert all(item["value"] != "PS / RM" for item in result.rejected_proposals)


def test_balance_and_grip_are_calculated_from_raw_tire_telemetry():
    df = pd.DataFrame({
        "Physics_slip_angle_front_left": [0.01, 0.01],
        "Physics_slip_angle_front_right": [0.01, 0.01],
        "Physics_slip_angle_rear_left": [0.20, 0.20],
        "Physics_slip_angle_rear_right": [0.20, 0.20],
        "Physics_slip_ratio_front_left": [0.01, 0.01],
        "Physics_slip_ratio_front_right": [0.01, 0.01],
        "Physics_slip_ratio_rear_left": [0.20, 0.20],
        "Physics_slip_ratio_rear_right": [0.20, 0.20],
        "trajectory_balance": [-1.0, -1.0],
        "driver_push_to_limit": [0.0, 0.0],
    })
    facts, _ = deterministic.calculate_facts(df, 0, 1)
    assert facts["balance.oversteer"] is True
    assert facts["balance.understeer"] is False
    assert facts["grip.over_limit"] is True


def test_section_name_comes_from_catalog_mapping():
    facts, _ = deterministic.calculate_facts(
        pd.DataFrame(index=range(3)), 0, 2, section_id="silverstone22"
    )
    assert facts["section.name"] == "Pit"


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


def test_rm_sub_labels_require_rm_parent_but_only_evaluate_own_facts(monkeypatch):
    def fake_facts(_df, start, end, **_kwargs):
        ranges = [(1, 4)] if (start, end) == (0, 5) else []
        return {"trajectory.converging": True}, ranges

    monkeypatch.setattr(deterministic, "calculate_facts", fake_facts)
    without_rm = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(6)),
        parent_start=0,
        parent_end=5,
        parent_main_labels=["EA"],
    )
    with_rm = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(6)),
        parent_start=0,
        parent_end=5,
        parent_main_labels=["RM"],
    )

    assert "RM7" not in without_rm.final_labels
    assert "RM7" in with_rm.final_labels


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
    assert "Failed — time_gap.total_change_abs_ms: unavailable" in rejected["EA"]
    assert "branch" not in rejected["EA"]
    assert " operator " not in rejected["EA"]


def test_annotation_flows_do_not_import_removed_retrieval_code():
    root = Path(__file__).parents[1]
    flow_text = "\n".join(
        path.read_text()
        for path in (root / "app/local_annotation_agent/workflow/flows").glob("*.py")
    )
    assert "preflight" not in flow_text
    assert "label_search" not in flow_text
    assert (root / "app/external_knowledge_base/_embedder.py").exists()
