import json
from pathlib import Path

import numpy as np
import pandas as pd

import app.local_annotation_agent.workflow as workflow
from app.local_annotation_agent.workflow import deterministic


def _expert_corner_dataframe() -> pd.DataFrame:
    theta = np.linspace(0.0, np.pi / 2.0, 31)
    return pd.DataFrame({
        "expert_optimal_player_pos_x": 20.0 * np.cos(theta),
        "expert_optimal_player_pos_y": 20.0 * np.sin(theta),
        "Physics_brake": np.zeros(theta.size),
        "expert_optimal_brake": np.zeros(theta.size),
        "Physics_gas": np.zeros(theta.size),
        "expert_optimal_throttle": np.zeros(theta.size),
    })


def _isolate_phase_fact_sources(monkeypatch) -> None:
    monkeypatch.setattr(deterministic, "_slope_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(deterministic, "_opponent_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(deterministic, "_trajectory_offset", lambda _segment: None)


def test_telemetry_is_smoothed_once_with_centered_three_sample_median():
    df = pd.DataFrame({"signal": [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]})

    telemetry = deterministic._smoothed_telemetry(df)
    values = deterministic._series(telemetry, "signal")

    assert values.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert df["signal"].tolist() == [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]


def test_all_dataframe_fact_sources_receive_centrally_smoothed_telemetry(monkeypatch):
    df = pd.DataFrame({
        "expert_time_difference": [0.0, 0.0, 40.0, 0.0, 0.0],
    })
    received = {}

    def capture(name, result):
        def fake(telemetry, *_args, **_kwargs):
            received[name] = telemetry["expert_time_difference"].tolist()
            return result
        return fake

    monkeypatch.setattr(deterministic, "_slope_facts", capture("slope", {}))
    monkeypatch.setattr(deterministic, "_shape_facts", capture("shape", ({}, [])))
    monkeypatch.setattr(deterministic, "_opponent_facts", capture("opponent", {}))
    monkeypatch.setattr(deterministic, "_trajectory_offset", capture("trajectory", None))

    deterministic.calculate_facts(df, 0, 4)

    assert received == {
        name: [0.0, 0.0, 0.0, 0.0, 0.0]
        for name in ("slope", "shape", "opponent", "trajectory")
    }


def test_closing_evidence_is_limited_to_decreasing_magnitude_runs():
    ranges = deterministic._decreasing_magnitude_ranges(
        np.array([5.0, 4.0, 3.0, 4.0, 2.0, 3.0]),
        np.arange(8, 14),
    )

    assert ranges == [(8, 10), (11, 12)]


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
            "time_gap.total_change_ms": -20,
            "time_gap.ending_direction": "falling",
        },
    )
    second = deterministic.evaluate_requirements(
        requirements,
        {
            "time_gap.total_change_ms": 50,
            "brake.similarity": 1.0,
            "throttle.similarity": 1.0,
        },
    )
    assert first.matched and first.branch == 0
    assert second.matched and second.branch == 1


def test_slope_facts_distinguish_start_middle_and_end_rises(monkeypatch):
    def fake_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 100}, {"value": 200}],
            "extra": {
                "delta_value": 200,
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
    evidence = {}
    facts = deterministic._slope_facts(df, 0, 10, evidence)

    assert facts["time_gap.starting_direction"] == "rising"
    assert facts["time_gap.ending_direction"] == "rising"
    assert facts["time_gap.rising_ranges"] == [[0, 2], [8, 10]]
    assert facts["time_gap.falling_ranges"] == [[2, 8]]
    assert facts["time_gap.flat_ranges"] == []
    assert evidence["time_gap.rising_ranges"] == [(0, 2), (8, 10)]
    assert evidence["time_gap.falling_ranges"] == [(2, 8)]
    assert evidence["time_gap.direction"] == [(0, 10)]
    assert facts["time_gap.middle_has_rise"] is False
    assert facts["time_gap.flattening_at_end"] is False
    assert facts["time_gap.overall_gap"] == 200
    assert "time_gap.significant" not in facts


def test_slope_facts_smooth_ending_slope_windows():
    step_slopes = [20.0] * 6 + [10.0] * 3 + [5.0, 5.0, 100.0]
    df = pd.DataFrame({
        "expert_time_difference": np.cumsum([0.0, *step_slopes]),
    })

    facts = deterministic._slope_facts(df, 0, 12)

    assert facts["time_gap.flattening_at_end"] is True


def test_slope_facts_preserve_insignificant_runs_without_selecting_mistake():
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 5, 10, 15, 20, 20, 20, 20],
    })

    facts = deterministic._slope_facts(df, 0, 9)

    assert facts["time_gap.rising_ranges"] == [[2, 6]]
    assert facts["time_gap.falling_ranges"] == []
    assert facts["time_gap.flat_ranges"] == [[0, 2], [6, 9]]


def test_slope_facts_do_not_select_local_rise_when_total_change_falls():
    df = pd.DataFrame({
        "expert_time_difference": [
            4240, 4230, 4220, 4195, 4145, 4070, 4020, 4000, 4050,
            4145, 4240, 4305, 4280, 4255, 4220, 4198, 4185,
        ],
    })
    evidence = {}

    facts = deterministic._slope_facts(df, 0, 16, evidence)
    msp = deterministic._requirements_for("MSP", deterministic.get_label("MSP"))

    assert facts["time_gap.total_change_ms"] == -55
    assert evidence["time_gap.total_change_ms"] == [(0, 16)]
    assert "time_gap.total_change_abs_ms" not in facts
    assert facts["time_gap.direction"] == "falling"
    assert not deterministic.evaluate_requirements(msp, facts).matched


def test_slope_facts_accept_steady_rise():
    for rate in (20, 200):
        df = pd.DataFrame({
            "expert_time_difference": np.arange(9) * rate,
        })

        facts = deterministic._slope_facts(df, 0, 8)

        assert facts["time_gap.rising_ranges"] == [[0, 8]]


def test_slope_facts_identify_accelerating_middle_rise_then_flattening(monkeypatch):
    def fake_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 100}, {"value": 200}],
            "extra": {
                "delta_value": 200,
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

    assert facts["time_gap.rising_ranges"] == [[0, 7]]
    assert facts["time_gap.flat_ranges"] == [[7, 10]]
    assert facts["time_gap.middle_has_rise"] is False
    assert facts["time_gap.flattening_at_end"] is True

    evaluation = deterministic.evaluate_requirements(
        {"any_of": [{"all_of": [{
            "fact": "time_gap.total_change_ms",
            "operator": "gte",
            "value": 150,
        }]}]},
        facts,
    )
    assert evaluation.passed == [
        "time_gap.total_change_ms: 200",
    ]

    def fake_rise_fall_flat_query(_df, _name, _args):
        return ({
            "samples": [{"value": 0}, {"value": 100}, {"value": 50}],
            "extra": {
                "delta_value": 50,
                "total_change_direction": "rising",
                "total_change_is_label_significant": False,
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

    assert facts["time_gap.rising_ranges"] == [[0, 3]]
    assert facts["time_gap.falling_ranges"] == [[3, 7]]
    assert facts["time_gap.flat_ranges"] == [[7, 10]]
    assert facts["time_gap.flattening_at_end"] is True


def test_slope_facts_identify_single_rise():
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 0, 200, 200, 200, 200, 200, 200],
    })

    facts = deterministic._slope_facts(df, 0, 9)

    assert facts["time_gap.total_change_ms"] == 200


def test_calculate_facts_smooths_single_sample_noise_before_slope_facts(monkeypatch):
    df = pd.DataFrame({
        "expert_time_difference": [0, 0, 0, 0, 40, 0, 0, 0, 0, 0],
    })
    monkeypatch.setattr(
        deterministic, "_shape_facts", lambda *_args, **_kwargs: ({}, [])
    )
    monkeypatch.setattr(deterministic, "_opponent_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(deterministic, "_trajectory_offset", lambda _segment: None)

    facts, _ = deterministic.calculate_facts(df, 0, 9)

    assert facts["time_gap.rising_ranges"] == []
    assert facts["time_gap.total_change_ms"] == 0
    assert facts["time_gap.middle_has_rise"] is False


def test_behavior_requirements_use_middle_rise_for_msp():
    msp = deterministic._requirements_for("MSP", deterministic.get_label("MSP"))
    rm = deterministic._requirements_for("RM", deterministic.get_label("RM"))

    no_middle_rise = {
        "time_gap.total_change_ms": 150,
        "time_gap.middle_has_rise": False,
    }
    middle_rise = {
        "time_gap.direction": "rising",
        "time_gap.total_change_ms": 150,
        "time_gap.middle_has_rise": True,
    }
    recovery_merge = {
        "time_gap.starting_direction": "rising",
        "time_gap.middle_has_rise": False,
        "time_gap.flattening_at_end": True,
    }

    assert not deterministic.evaluate_requirements(msp, no_middle_rise).matched
    assert deterministic.evaluate_requirements(msp, middle_rise).matched
    assert not deterministic.evaluate_requirements(msp, {
        **middle_rise, "time_gap.total_change_ms": 49,
    }).matched
    assert not deterministic.evaluate_requirements(msp, {
        **middle_rise, "time_gap.middle_has_rise": False,
    }).matched
    assert not deterministic.evaluate_requirements(rm, middle_rise).matched
    assert deterministic.evaluate_requirements(rm, recovery_merge).matched
    assert not deterministic.evaluate_requirements(rm, {
        **recovery_merge, "time_gap.starting_direction": "falling",
    }).matched
    assert not deterministic.evaluate_requirements(rm, {
        **recovery_merge, "time_gap.middle_has_rise": True,
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


def test_lap_omits_sub_labels_that_cover_only_part_of_segment(monkeypatch):
    facts = deterministic.FactSet(
        {
            "time_gap.starting_direction": "rising",
            "time_gap.middle_has_rise": False,
            "time_gap.flattening_at_end": True,
            "trajectory.peak_abs_offset_m": 2.0,
            "trajectory.converging": True,
            "speed.expert_faster": True,
            "speed.gap_peak_abs_kmh": 30.0,
            "speed.gap_closing": True,
        },
        evidence={
            "trajectory.peak_abs_offset_m": [(12, 12)],
            "trajectory.converging": [(10, 16)],
            "speed.expert_faster": [(18, 20)],
            "speed.gap_peak_abs_kmh": [(19, 19)],
            "speed.gap_closing": [(18, 22)],
        },
    )
    monkeypatch.setattr(
        deterministic, "calculate_facts", lambda *_args, **_kwargs: (facts, []),
    )
    monkeypatch.setattr(
        deterministic, "_resolve_circuit_sections",
        lambda *_args, **_kwargs: ("silverstone1", ["silverstone1"]),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(8, 30)),
        lap_start=8,
        lap_end=29,
        section_id="silverstone1",
        section_start=8,
        section_end=29,
        circuit_id="silverstone",
    )

    assert "RM selected for iloc range [8, 29]" in result.reasoning
    assert all(label not in result.label_ids for label in ("RM1", "RM5", "RM7"))
    assert result.reasoning.count("\n") == 0


def test_lap_omits_sub_label_that_is_not_fully_inside_segment(monkeypatch):
    facts = deterministic.FactSet(
        {
            "time_gap.starting_direction": "rising",
            "time_gap.middle_has_rise": False,
            "time_gap.flattening_at_end": True,
            "trajectory.converging": True,
        },
        evidence={"trajectory.converging": [(6, 12)]},
    )
    monkeypatch.setattr(
        deterministic, "calculate_facts", lambda *_args, **_kwargs: (facts, []),
    )
    monkeypatch.setattr(
        deterministic, "_resolve_circuit_sections",
        lambda *_args, **_kwargs: ("silverstone1", ["silverstone1"]),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(11)),
        lap_start=0,
        lap_end=10,
        section_id="silverstone1",
        section_start=0,
        section_end=10,
        circuit_id="silverstone",
    )

    assert "RM" in result.label_ids
    assert "RM7" not in result.label_ids
    assert "RM7 selected" not in result.reasoning


def test_lap_selects_sub_label_that_covers_entire_segment(monkeypatch):
    facts = deterministic.FactSet(
        {
            "time_gap.starting_direction": "rising",
            "time_gap.middle_has_rise": False,
            "time_gap.flattening_at_end": True,
            "trajectory.converging": True,
        },
        evidence={"trajectory.converging": [(0, 10)]},
    )
    monkeypatch.setattr(
        deterministic, "calculate_facts", lambda *_args, **_kwargs: (facts, []),
    )
    monkeypatch.setattr(
        deterministic, "_resolve_circuit_sections",
        lambda *_args, **_kwargs: ("silverstone1", ["silverstone1"]),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(11)),
        lap_start=0,
        lap_end=10,
        section_id="silverstone1",
        section_start=0,
        section_end=10,
        circuit_id="silverstone",
    )

    assert "RM7" in result.label_ids
    assert "RM7 selected for iloc range [0, 10]" in result.reasoning


def test_lap_supporting_evidence_cannot_expand_partial_required_match(monkeypatch):
    facts = deterministic.FactSet(
        {
            "time_gap.total_change_ms": 150,
            "time_gap.middle_has_rise": True,
            "grip.over_limit": True,
            "speed.gap_peak_abs_kmh": 20.0,
        },
        evidence={
            "grip.over_limit": [(3, 7)],
            "speed.gap_peak_abs_kmh": [(0, 10)],
        },
        phases={"entry": [(0, 10)]},
    )
    monkeypatch.setattr(
        deterministic, "calculate_facts", lambda *_args, **_kwargs: (facts, []),
    )
    monkeypatch.setattr(
        deterministic, "_resolve_circuit_sections",
        lambda *_args, **_kwargs: ("silverstone1", ["silverstone1"]),
    )

    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(11)),
        lap_start=0,
        lap_end=10,
        section_id="silverstone1",
        section_start=0,
        section_end=10,
        circuit_id="silverstone",
    )

    assert "MSP" in result.label_ids
    assert "MSP42" not in result.label_ids
    assert "MSP42 selected" not in result.reasoning


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
        facts = deterministic.FactSet(
            {"trajectory.converging": True},
            evidence={"trajectory.converging": [(start, end)]},
        )
        return facts, ranges

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


def test_sub_label_range_uses_required_evidence_and_phase_boundary(monkeypatch):
    def fake_facts(_df, start, end, **_kwargs):
        if (start, end) == (0, 10):
            return deterministic.FactSet(
                {}, phases={"entry": [(1, 5)], "exit": [(5, 9)]},
            ), [(1, 6)]
        return deterministic.FactSet(
            {
                "brake.application_onset_relation": "later",
                "brake.application_end_relation": "later",
            },
            evidence={
                "brake.application_onset_relation": [(0, 3)],
                "brake.application_end_relation": [(4, 6)],
            },
            phases={"entry": [(1, 5)]},
        ), []

    monkeypatch.setattr(deterministic, "calculate_facts", fake_facts)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(11)),
        parent_start=0,
        parent_end=10,
        parent_main_labels=["MSP"],
    )

    proposal = next(item for item in result.label_annotations if item["label_id"] == "MSP1")
    assert (proposal["start_index"], proposal["end_index"]) == (1, 5)
    assert "MSP1 selected for iloc range [1, 5]" in proposal["reasoning"]


def test_optional_supporting_evidence_expands_and_is_cited(monkeypatch):
    def fake_facts(_df, start, end, **_kwargs):
        if (start, end) == (0, 10):
            return deterministic.FactSet(
                {"time_gap.direction": "rising"},
                evidence={"time_gap.direction": [(7, 8)]},
                phases={"entry": [(1, 5)]},
            ), [(1, 5)]
        return deterministic.FactSet(
            {
                "brake.application_onset_relation": "later",
                "brake.application_end_relation": "later",
            },
            evidence={
                "brake.application_onset_relation": [(2, 3)],
                "brake.application_end_relation": [(3, 4)],
            },
            phases={"entry": [(1, 5)]},
        ), []

    monkeypatch.setattr(deterministic, "calculate_facts", fake_facts)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(11)),
        parent_start=0,
        parent_end=10,
        parent_main_labels=["MSP"],
    )

    proposal = next(item for item in result.label_annotations if item["label_id"] == "MSP1")
    assert (proposal["start_index"], proposal["end_index"]) == (2, 8)
    assert "Supporting — time_gap.direction: 'rising'" in proposal["reasoning"]


def test_sub_label_without_required_provenance_is_omitted(monkeypatch):
    def fake_facts(_df, start, end, **_kwargs):
        ranges = [(1, 4)] if (start, end) == (0, 5) else []
        return {"trajectory.converging": True}, ranges

    monkeypatch.setattr(deterministic, "calculate_facts", fake_facts)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(6)),
        parent_start=0,
        parent_end=5,
        parent_main_labels=["RM"],
    )

    assert "RM7" not in result.final_labels


def test_calculated_comparison_facts_keep_driver_and_expert_provenance(monkeypatch):
    monkeypatch.setattr(deterministic, "_slope_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(deterministic, "_shape_facts", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(deterministic, "_opponent_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        deterministic, "_trajectory_offset",
        lambda _segment: np.array([0.0, 0.2, 0.8, 1.2, 0.7, 0.3, 0.0]),
    )
    df = pd.DataFrame({
        "Physics_brake": [0, 0, 0.2, 1, 1, 0.2, 0],
        "expert_optimal_brake": [0, 0.2, 1, 1, 0.2, 0, 0],
        "Physics_gas": [0, 0, 0, 0, 0.2, 1, 1],
        "expert_optimal_throttle": [0, 0, 0, 0.2, 1, 1, 1],
        "Physics_steer_angle": [0, 0, 0.1, 0.5, 1, 0.2, 0],
        "expert_optimal_steering": [0, 0.1, 0.5, 1, 0.2, 0, 0],
        "Physics_gear": [2, 2, 2, 2, 2, 3, 3],
        "expert_optimal_gear": [2, 2, 2, 2, 3, 3, 3],
        "Physics_speed_kmh": [80, 82, 84, 86, 88, 90, 92],
        "expert_optimal_speed": [82, 84, 87, 90, 92, 94, 96],
    })

    facts, _ = deterministic.calculate_facts(df, 0, 6)

    comparison_facts = (
        "brake.application_onset_relation",
        "throttle.application_onset_relation",
        "turn.in_relation",
        "gear.upshift_relation",
    )
    for fact_id in comparison_facts:
        assert facts.evidence[fact_id][0][0] < facts.evidence[fact_id][0][1]
    for fact_id in (
        "speed.gap_peak_abs_kmh",
        "trajectory.position",
    ):
        assert facts.evidence[fact_id]


def test_phase_facts_come_from_expert_curvature_independently_of_controls(
    monkeypatch,
):
    _isolate_phase_fact_sources(monkeypatch)
    no_controls = _expert_corner_dataframe()
    changed_controls = no_controls.copy()
    changed_controls["Physics_brake"] = np.linspace(0.0, 1.0, len(changed_controls))
    changed_controls["expert_optimal_brake"] = 0.25
    changed_controls["Physics_gas"] = np.linspace(1.0, 0.0, len(changed_controls))
    changed_controls["expert_optimal_throttle"] = 0.75

    no_control_facts, _ = deterministic.calculate_facts(
        no_controls, 0, len(no_controls),
    )
    changed_control_facts, _ = deterministic.calculate_facts(
        changed_controls, 0, len(changed_controls),
    )

    assert no_control_facts.phases == changed_control_facts.phases
    for phase_name in ("entry", "apex", "exit"):
        fact_id = f"phase.{phase_name}"
        assert no_control_facts[fact_id] is True
        assert changed_control_facts[fact_id] is True
        assert no_control_facts.evidence[fact_id] == no_control_facts.phases[phase_name]
        assert changed_control_facts.evidence[fact_id] == changed_control_facts.phases[phase_name]
    assert no_control_facts["brake.peak_relation"] == "aligned"
    assert changed_control_facts["brake.peak_relation"] == "higher"


def test_controls_do_not_fallback_to_phase_facts_without_expert_curvature(
    monkeypatch,
):
    _isolate_phase_fact_sources(monkeypatch)
    controls = {
        "Physics_brake": np.ones(31),
        "expert_optimal_brake": np.full(31, 0.5),
        "Physics_gas": np.linspace(0.0, 1.0, 31),
        "expert_optimal_throttle": np.full(31, 0.5),
    }
    missing_expert = pd.DataFrame(controls)
    straight_expert = pd.DataFrame({
        **controls,
        "expert_optimal_player_pos_x": np.arange(31, dtype=float),
        "expert_optimal_player_pos_y": np.zeros(31),
    })

    for telemetry in (missing_expert, straight_expert):
        facts, phase_ranges = deterministic.calculate_facts(
            telemetry, 0, len(telemetry),
        )

        assert phase_ranges == []
        assert facts.phases == {}
        assert all(
            f"phase.{phase_name}" not in facts
            for phase_name in ("entry", "apex", "exit")
        )


def test_detailed_subrange_synchronizes_inherited_expert_phase_facts(monkeypatch):
    parent_facts = deterministic.FactSet(
        {"phase.entry": True},
        evidence={"phase.entry": [(1, 5)]},
        phases={"entry": [(1, 5)]},
    )
    candidate_facts = deterministic.FactSet(
        {"phase.exit": True},
        evidence={"phase.exit": [(2, 4)]},
        phases={"exit": [(2, 4)]},
    )
    calls = iter(((parent_facts, [(1, 5)]), (candidate_facts, [])))
    evaluated_facts = []

    monkeypatch.setattr(
        deterministic, "calculate_facts", lambda *_args, **_kwargs: next(calls),
    )
    monkeypatch.setattr(
        deterministic, "_candidate_ranges", lambda *_args, **_kwargs: [(2, 4)],
    )

    def capture_evaluation(_label_ids, facts):
        evaluated_facts.append(facts)
        return deterministic.LabelEvaluation([], {})

    monkeypatch.setattr(deterministic, "evaluate_labels", capture_evaluation)

    deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(6)),
        parent_start=0,
        parent_end=5,
        parent_main_labels=[],
    )

    assert evaluated_facts[0]["phase.entry"] is True
    assert evaluated_facts[0].phases["entry"] == [(2, 4)]
    assert evaluated_facts[0].evidence["phase.entry"] == [(2, 4)]
    assert "phase.exit" not in evaluated_facts[0]
    assert "exit" not in evaluated_facts[0].phases
    assert "phase.exit" not in evaluated_facts[0].evidence


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
            "time_gap.total_change_ms": -10,
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
    assert "time_gap.total_change_ms" in rejected["EA"]
    assert "Failed — time_gap.total_change_ms: unavailable" in rejected["EA"]
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
