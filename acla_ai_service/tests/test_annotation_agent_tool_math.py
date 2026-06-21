import numpy as np
import pandas as pd

from app.shared.annotation_agent_tools import (
    _classify_base_segment_shape,
    _query_compute_slope,
    build_graph,
    locate_circuit_section,
    measure_segment_shape,
    run_pipeline_query,
)
from app.local_annotation_agent.workflow.preflight import (
    PreflightContext,
    _prompt_block,
    _query_semantic_tags,
    _run_queries,
    _semantic_tool_output,
)
from app.local_annotation_agent.workflow import preflight_detailed, preflight_lap
from app.local_annotation_agent.workflow.flows import detailed as detailed_flow
from app.shared.contracts import Attachment


def _trajectory_df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "expert_optimal_player_pos_x": x,
        "expert_optimal_player_pos_y": y,
        "expert_optimal_player_pos_z": np.linspace(0.0, 1.0, len(x)),
        "Graphics_player_pos_x": x.copy(),
        "Graphics_player_pos_y": y.copy(),
        "Graphics_player_pos_z": np.linspace(0.0, 1.0, len(x)),
        "expert_optimal_speed": np.full(len(x), 80.0),
        "expert_optimal_steering": np.zeros(len(x)),
    })


def test_time_difference_uses_exact_expert_time_difference_column():
    df = pd.DataFrame(
        {"expert_time_difference": [100.0, 150.0, 225.0, 300.0]},
        index=[10, 11, 12, 13],
    )

    result = _query_compute_slope(df, 10, 13, "expert_time_difference")

    assert result is not None
    assert result["extra"]["unit"] == "ms"
    assert result["extra"]["delta_value"] == 200.0


def test_time_difference_to_expert_alias_is_not_used():
    df = pd.DataFrame(
        {"expert_time_difference": [100.0, 150.0, 225.0, 300.0]},
        index=[10, 11, 12, 13],
    )

    result = _query_compute_slope(df, 10, 13, "time_difference_to_expert")

    assert result is None


def test_time_difference_rising_concave_down_is_recovery_trend():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                0.0,
                150.0,
                280.0,
                380.0,
                450.0,
                500.0,
                530.0,
                550.0,
            ],
        },
        index=range(10, 18),
    )

    result = _query_compute_slope(df, 10, 17, "expert_time_difference")

    assert result is not None
    assert result["extra"]["total_change_direction"] == "rising"
    assert result["extra"]["slope_shape"] == "slope_decreasing_over_section"

    tags = _query_semantic_tags(
        {
            "graph_id": "time_delta",
            "query_id": "compute_slope",
            "params": {"column": "expert_time_difference"},
        },
        result,
    )
    assert "recovery trend" in tags
    assert "rate of losing time decreasing" in tags


def test_time_difference_rising_concave_up_is_losing_time_accelerating():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                0.0,
                20.0,
                50.0,
                90.0,
                150.0,
                240.0,
                360.0,
                500.0,
            ],
        },
        index=range(20, 28),
    )

    result = _query_compute_slope(df, 20, 27, "expert_time_difference")

    assert result is not None
    assert result["extra"]["total_change_direction"] == "rising"
    assert result["extra"]["slope_shape"] == "slope_increasing_over_section"

    tags = _query_semantic_tags(
        {
            "graph_id": "time_delta",
            "query_id": "compute_slope",
            "params": {"column": "expert_time_difference"},
        },
        result,
    )
    assert "losing time accelerating" in tags


def test_time_difference_linear_rise_has_steady_slope_shape():
    df = pd.DataFrame(
        {"expert_time_difference": [0.0, 100.0, 200.0, 300.0, 400.0]},
        index=range(30, 35),
    )

    result = _query_compute_slope(df, 30, 34, "expert_time_difference")

    assert result is not None
    assert result["extra"]["slope_shape"] == "slope_steady_over_section"


def test_time_difference_detects_reversal_to_falling_within_section():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                0.0,
                200.0,
                400.0,
                450.0,
                250.0,
                50.0,
            ],
        },
        index=range(40, 46),
    )

    result = _query_compute_slope(df, 40, 45, "expert_time_difference")

    assert result is not None
    assert (
        result["extra"]["slope_shape"]
        == "reversing_to_falling_within_section"
    )


def test_time_difference_detects_reversal_to_rising_within_section():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                500.0,
                300.0,
                100.0,
                50.0,
                250.0,
                450.0,
            ],
        },
        index=range(50, 56),
    )

    result = _query_compute_slope(df, 50, 55, "expert_time_difference")

    assert result is not None
    assert (
        result["extra"]["slope_shape"]
        == "reversing_to_rising_within_section"
    )


def test_trajectory_offset_uses_generic_slope_shape_detection():
    df = pd.DataFrame(
        {
            "trajectory_offset": [
                0.0,
                0.30,
                0.55,
                0.72,
                0.82,
                0.88,
                0.91,
                0.93,
            ],
        },
        index=range(60, 68),
    )

    result = _query_compute_slope(df, 60, 67, "trajectory_offset")

    assert result is not None
    assert result["extra"]["slope_shape"] == "slope_decreasing_over_section"


def test_locate_circuit_section_filters_by_circuit_id():
    df = pd.DataFrame(
        {"Graphics_normalized_car_position": [0.95, 0.96, 0.97]},
        index=[10, 11, 12],
    )

    brands = locate_circuit_section(df, "brands_hatch", 10, 13).content
    moza = locate_circuit_section(df, "moza", 10, 13).content

    assert brands["circuit_id"] == "brands_hatch"
    assert {match["label_id"] for match in brands["top_matches"]} == {
        "brands_hatch1",
        "brands_hatch17",
    }
    assert moza["circuit_id"] == "moza"
    assert [match["label_id"] for match in moza["top_matches"]] == ["moza1"]


def test_preflight_trajectory_offset_queries_repair_reset_segment_index():
    theta = np.linspace(0.0, np.pi / 2.0, 100)
    radius = 30.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))
    df["Graphics_player_pos_x"] = df["Graphics_player_pos_x"] + 0.5
    df["Graphics_player_pos_y"] = df["Graphics_player_pos_y"] + 0.2
    df["expert_time_difference"] = np.linspace(0.0, 100.0, len(df))
    df["speed_difference"] = np.linspace(-2.0, 4.0, len(df))

    results = dict(_run_queries(df, 1000, 1100))

    for tool_id in (
        "query_telemetry.find_extremum.trajectory_offset.max",
        "query_telemetry.find_extremum.trajectory_offset.min",
    ):
        content = results[tool_id]
        assert "error" not in content
        assert content["params"]["range"] == [1000, 1099]
        assert 1000 <= content["result"]["iloc"] <= 1099


def test_query_telemetry_does_not_derive_trajectory_offset_from_raw_dataframe():
    theta = np.linspace(0.0, np.pi / 2.0, 100)
    radius = 30.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))
    df["Graphics_player_pos_x"] = df["Graphics_player_pos_x"] + 0.5
    df["Graphics_player_pos_y"] = df["Graphics_player_pos_y"] + 0.2

    payload, error = run_pipeline_query(
        df,
        "find_extremum",
        {"range": [0, 99], "column": "trajectory_offset", "kind": "max"},
    )

    assert payload["extra"] is None
    assert error is not None
    assert "column 'trajectory_offset' is not in the graph table" in error


def test_trajectory_offset_builds_for_seven_sample_range():
    theta = np.linspace(0.0, np.pi / 8.0, 7)
    radius = 30.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))
    df["Graphics_player_pos_x"] = df["Graphics_player_pos_x"] + 0.5
    df["Graphics_player_pos_y"] = df["Graphics_player_pos_y"] + 0.2

    table = build_graph("trajectory_offset", df)

    assert table is not None
    assert len(table) == 7
    assert "trajectory_offset" in table.columns


def test_trajectory_offset_projects_player_to_expert_path():
    theta = np.linspace(0.0, np.pi / 2.0, 80)
    radius = 30.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    df = _trajectory_df(x, y)
    shift = 8
    df["Graphics_player_pos_x"] = np.concatenate([
        x[shift:],
        np.repeat(x[-1], shift),
    ])
    df["Graphics_player_pos_y"] = np.concatenate([
        y[shift:],
        np.repeat(y[-1], shift),
    ])

    table = build_graph("trajectory_offset", df)

    assert table is not None
    offset = table["trajectory_offset"].to_numpy(dtype=float)
    assert np.nanmax(np.abs(offset)) < 1e-6


def test_preflight_trajectory_offset_summary_separates_side_from_distance():
    df = pd.DataFrame(
        {"trajectory_offset": [-5.0, -3.0, -1.0]},
        index=range(3),
    )

    result = _query_compute_slope(df, 0, 2, "trajectory_offset")
    content = {
        "graph_id": "trajectory_offset",
        "query_id": "compute_slope",
        "params": {"column": "trajectory_offset"},
        "semantic_target": "trajectory offset",
        "semantic_tags": [],
        "result": result,
    }
    prompt = _prompt_block(
        "lap",
        0,
        2,
        [("query_telemetry.compute_slope.trajectory_offset", content)],
        [],
        [],
    )
    output = _semantic_tool_output(
        "query_telemetry.compute_slope.trajectory_offset",
        content,
    )

    assert result is not None
    assert "expert_line_relation=converging_to_expert_line" in prompt
    assert "absolute_offset_start=5.0 m" in prompt
    assert "absolute_offset_end=1.0 m" in prompt
    absolute_offset = output["analysis"]["absolute_offset"]
    assert absolute_offset["moves_toward_expert_line"] is True


def test_detailed_preflight_missing_query_tables_are_nonfatal(monkeypatch):
    captured = {}

    def fake_run_tools(df, start, end, tool_ids):
        captured["tool_ids"] = tool_ids
        return [
            ("compute_expert_phases", {"phases": []}),
            (
                "measure_segment_shape",
                {
                    "base_segment_shape": {
                        "segment_type_role": "base_segment_shape",
                        "shape_key": "straight",
                    },
                    "phases": [],
                },
            ),
        ]

    def fake_run_queries(df, start, end, query_specs):
        captured["query_specs"] = query_specs
        return [
            (
                "query_telemetry.compute_slope.trajectory_offset",
                {
                    "graph_id": "trajectory_offset",
                    "query_id": "compute_slope",
                    "params": {"column": "trajectory_offset", "range": [start, end]},
                    "error": "cannot build `trajectory_offset` graph table",
                    "analysis": {},
                    "semantic_tags": [],
                },
            )
        ]

    monkeypatch.setattr(preflight_detailed, "_run_tools", fake_run_tools)
    monkeypatch.setattr(preflight_detailed, "_run_queries", fake_run_queries)

    result = preflight_detailed.build_preflight_context(
        df=pd.DataFrame(),
        start=0,
        end=1,
        parent_main_labels=[],
        extra_query_terms=[],
    )

    assert result.label_candidates == []
    assert "preflight semantic candidates" not in result.prompt_block.lower()
    names = [attachment.name for attachment in result.attachments]
    assert "init.detailed_preflight_events" in names
    assert "init.preflight_label_candidates" not in names
    events_attachment = next(
        attachment
        for attachment in result.attachments
        if attachment.name == "init.detailed_preflight_events"
    )
    assert events_attachment.content["events"][0]["event"].startswith("on the straight")
    assert captured["query_specs"]


def test_detailed_preflight_events_capture_late_brake_widening_and_time_loss():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.18,
                0.30,
                0.42,
                0.54,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
            ],
            "expert_optimal_brake": [
                0.10,
                0.10,
                0.18,
                0.30,
                0.42,
                0.54,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
                0.62,
            ],
        },
        index=list(range(10, 31)),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {"phases": [{"entry": 10, "apex": 20, "exit": 30}]},
            ),
            (
                "query_telemetry.find_threshold_crossing.brake.onset",
                {
                    "result": {
                        "samples": [
                            {"column": "expert_optimal_brake", "iloc": 12},
                            {"column": "Physics_brake", "iloc": 18},
                        ],
                    },
                },
            ),
            (
                "query_telemetry.find_extremum.trajectory_offset.max",
                {"result": {"iloc": 17, "value": 0.85}},
            ),
            (
                "query_telemetry.compute_slope.trajectory_offset",
                {
                    "analysis": {
                        "total_change": {
                            "value": 0.7,
                            "domain_direction": "moving_wider",
                        },
                    },
                },
            ),
            (
                "query_telemetry.compute_slope.expert_time_difference",
                {
                    "analysis": {
                        "total_gap_change": {
                            "value": 350.0,
                            "gap_direction": "time_gap_rising",
                            "threshold_state": "label_threshold_met",
                        },
                        "slope_shape": "slope_steady_over_section",
                    },
                },
            ),
        ],
    )

    event_names = {event["event"] for event in events}
    assert "brake initiation onset later than expert" in event_names
    assert "trajectory wider than expert" in event_names
    assert "moving toward positive" in event_names
    assert "gap grows" in event_names
    assert "time loss" in event_names


def test_detailed_preflight_events_capture_recovery_and_speed_gap_closing():
    events = preflight_detailed._build_detailed_events(
        pd.DataFrame(),
        100,
        130,
        [
            (
                "query_telemetry.compute_slope.trajectory_offset",
                {
                    "analysis": {
                        "total_change": {
                            "value": -1.2,
                            "domain_direction": "moving_tighter",
                        },
                        "absolute_offset": {
                            "start": 2.0,
                            "end": 0.2,
                            "moves_toward_expert_line": True,
                        },
                    },
                },
            ),
            (
                "query_telemetry.find_extremum.speed_difference.max",
                {"result": {"iloc": 105, "value": 24.0}},
            ),
            (
                "query_telemetry.compute_slope.speed_difference",
                {
                    "analysis": {
                        "total_change": {
                            "value": -18.0,
                            "domain_direction": "speed_gap_decreasing",
                            "moves_toward_zero": True,
                        },
                    },
                },
            ),
            (
                "query_telemetry.find_extremum.player_speed.max",
                {"result": {"iloc": 120, "value": 181.0, "extra": {"unit": "km/h"}}},
            ),
            (
                "query_telemetry.find_extremum.player_speed.min",
                {"result": {"iloc": 101, "value": 96.0, "extra": {"unit": "km/h"}}},
            ),
            (
                "query_telemetry.find_trend_runs.player_speed",
                {
                    "analysis": {
                        "runs": [
                            {
                                "start_iloc": 100,
                                "end_iloc": 112,
                                "start_value": 96.0,
                                "end_value": 155.0,
                                "change": 59.0,
                                "unit": "km/h",
                                "slope": 4.9,
                                "direction": "rising",
                                "domain_direction": "rising",
                                "is_label_significant": True,
                            },
                            {
                                "start_iloc": 112,
                                "end_iloc": 130,
                                "start_value": 155.0,
                                "end_value": 140.0,
                                "change": -15.0,
                                "unit": "km/h",
                                "slope": -0.8,
                                "direction": "falling",
                                "domain_direction": "falling",
                                "is_label_significant": True,
                            },
                        ],
                    },
                },
            ),
            (
                "query_telemetry.compute_slope.player_speed",
                {
                    "analysis": {
                        "total_change": {
                            "value": 44.0,
                            "direction": "rising",
                            "domain_direction": "rising",
                            "is_label_significant": True,
                        },
                        "slope_shape": "reversing_to_falling_within_section",
                    },
                },
            ),
        ],
    )

    event_names = {event["event"] for event in events}
    assert "recovery toward expert line" in event_names
    assert "speed gap closing" in event_names
    assert "large speed gap over 20" in event_names
    assert "expert faster than player" in event_names
    assert "player speed maximum" in event_names
    assert "player speed minimum" in event_names
    assert "speed local curve rising" in event_names
    assert "speed local curve falling" in event_names
    assert "speed overall trend rising" in event_names


def test_detailed_preflight_events_capture_throttle_timing_and_peak():
    df = pd.DataFrame(
        {
            "Physics_gas": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.20,
                0.35,
                0.50,
                0.65,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
            ],
            "expert_optimal_throttle": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.20,
                0.35,
                0.50,
                0.65,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(
        df,
        0,
        20,
        [
            (
                "query_telemetry.find_threshold_crossing.throttle.onset",
                {
                    "result": {
                        "samples": [
                            {"column": "expert_optimal_throttle", "iloc": 12},
                            {"column": "Physics_gas", "iloc": 8},
                        ],
                    },
                },
            ),
            (
                "query_telemetry.find_extremum.throttle.player.max",
                {"result": {"iloc": 14, "value": 0.95}},
            ),
            (
                "query_telemetry.find_extremum.throttle.expert.max",
                {"result": {"iloc": 15, "value": 0.72}},
            ),
        ],
    )

    event_names = {event["event"] for event in events}
    assert "throttle application onset earlier than expert" in event_names
    assert "peak throttle pressure higher than expert" in event_names


def test_detailed_preflight_detects_fuzzy_brake_release_too_quickly():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.76,
                0.76,
                0.76,
                0.72,
                0.62,
                0.52,
                0.42,
                0.32,
                0.31,
                0.31,
                0.31,
                0.31,
            ],
            "expert_optimal_brake": [
                0.76,
                0.76,
                0.76,
                0.72,
                0.67,
                0.62,
                0.57,
                0.52,
                0.47,
                0.42,
                0.37,
                0.32,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 11, [])

    release = next(
        event
        for event in events
        if event["event"] == "brake release too quickly"
    )
    assert release["measurements"]["decision_basis"] == "fuzzy_change_speed_comparison"
    assert release["measurements"]["duration_delta_iloc"] <= -2
    assert release["measurements"]["slope_ratio"] >= 1.25


def test_detailed_preflight_ignores_old_thresholds_when_release_shape_matches():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.10,
                0.10,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
                0.00,
            ],
            "expert_optimal_brake": [
                0.80,
                0.80,
                0.72,
                0.64,
                0.56,
                0.48,
                0.40,
                0.32,
                0.24,
                0.16,
                0.08,
                0.00,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 11, [])
    event_names = {event["event"] for event in events}

    assert "brake release too quickly" not in event_names
    assert "brake release too slowly" not in event_names


def test_detailed_preflight_separates_early_timing_from_speed():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.10,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
            "expert_optimal_brake": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.50,
                0.50,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 11, [])
    event_names = {event["event"] for event in events}

    assert "brake initiation onset earlier than expert" in event_names
    assert "brake applied too quickly" not in event_names
    assert "brake applied too slowly" not in event_names


def test_detailed_preflight_separates_speed_from_aligned_timing():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.10,
                0.10,
                0.20,
                0.35,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
            "expert_optimal_brake": [
                0.10,
                0.10,
                0.18,
                0.26,
                0.34,
                0.42,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 9, [])
    event_names = {event["event"] for event in events}

    assert "brake applied too quickly" in event_names
    assert "brake initiation onset earlier than expert" not in event_names
    assert "brake initiation onset later than expert" not in event_names


def test_detailed_preflight_separates_throttle_speed_from_aligned_timing():
    df = pd.DataFrame(
        {
            "Physics_gas": [
                0.10,
                0.10,
                0.20,
                0.35,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
            "expert_optimal_throttle": [
                0.10,
                0.10,
                0.18,
                0.26,
                0.34,
                0.42,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 9, [])
    event_names = {event["event"] for event in events}

    assert "throttle applied too quickly" in event_names
    assert "throttle application onset earlier than expert" not in event_names
    assert "throttle application onset later than expert" not in event_names


def test_detailed_preflight_ignores_noisy_small_wiggles_as_actions():
    df = pd.DataFrame(
        {
            "Physics_brake": [
                0.50,
                0.51,
                0.49,
                0.50,
                0.51,
                0.50,
                0.49,
                0.50,
            ],
            "expert_optimal_brake": [
                0.50,
                0.49,
                0.50,
                0.51,
                0.50,
                0.49,
                0.50,
                0.50,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(df, 0, 7, [])
    event_names = {event["event"] for event in events}

    assert not any("brake initiation onset" in event for event in event_names)
    assert not any("brake release onset" in event for event in event_names)
    assert "brake applied too quickly" not in event_names
    assert "brake applied too slowly" not in event_names
    assert "brake release too quickly" not in event_names
    assert "brake release too slowly" not in event_names


def test_action_speed_requires_two_ilocs_of_duration_difference():
    player = max(
        preflight_detailed._change_episodes(
            [(0, 0.80), (1, 0.80), (2, 0.70), (3, 0.60), (4, 0.50), (5, 0.40)],
            "decrease",
        ),
        key=lambda episode: episode["total_movement"],
    )
    one_iloc_longer_expert = max(
        preflight_detailed._change_episodes(
            [
                (0, 0.80),
                (1, 0.80),
                (2, 0.72),
                (3, 0.64),
                (4, 0.56),
                (5, 0.48),
                (6, 0.40),
            ],
            "decrease",
        ),
        key=lambda episode: episode["total_movement"],
    )
    two_ilocs_longer_expert = max(
        preflight_detailed._change_episodes(
            [
                (0, 0.80),
                (1, 0.80),
                (2, 0.7333),
                (3, 0.6667),
                (4, 0.60),
                (5, 0.5333),
                (6, 0.4667),
                (7, 0.40),
            ],
            "decrease",
        ),
        key=lambda episode: episode["total_movement"],
    )

    assert preflight_detailed._compare_action_speed(
        player,
        one_iloc_longer_expert,
    ) is None
    assert preflight_detailed._compare_action_speed(
        player,
        two_ilocs_longer_expert,
    )["verdict"] == "quickly"


def test_detailed_preflight_events_capture_multiple_slip_balance_runs():
    df = pd.DataFrame(
        {
            "slip_balance": [
                0.0,
                0.03,
                0.04,
                0.01,
                0.03,
                0.06,
                0.01,
                -0.03,
                -0.04,
                -0.01,
                0.0,
            ],
        },
        index=list(range(10, 21)),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        20,
        [
            (
                "query_telemetry.find_extremum.trajectory_balance.max",
                {"result": {"iloc": 15, "value": 0.06}},
            ),
            (
                "query_telemetry.find_extremum.trajectory_balance.min",
                {"result": {"iloc": 18, "value": -0.04}},
            ),
        ],
    )

    oversteer_ranges = [
        event["range"]
        for event in events
        if event["event"] == "oversteer"
    ]
    understeer_ranges = [
        event["range"]
        for event in events
        if event["event"] == "understeer"
    ]
    assert [11, 12] in oversteer_ranges
    assert [14, 15] in oversteer_ranges
    assert [17, 18] in understeer_ranges
    assert [15, 15] not in oversteer_ranges
    assert [18, 18] not in understeer_ranges


def test_detailed_preflight_events_capture_opponent_outcomes():
    events = preflight_detailed._build_detailed_events(
        pd.DataFrame(),
        0,
        20,
        [
            (
                "classify_opponent_interaction",
                {
                    "outcome": "failed_attack",
                    "confidence_level": "high",
                    "primary_slot_for_role": 3,
                },
            )
        ],
    )

    assert events[0]["event"] == "failed attack"
    assert events[0]["confidence"] == "strong"
    assert events[0]["measurements"]["primary_slot_for_role"] == 3
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])
    assert "failed overtake attempt" in semantic_search_text
    assert "close opponent caused position or time loss" in semantic_search_text
    assert "MSR" not in semantic_search_text


def test_detailed_preflight_outputs_sentence_evidence_without_label_tool():
    semantic_search_text = preflight_detailed._semantic_search_text(
        [
            {
                "event": "brake initiation onset later than expert",
                "phase": "entry",
                "range": [2, 5],
                "confidence": "strong",
                "measurements": {
                    "player_start_index": 6,
                    "expert_start_index": 2,
                    "start_delta_iloc": 4,
                },
                "sources": [],
            },
            {
                "event": "trajectory wider than expert",
                "phase": "entry",
                "range": [4, 4],
                "confidence": "moderate",
                "measurements": {"value": 0.8},
                "sources": [],
            },
        ],
        ["MSP"],
        ["Mistake (Practice)"],
    )
    prompt = preflight_detailed._prompt_block(
        0,
        10,
        semantic_search_text,
    )

    assert "brake initiation onset later than expert" in semantic_search_text
    assert "the player began at iloc 6 while the expert began at iloc 2" in semantic_search_text
    assert "trajectory wider than expert" in semantic_search_text
    assert "the trajectory offset was 0.8 m" in semantic_search_text
    assert "start_delta_iloc" not in semantic_search_text
    assert "measurements=" not in semantic_search_text
    assert "{" not in semantic_search_text
    assert "Preflight evidence sentences" in prompt
    assert "Embedding search words" not in prompt
    assert "Required tool outputs" not in prompt
    assert "search_labels" not in prompt
    assert "preflight semantic candidates" not in prompt.lower()


def test_detailed_preflight_maps_shape_keys_to_evidence_sentences():
    events = preflight_detailed._build_detailed_events(
        pd.DataFrame(),
        0,
        20,
        [
            (
                "measure_segment_shape",
                {
                    "base_segment_shape": {
                        "segment_type_role": "base_segment_shape",
                        "shape_key": "in_corner",
                        "reason": "One curvature arc spans most of the segment.",
                    },
                    "corner_shape_refinement": {
                        "segment_type_role": "corner_shape_refinement",
                        "shape_key": "constant_radius",
                        "reason": "Curvature stays broadly steady.",
                    },
                    "altitude": {
                        "source_column": "expert_optimal_player_pos_z",
                        "entry": {
                            "start_iloc": 1,
                            "end_iloc": 8,
                            "trend": "uphill",
                            "delta_m": 1.5,
                        },
                    },
                    "phases": [],
                },
            ),
        ],
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert "in the corner" in semantic_search_text
    assert "full segment covers entire corner" in semantic_search_text
    assert "constant-radius corner" in semantic_search_text
    assert "smooth steady curvature" in semantic_search_text
    assert "entry altitude uphill" in semantic_search_text
    assert "altitude changed by 1.5 m" in semantic_search_text
    assert "label_id" not in semantic_search_text


def test_detailed_embedding_candidates_searches_parent_scoped_labels(monkeypatch):
    from app.internal_knowledge_base import label_search

    calls = []

    def fake_get_doc(label_id):
        return {"type": "main"} if label_id == "MSP" else {}

    def fake_search(query, *, top_k=8, min_score=0.0, filters=None):
        calls.append((query, top_k, filters))
        if filters == {"type": "segment_type"}:
            return [{
                "id": "ST3",
                "name": "Approach to corner",
                "type": "segment_type",
                "description": "Approach phase",
                "score": 0.7,
            }]
        if filters == {"parent": "MSP"}:
            return [{
                "id": "MSP2",
                "name": "Initiate the turn too late",
                "type": "sub",
                "parent": "MSP",
                "description": "Late turn-in",
                "score": 0.9,
            }]
        return []

    monkeypatch.setattr(label_search, "get_doc", fake_get_doc)
    monkeypatch.setattr(label_search, "search", fake_search)

    candidates = detailed_flow._embedding_label_candidates(
        evidence_text="brake initiation onset later than expert",
        parent_main_labels=["MSP"],
    )

    assert [candidate["id"] for candidate in candidates] == ["MSP2", "ST3"]
    assert calls == [
        ("brake initiation onset later than expert", 12, {"type": "segment_type"}),
        ("brake initiation onset later than expert", 12, {"parent": "MSP"}),
    ]


def test_detailed_build_request_adds_embedding_candidates_to_prompt(monkeypatch):
    preflight_attachment = Attachment(
        name="init.annotation_preflight_context",
        kind="structured",
        label="Annotation Preflight Context",
        content={
            "semantic_search_text": "brake initiation onset later than expert",
            "semantic_evidence_text": "raw evidence fallback",
            "tool_output_tags": ["brake later than expert"],
        },
        content_schema="annotation_preflight_context",
    )

    def fake_preflight(**_kwargs):
        return PreflightContext(
            prompt_block="#### Required Upfront Detailed Statistical Preflight",
            attachments=[preflight_attachment],
            label_candidates=[],
        )

    def fake_candidates(*, evidence_text, parent_main_labels):
        assert evidence_text == "brake initiation onset later than expert"
        assert parent_main_labels == ["MSP"]
        return [{
            "id": "MSP2",
            "name": "Initiate the turn too late",
            "type": "sub",
            "parent": "MSP",
            "description": "Late turn-in",
            "score": 0.9,
        }]

    monkeypatch.setattr(detailed_flow, "build_preflight_context", fake_preflight)
    monkeypatch.setattr(detailed_flow, "_embedding_label_candidates", fake_candidates)

    request = detailed_flow.build_request(
        provider_id="claude_cli",
        prompt_mode="tool_agent",
        df=pd.DataFrame(index=range(10)),
        parent_start=0,
        parent_end=10,
        parent_main_labels=["MSP"],
    )

    attachment_names = [attachment.name for attachment in request.initial_attachments]
    assert "init.preflight_label_candidates" not in attachment_names
    assert "Upfront Detailed Embedding Label Candidates" in request.planner_prompt
    assert "`MSP2`" in request.planner_prompt
    assert "not final labels" in request.planner_prompt
    assert "Required tool outputs" not in request.planner_prompt
    assert "search_labels" not in request.planner_prompt
    assert request.extra_state.get("tool_agent_extra_tools") is None


def test_trajectory_similarity_query_flags_persistent_positive_offset_rise():
    payload, error = run_pipeline_query(
        pd.DataFrame(
            {"trajectory_offset": [0.0, 0.2, 0.6, 1.1, 1.7]},
            index=range(5),
        ),
        "measure_trajectory_similarity",
        {
            "range": [0, 4],
            "column": "trajectory_offset",
            "smoothing_window": 1,
        },
    )

    assert error is None
    extra = payload["extra"]
    assert extra["off_track_trend"]["is_off_track_like"] is True
    assert (
        extra["off_track_trend"]["verdict"]
        == "offset_keeps_rising_positive_off_track_like"
    )
    assert extra["offset_delta_m"] > 0.5
    assert extra["rising_fraction"] == 1.0


def test_trajectory_similarity_query_does_not_flag_recovery_toward_zero():
    payload, error = run_pipeline_query(
        pd.DataFrame(
            {"trajectory_offset": [-3.0, -2.2, -1.5, -0.8, -0.2]},
            index=range(5),
        ),
        "measure_trajectory_similarity",
        {
            "range": [0, 4],
            "column": "trajectory_offset",
            "smoothing_window": 1,
        },
    )

    assert error is None
    extra = payload["extra"]
    assert extra["off_track_trend"]["is_off_track_like"] is False
    assert (
        extra["off_track_trend"]["verdict"]
        == "trajectory_not_persistently_rising_positive"
    )


def test_detailed_preflight_runs_trajectory_similarity_query():
    tool_ids = {
        spec["tool_id"]
        for spec in preflight_detailed.DETAILED_PREFLIGHT_QUERY_SPECS
    }

    assert (
        "query_telemetry.measure_trajectory_similarity.trajectory_offset"
        in tool_ids
    )


def test_lap_preflight_calculates_player_speed_investigation(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_build_shared_preflight_context(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        preflight_lap,
        "build_shared_preflight_context",
        fake_build_shared_preflight_context,
    )

    result = preflight_lap.build_preflight_context(
        df=pd.DataFrame(),
        start=0,
        end=1,
        eligible_behavior_label_ids=["PS", "RM", "MSP"],
        fixed_label_ids=[],
        extra_query_terms=[],
    )

    assert result is sentinel
    columns = [
        spec["params"]["column"]
        for spec in captured["query_specs"]
        if "column" in spec.get("params", {})
    ]
    assert "speed_difference" not in columns
    assert columns.count("Physics_speed_kmh") == 4
    assert columns.count("trajectory_offset") == 3
    assert any(
        spec["tool_id"] == "query_telemetry.find_extremum.player_speed.max"
        for spec in captured["query_specs"]
    )
    assert any(
        spec["tool_id"] == "query_telemetry.find_extremum.player_speed.min"
        for spec in captured["query_specs"]
    )
    assert any(
        spec["tool_id"] == "query_telemetry.find_trend_runs.player_speed"
        for spec in captured["query_specs"]
    )
    assert any(
        spec["tool_id"] == "query_telemetry.compute_slope.player_speed"
        for spec in captured["query_specs"]
    )
    assert any(
        spec["tool_id"] == "query_telemetry.compute_slope.trajectory_offset"
        for spec in captured["query_specs"]
    )


def test_preflight_expert_time_summary_uses_slope_shape():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                0.0,
                100.0,
                200.0,
                260.0,
                300.0,
                280.0,
                240.0,
                200.0,
            ],
            "speed_difference": np.linspace(0.0, 3.0, 8),
        },
        index=range(100, 108),
    )

    results = _run_queries(df, 100, 107)
    prompt = _prompt_block("lap", 100, 107, results, [], [])

    assert "do not decide mistake/recovery from the raw endpoint difference" in prompt
    assert "slope_shape=reversing_to_falling_within_section" in prompt
    assert "end_moves_toward_zero" not in prompt


def test_preflight_trend_run_summary_uses_time_delta_selected_terms():
    df = pd.DataFrame(
        {"expert_time_difference": [0.0, 100.0, 250.0, 500.0, 800.0]},
        index=range(10, 15),
    )

    results = _run_queries(df, 10, 14)
    prompt = _prompt_block("lap", 10, 14, results, [], [])

    assert "selected_gap_increase_run=" in prompt
    assert "selected_losing_time_run" not in prompt
    assert "losing_time_run" not in prompt
    assert "time_gap_rising_run" in prompt
    assert "strong" + "est" not in prompt.lower()


def test_preflight_time_delta_tags_do_not_reuse_offset_zero_terms():
    spec = {
        "graph_id": "time_delta",
        "query_id": "compute_slope",
        "params": {"column": "expert_time_difference"},
    }
    payload = {
        "extra": {
            "delta_value": 300.0,
            "total_change_direction": "rising",
            "total_change_is_label_significant": True,
            "near_zero_summary": {
                "starts_near_zero": True,
                "ends_near_zero": True,
                "moves_toward_zero": True,
            },
        }
    }

    tags = _query_semantic_tags(spec, payload)

    assert "time gap rising" in tags
    assert "starts near zero" not in tags
    assert "ends near zero" not in tags
    assert "moves toward zero" not in tags
    assert "recovery toward expert line" not in tags


def test_preflight_query_outputs_use_generic_analysis_key():
    content = {
        "graph_id": "time_delta",
        "query_id": "compute_slope",
        "params": {"column": "expert_time_difference"},
        "semantic_target": "time gap to expert",
        "semantic_tags": ["time gap rising"],
        "result": {
            "extra": {
                "unit": "ms",
                "delta_value": 100.0,
                "total_change_direction": "rising",
                "total_change_is_label_significant": True,
                "slope_shape": "slope_increasing_over_section",
            },
        },
    }

    output = _semantic_tool_output(
        "query_telemetry.compute_slope.expert_time_difference",
        content,
    )

    assert "analysis" in output
    assert "time_delta_analysis" not in output
    assert "result" not in output
    assert output["analysis"]["total_gap_change"]["gap_direction"] == "time_gap_rising"


def test_preflight_formats_non_time_graph_query_analysis():
    content = {
        "graph_id": "trajectory_offset",
        "query_id": "compute_slope",
        "params": {"column": "trajectory_offset"},
        "semantic_target": "trajectory offset",
        "semantic_tags": ["trajectory moving wider"],
        "result": {
            "extra": {
                "unit": "m",
                "delta_value": 0.75,
                "total_change_direction": "rising",
                "total_change_domain_direction": "moving_wider",
                "total_change_is_label_significant": True,
                "slope_shape": "slope_increasing_over_section",
            },
        },
    }

    output = _semantic_tool_output(
        "query_telemetry.compute_slope.trajectory_offset",
        content,
    )

    assert output["analysis"]["total_change"]["domain_direction"] == "moving_wider"
    assert output["analysis"]["slope_shape"] == "slope_increasing_over_section"
    assert "result" not in output


def test_preflight_threshold_tags_keep_only_observed_timing_direction():
    spec = {
        "graph_id": "brake",
        "query_id": "find_threshold_crossing",
        "params": {
            "columns": ["expert_optimal_brake", "Physics_brake"],
            "threshold": 0.05,
        },
        "tags": [
            "brake initiation onset",
            "brake earlier than expert",
            "brake later than expert",
        ],
    }
    payload = {
        "samples": [
            {"column": "expert_optimal_brake", "iloc": 20},
            {"column": "Physics_brake", "iloc": 15},
        ],
    }

    tags = _query_semantic_tags(spec, payload)

    assert "brake initiation onset" in tags
    assert "brake initiation onset earlier than expert" in tags
    assert "brake earlier than expert" not in tags
    assert "brake later than expert" not in tags


def test_straight_segment_shape_does_not_emit_corner_or_altitude_labels():
    x = np.linspace(0.0, 40.0, 80)
    y = 0.03 * np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
    df = _trajectory_df(x, y)

    attachment = measure_segment_shape(df, 0, len(df))
    content = attachment.content

    assert content["base_segment_shape"]["shape_key"] == "straight"
    assert "label_id" not in content["base_segment_shape"]
    assert content["corner_shape_refinement"] is None
    assert "labels" not in content["altitude"]
    assert "subsegment_label_candidates" not in content["altitude"]
    assert content["phases"] == []


def test_real_corner_segment_shape_still_emits_corner_phase():
    theta = np.linspace(0.0, np.pi / 2.0, 100)
    radius = 30.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))

    attachment = measure_segment_shape(df, 0, len(df))
    content = attachment.content

    assert content["base_segment_shape"]["shape_key"] == "in_corner"
    assert content["corner_shape_refinement"] is not None
    assert "turn angle" in content["corner_shape_refinement"]["reason"]
    assert content["phases"]
    assert content["phases"][0]["turn_angle_degrees"] >= 10.0


def test_approach_to_corner_requires_range_to_end_before_apex():
    result = _classify_base_segment_shape(
        [{"entry": 30, "apex": 60, "exit": 90}],
        100,
    )

    assert result["shape_key"] == "in_corner"


def test_approach_to_corner_when_range_starts_before_corner_and_ends_near_apex():
    result = _classify_base_segment_shape(
        [{"entry": 30, "apex": 96, "exit": 99}],
        100,
    )

    assert result["shape_key"] == "approach_to_corner"


def test_short_offset_corner_turn_angle_is_not_edge_inflated_to_hairpin():
    theta = np.linspace(0.0, np.pi / 2.0, 9)
    radius = 20.0
    df = _trajectory_df(
        100.0 + radius * np.cos(theta),
        100.0 + radius * np.sin(theta),
    )

    attachment = measure_segment_shape(df, 0, len(df))
    refinement = attachment.content["corner_shape_refinement"]

    assert refinement is not None
    assert refinement["turn_angle_degrees"] < 90.0
    assert refinement["shape_key"] != "hairpin"
    assert refinement["is_near_u_turn"] is False


def test_hairpin_shape_uses_turn_angle_without_radius_gate():
    theta = np.linspace(0.0, np.pi, 180)
    radius = 120.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))

    attachment = measure_segment_shape(df, 0, len(df))
    refinement = attachment.content["corner_shape_refinement"]

    assert refinement["shape_key"] == "hairpin"
    assert "label_id" not in refinement
    assert refinement["is_near_u_turn"] is True
    assert "is_tight" not in refinement
    assert "average_radius_m" not in refinement


def test_hairpin_shape_accepts_130_degree_turn_angle_gate():
    theta = np.linspace(0.0, np.deg2rad(140.0), 180)
    radius = 120.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))

    attachment = measure_segment_shape(df, 0, len(df))
    refinement = attachment.content["corner_shape_refinement"]

    assert refinement["shape_key"] == "hairpin"
    assert refinement["is_near_u_turn"] is True
