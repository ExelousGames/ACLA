import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.shared.annotation_agent_tools import (
    _classify_base_segment_shape,
    _query_compute_slope,
    build_graph,
    classify_opponent_interaction,
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
from app.local_annotation_agent.workflow import (
    formatters,
    preflight_detailed,
    preflight_lap,
)
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


RACING_DOCS = [
    "failed_late_brake_attack_at_entry.md",
    "failed_outside_line_sweep.md",
    "failed_switchback.md",
    "failed_slipstream_gain_on_straight.md",
    "inside_cover_broken_early_brake_defense.md",
    "defensive_lift_broken_on_straight.md",
    "failed_overtake_attempt_type_unclear.md",
    "defense_broken_type_unclear.md",
]


def test_time_difference_uses_exact_expert_time_difference_column():
    df = pd.DataFrame(
        {"expert_time_difference": [100.0, 150.0, 225.0, 300.0]},
        index=[10, 11, 12, 13],
    )

    result = _query_compute_slope(df, 10, 13, "expert_time_difference")

    assert result is not None
    assert result["extra"]["unit"] == "ms"
    assert result["extra"]["delta_value"] == 200.0
    assert result["extra"]["start_trend"]["start_iloc"] == 10
    assert result["extra"]["start_trend"]["end_iloc"] == 13
    assert result["extra"]["start_trend"]["direction"] == "rising"
    assert result["extra"]["overall_point_trend"]["direction"] == "rising"
    assert result["extra"]["rising_steps"] == 3


def test_compute_slope_uses_finite_point_by_point_samples_inside_range():
    df = pd.DataFrame(
        {"expert_time_difference": [np.nan, 100.0, 150.0, 120.0, np.nan]},
        index=range(20, 25),
    )

    result = _query_compute_slope(df, 20, 24, "expert_time_difference")

    assert result is not None
    assert result["samples"] == [
        {"iloc": 21, "value": 100.0},
        {"iloc": 23, "value": 120.0},
    ]
    assert result["extra"]["delta_iloc"] == 2.0
    assert result["extra"]["point_trend_runs"][0]["direction"] == "rising"
    assert result["extra"]["point_trend_runs"][1]["direction"] == "falling"
    assert result["extra"]["overall_point_trend"]["direction"] == "rising"


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


def test_speed_difference_uses_derivative_for_recovery_shape():
    df = pd.DataFrame(
        {
            "speed_difference": [
                25.0,
                18.0,
                12.0,
                7.0,
                3.0,
                0.0,
                -2.0,
                -3.0,
            ],
        },
        index=range(70, 78),
    )

    result = _query_compute_slope(df, 70, 77, "speed_difference")

    assert result is not None
    assert result["extra"]["total_change_domain_direction"] == "speed_gap_decreasing"
    assert result["extra"]["slope_shape"] == "slope_increasing_over_section"

    tags = _query_semantic_tags(
        {
            "graph_id": "speed_delta",
            "query_id": "compute_slope",
            "params": {"column": "speed_difference"},
        },
        result,
    )
    assert "speed gap closing" in tags
    assert "speed gap slope increasing over section" in tags


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
    assert "the expert-line relation is converging to expert line" in prompt
    assert "absolute offset starts at 5 m" in prompt
    assert "ends at 1 m" in prompt
    assert "Required tool outputs" not in prompt
    assert "```json" not in prompt
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
            "Graphics_current_time": [
                1000.0 + value
                for value in [
                    0.0,
                    20.0,
                    40.0,
                    60.0,
                    80.0,
                    100.0,
                    120.0,
                    140.0,
                    160.0,
                    180.0,
                    200.0,
                    220.0,
                    240.0,
                    260.0,
                    280.0,
                    300.0,
                    320.0,
                    340.0,
                    360.0,
                    380.0,
                    400.0,
                ]
            ],
            "expert_time_difference": [
                0.0,
                20.0,
                40.0,
                60.0,
                80.0,
                100.0,
                120.0,
                140.0,
                160.0,
                180.0,
                200.0,
                220.0,
                240.0,
                260.0,
                280.0,
                300.0,
                320.0,
                340.0,
                360.0,
                380.0,
                400.0,
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
        ],
    )

    event_names = {event["event"] for event in events}
    assert "brake initiation onset later than expert" in event_names
    assert "trajectory wider than expert" in event_names
    assert "moving toward positive" in event_names
    assert "gap grows" in event_names
    assert "time loss" in event_names


def test_detailed_preflight_outputs_corner_phase_boundaries_for_range_selection():
    events = preflight_detailed._build_detailed_events(
        pd.DataFrame(),
        10,
        30,
        [
            (
                "compute_expert_phases",
                {"phases": [{"entry": 10, "apex": 20, "exit": 30}]},
            ),
        ],
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert events == [
        {
            "event": "corner phase markers",
            "phase": "whole_range",
            "range": [10, 30],
            "measurements": {
                "entry_start_iloc": 10,
                "apex_iloc": 20,
                "exit_end_iloc": 30,
            },
            "confidence": "strong",
            "sources": ["compute_expert_phases"],
        }
    ]
    assert "entry starts at iloc 10" in semantic_search_text
    assert "apex is at iloc 20" in semantic_search_text
    assert "exit ends at iloc 30" in semantic_search_text
    assert "entry phase detected" not in semantic_search_text


def test_detailed_preflight_compares_player_apex_to_expert_apex():
    local = np.arange(21, dtype=float)
    df = pd.DataFrame(
        {
            "expert_optimal_player_pos_x": local,
            "expert_optimal_player_pos_y": 0.05 * (local - 10.0) ** 2,
            "Graphics_player_pos_x": local,
            "Graphics_player_pos_y": 0.05 * (local - 16.0) ** 2,
        },
        index=range(10, 31),
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
        ],
    )

    apex_event = next(
        event
        for event in events
        if event["event"] == "player reaches apex later than expert"
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert apex_event["phase"] == "apex"
    assert apex_event["measurements"]["expert_apex_iloc"] == 20
    assert apex_event["measurements"]["player_apex_iloc"] == 26
    assert apex_event["measurements"]["apex_delta_iloc"] == 6
    assert apex_event["measurements"]["expert_apex_range"] == [18, 22]
    assert apex_event["measurements"]["player_apex_range"] == [24, 28]
    assert apex_event["measurements"]["apex_boundary_gap_iloc"] == 2
    assert (
        "the player apex range was iloc 24 to 28 while the expert apex "
        "range was iloc 18 to 22"
        in semantic_search_text
    )
    assert "too late compared to expert apex" in semantic_search_text


def test_detailed_preflight_marks_similar_trajectory_phases_aligned():
    df = pd.DataFrame(
        {"trajectory_offset": np.full(21, 0.2)},
        index=range(10, 31),
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
                "query_telemetry.measure_trajectory_similarity.driver_expert_path",
                {"result": {"extra": {"similarity_score": 0.8}}},
            ),
        ],
    )

    aligned_events = {
        event["event"]: event
        for event in events
        if "trajectory aligned with expert" in event["event"]
    }
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert set(aligned_events) == {
        "entry trajectory aligned with expert",
        "apex trajectory aligned with expert",
        "exit trajectory aligned with expert",
    }
    assert aligned_events["entry trajectory aligned with expert"]["phase"] == "entry"
    assert aligned_events["apex trajectory aligned with expert"]["phase"] == "apex"
    assert aligned_events["exit trajectory aligned with expert"]["phase"] == "exit"
    assert (
        aligned_events["entry trajectory aligned with expert"]["measurements"][
            "similarity_score"
        ]
        == 0.8
    )
    assert "entry trajectory aligned with expert" in semantic_search_text


def test_detailed_preflight_phases_time_gap_percent_changes_at_corner_entry_or_exit():
    time_gap = [
        200.0,
        220.0,
        240.0,
        260.0,
        280.0,
        300.0,
        320.0,
        340.0,
        360.0,
        380.0,
        400.0,
        380.0,
        360.0,
        340.0,
        320.0,
        300.0,
        280.0,
        260.0,
        240.0,
        220.0,
        200.0,
    ]
    df = pd.DataFrame(
        {
            "Graphics_current_time": [1000.0 + value for value in time_gap],
            "expert_time_difference": time_gap,
        },
        index=range(10, 31),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {
                    "phases": [
                        {
                            "entry": 10,
                            "apex": 20,
                            "exit": 30,
                            "direction": "right",
                        }
                    ]
                },
            ),
        ],
    )

    rising = next(
        event for event in events if event["event"] == "time gap rising at entry"
    )
    falling = next(
        event for event in events if event["event"] == "time gap falling at exit"
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert rising["phase"] == "entry"
    assert rising["measurements"]["relative_gain_percent"] < 0.0
    assert falling["phase"] == "exit"
    assert falling["measurements"]["relative_gain_percent"] > 0.0
    assert "the evidence shows time gap rising at entry" in semantic_search_text
    assert "the evidence shows time gap falling at exit" in semantic_search_text
    assert "percentage points gained" in semantic_search_text


def test_detailed_preflight_phases_time_gap_percent_changes_at_corner_apex():
    df = pd.DataFrame(
        {
            "expert_time_difference": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                200.0,
                400.0,
                400.0,
                400.0,
                400.0,
                400.0,
                400.0,
                400.0,
                400.0,
                400.0,
            ],
            "Graphics_current_time": [
                1000.0 + value
                for value in [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    200.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                    400.0,
                ]
            ],
        },
        index=range(10, 31),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {
                    "phases": [
                        {
                            "entry": 10,
                            "apex": 20,
                            "exit": 30,
                            "direction": "right",
                        }
                    ]
                },
            ),
        ],
    )

    apex = next(
        event for event in events if event["event"] == "time gap rising at apex"
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert apex["phase"] == "apex"
    assert apex["range"] == [18, 22]
    assert apex["measurements"]["relative_gain_percent"] < 0.0
    assert apex["measurements"]["threshold_state"] == "label_threshold_met"
    assert "the evidence shows time gap rising at apex" in semantic_search_text


def test_detailed_preflight_phases_speed_gap_percent_changes_at_corner_entry_or_exit():
    speed_difference = [
        20.0,
        19.0,
        18.0,
        17.0,
        16.0,
        15.0,
        14.0,
        13.0,
        12.0,
        11.0,
        10.0,
        11.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
    ]
    df = pd.DataFrame(
        {
            "speed_difference": speed_difference,
            "expert_optimal_speed": [100.0] * len(speed_difference),
            "Physics_speed_kmh": [
                100.0 - value for value in speed_difference
            ],
        },
        index=range(10, 31),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {
                    "phases": [
                        {
                            "entry": 10,
                            "apex": 20,
                            "exit": 30,
                            "direction": "right",
                        }
                    ]
                },
            ),
        ],
    )

    closing = next(
        event for event in events if event["event"] == "speed gap closing at entry"
    )
    growing = next(
        event for event in events if event["event"] == "speed gap growing at exit"
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert closing["phase"] == "entry"
    assert (
        closing["measurements"]["start_abs_gap_percent"]
        > closing["measurements"]["end_abs_gap_percent"]
    )
    assert closing["measurements"]["relative_gain_percent"] > 0.0
    assert growing["phase"] == "exit"
    assert (
        growing["measurements"]["end_abs_gap_percent"]
        > growing["measurements"]["start_abs_gap_percent"]
    )
    assert growing["measurements"]["relative_gain_percent"] < 0.0
    assert "the evidence shows speed gap closing at entry" in semantic_search_text
    assert "the evidence shows speed gap growing at exit" in semantic_search_text
    assert "percentage points gained" in semantic_search_text


def test_detailed_preflight_phases_speed_gap_percent_changes_at_corner_apex():
    speed_difference = [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]
    df = pd.DataFrame(
        {
            "speed_difference": speed_difference,
            "expert_optimal_speed": [100.0] * len(speed_difference),
            "Physics_speed_kmh": [
                100.0 - value for value in speed_difference
            ],
        },
        index=range(10, 31),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {
                    "phases": [
                        {
                            "entry": 10,
                            "apex": 20,
                            "exit": 30,
                            "direction": "right",
                        }
                    ]
                },
            ),
        ],
    )

    apex = next(
        event for event in events if event["event"] == "speed gap growing at apex"
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert apex["phase"] == "apex"
    assert apex["range"] == [18, 22]
    assert apex["measurements"]["threshold_state"] == "label_threshold_met"
    assert "the evidence shows speed gap growing at apex" in semantic_search_text


def test_detailed_preflight_does_not_treat_carried_speed_gap_as_local_change():
    df = pd.DataFrame(
        {
            "speed_difference": [18.0] * 21,
            "expert_optimal_speed": [100.0] * 21,
            "Physics_speed_kmh": [82.0] * 21,
        },
        index=range(10, 31),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        30,
        [
            (
                "compute_expert_phases",
                {
                    "phases": [
                        {
                            "entry": 10,
                            "apex": 20,
                            "exit": 30,
                            "direction": "right",
                        }
                    ]
                },
            ),
        ],
    )

    speed_gap_events = [
        event
        for event in events
        if event["event"].startswith(("speed gap closing", "speed gap growing"))
    ]

    assert speed_gap_events == []


def test_detailed_preflight_events_capture_recovery_and_speed_gap_closing():
    speed_difference = np.linspace(24.0, 4.0, 31)
    df = pd.DataFrame(
        {
            "speed_difference": speed_difference,
            "expert_optimal_speed": np.full(31, 100.0),
            "Physics_speed_kmh": 100.0 - speed_difference,
        },
        index=range(100, 131),
    )
    events = preflight_detailed._build_detailed_events(
        df,
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
    assert "large speed percentage gap" in event_names
    assert "expert faster than player" in event_names
    assert "player speed maximum" in event_names
    assert "player speed minimum" in event_names
    assert "player accelerating" in event_names
    assert "player decelerating" in event_names
    assert "speed overall trend rising" in event_names


def test_detailed_preflight_events_capture_throttle_timing_and_lowest_pressure():
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
                "query_telemetry.find_extremum.throttle.player.min",
                {"result": {"iloc": 1, "value": 0.10}},
            ),
            (
                "query_telemetry.find_extremum.throttle.expert.min",
                {"result": {"iloc": 1, "value": 0.10}},
            ),
        ],
    )

    event_names = {event["event"] for event in events}
    assert "throttle application onset earlier than expert" in event_names
    assert "lowest throttle pressure about same as expert" in event_names
    timing_event = next(
        event
        for event in events
        if event["event"] == "throttle application onset earlier than expert"
    )
    assert timing_event["phase"] == "unknown"
    throttle_event = next(
        event
        for event in events
        if event["event"] == "lowest throttle pressure about same as expert"
    )
    assert throttle_event["phase"] == "unknown"
    sentence = preflight_detailed._event_sentence(throttle_event)
    assert not sentence.startswith("During ")
    assert "the player lowest was 0.1 versus expert lowest 0.1" in sentence
    assert "player peak" not in sentence


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


def test_detailed_preflight_reports_throttle_boundaries_instead_of_speed_verdict():
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

    assert "throttle applied too quickly" not in event_names
    assert "throttle applied too slowly" not in event_names
    assert "throttle application onset aligned with expert" in event_names
    assert "throttle application end earlier than expert" in event_names
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

    assert "brake initiation onset earlier than expert" not in event_names
    assert "brake initiation onset later than expert" not in event_names
    assert "brake release onset earlier than expert" not in event_names
    assert "brake release onset later than expert" not in event_names
    assert "brake applied too quickly" not in event_names
    assert "brake applied too slowly" not in event_names
    assert "brake release too quickly" not in event_names
    assert "brake release too slowly" not in event_names


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


def test_detailed_preflight_outputs_exit_gear_verdict():
    df = pd.DataFrame(
        {
            "Physics_gear": [3, 3, 3, 3, 3, 4],
            "expert_optimal_gear": [3, 3, 3, 3, 3, 5],
            "Physics_rpm": [7000, 7100, 7200, 7300, 7400, 7600],
        },
        index=range(10, 16),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        10,
        15,
        [
            (
                "compute_expert_phases",
                {"phases": [{"entry": 10, "apex": 12, "exit": 15}]},
            ),
        ],
    )

    exit_gear_events = [
        event
        for event in events
        if event["event"] in {"player gear low at exit", "player gear high at exit"}
    ]
    assert len(exit_gear_events) == 1
    event = exit_gear_events[0]
    assert event["event"] == "player gear low at exit"
    assert event["phase"] == "exit"
    assert event["range"] == [15, 15]
    assert event["measurements"]["player_gear"] == 4
    assert event["measurements"]["expert_gear"] == 5

    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])
    assert "player gear low at exit" in semantic_search_text
    assert "gear too low when accelerating" in semantic_search_text


def test_racing_detailed_preflight_outputs_indexed_opponent_facts():
    df = pd.DataFrame(
        {
            "Graphics_player_pos_x": [0.0, 1.0, 2.5, 4.5, 7.0, 10.0],
            "Graphics_player_pos_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Car_3_pos_x": [5.0, 5.8, 6.4, 6.8, 7.0, 7.1],
            "Car_3_pos_y": [0.0, 2.0, 2.0, 2.0, 1.5, 0.0],
        },
        index=range(100, 106),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        100,
        105,
        [
            (
                "classify_opponent_interaction",
                {
                    "outcome": "pass_completed",
                    "confidence_level": "high",
                    "primary_slot_for_role": 3,
                },
            )
        ],
        parent_main_labels=["O"],
    )
    text = preflight_detailed._semantic_search_text(events, ["O"], [])

    assert "the opponent started ahead of the driver, against opponent slot 3; at index 100" in text
    assert "the driver ended ahead of the opponent, against opponent slot 3; at index 105" in text
    assert "the gap flipped from opponent ahead to driver ahead" in text
    assert "the driver drew alongside the opponent, against opponent slot 3; from index" in text
    assert "the opponent was on the driver's left side" in text
    assert "the gap to the opponent shrank from" in text
    assert "the driver gained relative speed" in text
    assert "the driver accelerated better than the opponent" in text
    assert "wider than expert" not in text
    assert "later than expert" not in text
    assert "expert_time_difference" not in text


def test_racing_detailed_preflight_outputs_broken_defense_motion_facts():
    df = pd.DataFrame(
        {
            "Graphics_player_pos_x": [0.0, 4.0, 7.0, 9.0, 10.0, 10.5],
            "Graphics_player_pos_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Car_4_pos_x": [-5.0, -1.0, 3.0, 7.0, 10.0, 13.0],
            "Car_4_pos_y": [0.0, 2.0, 2.0, 2.0, 1.5, 0.0],
        },
        index=range(200, 206),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        200,
        205,
        [
            (
                "classify_opponent_interaction",
                {
                    "outcome": "broken_defense",
                    "confidence_level": "high",
                    "primary_slot_for_role": 4,
                },
            )
        ],
        parent_main_labels=["MSR"],
    )
    text = preflight_detailed._semantic_search_text(events, ["MSR"], [])

    assert "the driver started ahead of the opponent, against opponent slot 4; at index 200" in text
    assert "the opponent ended ahead of the driver, against opponent slot 4; at index 205" in text
    assert "the gap flipped from driver ahead to opponent ahead" in text
    assert "the opponent drew alongside the driver, against opponent slot 4; from index" in text
    assert "the driver slowed more than the opponent" in text
    assert "tighter than expert" not in text
    assert "earlier than expert" not in text
    assert "expert_time_difference" not in text


def test_opponent_classifier_rejects_marginal_lateral_nose_ahead_as_broken_defense():
    df = pd.DataFrame(
        {
            "Graphics_player_pos_x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "Graphics_player_pos_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Car_7_pos_x": [-5.3, 5.0, 15.0, 25.0, 35.0, 53.1],
            "Car_7_pos_y": [2.8, 3.0, 4.0, 5.0, 6.0, -6.3],
        },
        index=range(69, 75),
    )

    result = classify_opponent_interaction(df, 69, 74).content

    assert result["outcome"] != "broken_defense"
    assert result["candidates"][0]["exit_signed_long_gap_m"] == 3.1


def test_opponent_classifier_keeps_clear_completed_pass_as_broken_defense():
    df = pd.DataFrame(
        {
            "Graphics_player_pos_x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "Graphics_player_pos_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Car_7_pos_x": [-5.3, 5.0, 15.0, 25.0, 35.0, 55.0],
            "Car_7_pos_y": [2.8, 3.0, 4.0, 5.0, 4.0, 2.0],
        },
        index=range(69, 75),
    )

    result = classify_opponent_interaction(df, 69, 74).content

    assert result["outcome"] == "broken_defense"
    assert result["candidates"][0]["exit_signed_long_gap_m"] == 5.0


def test_racing_label_catalog_and_docs_use_reusable_opponent_phrases():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads(
        (root / "app/internal_knowledge_base/sub_label_annotation.json").read_text()
    )
    labels = catalog["labels"]
    racing_ids = [
        "O",
        "OD",
        "MSR",
        "O1",
        "O3",
        "O4",
        "O5",
        "OD1",
        "OD2",
        "MSR1",
        "MSR2",
        "MSR3",
        "MSR4",
        "MSR5",
        "MSR6",
        "MSR7",
        "MSR8",
    ]
    racing_text = "\n".join(
        str(labels[label_id].get("description", ""))
        + "\n"
        + str(labels[label_id].get("annotation_guideline", ""))
        for label_id in racing_ids
    )

    for phrase in (
        "opponent started ahead of the driver",
        "driver ended ahead of the opponent",
        "driver drew alongside the opponent",
        "opponent drew alongside the driver",
        "opponent was on the driver's left/right side",
        "gap to the opponent shrank",
        "gap flipped from opponent ahead to driver ahead",
        "gap flipped from driver ahead to opponent ahead",
        "driver gained relative speed",
        "driver accelerated better than the opponent",
        "driver slowed more/less than the opponent",
    ):
        assert phrase in racing_text

    for forbidden in (
        "wider than expert",
        "tighter than expert",
        "expert_time_difference",
        "later than expert",
        "earlier than expert",
        "reference-lap",
    ):
        assert forbidden not in racing_text

    docs_root = root / "app/external_knowledge_base/labels"
    docs_text = "\n".join((docs_root / name).read_text() for name in RACING_DOCS)
    for forbidden in (
        "wider than expert",
        "tighter than expert",
        "expert_time_difference",
        "time difference to the expert",
        "at or below expert",
        "expert reference",
    ):
        assert forbidden not in docs_text


def test_detailed_preflight_outputs_sentence_evidence_without_label_tool():
    semantic_search_text = preflight_detailed._semantic_search_text(
        [
            {
                "event": "brake initiation onset later than expert",
                "phase": "unknown",
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
    assert (
        "the player onset was at iloc 6 while the expert onset was at iloc 2"
        in semantic_search_text
    )
    assert "trajectory wider than expert" in semantic_search_text
    assert (
        "During entry, the evidence shows trajectory wider than expert"
        in semantic_search_text
    )
    assert "the trajectory offset was 0.8 m" in semantic_search_text
    assert "During entry, the evidence shows brake initiation" not in semantic_search_text
    assert "start_delta_iloc" not in semantic_search_text
    assert "measurements=" not in semantic_search_text
    assert "{" not in semantic_search_text
    assert "Preflight evidence sentences" in prompt
    assert "Embedding search words" not in prompt
    assert "Required tool outputs" not in prompt
    assert "search_labels" not in prompt
    assert "preflight semantic candidates" not in prompt.lower()


def test_detailed_preflight_input_timing_sentences_omit_episode_spans():
    events = [
        {
            "event": "brake initiation onset later than expert",
            "phase": "unknown",
            "range": [42, 44],
            "confidence": "strong",
            "measurements": {
                "player_start_index": 44,
                "player_end_index": 48,
                "expert_start_index": 42,
                "expert_end_index": 50,
                "start_delta_iloc": 2,
            },
            "sources": [],
        },
        {
            "event": "throttle application end earlier than expert",
            "phase": "unknown",
            "range": [46, 50],
            "confidence": "strong",
            "measurements": {
                "player_start_index": 42,
                "player_end_index": 46,
                "expert_start_index": 44,
                "expert_end_index": 50,
                "end_delta_iloc": -4,
            },
            "sources": [],
        },
    ]

    semantic_search_text = preflight_detailed._semantic_search_text(
        events,
        ["MSP"],
        ["Mistake (Practice)"],
    )

    assert "episode spans" not in semantic_search_text
    assert (
        "the player onset was at iloc 44 while the expert onset was at iloc 42"
        in semantic_search_text
    )
    assert (
        "the player end was at iloc 46 while the expert end was at iloc 50"
        in semantic_search_text
    )


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
                            "slope_angle_degrees": 3.2,
                            "horizontal_distance_units": 26.8,
                            "delta_m": 1.5,
                        },
                    },
                    "phases": [],
                },
            ),
        ],
    )
    semantic_search_text = preflight_detailed._semantic_search_text(events, [], [])

    assert (
        "the segment is in the corner, detected from iloc 0 to 20; "
        "this matches label vocabulary for full segment covers entire corner, "
        "driver turning throughout, and single curve hairpin continuous arc, "
        "with strong confidence"
    ) in semantic_search_text
    assert "in the corner; full segment covers entire corner" not in semantic_search_text
    assert "in the corner" in semantic_search_text
    assert "full segment covers entire corner" in semantic_search_text
    assert "constant-radius corner" in semantic_search_text
    assert "smooth steady curvature" in semantic_search_text
    assert "entry altitude uphill" in semantic_search_text
    assert "slope angle was 3.2 degrees" in semantic_search_text
    assert "horizontal path distance was 26.8 telemetry units" in semantic_search_text
    assert "altitude changed by 1.5 m for citation only" in semantic_search_text
    assert "label_id" not in semantic_search_text


def test_detailed_embedding_candidates_searches_parent_scoped_labels(monkeypatch):
    from app.internal_knowledge_base import label_reranker
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
    monkeypatch.setattr(
        label_reranker.settings,
        "annotation_label_reranker_enabled",
        False,
    )

    candidates = detailed_flow._embedding_label_candidates(
        evidence_text="brake initiation onset later than expert",
        parent_main_labels=["MSP"],
    )

    assert [candidate["id"] for candidate in candidates] == ["MSP2", "ST3"]
    assert calls == [
        ("brake initiation onset later than expert", 12, {"type": "segment_type"}),
        ("brake initiation onset later than expert", 12, {"parent": "MSP"}),
    ]


def test_label_reranker_scores_evidence_label_pairs(monkeypatch):
    from app.internal_knowledge_base import label_reranker

    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs[0][0] == "driver ended ahead of the opponent"
            assert "driver ended ahead" in pairs[0][1]
            assert "opponent ended ahead" in pairs[1][1]
            return [0.95, 0.12]

    monkeypatch.setattr(
        label_reranker.settings,
        "annotation_label_reranker_enabled",
        True,
    )
    monkeypatch.setattr(
        label_reranker.settings,
        "annotation_label_reranker_top_k",
        1,
    )
    monkeypatch.setattr(
        label_reranker.settings,
        "annotation_label_reranker_min_score",
        None,
    )
    monkeypatch.setattr(label_reranker, "_get_cross_encoder", lambda: FakeCrossEncoder())

    docs = [
        {
            "id": "O1",
            "name": "Late-brake attack",
            "description": "Use when the driver ended ahead of the opponent.",
            "score": 0.2,
        },
        {
            "id": "OD1",
            "name": "Inside cover",
            "description": "Use when the opponent ended ahead of the driver.",
            "score": 0.9,
        },
    ]

    reranked = label_reranker.rerank_label_docs(
        "driver ended ahead of the opponent",
        docs,
    )

    assert [doc["id"] for doc in reranked] == ["O1"]
    assert reranked[0]["score"] == 0.95
    assert reranked[0]["reranker_score"] == 0.95
    assert reranked[0]["embedding_score"] == 0.2


def test_detailed_embedding_candidates_uses_reranker_scores(monkeypatch):
    from app.internal_knowledge_base import label_reranker
    from app.internal_knowledge_base import label_search

    def fake_get_doc(label_id):
        return {"type": "main"} if label_id == "O" else {}

    def fake_search(query, *, top_k=8, min_score=0.0, filters=None):
        if filters == {"type": "segment_type"}:
            return []
        if filters == {"parent": "O"}:
            return [
                {
                    "id": "O1",
                    "name": "Late-brake attack",
                    "type": "sub",
                    "parent": "O",
                    "description": "Driver completes the pass and ends ahead.",
                    "annotation_guideline": "The gap flips to driver ahead.",
                    "score": 0.2,
                },
                {
                    "id": "O3",
                    "name": "Outside-line sweep",
                    "type": "sub",
                    "parent": "O",
                    "description": "Driver stays outside and completes the pass.",
                    "score": 0.8,
                },
            ]
        return []

    def fake_rerank(query, docs, *, top_k=None, min_score=None):
        assert query == "gap flipped from opponent ahead to driver ahead"
        assert any(
            doc.get("annotation_guideline") == "The gap flips to driver ahead."
            for doc in docs
        )
        by_id = {doc["id"]: doc for doc in docs}
        return [
            {
                **by_id["O1"],
                "score": 0.97,
                "embedding_score": 0.2,
                "reranker_score": 0.97,
            },
            {
                **by_id["O3"],
                "score": 0.21,
                "embedding_score": 0.8,
                "reranker_score": 0.21,
            },
        ]

    monkeypatch.setattr(label_search, "get_doc", fake_get_doc)
    monkeypatch.setattr(label_search, "search", fake_search)
    monkeypatch.setattr(label_reranker, "rerank_label_docs", fake_rerank)

    candidates = detailed_flow._embedding_label_candidates(
        evidence_text="gap flipped from opponent ahead to driver ahead",
        parent_main_labels=["O"],
    )

    assert [candidate["id"] for candidate in candidates] == ["O1", "O3"]
    assert candidates[0]["score"] == 0.97
    assert candidates[0]["embedding_score"] == 0.2
    assert candidates[0]["reranker_score"] == 0.97


def test_label_verifier_uses_reranker_scores(monkeypatch):
    from app.local_annotation_agent.sub_agents import label_verifier

    def fake_get_doc(label_id):
        return {"type": "main"} if label_id == "O" else {}

    def fake_search(query, *, top_k=8, min_score=0.0, filters=None):
        if filters == {"type": "segment_type"}:
            return []
        if filters == {"parent": "O"}:
            return [
                {
                    "id": "O1",
                    "name": "Late-brake attack",
                    "description": "Driver ends ahead of the opponent.",
                    "score": 0.3,
                },
                {
                    "id": "O5",
                    "name": "Slipstream pass",
                    "description": "Driver passes on the straight.",
                    "score": 0.7,
                },
            ]
        return []

    def fake_rerank(query, docs, *, top_k=None, min_score=None):
        by_id = {doc["id"]: doc for doc in docs}
        return [{
            **by_id["O1"],
            "score": 0.91,
            "embedding_score": 0.3,
            "reranker_score": 0.91,
        }]

    monkeypatch.setattr(label_verifier, "get_doc", fake_get_doc)
    monkeypatch.setattr(label_verifier, "search", fake_search)
    monkeypatch.setattr(label_verifier, "rerank_label_docs", fake_rerank)
    monkeypatch.setattr(
        label_verifier.settings,
        "annotation_label_reranker_enabled",
        True,
    )
    monkeypatch.setattr(
        label_verifier.settings,
        "annotation_label_reranker_top_k",
        16,
    )

    verified, all_scored = label_verifier.compute_verified_labels(
        ["O"],
        "driver ended ahead of the opponent",
    )

    assert [item["label_id"] for item in verified] == ["O1"]
    assert all_scored == verified
    assert verified[0]["similarity"] == 0.91
    assert verified[0]["embedding_similarity"] == 0.3
    assert verified[0]["reranker_score"] == 0.91


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
    assert "init.preflight_label_candidates" in attachment_names
    candidate_attachment = next(
        attachment
        for attachment in request.initial_attachments
        if attachment.name == "init.preflight_label_candidates"
    )
    assert candidate_attachment.label == "Upfront Detailed Embedding Label Candidates"
    assert candidate_attachment.content_schema == "annotation_preflight_labels"
    assert candidate_attachment.content["candidates"][0]["id"] == "MSP2"
    context_attachment = next(
        attachment
        for attachment in request.initial_attachments
        if attachment.name == "init.annotation_preflight_context"
    )
    assert context_attachment.content["label_candidate_ids"] == ["MSP2"]
    assert "Upfront Detailed Embedding Label Candidates" in request.planner_prompt
    assert "`MSP2`" in request.planner_prompt
    assert "not final labels" in request.planner_prompt
    assert "Required tool outputs" not in request.planner_prompt
    assert "search_labels" not in request.planner_prompt
    assert request.extra_state.get("tool_agent_extra_tools") is None


def test_trajectory_similarity_query_measures_driver_expert_path_divergence():
    x = np.arange(5, dtype=float)
    df = _trajectory_df(x, np.zeros_like(x))
    df["Graphics_player_pos_y"] = [0.0, 0.3, 0.8, 1.4, 2.0]

    payload, error = run_pipeline_query(
        df,
        "measure_trajectory_similarity",
        {
            "range": [0, 4],
            "smoothing_window": 1,
        },
    )

    assert error is None
    extra = payload["extra"]
    assert 0.0 < extra["similarity_score"] < 1.0
    assert extra["line_separation_gain_m"] > 0.5
    assert extra["widening_fraction"] == 1.0
    assert extra["peak_line_separation"]["value_m"] >= 1.25


def test_trajectory_similarity_query_scores_close_driver_expert_path_highly():
    x = np.arange(5, dtype=float)
    df = _trajectory_df(x, np.zeros_like(x))
    df["Graphics_player_pos_y"] = [0.0, 0.05, 0.1, 0.05, 0.0]

    payload, error = run_pipeline_query(
        df,
        "measure_trajectory_similarity",
        {
            "range": [0, 4],
            "smoothing_window": 1,
        },
    )

    assert error is None
    extra = payload["extra"]
    assert extra["similarity_score"] > 0.9
    assert extra["mean_line_separation_m"] < 0.1


def test_detailed_preflight_runs_trajectory_similarity_query():
    tool_ids = {
        spec["tool_id"]
        for spec in preflight_detailed.DETAILED_PREFLIGHT_QUERY_SPECS
    }

    assert (
        "query_telemetry.measure_trajectory_similarity.driver_expert_path"
        in tool_ids
    )


def test_detailed_preflight_does_not_use_fixed_threshold_for_input_timing():
    tool_ids = {
        spec["tool_id"]
        for spec in preflight_detailed.DETAILED_PREFLIGHT_QUERY_SPECS
    }

    assert "query_telemetry.find_threshold_crossing.brake.onset" not in tool_ids
    assert "query_telemetry.find_threshold_crossing.brake.release" not in tool_ids
    assert "query_telemetry.find_threshold_crossing.throttle.onset" not in tool_ids
    assert "query_telemetry.find_threshold_crossing.throttle.release" not in tool_ids


def test_detailed_preflight_shape_comparison_outputs_all_input_timing_events():
    df = pd.DataFrame(
        {
            "expert_optimal_brake": [
                0.0, 0.0, 0.0, 0.15, 0.45, 0.75, 0.75, 0.75, 0.60,
                0.40, 0.20, 0.0, 0.0, 0.0, 0.0,
            ],
            "Physics_brake": [
                0.0, 0.0, 0.0, 0.0, 0.15, 0.45, 0.75, 0.75, 0.75,
                0.60, 0.40, 0.20, 0.0, 0.0, 0.0,
            ],
            "expert_optimal_throttle": [
                1.0, 1.0, 1.0, 0.85, 0.65, 0.45, 0.25, 0.20, 0.35,
                0.55, 0.75, 0.95, 1.0, 1.0, 1.0,
            ],
            "Physics_gas": [
                1.0, 1.0, 1.0, 1.0, 0.85, 0.60, 0.35, 0.10, 0.0,
                0.20, 0.45, 0.70, 0.95, 1.0, 1.0,
            ],
        }
    )

    events = preflight_detailed._build_detailed_events(
        df,
        0,
        14,
        [],
    )

    event_names = {event["event"] for event in events}

    assert "brake initiation onset later than expert" in event_names
    assert "brake release onset later than expert" in event_names
    assert "throttle application onset later than expert" in event_names
    assert "throttle release onset later than expert" in event_names
    assert "throttle application end aligned with expert" in event_names
    assert "throttle release end later than expert" in event_names
    input_events = [
        event
        for event in events
        if event["event"].startswith((
            "brake initiation",
            "brake release",
            "throttle application",
            "throttle release",
        ))
    ]
    assert input_events
    assert {event["phase"] for event in input_events} == {"unknown"}

    evidence_text = preflight_detailed._event_text(events, [], [])
    assert "During entry, the evidence shows brake initiation" not in evidence_text
    assert "During apex, the evidence shows brake release" not in evidence_text
    assert "During exit, the evidence shows throttle application" not in evidence_text


def test_detailed_preflight_detects_short_brake_initiation_after_smoothing():
    df = pd.DataFrame(
        {
            "expert_optimal_brake": [
                0.0, 0.0, 0.33, 0.83, 0.79, 0.25, 0.02, 0.0, 0.0,
            ],
            "Physics_brake": [
                0.0, 0.0, 0.0, 0.86, 1.0, 0.60, 0.19, 0.0, 0.0,
            ],
        },
        index=range(342, 351),
    )

    events = preflight_detailed._build_detailed_events(
        df,
        342,
        350,
        [],
    )

    event_names = {event["event"] for event in events}

    assert "brake initiation onset comparison unavailable" not in event_names
    assert "brake initiation onset later than expert" in event_names


def test_detailed_preflight_keeps_shape_categories_when_action_missing():
    events = preflight_detailed._build_detailed_events(
        pd.DataFrame(),
        0,
        20,
        [],
    )

    event_names = [event["event"] for event in events]
    evidence_text = preflight_detailed._event_text(events, [], [])

    assert "brake initiation onset comparison unavailable" in event_names
    assert "brake release onset comparison unavailable" in event_names
    assert "throttle application onset comparison unavailable" in event_names
    assert "throttle release onset comparison unavailable" in event_names
    assert "throttle application end comparison unavailable" in event_names
    assert "throttle release end comparison unavailable" in event_names
    assert "searched for a rising episode" in evidence_text
    assert "searched for a falling episode" in evidence_text


def test_preflight_context_formatter_displays_evidence_sentences():
    rendered = formatters._format_preflight_context({
        "flow": "detailed",
        "range": [1, 2],
        "required_tools": [],
        "semantic_evidence_text": (
            "The evidence shows brake initiation onset comparison unavailable.\n"
            "The evidence shows throttle release onset later than expert."
        ),
    })

    assert "Evidence sentences:" in rendered
    assert "brake initiation onset comparison unavailable" in rendered
    assert "throttle release onset later than expert" in rendered


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
    assert columns.count("speed_difference") == 2
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
        spec["tool_id"] == "query_telemetry.find_trend_runs.speed_difference"
        for spec in captured["query_specs"]
    )
    assert any(
        spec["tool_id"] == "query_telemetry.compute_slope.speed_difference"
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

    assert "Start trend: the time gap is trending up from index 100 to 104" in prompt
    assert "Growth index ranges: index 100 to 104" in prompt
    assert "Shrink index ranges: index 104 to 107" in prompt
    assert "Overall trend: the time gap is trending up" in prompt
    assert "Slope shape: reversing to falling within section" in prompt
    assert "end_moves_toward_zero" not in prompt
    assert "tool_results_json" not in prompt


def test_preflight_speed_gap_summary_uses_point_trend_indexes():
    df = pd.DataFrame(
        {
            "speed_difference": [
                20.0,
                15.0,
                10.0,
                12.0,
                14.0,
                8.0,
                4.0,
                2.0,
            ],
        },
        index=range(200, 208),
    )
    query_specs = (
        {
            "tool_id": "query_telemetry.compute_slope.speed_difference",
            "graph_id": "speed_delta",
            "query_id": "compute_slope",
            "params": {"column": "speed_difference"},
        },
    )

    results = _run_queries(df, 200, 207, query_specs=query_specs)
    prompt = _prompt_block("lap", 200, 207, results, [], [])

    assert "Start trend: the speed gap is trending down from index 200 to 202" in prompt
    assert "Growth index ranges: index 202 to 204" in prompt
    assert "Shrink index ranges: index 200 to 202" in prompt
    assert "index 204 to 207" in prompt
    assert "Overall trend: the speed gap is trending down" in prompt


def test_preflight_trend_run_summary_uses_time_delta_selected_terms():
    df = pd.DataFrame(
        {"expert_time_difference": [0.0, 100.0, 250.0, 500.0, 800.0]},
        index=range(10, 15),
    )

    results = _run_queries(df, 10, 14)
    prompt = _prompt_block("lap", 10, 14, results, [], [])

    assert "The selected losing time run spans" in prompt
    assert "selected_losing_time_run" not in prompt
    assert "losing_time_run" not in prompt
    assert "The time-gap trend verdict was time gap rising" in prompt
    assert "time_gap_rising_run" not in prompt
    assert "Required tool outputs" not in prompt
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


def test_altitude_labels_use_slope_angle_not_raw_delta():
    x = np.linspace(0.0, 40.0, 100)
    y = np.zeros(100)
    df = _trajectory_df(x, y)
    df["expert_optimal_player_pos_y"] = 18.0 * np.sin(np.linspace(0.0, np.pi, 100))
    df["Graphics_player_pos_y"] = df["expert_optimal_player_pos_y"]
    df["expert_optimal_player_pos_z"] = np.linspace(0.0, 0.1, 100)
    df["Graphics_player_pos_z"] = df["expert_optimal_player_pos_z"]

    attachment = measure_segment_shape(df, 0, len(df))
    entry = attachment.content["altitude"]["entry"]

    assert entry["delta_m"] > 0.0
    assert entry["slope_angle_degrees"] < 1.0
    assert entry["trend"] == "level"
    assert "horizontal_distance_units" in entry


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
