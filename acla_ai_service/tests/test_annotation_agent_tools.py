"""Tests for deterministic annotation-agent telemetry tools."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.shared.annotation_agent_tools import (
    _relative_position_frame,
    _smoothed_expert_kinematics,
    classify_opponent_interaction,
    find_nearest_opponent,
    split_lap_by_circuit_sections,
)


def _curve_slice(n: int = 60) -> pd.DataFrame:
    theta = np.linspace(0.0, np.pi, n)
    radius = 40.0
    return pd.DataFrame({
        "Graphics_normalized_car_position": np.linspace(0.12, 0.18, n),
        "Graphics_player_pos_x": radius * np.cos(theta),
        "Graphics_player_pos_y": radius * np.sin(theta),
    })


def _brands_hatch_slice(n: int = 120) -> pd.DataFrame:
    """Small Brands Hatch slice crossing two measured section ranges."""
    player_x = np.arange(n, dtype=float)
    player_y = np.zeros(n, dtype=float)
    return pd.DataFrame({
        "Graphics_normalized_car_position": np.linspace(0.12, 0.24, n),
        "Graphics_player_pos_x": player_x,
        "Graphics_player_pos_y": player_y,
    })


def _silverstone_unmeasured_slice(n: int = 120) -> pd.DataFrame:
    """Slice on a circuit whose section ranges are intentionally TBD."""
    player_x = np.arange(n, dtype=float)
    player_y = np.zeros(n, dtype=float)
    return pd.DataFrame({
        "Graphics_normalized_car_position": np.linspace(0.12, 0.24, n),
        "Graphics_player_pos_x": player_x,
        "Graphics_player_pos_y": player_y,
    })


def test_split_expands_section_for_close_overtake_context() -> None:
    df = _brands_hatch_slice()
    n = len(df)
    signed_gap = np.linspace(20.0, -20.0, n)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + signed_gap
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    interaction_segments = [
        s for s in segments
        if s.get("split_basis") == "opponent_interaction"
    ]

    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert interaction_segments
    assert att.content["interaction_windows"]
    assert any(
        w["passed_by_player"]
        for s in interaction_segments
        for w in s["opponent_interaction"]["windows"]
    )


def test_split_stays_section_only_without_opponents() -> None:
    df = _brands_hatch_slice()

    att = split_lap_by_circuit_sections(
        df, 0, len(df), circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    assert att.content["interaction_windows"] == []
    assert att.content["opponent_session"] is False
    assert att.content["split_mode"] == "circuit_sections"
    assert {s["split_basis"] for s in att.content["segments"]} == {"circuit_section"}


def test_player_car_slot_is_not_treated_as_opponent() -> None:
    df = _brands_hatch_slice()
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]

    att = split_lap_by_circuit_sections(
        df, 0, len(df), circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    assert att.content["opponent_session"] is False
    assert att.content["split_mode"] == "circuit_sections"
    assert att.content["interaction_windows"] == []
    assert {s["split_basis"] for s in att.content["segments"]} == {"circuit_section"}


def test_opponent_session_without_close_engagement_returns_no_work_units() -> None:
    df = _brands_hatch_slice()
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + 200.0
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, len(df), circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert att.content["interaction_windows"] == []
    assert att.content["segments"] == []


def test_curved_path_gap_uses_along_track_order_not_local_tangent() -> None:
    df = _curve_slice(60)
    player_x = df["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = df["Graphics_player_pos_y"].to_numpy(dtype=float)
    lag = 4
    behind_x = np.r_[np.full(lag, np.nan), player_x[:-lag]]
    behind_y = np.r_[np.full(lag, np.nan), player_y[:-lag]]
    ahead_x = np.r_[player_x[lag:], np.full(lag, np.nan)]
    ahead_y = np.r_[player_y[lag:], np.full(lag, np.nan)]

    behind_long, _behind_lat, _s, _d, frame = _relative_position_frame(
        df, player_x, player_y, behind_x, behind_y,
    )
    ahead_long, _ahead_lat, _s, _d, _frame = _relative_position_frame(
        df, player_x, player_y, ahead_x, ahead_y,
    )

    valid = slice(lag + 2, -lag - 2)
    assert frame == "player_local_path_projection"
    assert np.nanmax(behind_long[valid]) < 0.0
    assert np.nanmin(ahead_long[valid]) > 0.0


def test_local_path_projection_extrapolates_opponent_beyond_player_exit() -> None:
    df = _curve_slice(40)
    player_x = df["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = df["Graphics_player_pos_y"].to_numpy(dtype=float)
    dx = player_x[-1] - player_x[-3]
    dy = player_y[-1] - player_y[-3]
    norm = np.sqrt(dx * dx + dy * dy)
    ahead_x = player_x.copy()
    ahead_y = player_y.copy()
    ahead_x[-1] = player_x[-1] + dx / norm * 12.0
    ahead_y[-1] = player_y[-1] + dy / norm * 12.0

    long_gap, _lat, _s, _d, frame = _relative_position_frame(
        df, player_x, player_y, ahead_x, ahead_y,
    )

    assert frame == "player_local_path_projection"
    assert long_gap[-1] > 5.0


def test_side_by_side_requires_tighter_overlap() -> None:
    df = _brands_hatch_slice(20)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"] + 6.0

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["min_distance_m"] == 6.0
    assert candidate["side_by_side_iloc_count"] == 0
    assert candidate["outcome"] == "incidental"


def test_relative_velocity_can_gate_attack_pressure() -> None:
    df = _brands_hatch_slice(12)
    df["Graphics_current_time"] = np.arange(len(df), dtype=float) * 100.0
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + np.linspace(10.0, 8.0, len(df))
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["outcome"] == "failed_attack"
    assert candidate["recommended_label"] == "MSR"
    assert candidate["attack_relative_long_gap_velocity"] < -0.5
    assert candidate["relative_velocity_units"] == "m/s"


def test_far_signed_gap_flip_does_not_count_as_completed_pass() -> None:
    df = _brands_hatch_slice(20)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + np.linspace(14.0, -18.0, len(df))
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"] + 20.0

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["passed_by_player"] is False
    assert candidate["outcome"] != "pass_completed"
    assert candidate["recommended_label"] is None
    assert att.content["gates"]["O"] is False


def test_local_path_projection_keeps_rear_car_behind_on_curved_path() -> None:
    n = 40
    theta = np.linspace(0.0, np.pi, n)
    player_x = 100.0 * np.cos(theta)
    player_y = 100.0 * np.sin(theta)
    dx = np.gradient(player_x)
    dy = np.gradient(player_y)
    norm = np.sqrt(dx * dx + dy * dy)
    hx = dx / norm
    hy = dy / norm

    df = pd.DataFrame({
        "Graphics_normalized_car_position": np.linspace(0.10, 0.18, n),
        "Graphics_player_pos_x": player_x,
        "Graphics_player_pos_y": player_y,
        "expert_optimal_player_pos_x": player_x,
        "expert_optimal_player_pos_y": player_y,
    })
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = player_x - hx * 5.0
    df["Car_2_pos_y"] = player_y - hy * 5.0

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["coordinate_frame"] == "player_local_path_projection"
    assert candidate["role"] == "following"
    assert candidate["outcome"] == "close_following"
    assert candidate["entry_signed_long_gap_m"] < 0
    assert candidate["exit_signed_long_gap_m"] < 0


def test_opponent_from_level_to_ahead_is_broken_defense() -> None:
    df = _brands_hatch_slice(20)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + np.linspace(0.0, 10.0, len(df))
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"] + 2.0

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["role"] == "defense"
    assert candidate["outcome"] == "broken_defense"
    assert candidate["recommended_label"] == "MSR"
    assert candidate["got_passed_by_opponent"] is True
    assert att.content["gates"]["MSR"] is True


def test_close_trailing_car_is_identified_as_following_context() -> None:
    df = _brands_hatch_slice(20)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] - 24.0
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert att.content["primary_slot_for_role"] == 2
    assert candidate["role"] == "following"
    assert candidate["outcome"] == "close_following"
    assert candidate["recommended_label"] is None
    assert candidate["trailing_pressure_iloc_count"] == len(df)
    assert candidate["close_following_iloc_count"] == len(df)
    assert att.content["gates"] == {"O": False, "OD": False, "MSR": False}
    assert att.content["label_gates"]["OD"] is False


def test_inline_bumper_pressure_is_not_side_by_side_defense() -> None:
    df = _brands_hatch_slice(20)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] - 3.0
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["role"] == "following"
    assert candidate["outcome"] == "close_following"
    assert candidate["recommended_label"] is None
    assert candidate["side_by_side_iloc_count"] == 0
    assert candidate["trailing_pressure_iloc_count"] == len(df)
    assert att.content["gates"]["OD"] is False


def test_brief_rear_overlap_then_fall_back_stays_close_following() -> None:
    df = _brands_hatch_slice(30)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    signed_gap = np.concatenate((
        np.array([-0.93, 0.0]),
        np.linspace(-6.72, -13.01, len(df) - 2),
    ))
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + signed_gap
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = classify_opponent_interaction(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["role"] == "following"
    assert candidate["outcome"] == "close_following"
    assert candidate["recommended_label"] is None
    assert candidate["side_by_side_iloc_count"] < 3
    assert candidate["defense_threat_gain_m"] < 4.0
    assert att.content["gates"]["OD"] is False
    assert att.content["label_gates"]["OD"] is False


def test_split_detects_inline_trailing_pressure_window() -> None:
    df = _brands_hatch_slice(60)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] - 24.0
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = split_lap_by_circuit_sections(
        df, 0, len(df), circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert segments
    window = segments[0]["opponent_interaction"]["windows"][0]
    assert window["slot"] == 2
    assert window["event_role"] == "following"
    assert window["event_outcome"] == "close_following"
    assert window["close_following_iloc_count"] > 0
    assert window["trailing_pressure_iloc_count"] > 0


def test_split_keeps_brief_rear_overlap_as_following_window() -> None:
    df = _brands_hatch_slice(60)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    signed_gap = np.concatenate((
        np.array([-0.93, 0.0]),
        np.linspace(-6.72, -13.01, len(df) - 2),
    ))
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + signed_gap
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = split_lap_by_circuit_sections(
        df, 0, len(df), circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    window = att.content["segments"][0]["opponent_interaction"]["windows"][0]
    assert window["event_role"] == "following"
    assert window["event_outcome"] == "close_following"
    assert window["side_by_side_iloc_count"] < 3


def test_nearest_opponent_reports_close_following_counts() -> None:
    df = _brands_hatch_slice(12)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] - 18.0
    df["Car_2_pos_y"] = df["Graphics_player_pos_y"]

    att = find_nearest_opponent(df, 0, len(df))
    candidate = att.content["candidates"][0]

    assert candidate["slot"] == 2
    assert candidate["close_following_iloc_count"] == len(df)
    assert candidate["trailing_pressure_iloc_count"] == len(df)
    assert candidate["leading_draft_iloc_count"] == 0


def test_long_inline_draft_compresses_to_following_window() -> None:
    n = 160
    player_x = np.arange(n, dtype=float)
    player_y = np.zeros(n, dtype=float)
    df = pd.DataFrame({
        "Graphics_normalized_car_position": np.linspace(0.09, 0.25, n),
        "Graphics_player_pos_x": player_x,
        "Graphics_player_pos_y": player_y,
    })
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + np.linspace(10.0, 2.0, n)
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert segments
    assert len(segments) < 3
    assert {s["circuit_section_id"] for s in segments} == {"interaction_window"}
    assert all(s["start_index"] < s["end_index"] for s in segments)
    assert all(s["end_index"] - s["start_index"] <= 80 for s in segments)
    assert all(
        segment["opponent_interaction"]["windows"][0]["event_role"] == "following"
        for segment in segments
    )
    assert all(
        segment["opponent_interaction"]["windows"][0]["event_outcome"] == "close_following"
        for segment in segments
    )
    assert all(
        segment["opponent_interaction"]["windows"][0]["leading_draft_iloc_count"] > 0
        for segment in segments
    )


def test_same_section_interaction_is_split_at_lap_boundaries() -> None:
    n = 300
    player_x = np.arange(n, dtype=float)
    player_y = np.zeros(n, dtype=float)
    df = pd.DataFrame({
        "Graphics_normalized_car_position": np.tile(np.linspace(0.94, 0.99, 100), 3),
        "Graphics_completed_lap": np.repeat([0, 1, 2], 100),
        "Graphics_player_pos_x": player_x,
        "Graphics_player_pos_y": player_y,
    })
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + np.tile(np.linspace(20.0, -20.0, 100), 3)
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert len(segments) == 3
    assert all(s["start_index"] < s["end_index"] for s in segments)
    assert all(
        s["start_index"] < boundary < s["end_index"]
        for s, boundary in zip(segments, [50, 150, 250])
    )
    assert all(
        not (s["start_index"] < boundary < s["end_index"])
        for s in segments
        for boundary in [100, 200]
    )
    assert {s["circuit_section_id"] for s in segments} == {"interaction_window"}
    assert all(
        s["opponent_interaction"]["section_context"][0]["circuit_section_id"] == "brands_hatch1"
        for s in segments
    )


def test_racing_interaction_is_not_blocked_by_unmeasured_sections() -> None:
    df = _silverstone_unmeasured_slice()
    n = len(df)
    signed_gap = np.linspace(20.0, -20.0, n)
    df["Car_1_pos_x"] = df["Graphics_player_pos_x"]
    df["Car_1_pos_y"] = df["Graphics_player_pos_y"]
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + signed_gap
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="silverstone",
        include_interaction_windows=True,
    )

    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert att.content["warning"]
    assert att.content["segments"]
    assert att.content["segments"][0]["circuit_section_id"] == "interaction_window"
    assert att.content["segments"][0]["split_basis"] == "opponent_interaction"


def test_smoothed_expert_kinematics_suppresses_zero_denominator_warning() -> None:
    df = pd.DataFrame({
        "expert_optimal_player_pos_x": np.ones(30),
        "expert_optimal_player_pos_y": np.ones(30),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        kin = _smoothed_expert_kinematics(df)

    assert kin is not None
    _x_s, _y_s, _dx, _dy, kappa, _window = kin
    assert np.all(kappa == 0.0)
