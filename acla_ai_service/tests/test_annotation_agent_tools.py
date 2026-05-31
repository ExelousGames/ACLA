"""Tests for deterministic annotation-agent telemetry tools."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.shared.annotation_agent_tools import (
    _smoothed_expert_kinematics,
    split_lap_by_circuit_sections,
)


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


def test_interaction_window_keeps_sections_as_context_only() -> None:
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
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + 2.0
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert len(segments) == 1
    assert segments[0]["start_index"] == 0
    assert segments[0]["end_index"] == n
    assert segments[0]["circuit_section_id"] == "interaction_window"
    section_context = segments[0]["opponent_interaction"]["section_context"]
    assert {s["circuit_section_id"] for s in section_context} >= {
        "brands_hatch3",
        "brands_hatch4",
    }


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
    df["Car_2_pos_x"] = df["Graphics_player_pos_x"] + 2.0
    df["Car_2_pos_y"] = 0.0

    att = split_lap_by_circuit_sections(
        df, 0, n, circuit_id="brands_hatch",
        include_interaction_windows=True,
    )

    segments = att.content["segments"]
    assert att.content["opponent_session"] is True
    assert att.content["split_mode"] == "opponent_interactions_only"
    assert [s["start_index"] for s in segments] == [0, 100, 200]
    assert [s["end_index"] for s in segments] == [100, 200, 300]
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
