from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import runtime_segment_splitter
from app.services.runtime_segment_splitter import (
    RuntimeSegmentSplitError,
    resolve_runtime_circuit,
    split_runtime_segments,
)


def _solo_frame(positions, **columns):
    return pd.DataFrame({
        "Graphics_normalized_car_position": positions,
        **columns,
    })


def _opponent_frame(opponent_gap):
    gap = np.asarray(opponent_gap, dtype=float)
    player_x = np.arange(gap.size, dtype=float)
    zeros = np.zeros(gap.size, dtype=float)
    return _solo_frame(
        np.linspace(0.12, 0.17, gap.size),
        Graphics_player_pos_x=player_x,
        Graphics_player_pos_y=zeros,
        Car_1_pos_x=player_x,
        Car_1_pos_y=zeros,
        Car_2_pos_x=player_x + gap,
        Car_2_pos_y=np.full(gap.size, 2.0),
    )


def test_circuit_section_changes_create_ordered_ranges():
    result = split_runtime_segments(
        _solo_frame([0.12, 0.15, 0.20, 0.23]),
        "brands_hatch",
    )

    assert [
        (segment["start_index"], segment["end_index"], segment["circuit_section_id"])
        for segment in result["segments"]
    ] == [
        (0, 2, "brands_hatch3"),
        (2, 4, "brands_hatch4"),
    ]


@pytest.mark.parametrize("positions", [[0.95, 0.98], [0.01, 0.05]])
def test_wrap_around_section_matches_both_sides_of_start_finish(positions):
    result = split_runtime_segments(_solo_frame(positions), "moza")

    assert len(result["segments"]) == 1
    assert result["segments"][0]["circuit_section_id"] == "moza1"


def test_lap_boundary_splits_a_wrap_around_section():
    result = split_runtime_segments(
        _solo_frame([0.95, 0.98, 0.01, 0.05]),
        "moza",
    )

    assert [
        (segment["start_index"], segment["end_index"])
        for segment in result["segments"]
    ] == [(0, 2), (2, 4)]


def test_opponent_pass_creates_interaction_ranges():
    result = split_runtime_segments(
        _opponent_frame(np.linspace(8.0, -8.0, 40)),
        "brands_hatch",
    )

    assert result["opponent_session"] is True
    assert result["split_mode"] == "opponent_interactions_only"
    assert result["segments"]
    assert all(
        segment["split_basis"] == "opponent_interaction"
        for segment in result["segments"]
    )


def test_following_only_opponent_session_returns_no_ranges(monkeypatch):
    dataframe = _opponent_frame(np.full(40, 10.0))
    monkeypatch.setattr(
        runtime_segment_splitter,
        "_detect_opponent_interaction_windows",
        lambda dataframe, start_index, end_index: [{
            "start_index": 5,
            "end_index": 20,
            "slot": 2,
            "event_role": "following",
            "event_outcome": "close_following",
        }],
    )
    monkeypatch.setattr(
        runtime_segment_splitter,
        "_align_interaction_windows_with_classifier",
        lambda dataframe, windows: windows,
    )
    result = split_runtime_segments(dataframe, "brands_hatch")

    assert result["opponent_session"] is True
    assert result["segments"] == []
    assert result["following_windows_filtered"] >= 1


def test_explicit_circuit_precedes_static_track_and_errors_when_unsupported():
    dataframe = _solo_frame([0.12], Static_track=["brands_hatch"])

    assert resolve_runtime_circuit(dataframe) == "brands_hatch"
    with pytest.raises(RuntimeSegmentSplitError, match="unsupported circuit"):
        resolve_runtime_circuit(dataframe, "spa")


def test_missing_normalized_position_is_rejected():
    with pytest.raises(RuntimeSegmentSplitError, match="missing from telemetry"):
        split_runtime_segments(pd.DataFrame({"Static_track": ["brands_hatch"]}), None)
