"""Tests for training-pipeline telemetry cleaning helpers."""

from __future__ import annotations

import pandas as pd

from app.pipelines.training.pipeline.cleaning import _clean_position_anomalies


def test_position_cleanup_removes_anomalous_player_row_and_resets_index() -> None:
    df = pd.DataFrame({
        "Graphics_current_time": [0, 100, 200, 300, 400],
        "Graphics_player_pos_x": [0.0, 1.0, 9_000_000.0, 3.0, 4.0],
        "Graphics_player_pos_y": [0.0, 0.0, 9_000_000.0, 0.0, 0.0],
        "Graphics_player_pos_z": [0.0, 0.0, 9_000_000.0, 0.0, 0.0],
        "Car_1_pos_x": [0.0, 1.0, 9_000_000.0, 3.0, 4.0],
        "Car_1_pos_y": [0.0, 0.0, 9_000_000.0, 0.0, 0.0],
        "Car_1_pos_z": [0.0, 0.0, 9_000_000.0, 0.0, 0.0],
    })

    cleaned = _clean_position_anomalies(df)

    assert cleaned.index.tolist() == [0, 1, 2, 3]
    assert cleaned["Graphics_player_pos_x"].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_position_cleanup_clears_bad_opponent_slot_without_dropping_row() -> None:
    df = pd.DataFrame({
        "Graphics_current_time": [0, 100, 200, 300, 400],
        "Graphics_player_pos_x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "Graphics_player_pos_y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Graphics_player_pos_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Car_2_pos_x": [10.0, 11.0, 8_000_000.0, 13.0, 14.0],
        "Car_2_pos_y": [0.0, 0.0, 8_000_000.0, 0.0, 0.0],
        "Car_2_pos_z": [0.0, 0.0, 8_000_000.0, 0.0, 0.0],
    })

    cleaned = _clean_position_anomalies(df)

    assert len(cleaned) == len(df)
    assert cleaned.loc[
        2,
        ["Car_2_pos_x", "Car_2_pos_y", "Car_2_pos_z"],
    ].tolist() == [0.0, 0.0, 0.0]
    assert cleaned["Graphics_player_pos_x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_position_cleanup_keeps_session_start_relocation() -> None:
    df = pd.DataFrame({
        "Graphics_current_time": [0, 100, 200, 300],
        "Graphics_player_pos_x": [-279.0, -259.0, -258.0, -257.0],
        "Graphics_player_pos_y": [-85.0, -335.0, -336.0, -337.0],
        "Graphics_player_pos_z": [1.8, -5.0, -5.0, -5.0],
        "Car_1_pos_x": [-279.0, -259.0, -258.0, -257.0],
        "Car_1_pos_y": [-85.0, -335.0, -336.0, -337.0],
        "Car_1_pos_z": [1.8, -5.0, -5.0, -5.0],
    })

    cleaned = _clean_position_anomalies(df)

    assert len(cleaned) == len(df)
    assert cleaned["Graphics_player_pos_x"].tolist() == [-279.0, -259.0, -258.0, -257.0]
    assert cleaned["Graphics_player_pos_y"].tolist() == [-85.0, -335.0, -336.0, -337.0]
