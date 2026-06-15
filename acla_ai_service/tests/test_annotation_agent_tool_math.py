import numpy as np
import pandas as pd

from app.shared.annotation_agent_tools import (
    _query_compute_slope,
    measure_segment_shape,
)


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


def test_straight_segment_shape_does_not_emit_corner_or_altitude_labels():
    x = np.linspace(0.0, 40.0, 80)
    y = 0.03 * np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
    df = _trajectory_df(x, y)

    attachment = measure_segment_shape(df, 0, len(df))
    content = attachment.content

    assert content["base_segment_shape"]["label_id"] == "ST2"
    assert content["corner_shape_refinement"] is None
    assert content["altitude"]["labels"] == []
    assert content["phases"] == []


def test_real_corner_segment_shape_still_emits_corner_phase():
    theta = np.linspace(0.0, np.pi / 2.0, 100)
    radius = 30.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))

    attachment = measure_segment_shape(df, 0, len(df))
    content = attachment.content

    assert content["base_segment_shape"]["label_id"] == "ST1"
    assert content["corner_shape_refinement"] is not None
    assert content["phases"]
    assert content["phases"][0]["turn_angle_degrees"] >= 10.0


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
    assert refinement["label_id"] != "ST10"
    assert refinement["is_near_u_turn"] is False


def test_hairpin_shape_uses_turn_angle_without_radius_gate():
    theta = np.linspace(0.0, np.pi, 180)
    radius = 120.0
    df = _trajectory_df(radius * np.cos(theta), radius * np.sin(theta))

    attachment = measure_segment_shape(df, 0, len(df))
    refinement = attachment.content["corner_shape_refinement"]

    assert refinement["label_id"] == "ST10"
    assert refinement["is_near_u_turn"] is True
    assert "is_tight" not in refinement
    assert "average_radius_m" not in refinement
