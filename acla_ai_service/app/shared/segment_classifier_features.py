"""Explicit raw feature allowlist for the segment classifier."""

from typing import List


SEGMENT_CLASSIFIER_FEATURES: List[str] = [
    "Graphics_normalized_car_position",
    "Graphics_player_pos_x",
    "Graphics_player_pos_y",
    "Graphics_player_pos_z",
    "Physics_speed_kmh",
    "Physics_gas",
    "Physics_brake",
    "Physics_steer_angle",
    "Physics_gear",
    "Physics_rpm",
    "Physics_roll",
    "Physics_pitch",
    "Physics_g_force_x",
    "Physics_g_force_y",
    "Physics_g_force_z",
    "Physics_tyre_core_temp_front_left",
    "Physics_tyre_core_temp_front_right",
    "Physics_tyre_core_temp_rear_left",
    "Physics_tyre_core_temp_rear_right",
    "Physics_brake_temp_front_left",
    "Physics_brake_temp_front_right",
    "Physics_brake_temp_rear_left",
    "Physics_brake_temp_rear_right",
    "Physics_wheel_pressure_front_left",
    "Physics_wheel_pressure_front_right",
    "Physics_wheel_pressure_rear_left",
    "Physics_wheel_pressure_rear_right",
    "Physics_velocity_x",
    "Physics_velocity_y",
    "Physics_velocity_z",
    "Graphics_track_grip_status",
    "expert_optimal_player_pos_x",
    "expert_optimal_player_pos_y",
    "expert_optimal_player_pos_z",
    "expert_optimal_speed",
    "expert_optimal_throttle",
    "expert_optimal_brake",
    "expert_optimal_gear",
    "expert_optimal_time",
    "expert_velocity_alignment",
    "speed_difference",
    "distance_to_expert_line",
    "expert_time_difference",
]


__all__ = ["SEGMENT_CLASSIFIER_FEATURES"]
