---
name: query_telemetry_metric
title: Querying telemetry
description: >
  Read reduced telemetry metrics over a live scope. Use for direct numeric
  questions about recent or scoped telemetry.
parameters:
  fields:
    description: Use actual telemetry row names exactly as listed below. Do not invent display aliases.
  scope:
    description: Time window or event window to query.
  reduce:
    description: Aggregation to return. stats means avg, min, max, and stddev.
---

## Usage notes

This tool returns reduced metrics only. 

Use the actual telemetry row names directly in `fields`. Do not use UI/display
aliases or camel-case convenience names; those are not telemetry rows. 

## For fuel 
- `Physics_fuel`

## tyre-pressure questions
- `Physics_wheel_pressure_front_left`
- `Physics_wheel_pressure_front_right`
- `Physics_wheel_pressure_rear_left`
- `Physics_wheel_pressure_rear_right`


## Weather 
- `Graphcis_rain_intensity`
  values: no rain - 0, drizzle - 1, light rain - 2, medium rain - 3, heavy rain - 4, thunderstrom -5


## Common row names

- Speed: `Physics_speed_kmh`
- Throttle: `Physics_gas`
- Brake: `Physics_brake`
- Gear: `Physics_gear`
- Steering: `Physics_steer_angle`
- RPM: `Physics_rpm`
- Fuel: `Physics_fuel`
- Track position: `Graphics_normalized_car_position`
- Race position: `Graphics_position`
- Lap time strings: `Graphics_current_time_str`, `Graphics_last_time_str`, `Graphics_best_time_str`
- Tyre pressures: `Physics_wheel_pressure_front_left`, `Physics_wheel_pressure_front_right`, `Physics_wheel_pressure_rear_left`, `Physics_wheel_pressure_rear_right`
- Tyre core temperatures: `Physics_tyre_core_temp_front_left`, `Physics_tyre_core_temp_front_right`, `Physics_tyre_core_temp_rear_left`, `Physics_tyre_core_temp_rear_right`
- Brake temperatures: `Physics_brake_temp_front_left`, `Physics_brake_temp_front_right`, `Physics_brake_temp_rear_left`, `Physics_brake_temp_rear_right`
- Wheel slip: `Physics_wheel_slip_front_left`, `Physics_wheel_slip_front_right`, `Physics_wheel_slip_rear_left`, `Physics_wheel_slip_rear_right`
- G force: `Physics_g_force_x`, `Physics_g_force_y`, `Physics_g_force_z`
- Suspension travel: `Physics_suspension_travel_front_left`, `Physics_suspension_travel_front_right`, `Physics_suspension_travel_rear_left`, `Physics_suspension_travel_rear_right`

## Full telemetry row catalog

### Physics rows

- `Physics_pad_life_front_left`
- `Physics_wheel_angular_s_front_left`
- `Physics_brake_pressure_rear_left`
- `Physics_starter_engine_on`
- `Physics_is_engine_running`
- `Physics_tyre_contact_point_rear_right_z`
- `Physics_tyre_contact_normal_rear_right_x`
- `Physics_slip_angle_rear_left`
- `Physics_tyre_core_temp_front_left`
- `Physics_suspension_damage_rear_left`
- `Physics_tyre_contact_heading_rear_left_x`
- `Physics_rear_brake_compound`
- `Physics_local_angular_vel_x`
- `Physics_final_ff`
- `Physics_disc_life_rear_right`
- `Physics_tyre_core_temp_front_right`
- `Physics_tyre_contact_normal_front_right_z`
- `Physics_g_vibration`
- `Physics_brake_bias`
- `Physics_tyre_contact_point_front_right_x`
- `Physics_pad_life_front_right`
- `Physics_local_velocity_x`
- `Physics_brake_temp_rear_left`
- `Physics_tyre_contact_point_rear_left_y`
- `Physics_heading`
- `Physics_tyre_contact_heading_rear_right_z`
- `Physics_fuel`
- `Physics_tyre_contact_heading_front_left_z`
- `Physics_slip_vibration`
- `Physics_disc_life_front_left`
- `Physics_suspension_travel_front_right`
- `Physics_disc_life_rear_left`
- `Physics_slip_angle_front_right`
- `Physics_g_force_x`
- `Physics_rpm`
- `Physics_g_force_z`
- `Physics_car_damage_rear`
- `Physics_slip_ratio_front_left`
- `Physics_tyre_contact_heading_front_left_y`
- `Physics_tyre_contact_point_rear_right_y`
- `Physics_velocity_x`
- `Physics_tc`
- `Physics_wheel_pressure_front_right`
- `Physics_suspension_travel_front_left`
- `Physics_tyre_contact_heading_rear_right_y`
- `Physics_clutch`
- `Physics_road_temp`
- `Physics_wheel_pressure_front_left`
- `Physics_local_velocity_z`
- `Physics_wheel_angular_s_rear_right`
- `Physics_brake_temp_front_right`
- `Physics_tyre_contact_point_rear_left_x`
- `Physics_tyre_contact_heading_front_left_x`
- `Physics_air_temp`
- `Physics_g_force_y`
- `Physics_autoshifter_on`
- `Physics_brake_temp_rear_right`
- `Physics_abs_vibration`
- `Physics_gear`
- `Physics_wheel_pressure_rear_right`
- `Physics_tyre_contact_point_rear_left_z`
- `Physics_tyre_contact_heading_front_right_y`
- `Physics_suspension_travel_rear_right`
- `Physics_local_angular_vel_z`
- `Physics_tyre_contact_point_front_left_z`
- `Physics_brake_pressure_rear_right`
- `Physics_kerb_vibration`
- `Physics_tyre_contact_heading_rear_right_x`
- `Physics_tyre_contact_heading_front_right_z`
- `Physics_tyre_contact_heading_rear_left_z`
- `Physics_wheel_slip_rear_left`
- `Physics_slip_ratio_front_right`
- `Physics_tyre_contact_point_front_right_y`
- `Physics_steer_angle`
- `Physics_is_ai_controlled`
- `Physics_car_damage_left`
- `Physics_wheel_pressure_rear_left`
- `Physics_wheel_angular_s_rear_left`
- `Physics_pad_life_rear_right`
- `Physics_ignition_on`
- `Physics_car_damage_right`
- `Physics_tyre_contact_normal_rear_right_z`
- `Physics_velocity_z`
- `Physics_wheel_slip_rear_right`
- `Physics_tyre_contact_point_front_left_y`
- `Physics_tyre_core_temp_rear_left`
- `Physics_tyre_contact_point_front_right_z`
- `Physics_brake`
- `Physics_gas`
- `Physics_speed_kmh`
- `Physics_slip_angle_front_left`
- `Physics_slip_ratio_rear_right`
- `Physics_brake_pressure_front_right`
- `Physics_abs`
- `Physics_pitch`
- `Physics_tyre_contact_normal_rear_left_z`
- `Physics_roll`
- `Physics_tyre_contact_normal_rear_left_x`
- `Physics_pad_life_rear_left`
- `Physics_tyre_contact_normal_front_right_y`
- `Physics_local_angular_vel_y`
- `Physics_tyre_contact_normal_front_left_x`
- `Physics_suspension_travel_rear_left`
- `Physics_brake_temp_front_left`
- `Physics_slip_angle_rear_right`
- `Physics_slip_ratio_rear_left`
- `Physics_wheel_slip_front_right`
- `Physics_tyre_contact_heading_front_right_x`
- `Physics_suspension_damage_rear_right`
- `Physics_tyre_core_temp_rear_right`
- `Physics_tyre_contact_normal_rear_right_y`
- `Physics_tyre_contact_heading_rear_left_y`
- `Physics_disc_life_front_right`
- `Physics_wheel_angular_s_front_right`
- `Physics_tyre_contact_point_front_left_x`
- `Physics_tyre_contact_normal_front_right_x`
- `Physics_car_damage_front`
- `Physics_turbo_boost`
- `Physics_local_velocity_y`
- `Physics_water_temp`
- `Physics_tyre_contact_normal_front_left_z`
- `Physics_car_damage_center`
- `Physics_suspension_damage_front_left`
- `Physics_velocity_y`
- `Physics_tyre_contact_normal_front_left_y`
- `Physics_packed_id`
- `Physics_wheel_slip_front_left`
- `Physics_front_brake_compound`
- `Physics_suspension_damage_front_right`
- `Physics_brake_pressure_front_left`
- `Physics_tyre_contact_point_rear_right_x`
- `Physics_tyre_contact_normal_rear_left_y`

### Graphics rows

- `Graphics_ideal_line_on`
- `Graphics_is_valid_lap`
- `Graphics_packed_id`
- `Graphics_delta_lap_time_str`
- `Graphics_mfd_tyre_pressure_rear_left`
- `Graphics_mfd_tyre_pressure_front_right`
- `Graphics_rain_light`
- `Graphics_current_tyre_set`
- `Graphics_flashing_light`
- `Graphics_wiper_stage`
- `Graphics_mfd_tyre_pressure_rear_right`
- `Graphics_missing_mandatory_pits`
- `Graphics_best_time_str`
- `Graphics_player_car_id`
- `Graphics_is_delta_positive`
- `Graphics_mfd_fuel_to_add`
- `Graphics_driver_stint_total_time_left`
- `Graphics_tyre_compound`
- `Graphics_session_index`
- `Graphics_driver_stint_time_left`
- `Graphics_global_green`
- `Graphics_global_chequered`
- `Graphics_global_red`
- `Graphics_current_sector_index`
- `Graphics_direction_light_right`
- `Graphics_gap_ahead`
- `Graphics_global_white`
- `Graphics_last_time`
- `Graphics_clock`
- `Graphics_last_time_str`
- `Graphics_wind_direction`
- `Graphics_gap_behind`
- `Graphics_abs_level`
- `Graphics_delta_lap_time`
- `Graphics_used_fuel`
- `Graphics_global_yellow_s3`
- `Graphics_car_coordinates`
- `Graphics_mfd_tyre_set`
- `Graphics_normalized_car_position`
- `Graphics_wind_speed`
- `Graphics_current_time_str`
- `Graphics_last_sector_time_str`
- `Graphics_mfd_tyre_pressure_front_left`
- `Graphics_penalty_time`
- `Graphics_mandatory_pit_done`
- `Graphics_tc_level`
- `Graphics_strategy_tyre_set`
- `Graphics_last_sector_time`
- `Graphics_fuel_estimated_laps`
- `Graphics_direction_light_left`
- `Graphics_session_time_left`
- `Graphics_fuel_per_lap`
- `Graphics_track_status`
- `Graphics_number_of_laps`
- `Graphics_is_setup_menu_visible`
- `Graphics_position`
- `Graphics_rain_tyres`
- `Graphics_global_yellow_s2`
- `Graphics_car_id`
- `Graphics_best_time`
- `Graphics_is_in_pit`
- `Graphics_exhaust_temp`
- `Graphics_estimated_lap_time`
- `Graphics_secondary_display_index`
- `Graphics_global_yellow_s1`
- `Graphics_completed_lap`
- `Graphics_distance_traveled`
- `Graphics_main_display_index`
- `Graphics_light_stage`
- `Graphics_global_yellow`
- `Graphics_engine_map`
- `Graphics_active_cars`
- `Graphics_tc_cut_level`
- `Graphics_estimated_lap_time_str`
- `Graphics_current_time`

### Static rows

- `Static_sector_count`
- `Static_pit_window_start`
- `Static_max_rpm`
- `Static_pit_window_end`
- `Static_aid_auto_clutch`
- `Static_track`
- `Static_number_of_session`
- `Static_aid_stability`
- `Static_max_fuel`
- `Static_ac_version`
- `Static_num_cars`
- `Static_aid_tyre_rate`
- `Static_sm_version`
- `Static_player_name`
- `Static_penalty_enabled`
- `Static_dry_tyres_name`
- `Static_player_surname`
- `Static_is_online`
- `Static_car_model`
- `Static_aid_mechanical_damage`
- `Static_wet_tyres_name`
- `Static_aid_fuel_rate`
