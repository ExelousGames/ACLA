// The authoritative field table for every simulator reader and adapter.
// Add telemetry fields here before emitting them. Downstream accepts only rows
// validated against this table. Readers convert to the documented units and
// enum meanings before emitting a row.
// Missing source values are omitted, never filled with aliases or legacy data.
// Coordinates and car IDs each have 60 entries; all other values are scalar.
// Units and enum meanings are documented in tmp/telemetry-fields.md.
const LIVE_TELEMETRY_DATASET = Object.freeze(/** @type {const} */ ({
  // Physics
  Physics_abs: 'number',
  Physics_abs_vibration: 'number',
  Physics_air_temp: 'number',
  Physics_autoshifter_on: 'boolean',
  Physics_brake: 'number',
  Physics_brake_bias: 'number',
  Physics_brake_pressure_front_left: 'number',
  Physics_brake_pressure_front_right: 'number',
  Physics_brake_pressure_rear_left: 'number',
  Physics_brake_pressure_rear_right: 'number',
  Physics_brake_temp_front_left: 'number',
  Physics_brake_temp_front_right: 'number',
  Physics_brake_temp_rear_left: 'number',
  Physics_brake_temp_rear_right: 'number',
  Physics_car_damage_center: 'number',
  Physics_car_damage_front: 'number',
  Physics_car_damage_left: 'number',
  Physics_car_damage_rear: 'number',
  Physics_car_damage_right: 'number',
  Physics_clutch: 'number',
  Physics_disc_life_front_left: 'number',
  Physics_disc_life_front_right: 'number',
  Physics_disc_life_rear_left: 'number',
  Physics_disc_life_rear_right: 'number',
  Physics_final_ff: 'number',
  Physics_front_brake_compound: 'integer',
  Physics_fuel: 'number',
  Physics_g_force_x: 'number',
  Physics_g_force_y: 'number',
  Physics_g_force_z: 'number',
  Physics_g_vibration: 'number',
  Physics_gas: 'number',
  Physics_gear: 'integer',
  Physics_heading: 'number',
  Physics_ignition_on: 'boolean',
  Physics_is_ai_controlled: 'boolean',
  Physics_is_engine_running: 'boolean',
  Physics_kerb_vibration: 'number',
  Physics_local_angular_vel_x: 'number',
  Physics_local_angular_vel_y: 'number',
  Physics_local_angular_vel_z: 'number',
  Physics_local_velocity_x: 'number',
  Physics_local_velocity_y: 'number',
  Physics_local_velocity_z: 'number',
  Physics_packed_id: 'integer',
  Physics_pad_life_front_left: 'number',
  Physics_pad_life_front_right: 'number',
  Physics_pad_life_rear_left: 'number',
  Physics_pad_life_rear_right: 'number',
  Physics_pit_limiter_on: 'boolean',
  Physics_pitch: 'number',
  Physics_rear_brake_compound: 'integer',
  Physics_road_temp: 'number',
  Physics_roll: 'number',
  Physics_rpm: 'integer',
  Physics_slip_angle_front_left: 'number',
  Physics_slip_angle_front_right: 'number',
  Physics_slip_angle_rear_left: 'number',
  Physics_slip_angle_rear_right: 'number',
  Physics_slip_ratio_front_left: 'number',
  Physics_slip_ratio_front_right: 'number',
  Physics_slip_ratio_rear_left: 'number',
  Physics_slip_ratio_rear_right: 'number',
  Physics_slip_vibration: 'number',
  Physics_speed_kmh: 'number',
  Physics_starter_engine_on: 'boolean',
  Physics_steer_angle: 'number',
  Physics_suspension_damage_front_left: 'number',
  Physics_suspension_damage_front_right: 'number',
  Physics_suspension_damage_rear_left: 'number',
  Physics_suspension_damage_rear_right: 'number',
  Physics_suspension_travel_front_left: 'number',
  Physics_suspension_travel_front_right: 'number',
  Physics_suspension_travel_rear_left: 'number',
  Physics_suspension_travel_rear_right: 'number',
  Physics_tc: 'number',
  Physics_turbo_boost: 'number',
  Physics_tyre_contact_heading_front_left_x: 'number',
  Physics_tyre_contact_heading_front_left_y: 'number',
  Physics_tyre_contact_heading_front_left_z: 'number',
  Physics_tyre_contact_heading_front_right_x: 'number',
  Physics_tyre_contact_heading_front_right_y: 'number',
  Physics_tyre_contact_heading_front_right_z: 'number',
  Physics_tyre_contact_heading_rear_left_x: 'number',
  Physics_tyre_contact_heading_rear_left_y: 'number',
  Physics_tyre_contact_heading_rear_left_z: 'number',
  Physics_tyre_contact_heading_rear_right_x: 'number',
  Physics_tyre_contact_heading_rear_right_y: 'number',
  Physics_tyre_contact_heading_rear_right_z: 'number',
  Physics_tyre_contact_normal_front_left_x: 'number',
  Physics_tyre_contact_normal_front_left_y: 'number',
  Physics_tyre_contact_normal_front_left_z: 'number',
  Physics_tyre_contact_normal_front_right_x: 'number',
  Physics_tyre_contact_normal_front_right_y: 'number',
  Physics_tyre_contact_normal_front_right_z: 'number',
  Physics_tyre_contact_normal_rear_left_x: 'number',
  Physics_tyre_contact_normal_rear_left_y: 'number',
  Physics_tyre_contact_normal_rear_left_z: 'number',
  Physics_tyre_contact_normal_rear_right_x: 'number',
  Physics_tyre_contact_normal_rear_right_y: 'number',
  Physics_tyre_contact_normal_rear_right_z: 'number',
  Physics_tyre_contact_point_front_left_x: 'number',
  Physics_tyre_contact_point_front_left_y: 'number',
  Physics_tyre_contact_point_front_left_z: 'number',
  Physics_tyre_contact_point_front_right_x: 'number',
  Physics_tyre_contact_point_front_right_y: 'number',
  Physics_tyre_contact_point_front_right_z: 'number',
  Physics_tyre_contact_point_rear_left_x: 'number',
  Physics_tyre_contact_point_rear_left_y: 'number',
  Physics_tyre_contact_point_rear_left_z: 'number',
  Physics_tyre_contact_point_rear_right_x: 'number',
  Physics_tyre_contact_point_rear_right_y: 'number',
  Physics_tyre_contact_point_rear_right_z: 'number',
  Physics_tyre_core_temp_front_left: 'number',
  Physics_tyre_core_temp_front_right: 'number',
  Physics_tyre_core_temp_rear_left: 'number',
  Physics_tyre_core_temp_rear_right: 'number',
  Physics_velocity_x: 'number',
  Physics_velocity_y: 'number',
  Physics_velocity_z: 'number',
  Physics_water_temp: 'number',
  Physics_wheel_angular_s_front_left: 'number',
  Physics_wheel_angular_s_front_right: 'number',
  Physics_wheel_angular_s_rear_left: 'number',
  Physics_wheel_angular_s_rear_right: 'number',
  Physics_wheel_pressure_front_left: 'number',
  Physics_wheel_pressure_front_right: 'number',
  Physics_wheel_pressure_rear_left: 'number',
  Physics_wheel_pressure_rear_right: 'number',
  Physics_wheel_slip_front_left: 'number',
  Physics_wheel_slip_front_right: 'number',
  Physics_wheel_slip_rear_left: 'number',
  Physics_wheel_slip_rear_right: 'number',

  // Graphics
  Graphics_abs_level: 'integer',
  Graphics_active_cars: 'integer',
  Graphics_best_time: 'integer',
  Graphics_best_time_str: 'string',
  Graphics_car_coordinates: 'coordinates',
  Graphics_car_id: 'integer-array',
  Graphics_clock: 'number',
  Graphics_completed_lap: 'integer',
  Graphics_current_sector_index: 'integer',
  Graphics_current_time: 'integer',
  Graphics_current_time_str: 'string',
  Graphics_current_tyre_set: 'integer',
  Graphics_delta_lap_time: 'integer',
  Graphics_delta_lap_time_str: 'string',
  Graphics_direction_light_left: 'boolean',
  Graphics_direction_light_right: 'boolean',
  Graphics_distance_traveled: 'number',
  Graphics_driver_stint_time_left: 'integer',
  Graphics_driver_stint_total_time_left: 'integer',
  Graphics_engine_map: 'integer',
  Graphics_estimated_lap_time: 'integer',
  Graphics_estimated_lap_time_str: 'string',
  Graphics_exhaust_temp: 'number',
  Graphics_flag: 'integer',
  Graphics_flashing_light: 'boolean',
  Graphics_fuel_estimated_laps: 'number',
  Graphics_fuel_per_lap: 'number',
  Graphics_gap_ahead: 'integer',
  Graphics_gap_behind: 'integer',
  Graphics_global_chequered: 'boolean',
  Graphics_global_green: 'boolean',
  Graphics_global_red: 'boolean',
  Graphics_global_white: 'boolean',
  Graphics_global_yellow: 'boolean',
  Graphics_global_yellow_s1: 'boolean',
  Graphics_global_yellow_s2: 'boolean',
  Graphics_global_yellow_s3: 'boolean',
  Graphics_ideal_line_on: 'boolean',
  Graphics_is_delta_positive: 'boolean',
  Graphics_is_in_pit: 'boolean',
  Graphics_is_in_pit_lane: 'boolean',
  Graphics_is_setup_menu_visible: 'boolean',
  Graphics_is_valid_lap: 'boolean',
  Graphics_last_sector_time: 'integer',
  Graphics_last_sector_time_str: 'integer',
  Graphics_last_time: 'integer',
  Graphics_last_time_str: 'string',
  Graphics_light_stage: 'integer',
  Graphics_main_display_index: 'integer',
  Graphics_mandatory_pit_done: 'boolean',
  Graphics_mfd_fuel_to_add: 'number',
  Graphics_mfd_tyre_pressure_front_left: 'number',
  Graphics_mfd_tyre_pressure_front_right: 'number',
  Graphics_mfd_tyre_pressure_rear_left: 'number',
  Graphics_mfd_tyre_pressure_rear_right: 'number',
  Graphics_mfd_tyre_set: 'integer',
  Graphics_missing_mandatory_pits: 'integer',
  Graphics_normalized_car_position: 'number',
  Graphics_number_of_laps: 'integer',
  Graphics_packed_id: 'integer',
  Graphics_penalty: 'integer',
  Graphics_penalty_time: 'number',
  Graphics_player_car_id: 'integer',
  Graphics_position: 'integer',
  Graphics_rain_intensity: 'integer',
  Graphics_rain_intensity_in_10min: 'integer',
  Graphics_rain_intensity_in_30min: 'integer',
  Graphics_rain_light: 'boolean',
  Graphics_rain_tyres: 'integer',
  Graphics_secondary_display_index: 'integer',
  Graphics_session_index: 'integer',
  Graphics_session_time_left: 'number',
  Graphics_session_type: 'integer',
  Graphics_status: 'integer',
  Graphics_strategy_tyre_set: 'integer',
  Graphics_tc_cut_level: 'integer',
  Graphics_tc_level: 'integer',
  Graphics_track_grip_status: 'integer',
  Graphics_track_status: 'string',
  Graphics_tyre_compound: 'string',
  Graphics_used_fuel: 'number',
  Graphics_wind_direction: 'number',
  Graphics_wind_speed: 'number',
  Graphics_wiper_stage: 'integer',

  // Static
  Static_ac_version: 'string',
  Static_aid_auto_clutch: 'boolean',
  Static_aid_fuel_rate: 'number',
  Static_aid_mechanical_damage: 'number',
  Static_aid_stability: 'number',
  Static_aid_tyre_rate: 'number',
  Static_car_model: 'string',
  Static_dry_tyres_name: 'string',
  Static_is_online: 'boolean',
  Static_max_fuel: 'number',
  Static_max_rpm: 'integer',
  Static_num_cars: 'integer',
  Static_number_of_session: 'integer',
  Static_penalty_enabled: 'boolean',
  Static_pit_window_end: 'integer',
  Static_pit_window_start: 'integer',
  Static_player_name: 'string',
  Static_player_nick: 'string',
  Static_player_surname: 'string',
  Static_sector_count: 'integer',
  Static_sm_version: 'string',
  Static_track: 'string',
  Static_wet_tyres_name: 'string',
}));

const LIVE_TELEMETRY_FIELDS = Object.freeze(Object.keys(LIVE_TELEMETRY_DATASET));

function isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function validateCoordinates(value) {
  return Array.isArray(value)
    && value.length === 60
    && value.every((point) => isPlainObject(point)
      && Object.keys(point).length === 3
      && Object.prototype.hasOwnProperty.call(point, 'x')
      && Object.prototype.hasOwnProperty.call(point, 'y')
      && Object.prototype.hasOwnProperty.call(point, 'z')
      && isFiniteNumber(point.x)
      && isFiniteNumber(point.y)
      && isFiniteNumber(point.z));
}

function validateLiveTelemetryField(field, value) {
  switch (LIVE_TELEMETRY_DATASET[field]) {
    case 'boolean': return typeof value === 'boolean';
    case 'integer': return Number.isSafeInteger(value);
    case 'number': return isFiniteNumber(value);
    case 'string': return typeof value === 'string';
    case 'coordinates': return validateCoordinates(value);
    case 'integer-array': return Array.isArray(value)
      && value.length === 60
      && value.every(Number.isSafeInteger);
    default: return false;
  }
}

function validateLiveTelemetryRow(sample) {
  if (!isPlainObject(sample)) {
    return { ok: false, error: 'Live telemetry row must be a flat object.' };
  }
  const keys = Object.keys(sample);
  if (keys.length === 0) {
    return { ok: false, error: 'Live telemetry row must contain at least one dataset field.' };
  }
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(LIVE_TELEMETRY_DATASET, key)) {
      return { ok: false, error: `Unknown live telemetry dataset field: ${key}` };
    }
    if (!validateLiveTelemetryField(key, sample[key])) {
      return { ok: false, error: `Invalid value for live telemetry dataset field: ${key}` };
    }
  }
  return { ok: true, value: sample };
}

function assertLiveTelemetryRow(sample) {
  const result = validateLiveTelemetryRow(sample);
  if (!result.ok) throw new TypeError(result.error);
  return sample;
}

module.exports = {
  LIVE_TELEMETRY_DATASET,
  LIVE_TELEMETRY_FIELDS,
  assertLiveTelemetryRow,
  isPlainObject,
  validateLiveTelemetryField,
  validateLiveTelemetryRow,
};
