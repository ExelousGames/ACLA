'use strict';

const { DESKTOP_GAME_SET } = require('./recording-protocol');

const BOOLEAN_FIELDS = `
Physics_pit_limiter_on Physics_autoshifter_on Physics_is_ai_controlled Physics_ignition_on Physics_starter_engine_on Physics_is_engine_running
Graphics_is_in_pit Graphics_ideal_line_on Graphics_is_in_pit_lane Graphics_mandatory_pit_done Graphics_is_setup_menu_visible Graphics_rain_light Graphics_flashing_light Graphics_is_delta_positive Graphics_is_valid_lap Graphics_direction_light_left Graphics_direction_light_right Graphics_global_yellow Graphics_global_yellow_s1 Graphics_global_yellow_s2 Graphics_global_yellow_s3 Graphics_global_white Graphics_global_green Graphics_global_chequered Graphics_global_red
Static_penalty_enabled Static_aid_auto_clutch Static_is_online
`.trim().split(/\s+/);

const INTEGER_FIELDS = `
Physics_packed_id Physics_gear Physics_rpm Physics_front_brake_compound Physics_rear_brake_compound
Graphics_packed_id Graphics_status Graphics_session_type Graphics_last_sector_time_str Graphics_completed_lap Graphics_position Graphics_current_time Graphics_last_time Graphics_best_time Graphics_current_sector_index Graphics_last_sector_time Graphics_number_of_laps Graphics_active_cars Graphics_player_car_id Graphics_flag Graphics_penalty Graphics_main_display_index Graphics_secondary_display_index Graphics_tc_level Graphics_tc_cut_level Graphics_engine_map Graphics_abs_level Graphics_light_stage Graphics_wiper_stage Graphics_driver_stint_total_time_left Graphics_driver_stint_time_left Graphics_rain_tyres Graphics_session_index Graphics_delta_lap_time Graphics_estimated_lap_time Graphics_missing_mandatory_pits Graphics_mfd_tyre_set Graphics_track_grip_status Graphics_rain_intensity Graphics_rain_intensity_in_10min Graphics_rain_intensity_in_30min Graphics_current_tyre_set Graphics_strategy_tyre_set Graphics_gap_ahead Graphics_gap_behind
Static_number_of_session Static_num_cars Static_sector_count Static_max_rpm Static_pit_window_start Static_pit_window_end
`.trim().split(/\s+/);

const NUMBER_FIELDS = `
Physics_gas Physics_brake Physics_fuel Physics_steer_angle Physics_speed_kmh Physics_velocity_x Physics_velocity_y Physics_velocity_z Physics_g_force_x Physics_g_force_y Physics_g_force_z Physics_wheel_slip_front_left Physics_wheel_slip_front_right Physics_wheel_slip_rear_left Physics_wheel_slip_rear_right Physics_wheel_pressure_front_left Physics_wheel_pressure_front_right Physics_wheel_pressure_rear_left Physics_wheel_pressure_rear_right Physics_wheel_angular_s_front_left Physics_wheel_angular_s_front_right Physics_wheel_angular_s_rear_left Physics_wheel_angular_s_rear_right Physics_tyre_core_temp_front_left Physics_tyre_core_temp_front_right Physics_tyre_core_temp_rear_left Physics_tyre_core_temp_rear_right Physics_suspension_travel_front_left Physics_suspension_travel_front_right Physics_suspension_travel_rear_left Physics_suspension_travel_rear_right Physics_tc Physics_heading Physics_pitch Physics_roll Physics_car_damage_front Physics_car_damage_rear Physics_car_damage_left Physics_car_damage_right Physics_car_damage_center Physics_abs Physics_turbo_boost Physics_air_temp Physics_road_temp Physics_local_angular_vel_x Physics_local_angular_vel_y Physics_local_angular_vel_z Physics_final_ff Physics_brake_temp_front_left Physics_brake_temp_front_right Physics_brake_temp_rear_left Physics_brake_temp_rear_right Physics_clutch Physics_tyre_contact_point_front_left_x Physics_tyre_contact_point_front_left_y Physics_tyre_contact_point_front_left_z Physics_tyre_contact_point_front_right_x Physics_tyre_contact_point_front_right_y Physics_tyre_contact_point_front_right_z Physics_tyre_contact_point_rear_left_x Physics_tyre_contact_point_rear_left_y Physics_tyre_contact_point_rear_left_z Physics_tyre_contact_point_rear_right_x Physics_tyre_contact_point_rear_right_y Physics_tyre_contact_point_rear_right_z Physics_tyre_contact_normal_front_left_x Physics_tyre_contact_normal_front_left_y Physics_tyre_contact_normal_front_left_z Physics_tyre_contact_normal_front_right_x Physics_tyre_contact_normal_front_right_y Physics_tyre_contact_normal_front_right_z Physics_tyre_contact_normal_rear_left_x Physics_tyre_contact_normal_rear_left_y Physics_tyre_contact_normal_rear_left_z Physics_tyre_contact_normal_rear_right_x Physics_tyre_contact_normal_rear_right_y Physics_tyre_contact_normal_rear_right_z Physics_tyre_contact_heading_front_left_x Physics_tyre_contact_heading_front_left_y Physics_tyre_contact_heading_front_left_z Physics_tyre_contact_heading_front_right_x Physics_tyre_contact_heading_front_right_y Physics_tyre_contact_heading_front_right_z Physics_tyre_contact_heading_rear_left_x Physics_tyre_contact_heading_rear_left_y Physics_tyre_contact_heading_rear_left_z Physics_tyre_contact_heading_rear_right_x Physics_tyre_contact_heading_rear_right_y Physics_tyre_contact_heading_rear_right_z Physics_brake_bias Physics_local_velocity_x Physics_local_velocity_y Physics_local_velocity_z Physics_slip_ratio_front_left Physics_slip_ratio_front_right Physics_slip_ratio_rear_left Physics_slip_ratio_rear_right Physics_slip_angle_front_left Physics_slip_angle_front_right Physics_slip_angle_rear_left Physics_slip_angle_rear_right Physics_suspension_damage_front_left Physics_suspension_damage_front_right Physics_suspension_damage_rear_left Physics_suspension_damage_rear_right Physics_water_temp Physics_brake_pressure_front_left Physics_brake_pressure_front_right Physics_brake_pressure_rear_left Physics_brake_pressure_rear_right Physics_pad_life_front_left Physics_pad_life_front_right Physics_pad_life_rear_left Physics_pad_life_rear_right Physics_disc_life_front_left Physics_disc_life_front_right Physics_disc_life_rear_left Physics_disc_life_rear_right Physics_kerb_vibration Physics_slip_vibration Physics_g_vibration Physics_abs_vibration
Graphics_session_time_left Graphics_distance_traveled Graphics_normalized_car_position Graphics_penalty_time Graphics_wind_speed Graphics_wind_direction Graphics_fuel_per_lap Graphics_exhaust_temp Graphics_used_fuel Graphics_fuel_estimated_laps Graphics_clock Graphics_mfd_fuel_to_add Graphics_mfd_tyre_pressure_front_left Graphics_mfd_tyre_pressure_front_right Graphics_mfd_tyre_pressure_rear_left Graphics_mfd_tyre_pressure_rear_right
Static_max_fuel Static_aid_fuel_rate Static_aid_tyre_rate Static_aid_mechanical_damage Static_aid_stability
`.trim().split(/\s+/);

const STRING_FIELDS = `
Graphics_current_time_str Graphics_last_time_str Graphics_best_time_str Graphics_tyre_compound Graphics_delta_lap_time_str Graphics_estimated_lap_time_str Graphics_track_status
Static_sm_version Static_ac_version Static_car_model Static_track Static_player_name Static_player_surname Static_player_nick Static_dry_tyres_name Static_wet_tyres_name
`.trim().split(/\s+/);

const FIELD_TYPES = Object.freeze({
  ...Object.fromEntries(BOOLEAN_FIELDS.map((field) => [field, 'boolean'])),
  ...Object.fromEntries(INTEGER_FIELDS.map((field) => [field, 'integer'])),
  ...Object.fromEntries(NUMBER_FIELDS.map((field) => [field, 'number'])),
  ...Object.fromEntries(STRING_FIELDS.map((field) => [field, 'string'])),
  Graphics_car_coordinates: 'coordinates',
  Graphics_car_id: 'integer-array',
});

const STANDARD_TELEMETRY_FIELDS = Object.freeze(Object.keys(FIELD_TYPES));
const STANDARD_TELEMETRY_FIELD_SET = new Set(STANDARD_TELEMETRY_FIELDS);

if (STANDARD_TELEMETRY_FIELDS.length !== 240) {
  throw new Error(`Standard telemetry catalog must contain 240 fields, found ${STANDARD_TELEMETRY_FIELDS.length}`);
}

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

function validateFieldValue(field, value) {
  switch (FIELD_TYPES[field]) {
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

function validateStandardTelemetrySample(sample) {
  if (!isPlainObject(sample)) {
    return { ok: false, error: 'Telemetry sample must be a flat object.' };
  }
  const keys = Object.keys(sample);
  if (keys.length === 0) {
    return { ok: false, error: 'Telemetry sample must contain at least one standard field.' };
  }
  for (const key of keys) {
    if (!STANDARD_TELEMETRY_FIELD_SET.has(key)) {
      return { ok: false, error: `Unknown standard telemetry field: ${key}` };
    }
    if (!validateFieldValue(key, sample[key])) {
      return { ok: false, error: `Invalid value for standard telemetry field: ${key}` };
    }
  }
  return { ok: true, value: sample };
}

function assertStandardTelemetrySample(sample) {
  const result = validateStandardTelemetrySample(sample);
  if (!result.ok) throw new TypeError(result.error);
  return sample;
}

function validateSourceFrame(frame, expectedGame) {
  if (!isPlainObject(frame) || !DESKTOP_GAME_SET.has(frame.game)) {
    return { ok: false, error: 'Source frame must contain a recognized game.' };
  }
  if (expectedGame !== undefined && frame.game !== expectedGame) {
    return { ok: false, error: `Source frame game ${frame.game} does not match ${expectedGame}.` };
  }
  const sampleResult = validateStandardTelemetrySample(frame.sample);
  if (!sampleResult.ok) return sampleResult;
  if (Object.keys(frame).some((key) => key !== 'game' && key !== 'sample')) {
    return { ok: false, error: 'Source frame contains unsupported transport fields.' };
  }
  return { ok: true, value: frame };
}

function assertSourceFrame(frame, expectedGame) {
  const result = validateSourceFrame(frame, expectedGame);
  if (!result.ok) throw new TypeError(result.error);
  return frame;
}

module.exports = {
  FIELD_TYPES,
  STANDARD_TELEMETRY_FIELDS,
  STANDARD_TELEMETRY_FIELD_SET,
  assertSourceFrame,
  assertStandardTelemetrySample,
  isPlainObject,
  validateFieldValue,
  validateSourceFrame,
  validateStandardTelemetrySample,
};
