'use strict';

const { load, JSON_SCHEMA } = require('js-yaml');
const { IRacingHistory, trackLength } = require('./iracing-history');
const {
  LIVE_TELEMETRY_FIELDS, assertLiveTelemetryRow, isPlainObject, validateLiveTelemetryField,
} = require('../../../../src/data/live-telemetry-dataset');

const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const positive = (value) => finite(value) && value >= 0 ? value : undefined;
const integer = (value) => finite(value) ? Math.round(value) : undefined;
const milliseconds = (value) => finite(value) && value >= 0 ? Math.round(value * 1000) : undefined;
const fraction = (value) => finite(value) && value >= 0 && value <= 1 ? value : undefined;
const rules = {};

function rule(target, source, convert = (value) => value) {
  rules[target] = { source, convert };
}

rule('Physics_gas', 'Throttle', fraction);
// Brake includes automatic brake hold; BrakeRaw follows the driver's pedal.
rule('Physics_brake', 'BrakeRaw', fraction);
rule('Physics_clutch', 'Clutch', fraction); // Both SDKs report clutch engagement (1 = engaged).
rule('Physics_fuel', 'FuelLevel', positive);
rule('Physics_speed_kmh', 'Speed', (value) => finite(value) ? value * 3.6 : undefined);
rule('Physics_gear', 'Gear', (value) => Number.isInteger(value) && value >= -1 ? value + 1 : undefined);
rule('Physics_rpm', 'RPM', (value) => integer(positive(value)));
rule('Physics_air_temp', 'AirTemp');
rule('Physics_road_temp', 'TrackTempCrew');
rule('Physics_water_temp', 'WaterTemp');
rule('Physics_brake_bias', 'dcBrakeBias', (value) => fraction(value / 100));
rule('Physics_abs', 'BrakeABSactive', (value) => typeof value === 'boolean' ? Number(value) : undefined);
rule('Physics_final_ff', 'SteeringWheelPctTorqueSign');
rule('Physics_pit_limiter_on', 'EngineWarnings', (value) => Number.isInteger(value) ? Boolean(value & 0x10) : undefined);
// Body axes: IRSDK forward/left/up -> standard right/up/forward.
// Angular velocity is an axial vector, so this reflection also reverses its sign.
for (const [axis, source, sign] of [['x', 'VelocityY', -1], ['y', 'VelocityZ', 1], ['z', 'VelocityX', 1]]) {
  rule(`Physics_local_velocity_${axis}`, source, (value) => finite(value) ? sign * value : undefined);
}
for (const [axis, source, sign] of [['x', 'PitchRate', 1], ['y', 'YawRate', -1], ['z', 'RollRate', -1]]) {
  rule(`Physics_local_angular_vel_${axis}`, source, (value) => finite(value) ? sign * value : undefined);
}
for (const [axis, source, sign] of [['x', 'LatAccel', -1], ['y', 'VertAccel', 1], ['z', 'LongAccel', 1]]) {
  rule(`Physics_g_force_${axis}`, source, (value) => finite(value) ? sign * value / 9.80665 : undefined);
}
rule('Physics_heading', 'Yaw', (value) => finite(value) ? Math.atan2(-Math.sin(value), Math.cos(value)) : undefined);
rule('Physics_pitch', 'Pitch');
rule('Physics_roll', 'Roll', (value) => finite(value) ? -value : undefined);
rule('Graphics_completed_lap', 'LapCompleted', positive);
rule('Graphics_position', 'PlayerCarPosition', (value) => value > 0 ? value : undefined);
rule('Graphics_player_car_id', 'PlayerCarIdx', positive);
rule('Graphics_normalized_car_position', 'LapDistPct', fraction);
rule('Graphics_session_index', 'SessionNum', positive);
rule('Graphics_session_time_left', 'SessionTimeRemain', (value) => value < 604800 ? positive(value) : undefined);
rule('Graphics_is_in_pit_lane', 'OnPitRoad');
rule('Graphics_is_in_pit', 'PlayerTrackSurface', (value) => Number.isInteger(value) ? value === 1 : undefined);
rule('Graphics_wind_speed', 'WindVel', positive);
rule('Graphics_wind_direction', 'WindDir');
rule('Graphics_clock', 'SessionTimeOfDay', positive);
rule('Graphics_tc_level', 'dcTractionControl', integer);
rule('Graphics_tc_cut_level', 'dcTractionControl2', integer);
rule('Graphics_abs_level', 'dcABS', integer);
rule('Graphics_mfd_fuel_to_add', 'PitSvFuel', positive);
for (const [standard, sdk] of Object.entries({ current: 'LapCurrentLapTime', last: 'LapLastLapTime', best: 'LapBestLapTime' })) {
  rule(`Graphics_${standard}_time`, sdk, milliseconds);
}
for (const [corner, sdk] of Object.entries({ front_left: 'LF', front_right: 'RF', rear_left: 'LR', rear_right: 'RR' })) {
  // Only hot pressure channels are equivalent. Never substitute cold pit measurements.
  rule(`Physics_wheel_pressure_${corner}`, `${sdk}pressure`, (value) => finite(value) && value >= 0 ? value / 6.894757293168 : undefined);
  rule(`Physics_suspension_travel_${corner}`, `${sdk}shockDefl`);
  rule(`Graphics_mfd_tyre_pressure_${corner}`, `PitSv${sdk}P`, (value) => finite(value) && value >= 0 ? value / 6.894757293168 : undefined);
}

const derived = {
  Physics_packed_id: 'SDK row tick',
  Graphics_packed_id: 'SDK row tick',
  Physics_steer_angle: 'SteeringWheelAngle / (SteeringWheelAngleMax / 2), normalized input',
  Physics_velocity_x: 'VelocityX/Y/Z rotated by Yaw/Pitch/Roll -> track-frame -Y, m/s',
  Physics_velocity_y: 'VelocityX/Y/Z rotated by Yaw/Pitch/Roll -> track-frame Z (up), m/s',
  Physics_velocity_z: 'VelocityX/Y/Z rotated by Yaw/Pitch/Roll -> track-frame X, m/s',
  Graphics_status: 'IsOnTrack / IsReplayPlaying -> standard status enum',
  Graphics_current_time_str: 'LapCurrentLapTime -> lap-time string',
  Graphics_last_time_str: 'LapLastLapTime -> lap-time string',
  Graphics_best_time_str: 'LapBestLapTime -> lap-time string',
  Graphics_delta_lap_time: 'LapDeltaToSessionBestLap, gated by LapDeltaToSessionBestLap_OK -> ms',
  Graphics_delta_lap_time_str: 'Validated session-best delta -> signed lap-time string',
  Graphics_is_delta_positive: 'Validated session-best delta sign',
  Graphics_estimated_lap_time: 'LapBestLapTime + validated session-best delta -> ms',
  Graphics_estimated_lap_time_str: 'Estimated lap time -> lap-time string',
  Graphics_active_cars: 'Count CarIdxTrackSurface entries in the world',
  Graphics_number_of_laps: 'LapCompleted after checkered/cooldown only',
  Graphics_current_sector_index: 'LapDistPct against SplitTimeInfo.Sectors',
  Graphics_session_type: 'SessionInfo.Sessions[SessionNum].SessionType -> standard enum',
  Graphics_flag: 'SessionFlags -> standard flag enum (not the SDK bitmask)',
  Graphics_global_yellow: 'SessionFlags yellow/waving/caution bits',
  Graphics_global_white: 'SessionFlags white bit',
  Graphics_global_green: 'SessionFlags green/held bits',
  Graphics_global_chequered: 'SessionFlags checkered bit',
  Graphics_global_red: 'SessionFlags red bit',
  Graphics_tyre_compound: 'PlayerTireCompound resolved through DriverInfo.DriverTires',
  Graphics_rain_tyres: 'Fitted TireCompoundType -> wet/dry integer, when known',
  Graphics_distance_traveled: 'Speed integrated over continuous samples since an observed pit start/refuel, meters',
  Graphics_used_fuel: 'FuelLevel consumption since an observed pit start/refuel, liters',
  Graphics_fuel_per_lap: 'Mean fuel consumption over up to five fully observed non-pit laps, liters',
  Graphics_fuel_estimated_laps: 'FuelLevel / observed mean fuel consumption per lap',
  Graphics_last_sector_time: 'Interpolated SessionTime at consecutive sector crossings -> ms',
  Graphics_last_sector_time_str: 'Same integer milliseconds as Graphics_last_sector_time (catalog type)',
  Graphics_gap_ahead: 'Time since nearest car ahead passed the player current lap position -> ms',
  Graphics_gap_behind: 'Time since player passed nearest car behind current lap position -> ms',
  Static_track: 'WeekendInfo.TrackName + TrackConfigName',
  Static_car_model: 'DriverInfo.Drivers[DriverCarIdx].CarScreenName',
  Static_player_name: 'DriverInfo.Drivers[DriverCarIdx].UserName (unsplit display name)',
  Static_player_nick: 'DriverInfo.Drivers[DriverCarIdx].AbbrevName',
  Static_max_rpm: 'DriverInfo.DriverCarRedLine',
  Static_max_fuel: 'DriverInfo.DriverCarFuelMaxLtr',
  Static_number_of_session: 'SessionInfo.Sessions.length',
  Static_num_cars: 'DriverInfo.Drivers excluding spectators and pace cars',
  Static_sector_count: 'SplitTimeInfo.Sectors.length',
};

// Account for the entire catalog without inventing values for absent SDK channels.
const IRACING_FIELD_COVERAGE = Object.freeze(Object.fromEntries(LIVE_TELEMETRY_FIELDS.map((field) => [
  field,
  Object.freeze(rules[field]
    ? { supported: true, source: rules[field].source }
    : derived[field] ? { supported: true, source: derived[field] }
      : { supported: false, reason: 'No verified equivalent in live IRSDK with the standard units and semantics.' }),
])));

const IRACING_VARIABLES = Object.freeze([...new Set([
  ...Object.values(rules).map(({ source }) => source),
  'IsOnTrack', 'IsReplayPlaying', 'SteeringWheelAngle', 'SteeringWheelAngleMax',
  'LapDeltaToSessionBestLap', 'LapDeltaToSessionBestLap_OK', 'SessionState',
  'SessionFlags', 'CarIdxTrackSurface',
  'SessionTime', 'SessionUniqueID', 'PlayerTireCompound', 'IsInGarage', 'PlayerCarTowTime',
  'CarIdxLapDistPct', 'CarIdxOnPitRoad',
])]);

function lapTime(ms, signed = false) {
  const absolute = Math.abs(ms);
  return `${ms < 0 ? '-' : signed ? '+' : ''}${Math.floor(absolute / 60000)}:${String(Math.floor(absolute / 1000) % 60).padStart(2, '0')}.${String(absolute % 1000).padStart(3, '0')}`;
}

function sessionType(value) {
  return ({ Practice: 0, 'Open Qualify': 1, 'Lone Qualify': 1, Qualify: 1, Race: 2, Warmup: 0, Testing: 0 })[value];
}

class IRacingAdapter {
  constructor() { this.reset(); }

  reset() {
    this.session = {};
    this.staticFields = {};
    this.playerIndex = undefined;
    this.history = new IRacingHistory();
    this.historyKey = undefined;
  }

  updateSession(text) {
    if (typeof text !== 'string' || Buffer.byteLength(text, 'utf8') > 4 * 1024 * 1024) {
      throw new TypeError('Invalid iRacing session metadata.');
    }
    const parsed = load(text, { schema: JSON_SCHEMA });
    if (parsed !== undefined && parsed !== null && !isPlainObject(parsed)) {
      throw new TypeError('iRacing session metadata must be a YAML mapping.');
    }
    this.session = parsed || {};
    this.staticFields = {};
    this.playerIndex = undefined;
  }

  buildStaticFields(playerIndex) {
    const sample = {};
    const put = (key, value) => { if (validateLiveTelemetryField(key, value)) sample[key] = value; };
    const weekend = this.session.WeekendInfo || {};
    const info = this.session.DriverInfo || {};
    const drivers = Array.isArray(info.Drivers) ? info.Drivers : [];
    const player = drivers.find((driver) => driver.CarIdx === playerIndex);
    if (typeof weekend.TrackName === 'string' && weekend.TrackName) {
      const config = weekend.TrackConfigName;
      put('Static_track', `${weekend.TrackName}${typeof config === 'string' && config && config !== 'default' ? ` - ${config}` : ''}`);
    }
    put('Static_car_model', player?.CarScreenName);
    put('Static_player_name', player?.UserName);
    put('Static_player_nick', player?.AbbrevName);
    put('Static_max_rpm', integer(positive(info.DriverCarRedLine)));
    if (info.DriverCarIsElectric !== true && info.DriverCarIsElectric !== 1) put('Static_max_fuel', positive(info.DriverCarFuelMaxLtr));
    if (Array.isArray(info.Drivers)) put('Static_num_cars', drivers.filter((driver) => !driver.CarIsPaceCar && !driver.IsSpectator).length);
    if (Array.isArray(this.session.SessionInfo?.Sessions)) put('Static_number_of_session', this.session.SessionInfo.Sessions.length);
    if (Array.isArray(this.session.SplitTimeInfo?.Sectors)) put('Static_sector_count', this.session.SplitTimeInfo.Sectors.length);
    this.staticFields = sample;
    this.playerIndex = playerIndex;
  }

  // The boundary from raw simulator data to the live telemetry dataset.
  adapt(packet) {
    if (!isPlainObject(packet) || packet.type !== 'sample' || !Number.isInteger(packet.tick) || !isPlainObject(packet.values)) {
      throw new TypeError('Invalid iRacing SDK sample.');
    }
    if (Object.prototype.hasOwnProperty.call(packet, 'sessionInfo')) this.updateSession(packet.sessionInfo);
    const values = packet.values;
    const playerIndex = Number.isInteger(values.PlayerCarIdx) ? values.PlayerCarIdx : this.session.DriverInfo?.DriverCarIdx;
    if (this.playerIndex !== playerIndex || Object.keys(this.staticFields).length === 0) this.buildStaticFields(playerIndex);
    const sample = { ...this.staticFields, Physics_packed_id: packet.tick, Graphics_packed_id: packet.tick };
    const put = (key, value) => { if (validateLiveTelemetryField(key, value)) sample[key] = value; };
    const driving = values.IsOnTrack === true && values.IsReplayPlaying !== true;
    put('Graphics_status', values.IsReplayPlaying === true ? 1 : driving ? 2 : 0);
    const sessions = this.session.SessionInfo?.Sessions;
    const session = Array.isArray(sessions) ? sessions.find((item) => item.SessionNum === values.SessionNum) : undefined;
    put('Graphics_session_type', sessionType(session?.SessionType));
    // Outside the cockpit, SDK player channels can be stale or reflect a replay.
    if (!driving) {
      this.history.reset();
      this.historyKey = undefined;
      return assertLiveTelemetryRow(sample);
    }

    const info = this.session.DriverInfo || {};
    const fuelInLiters = info.DriverCarIsElectric !== true && info.DriverCarIsElectric !== 1;

    for (const [field, { source, convert }] of Object.entries(rules)) {
      if (!fuelInLiters && (field === 'Physics_fuel' || field === 'Graphics_mfd_fuel_to_add')) continue;
      if (Object.prototype.hasOwnProperty.call(values, source)) put(field, convert(values[source]));
    }
    const { VelocityX: vx, VelocityY: vy, VelocityZ: vz, Yaw: yaw, Pitch: pitch, Roll: roll } = values;
    if ([vx, vy, vz, yaw, pitch, roll].every(finite)) {
      // Body -> track: Rz(yaw) * Ry(pitch) * Rx(roll) in native IRSDK axes.
      // Apply roll, then pitch, then yaw to the vector, then map world axes
      // to standard (-Y, Z, X), matching the existing local/orientation basis.
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const cr = Math.cos(roll), sr = Math.sin(roll);
      const rolledY = cr * vy - sr * vz;
      const rolledZ = sr * vy + cr * vz;
      const pitchedX = cp * vx + sp * rolledZ;
      const pitchedZ = -sp * vx + cp * rolledZ;
      put('Physics_velocity_x', -(sy * pitchedX + cy * rolledY));
      put('Physics_velocity_y', pitchedZ);
      put('Physics_velocity_z', cy * pitchedX - sy * rolledY);
    }
    if (finite(values.SteeringWheelAngle) && finite(values.SteeringWheelAngleMax) && values.SteeringWheelAngleMax > 0) {
      put('Physics_steer_angle', Math.max(-1, Math.min(1, 2 * values.SteeringWheelAngle / values.SteeringWheelAngleMax)));
    }
    for (const name of ['current', 'last', 'best']) {
      const ms = sample[`Graphics_${name}_time`];
      if (ms !== undefined) put(`Graphics_${name}_time_str`, lapTime(ms));
    }
    if (values.LapDeltaToSessionBestLap_OK === true && finite(values.LapDeltaToSessionBestLap)) {
      const delta = Math.round(values.LapDeltaToSessionBestLap * 1000);
      put('Graphics_delta_lap_time', delta);
      put('Graphics_delta_lap_time_str', lapTime(delta, true));
      put('Graphics_is_delta_positive', delta > 0);
      if (sample.Graphics_best_time > 0 && sample.Graphics_best_time + delta >= 0) {
        const estimate = sample.Graphics_best_time + delta;
        put('Graphics_estimated_lap_time', estimate);
        put('Graphics_estimated_lap_time_str', lapTime(estimate));
      }
    }
    if (Array.isArray(values.CarIdxTrackSurface)) {
      put('Graphics_active_cars', values.CarIdxTrackSurface.filter((surface) => Number.isInteger(surface) && surface >= 0).length);
    }
    if (values.SessionState === 5 || values.SessionState === 6) put('Graphics_number_of_laps', positive(values.LapCompleted));
    const sectors = this.session.SplitTimeInfo?.Sectors;
    if (Array.isArray(sectors) && fraction(values.LapDistPct) !== undefined) {
      const sector = sectors.filter((item) => finite(item.SectorStartPct) && item.SectorStartPct <= values.LapDistPct)
        .sort((a, b) => b.SectorStartPct - a.SectorStartPct)[0];
      put('Graphics_current_sector_index', sector?.SectorNum);
    }
    const tires = Array.isArray(info.DriverTires) ? info.DriverTires : [];
    const tire = Number.isInteger(values.PlayerTireCompound) && values.PlayerTireCompound >= 0
      ? tires.find((item) => item?.TireIndex === values.PlayerTireCompound) : undefined;
    if (typeof tire?.TireCompoundType === 'string' && tire.TireCompoundType.trim()) {
      const compound = tire.TireCompoundType.trim();
      put('Graphics_tyre_compound', compound);
      if (/^(wet|rain)$/i.test(compound)) put('Graphics_rain_tyres', 1);
      else if (/^(dry|slick|soft|medium|hard|qualifying)$/i.test(compound)) put('Graphics_rain_tyres', 0);
    }
    const drivers = Array.isArray(info.Drivers) ? info.Drivers.filter(isPlainObject) : [];
    const player = drivers.find((driver) => driver.CarIdx === playerIndex);
    const weekend = this.session.WeekendInfo || {};
    const validSectors = Array.isArray(sectors) && sectors.length
      && sectors.every((sector, index) => isPlainObject(sector) && sector.SectorNum === index
        && finite(sector.SectorStartPct) && sector.SectorStartPct >= 0 && sector.SectorStartPct < 1
        && (index === 0 ? sector.SectorStartPct === 0 : sector.SectorStartPct > sectors[index - 1].SectorStartPct))
      ? sectors : [];
    // Routine YAML updates must not discard an otherwise continuous lap.
    const historyKey = JSON.stringify([weekend.SessionID, weekend.SubSessionID, weekend.TrackName,
      weekend.TrackConfigName, weekend.TrackLength, values.SessionNum, values.SessionUniqueID,
      playerIndex, player?.UserID, player?.CarID, fuelInLiters, validSectors]);
    if (historyKey !== this.historyKey) {
      this.history.reset();
      this.historyKey = historyKey;
    }
    this.history.update(values, { length: trackLength(weekend.TrackLength), sectors: validSectors,
      drivers, playerIndex, fuelInLiters: info.DriverCarIsElectric === false || info.DriverCarIsElectric === 0 }, put);
    if (Number.isInteger(values.SessionFlags)) {
      const flags = values.SessionFlags;
      put('Graphics_global_yellow', Boolean(flags & 0xC108));
      put('Graphics_global_white', Boolean(flags & 0x02));
      put('Graphics_global_green', Boolean(flags & 0x404));
      put('Graphics_global_chequered', Boolean(flags & 0x01));
      put('Graphics_global_red', Boolean(flags & 0x10));
      const flag = flags & 0x30000 ? 3 : flags & 0x01 ? 5 : flags & 0xC108 ? 2
        : flags & 0x20 ? 1 : flags & 0x02 ? 4 : flags & 0x404 ? 7 : 0;
      put('Graphics_flag', flag);
    }
    return assertLiveTelemetryRow(sample);
  }
}

module.exports = { IRacingAdapter, IRACING_FIELD_COVERAGE, IRACING_VARIABLES };
