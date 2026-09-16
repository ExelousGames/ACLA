'use strict';

const { IRacingAdapter, IRACING_VARIABLES, IRACING_FIELD_COVERAGE } = require('./iracing-adapter');
const { createPositionReference, playerPosition } = require('./iracing-ibt-position');
const { LIVE_TELEMETRY_FIELDS, assertLiveTelemetryRow, isPlainObject, validateLiveTelemetryField } = require('../../../../src/data/live-telemetry-dataset');

const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const nonnegative = (value) => finite(value) && value >= 0;
const rules = {};
const rule = (field, source, unit, valid = finite) => { rules[field] = { source, unit, valid }; };

// Disk measurements have their own contract; never request these through the live reader.
// SDK units are checked against the IBT variable descriptors before mapping a value.
// https://support.iracing.com/support/solutions/articles/31000170128-2023-season-3-release-notes-2023-06-05-02-
rule('Physics_abs_cut', 'BrakeABSCutPct', '%', (value) => nonnegative(value) && value <= 1);
rule('Physics_oil_temp', 'OilTemp', 'C');
rule('Physics_oil_pressure', 'OilPress', 'bar', nonnegative);
rule('Physics_oil_level', 'OilLevel', 'l', nonnegative);
rule('Physics_fuel_pressure', 'FuelPress', 'bar', nonnegative);
rule('Physics_manifold_pressure', 'ManifoldPress', 'bar', nonnegative);
for (const [corner, sdk] of Object.entries({ front_left: 'LF', front_right: 'RF', rear_left: 'LR', rear_right: 'RR' })) {
  rule(`Physics_brake_pressure_${corner}`, `${sdk}brakeLinePress`, 'bar', nonnegative);
  rule(`Physics_suspension_velocity_${corner}`, `${sdk}shockVel`, 'm/s');
  rule(`Physics_ride_height_${corner}`, `${sdk}rideHeight`, 'm');
  // Linear wheel speed cannot populate the rad/s wheel_angular_s fields without a rolling radius.
  rule(`Physics_wheel_speed_${corner}`, `${sdk}speed`, 'm/s');
  const left = sdk.startsWith('L');
  for (const [position, suffix] of [['inner', left ? 'R' : 'L'], ['middle', 'M'], ['outer', left ? 'L' : 'R']]) {
    rule(`Physics_tyre_surface_temp_${corner}_${position}`, `${sdk}temp${suffix}`, 'C');
  }
  // *tempC* and *wear* are pit measurements, not continuous core temperature or tire wear.
}

const IRACING_IBT_VARIABLES = Object.freeze([...new Set([
  ...IRACING_VARIABLES, 'SessionTick', 'Lat', 'Lon', 'Alt', ...Object.values(rules).map(({ source }) => source),
])]);

const derived = {
  Graphics_car_coordinates: 'Lat/Lon/Alt -> WGS84 track-referenced meters (east/up/north); player only',
  Graphics_car_id: 'PlayerCarIdx or DriverInfo.DriverCarIdx in coordinate slot 0; unused slots -1',
};

const IRACING_IBT_FIELD_COVERAGE = Object.freeze(Object.fromEntries(LIVE_TELEMETRY_FIELDS.map((field) => [
  field, rules[field] ? Object.freeze({ supported: true, source: rules[field].source, unit: rules[field].unit })
    : derived[field] ? Object.freeze({ supported: true, source: derived[field] })
    : IRACING_FIELD_COVERAGE[field].supported ? IRACING_FIELD_COVERAGE[field]
      : Object.freeze({ supported: false, reason: 'No verified continuous IBT measurement with the standard units and semantics.' }),
])));

class IRacingIBTAdapter extends IRacingAdapter {
  constructor(variables) {
    super();
    this.variableUnits = new Map(variables.map(({ name, unit }) => [name, unit]));
  }

  reset() {
    super.reset();
    this.positionReference = undefined;
  }

  updateSession(text) {
    super.updateSession(text);
    this.positionReference = createPositionReference(this.session.WeekendInfo || {});
  }

  adapt(packet) {
    // Finalized disk telemetry is captured in the cockpit. Preserve explicit flags,
    // but do not require the live-only flags to exist in older/car-specific files.
    const recordedPacket = isPlainObject(packet) && isPlainObject(packet.values)
      ? { ...packet, values: { IsOnTrack: true, IsReplayPlaying: false, ...packet.values } } : packet;
    const sample = super.adapt(recordedPacket);
    if (sample.Graphics_status !== 2) return sample;
    for (const [field, { source, unit, valid }] of Object.entries(rules)) {
      const value = recordedPacket.values[source];
      if (this.variableUnits.get(source) === unit && valid(value) && validateLiveTelemetryField(field, value)) {
        sample[field] = value;
      }
    }
    const position = playerPosition(recordedPacket.values, this.variableUnits, this.positionReference);
    if (position && Number.isSafeInteger(this.playerIndex) && this.playerIndex >= 0) {
      // Coordinates are player-only. Compact into slot 0 so SDK car indices
      // 60-63 also fit the standard's 60-slot arrays without changing identity.
      sample.Graphics_car_coordinates = Array.from({ length: 60 }, (_, slot) => (
        slot === 0 ? position : { x: 0, y: 0, z: 0 }
      ));
      sample.Graphics_car_id = Array.from({ length: 60 }, (_, slot) => slot === 0 ? this.playerIndex : -1);
      sample.Graphics_player_car_id = this.playerIndex;
    }
    return assertLiveTelemetryRow(sample);
  }
}

module.exports = { IRacingIBTAdapter, IRACING_IBT_VARIABLES, IRACING_IBT_FIELD_COVERAGE };
