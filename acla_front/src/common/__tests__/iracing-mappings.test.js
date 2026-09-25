/** @jest-environment node */
const { IRacingAdapter, IRACING_VARIABLES, IRACING_FIELD_COVERAGE } = require('../../../electron/recording/readers/iracing/iracing-adapter');
const { validateLiveTelemetryRow } = require('../../data/live-telemetry-dataset');
const { dump } = require('js-yaml');

const metadata = () => ({
  WeekendInfo: { SessionID: 10, SubSessionID: 20, TrackName: 'test', TrackLength: '1.0000 km' },
  DriverInfo: { DriverCarIdx: 4, DriverCarIsElectric: 0, DriverCarFuelMaxLtr: 100,
    DriverTires: [{ TireIndex: 7, TireCompoundType: 'Wet' }, { TireIndex: 1, TireCompoundType: 'Soft' }],
    Drivers: [
      { CarIdx: 4, UserID: 40, CarID: 1 }, { CarIdx: 5, UserID: 50, CarID: 1 },
      { CarIdx: 6, UserID: 60, CarID: 1 }, { CarIdx: 0, CarIsPaceCar: 1 },
    ],
  },
  SplitTimeInfo: { Sectors: [
    { SectorNum: 0, SectorStartPct: 0 }, { SectorNum: 1, SectorStartPct: 0.25 },
    { SectorNum: 2, SectorStartPct: 0.6 },
  ] },
});
const positionAt = (time, offset = 0) => ((0.2 + time / 10 + offset) % 1 + 1) % 1;
function packet(time, overrides = {}, session) {
  return { type: 'sample', tick: Math.round((time + 10) * 60), values: {
    IsOnTrack: true, IsReplayPlaying: false, PlayerCarIdx: 4, SessionNum: 1,
    SessionTime: time + 10, Speed: 100, LapDistPct: positionAt(time),
    LapCompleted: Math.floor(0.2 + time / 10), FuelLevel: 50 - time / 10,
    OnPitRoad: false, PlayerTrackSurface: 3, ...overrides,
  }, ...(session ? { sessionInfo: dump(session) } : {}) };
}
function drive(adapter, from, to, overrides = () => ({})) {
  let sample;
  for (let t = from; t <= to; t += 0.5) sample = adapter.adapt(packet(t, overrides(t)));
  return sample;
}

describe('additional iRacing mappings', () => {
  it('requests every new live source through the existing capture boundary', () => {
    expect(IRACING_VARIABLES).toEqual(expect.arrayContaining([
      'BrakeRaw', 'LatAccel', 'LongAccel', 'VertAccel', 'Yaw', 'Pitch', 'Roll', 'VelocityX', 'VelocityY', 'VelocityZ',
      'YawRate', 'PitchRate', 'RollRate', 'SessionTime', 'PlayerTireCompound',
      'CarIdxLapDistPct', 'CarIdxOnPitRoad', 'PlayerCarTowTime', 'IsInGarage',
    ]));
    expect(Object.values(IRACING_FIELD_COVERAGE).filter((entry) => entry.supported)).toHaveLength(101);
    for (const axis of ['x', 'y', 'z']) expect(IRACING_FIELD_COVERAGE[`Physics_velocity_${axis}`].supported).toBe(true);
  });

  it('converts polar and axial vectors separately and uses radians and g', () => {
    const row = new IRacingAdapter().adapt(packet(0, {
      VelocityX: 40, VelocityY: 2, VelocityZ: 3,
      PitchRate: 0.1, YawRate: 0.2, RollRate: 0.3,
      LatAccel: 9.80665, VertAccel: 19.6133, LongAccel: -9.80665,
      Yaw: Math.PI / 2, Pitch: 0.1, Roll: 0.2,
    }));
    expect(row).toMatchObject({
      Physics_local_velocity_x: -2, Physics_local_velocity_y: 3, Physics_local_velocity_z: 40,
      Physics_local_angular_vel_x: 0.1, Physics_local_angular_vel_y: -0.2, Physics_local_angular_vel_z: -0.3,
      Physics_g_force_x: -1, Physics_g_force_y: 2, Physics_g_force_z: -1,
      Physics_heading: -Math.PI / 2, Physics_pitch: 0.1, Physics_roll: -0.2,
    });
    expect(validateLiveTelemetryRow(row).ok).toBe(true);
  });

  it.each([
    ['zero orientation', 0, 0, 0, [-2, 3, 40]],
    ['positive quarter-turn yaw', Math.PI / 2, 0, 0, [-40, 3, -2]],
    ['negative quarter-turn yaw', -Math.PI / 2, 0, 0, [40, 3, 2]],
    ['half-turn yaw', Math.PI, 0, 0, [2, 3, -40]],
    ['wrapped yaw', 5 * Math.PI / 2, 0, 0, [-40, 3, -2]],
    ['positive pitch', 0, Math.PI / 2, 0, [-2, -40, 3]],
    ['negative pitch', 0, -Math.PI / 2, 0, [-2, 40, -3]],
    ['positive roll', 0, 0, Math.PI / 2, [3, 2, 40]],
    ['negative roll', 0, 0, -Math.PI / 2, [-3, -2, 40]],
    ['combined yaw/pitch/roll', Math.PI / 2, Math.PI / 6, Math.PI / 2,
      [-20 * Math.sqrt(3) - 1, Math.sqrt(3) - 20, 3]],
  ])('rotates the full velocity vector into world space: %s', (_name, Yaw, Pitch, Roll, expected) => {
    const row = new IRacingAdapter().adapt(packet(0, { VelocityX: 40, VelocityY: 2, VelocityZ: 3, Yaw, Pitch, Roll }));
    const actual = ['x', 'y', 'z'].map((axis) => row[`Physics_velocity_${axis}`]);
    actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 10));
    expect(Math.hypot(...actual)).toBeCloseTo(Math.hypot(40, 2, 3), 10);
    expect(row).toMatchObject({ Physics_local_velocity_x: -2, Physics_local_velocity_y: 3, Physics_local_velocity_z: 40 });
    expect(validateLiveTelemetryRow(row).ok).toBe(true);
  });

  it('emits zero world velocity for a stationary car at any orientation', () => {
    const row = new IRacingAdapter().adapt(packet(0, {
      VelocityX: 0, VelocityY: 0, VelocityZ: 0, Yaw: 1.2, Pitch: -0.4, Roll: 0.7,
    }));
    for (const axis of ['x', 'y', 'z']) expect(row[`Physics_velocity_${axis}`]).toBeCloseTo(0, 10);
  });

  it.each(['VelocityX', 'VelocityY', 'VelocityZ', 'Yaw', 'Pitch', 'Roll'])(
    'omits the whole world vector when %s is missing or invalid, without retaining the previous vector', (source) => {
      const adapter = new IRacingAdapter();
      const values = { VelocityX: 40, VelocityY: 2, VelocityZ: 3, Yaw: 1, Pitch: 0.1, Roll: 0.2 };
      expect(adapter.adapt(packet(0, values))).toHaveProperty('Physics_velocity_x');
      for (const invalid of [undefined, null, NaN, Infinity, -Infinity, '0', false]) {
        const row = adapter.adapt(packet(0.5, { ...values, [source]: invalid }));
        for (const axis of ['x', 'y', 'z']) expect(row).not.toHaveProperty(`Physics_velocity_${axis}`);
        expect(validateLiveTelemetryRow(row).ok).toBe(true);
      }
      const missing = { ...values };
      delete missing[source];
      const row = adapter.adapt(packet(1, missing));
      for (const axis of ['x', 'y', 'z']) expect(row).not.toHaveProperty(`Physics_velocity_${axis}`);
      expect(adapter.adapt(packet(1.5, values))).toHaveProperty('Physics_velocity_x');
    },
  );

  it.each([{ IsOnTrack: false }, { IsReplayPlaying: true }])('omits world velocity outside live driving: %p', (status) => {
    const adapter = new IRacingAdapter();
    const values = { VelocityX: 40, VelocityY: 2, VelocityZ: 3, Yaw: 1, Pitch: 0.1, Roll: 0.2 };
    expect(adapter.adapt(packet(0, values))).toHaveProperty('Physics_velocity_x');
    const row = adapter.adapt(packet(0.5, { ...values, ...status }));
    for (const axis of ['x', 'y', 'z']) expect(row).not.toHaveProperty(`Physics_velocity_${axis}`);
  });

  it('does not fabricate absent vectors or use nonfinite values', () => {
    const row = new IRacingAdapter().adapt(packet(0, { VelocityX: 0, VelocityY: NaN, Yaw: Infinity, PitchRate: '1' }));
    expect(row.Physics_local_velocity_z).toBe(0);
    for (const name of ['local_velocity_x', 'local_velocity_y', 'heading', 'local_angular_vel_x', 'g_force_x']) {
      expect(row).not.toHaveProperty(`Physics_${name}`);
    }
  });

  it('resolves fitted tires by metadata index, not pit selection or a hard-coded wet index', () => {
    const adapter = new IRacingAdapter();
    expect(adapter.adapt(packet(0, { PlayerTireCompound: 7, PitSvTireCompound: 1 }, metadata())))
      .toMatchObject({ Graphics_tyre_compound: 'Wet', Graphics_rain_tyres: 1 });
    expect(adapter.adapt(packet(0.5, { PlayerTireCompound: 1, WeatherDeclaredWet: true })))
      .toMatchObject({ Graphics_tyre_compound: 'Soft', Graphics_rain_tyres: 0 });
    expect(adapter.adapt(packet(1, { PlayerTireCompound: 2 }))).not.toHaveProperty('Graphics_tyre_compound');
    const session = metadata();
    session.DriverInfo.DriverTires[0].TireCompoundType = 'All-Purpose';
    const row = adapter.adapt(packet(1.5, { PlayerTireCompound: 7 }, session));
    expect(row.Graphics_tyre_compound).toBe('All-Purpose');
    expect(row).not.toHaveProperty('Graphics_rain_tyres');
    expect(new IRacingAdapter().adapt(packet(0, { PlayerTireCompound: 7 }))).not.toHaveProperty('Graphics_rain_tyres');
  });

  it('waits for a complete sector and interpolates boundary times, including the finish line', () => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, {}, metadata()));
    expect(drive(adapter, 0.5, 3.5)).not.toHaveProperty('Graphics_last_sector_time');
    expect(drive(adapter, 4, 4.5)).toMatchObject({ Graphics_last_sector_time: 3500, Graphics_last_sector_time_str: 3500 });
    expect(drive(adapter, 5, 8.5).Graphics_last_sector_time).toBe(4000);
    expect(drive(adapter, 9, 11).Graphics_last_sector_time).toBe(2500);
  });

  it('waits for a full lap before estimating consumption and never invents a mid-stint total', () => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, {}, metadata()));
    const early = drive(adapter, 0.5, 17.5);
    expect(early).not.toHaveProperty('Graphics_fuel_per_lap');
    const row = drive(adapter, 18, 20);
    expect(row.Graphics_fuel_per_lap).toBeCloseTo(1);
    expect(row.Graphics_fuel_estimated_laps).toBeCloseTo(48);
    expect(row).not.toHaveProperty('Graphics_used_fuel');
    expect(row).not.toHaveProperty('Graphics_distance_traveled');
  });

  it('accumulates distance and consumption from an observed pit start and resets on refueling', () => {
    const adapter = new IRacingAdapter();
    expect(adapter.adapt(packet(0, { Speed: 0, OnPitRoad: true, PlayerTrackSurface: 1 }, metadata())))
      .toMatchObject({ Graphics_distance_traveled: 0, Graphics_used_fuel: 0 });
    const row = drive(adapter, 0.5, 2);
    expect(row.Graphics_distance_traveled).toBeCloseTo(175);
    expect(row.Graphics_used_fuel).toBeCloseTo(0.2);
    const refuel = adapter.adapt(packet(2.5, { FuelLevel: 60, OnPitRoad: true }));
    expect(refuel).toMatchObject({ Graphics_distance_traveled: 0, Graphics_used_fuel: 0 });
    expect(refuel).not.toHaveProperty('Graphics_fuel_per_lap');
    expect(adapter.adapt(packet(3, { FuelLevel: 59.9 })).Graphics_used_fuel).toBeCloseTo(0.1);
  });

  it.each([
    { PlayerCarTowTime: 10 }, { IsInGarage: true }, { IsOnTrack: false }, { IsReplayPlaying: true },
    { LapDistPct: 0.9 }, { LapCompleted: 0 }, { SessionNum: 2 }, { SessionTime: 1 },
    { SessionTime: 100 }, { LapDistPct: undefined }, { OnPitRoad: undefined }, { PlayerTrackSurface: undefined },
  ])('invalidates derived history after a discontinuity: %j', (override) => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, {}, metadata()));
    expect(drive(adapter, 0.5, 20).Graphics_fuel_per_lap).toBeCloseTo(1);
    const row = adapter.adapt(packet(20.5, override));
    expect(row).not.toHaveProperty('Graphics_fuel_per_lap');
    expect(row).not.toHaveProperty('Graphics_last_sector_time');
    expect(adapter.adapt(packet(21))).not.toHaveProperty('Graphics_fuel_per_lap');
  });

  it('keeps valid history through routine YAML updates but clears it on a driver change or reconnect', () => {
    const adapter = new IRacingAdapter();
    const session = metadata();
    adapter.adapt(packet(0, {}, session));
    drive(adapter, 0.5, 17.5);
    session.WeekendInfo.TrackAirTemp = '25 C';
    expect(adapter.adapt(packet(18, {}, session)).Graphics_fuel_per_lap).toBeCloseTo(1);
    session.DriverInfo.Drivers[0].UserID = 99;
    expect(adapter.adapt(packet(18.5, {}, session))).not.toHaveProperty('Graphics_fuel_per_lap');
    adapter.reset();
    expect(adapter.adapt(packet(19))).not.toHaveProperty('Graphics_last_sector_time');
  });

  it('does not reuse a lap that included a pit visit, reversal, or unavailable fuel', () => {
    for (const override of [{ OnPitRoad: true }, { LapDistPct: 0.19 }, { FuelLevel: undefined }]) {
      const adapter = new IRacingAdapter();
      adapter.adapt(packet(0, {}, metadata()));
      drive(adapter, 0.5, 10);
      adapter.adapt(packet(10.5, override));
      expect(drive(adapter, 11, 18)).not.toHaveProperty('Graphics_fuel_per_lap');
      expect(drive(adapter, 18.5, 28).Graphics_fuel_per_lap).toBeCloseTo(1);
    }
  });

  it('does not label electric energy as liters', () => {
    const session = metadata();
    session.DriverInfo.DriverCarIsElectric = 1;
    const adapter = new IRacingAdapter();
    const first = adapter.adapt(packet(0, { PitSvFuel: 20, OnPitRoad: true, PlayerTrackSurface: 1, Speed: 0 }, session));
    const row = drive(adapter, 0.5, 20);
    for (const key of ['Physics_fuel', 'Static_max_fuel', 'Graphics_mfd_fuel_to_add', 'Graphics_used_fuel', 'Graphics_fuel_per_lap', 'Graphics_fuel_estimated_laps']) {
      expect(first).not.toHaveProperty(key);
      expect(row).not.toHaveProperty(key);
    }
    expect(row.Graphics_distance_traveled).toBeGreaterThan(0);
  });

  it('waits for fuel-type metadata before deriving liter-based consumption', () => {
    const session = metadata();
    delete session.DriverInfo.DriverCarIsElectric;
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, {}, session));
    expect(drive(adapter, 0.5, 20)).not.toHaveProperty('Graphics_fuel_per_lap');
  });

  it('rejects malformed track lengths and sector definitions without stopping capture', () => {
    const session = metadata();
    session.WeekendInfo.TrackLength = '1 mile';
    const adapter = new IRacingAdapter();
    expect(adapter.adapt(packet(0, {}, session)).Physics_speed_kmh).toBe(360);
    expect(drive(adapter, 0.5, 20)).not.toHaveProperty('Graphics_fuel_per_lap');
    session.WeekendInfo.TrackLength = '1000 m';
    session.SplitTimeInfo.Sectors[1].SectorStartPct = 0;
    adapter.adapt(packet(20.5, {}, session));
    const row = drive(adapter, 21, 40);
    expect(row).not.toHaveProperty('Graphics_last_sector_time');
    expect(row.Graphics_fuel_per_lap).toBeCloseTo(1);
  });

  function traffic(time, extra = {}) {
    const positions = Array(7).fill(-1);
    const surfaces = Array(7).fill(-1);
    const pits = Array(7).fill(false);
    for (const [index, offset] of [[4, 0], [5, 0.05], [6, -0.03], [0, 0.01]]) {
      positions[index] = positionAt(time, offset);
      surfaces[index] = 3;
    }
    return { CarIdxLapDistPct: positions, CarIdxTrackSurface: surfaces, CarIdxOnPitRoad: pits, ...extra };
  }

  it('measures relative gaps from crossing history, excluding the pace car and handling lap wrap', () => {
    const adapter = new IRacingAdapter();
    const first = adapter.adapt(packet(0, traffic(0), metadata()));
    expect(first).not.toHaveProperty('Graphics_gap_ahead');
    expect(first).not.toHaveProperty('Graphics_gap_behind');
    const row = drive(adapter, 0.5, 10, traffic);
    expect(row.Graphics_gap_ahead).toBe(500);
    expect(row.Graphics_gap_behind).toBe(300);
    // No history means no gap, even when a plausible SDK estimated time is present.
    const missing = adapter.adapt(packet(10.5, { CarIdxEstTime: [0, 0, 0, 0, 30, 20, 40] }));
    expect(missing).not.toHaveProperty('Graphics_gap_ahead');
    expect(missing).not.toHaveProperty('Graphics_gap_behind');
    expect(adapter.history.cars.size).toBe(0);
  });

  it('drops history when an opponent goes to the pits, disappears, reverses, or changes driver', () => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, traffic(0), metadata()));
    drive(adapter, 0.5, 3, traffic);
    const data = traffic(3.5);
    data.CarIdxOnPitRoad[5] = true;
    adapter.adapt(packet(3.5, data));
    expect(adapter.history.cars.has(5)).toBe(false);
    const returning = adapter.adapt(packet(4, traffic(4)));
    expect(returning).not.toHaveProperty('Graphics_gap_ahead');
    const changed = metadata();
    changed.DriverInfo.Drivers[1].UserID = 999;
    expect(adapter.adapt(packet(4.5, traffic(4.5), changed))).not.toHaveProperty('Graphics_gap_ahead');
  });

  it('bounds relative history during long sessions', () => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet(0, traffic(0), metadata()));
    for (let n = 1; n <= 13000; n += 1) adapter.adapt(packet(n / 60, traffic(n / 60)));
    expect(adapter.history.cars.size).toBe(3);
    for (const { history } of adapter.history.cars.values()) {
      expect(history.length).toBeLessThanOrEqual(1802);
      expect(history[history.length - 1].time - history[0].time).toBeLessThanOrEqual(180);
    }
  });
});
