/** @jest-environment node */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');
const { IBTFile, convertIBTFiles } = require('../../../electron/recording/readers/iracing/iracing-ibt');
const { IRacingIBTAdapter, IRACING_IBT_VARIABLES, IRACING_IBT_FIELD_COVERAGE } = require('../../../electron/recording/readers/iracing/iracing-ibt-adapter');
const { IRacingAdapter, IRACING_VARIABLES, IRACING_FIELD_COVERAGE } = require('../../../electron/recording/readers/iracing/iracing-adapter');
const { LIVE_TELEMETRY_FIELDS, validateLiveTelemetryRow } = require('../../data/live-telemetry-dataset');

function fixture({ track = 'spa', sessionId = 42, speeds = [50, 60], encoding = 'UTF8', channels = [],
  reference = '', playerIndex = 4 } = {}) {
  const session = Buffer.from(`WeekendInfo:\n Encoding: ${encoding}\n SessionID: ${sessionId}\n SubSessionID: 1\n TrackName: ${track}\n${reference}DriverInfo:\n DriverCarIdx: ${playerIndex}\n Drivers:\n - CarIdx: ${playerIndex}\n   CarScreenName: GT3\n   UserName: Zoë Driver\n\0`, encoding === 'UTF8' ? 'utf8' : 'latin1');
  const variables = [
    ['Speed', 4, 0, 1], ['Gear', 2, 4, 1], ['SessionTick', 2, 8, 1], ['SessionTime', 5, 12, 1],
    ['RPM', 4, 20, 3, true], ['BrakeRaw', 4, 32, 1], ['CarIdxTrackSurface', 2, 36, 3],
    ['SessionFlags', 3, 48, 1], ['OnPitRoad', 1, 52, 1], ['FuelLevel', 4, 56, 1],
  ];
  let rowLength = 64;
  for (const { name, unit, value, type = 4, countAsTime = false } of channels) {
    const count = Array.isArray(value) ? value.length : 1;
    variables.push([name, type, rowLength, count, countAsTime, unit]);
    rowLength += count * [1, 1, 4, 4, 4, 8][type];
  }
  const sessionOffset = 144 + variables.length * 144;
  const dataOffset = sessionOffset + session.length;
  const buffer = Buffer.alloc(dataOffset + speeds.length * rowLength);
  [2, 0, 60, 1, session.length, sessionOffset, variables.length, 144, 1, rowLength].forEach((value, i) => buffer.writeInt32LE(value, i * 4));
  buffer.writeInt32LE(100, 48);
  buffer.writeInt32LE(dataOffset, 52);
  buffer.writeInt32LE(speeds.length, 140);
  variables.forEach(([name, type, offset, count, asTime, unit], index) => {
    const start = 144 + index * 144;
    buffer.writeInt32LE(type, start); buffer.writeInt32LE(offset, start + 4); buffer.writeInt32LE(count, start + 8);
    buffer[start + 12] = asTime ? 1 : 0;
    buffer.write(name, start + 16, 'ascii');
    if (unit) buffer.write(unit, start + 112, 'ascii');
  });
  session.copy(buffer, sessionOffset);
  speeds.forEach((speed, index) => {
    const start = dataOffset + index * rowLength;
    buffer.writeFloatLE(speed, start); buffer.writeInt32LE(2, start + 4); buffer.writeInt32LE(100 + index, start + 8);
    buffer.writeDoubleLE(10 + index / 60, start + 12);
    [5000, 5500, 6000].forEach((value, i) => buffer.writeFloatLE(value, start + 20 + i * 4));
    buffer.writeFloatLE(0.25, start + 32);
    [3, -1, 1].forEach((value, i) => buffer.writeInt32LE(value, start + 36 + i * 4));
    buffer.writeUInt32LE(0x80000004, start + 48); buffer[start + 52] = 1;
    buffer.writeFloatLE(NaN, start + 56);
    channels.forEach(({ value }, channelIndex) => {
      const [, type, offset] = variables[10 + channelIndex];
      (Array.isArray(value) ? value : [value]).forEach((entry, i) => {
        if (type === 1) buffer[start + offset + i] = Number(entry);
        else if (type === 5) buffer.writeDoubleLE(entry, start + offset + i * 8);
        else buffer.writeFloatLE(entry, start + offset + i * 4);
      });
    });
  });
  return buffer;
}

const diskChannels = () => [
  { name: 'BrakeABSCutPct', unit: '%', value: 0.25 },
  { name: 'OilTemp', unit: 'C', value: 105 },
  { name: 'OilPress', unit: 'bar', value: 4.5 },
  { name: 'OilLevel', unit: 'l', value: 8 },
  { name: 'FuelPress', unit: 'bar', value: 7.5 },
  { name: 'ManifoldPress', unit: 'bar', value: 1.5 },
  ...['LF', 'RF', 'LR', 'RR'].flatMap((corner, i) => [
    { name: `${corner}brakeLinePress`, unit: 'bar', value: 40 + i },
    { name: `${corner}pressure`, unit: 'kPa', value: 200 + i },
    { name: `${corner}shockVel`, unit: 'm/s', value: -0.5 + i },
    { name: `${corner}rideHeight`, unit: 'm', value: 0.02 + i / 100 },
    { name: `${corner}speed`, unit: 'm/s', value: 50 + i },
    ...['L', 'M', 'R'].map((side, j) => ({ name: `${corner}temp${side}`, unit: 'C', value: 80 + i * 10 + j })),
  ]),
];

const positionChannels = (lat = 53.80939444, lon = 2.12955, alt = 73) => [
  { name: 'Lat', unit: 'deg', type: 5, value: lat },
  { name: 'Lon', unit: 'deg', type: 5, value: lon },
  { name: 'Alt', unit: 'm', value: alt },
];
const trackReference = (lat = '55 deg', lon = '5 deg', alt = '200 m') => (
  ` TrackLatitude: ${lat}\n TrackLongitude: ${lon}\n TrackAltitude: ${alt}\n`
);

describe('iRacing .ibt import', () => {
  let directory;
  let live;
  const writeFixture = (name, options) => {
    const filePath = path.join(directory, name);
    fs.writeFileSync(filePath, fixture(options));
    return filePath;
  };
  beforeEach(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-ibt-'));
    live = { sample: { Static_track: 'spa', Static_car_model: 'GT3', Static_player_name: 'Zoë Driver' },
      startedAt: Date.now() - 1000, endedAt: Date.now() + 1000 };
  });
  afterEach(() => fs.rmSync(directory, { recursive: true, force: true }));

  it.each(['UTF8', 'CP1252'])('converts native disk data and %s metadata into standard telemetry', async (encoding) => {
    const ibt = await IBTFile.open(writeFixture('session.ibt', { encoding }));
    try {
      const rows = [];
      for await (const row of ibt.rows()) rows.push(row);
      expect(rows).toHaveLength(2);
      expect(rows[0]).toMatchObject({ Physics_speed_kmh: 180, Physics_gear: 3, Physics_rpm: 6000,
        Physics_brake: 0.25, Physics_packed_id: 100, Graphics_status: 2, Graphics_active_cars: 2,
        Graphics_global_green: true, Graphics_is_in_pit_lane: true, Static_player_name: 'Zoë Driver' });
      expect(rows[1].Physics_speed_kmh).toBe(216);
      expect(rows[0]).not.toHaveProperty('Physics_fuel');
    } finally { await ibt.close(); }
  });

  it('imports disk measurements with corner orientation and physical units intact', async () => {
    const ibt = await IBTFile.open(writeFixture('rich.ibt', { channels: diskChannels() }));
    try {
      expect(ibt.adapter).toBeInstanceOf(IRacingIBTAdapter);
      const { value: row } = await ibt.rows().next();
      expect(row).toMatchObject({ Physics_abs_cut: 0.25, Physics_oil_temp: 105, Physics_oil_pressure: 4.5,
        Physics_oil_level: 8, Physics_fuel_pressure: 7.5, Physics_manifold_pressure: 1.5 });
      ['front_left', 'front_right', 'rear_left', 'rear_right'].forEach((corner, i) => {
        expect(row[`Physics_brake_pressure_${corner}`]).toBe(40 + i);
        expect(row[`Physics_wheel_pressure_${corner}`]).toBeCloseTo((200 + i) / 6.894757293168);
        expect(row[`Physics_suspension_velocity_${corner}`]).toBe(-0.5 + i);
        expect(row[`Physics_ride_height_${corner}`]).toBeCloseTo(0.02 + i / 100);
        expect(row[`Physics_wheel_speed_${corner}`]).toBe(50 + i);
        expect(row[`Physics_tyre_surface_temp_${corner}_inner`]).toBe(80 + i * 10 + (i % 2 === 0 ? 2 : 0));
        expect(row[`Physics_tyre_surface_temp_${corner}_middle`]).toBe(81 + i * 10);
        expect(row[`Physics_tyre_surface_temp_${corner}_outer`]).toBe(80 + i * 10 + (i % 2 === 0 ? 0 : 2));
        expect(row).not.toHaveProperty(`Physics_tyre_core_temp_${corner}`);
        expect(row).not.toHaveProperty(`Physics_wheel_angular_s_${corner}`);
      });
      expect(validateLiveTelemetryRow(row).ok).toBe(true);
      expect(row).not.toHaveProperty('LFtempL');
    } finally { await ibt.close(); }
  });

  it('omits missing, invalid and wrong-unit disk measurements without substituting pit values', async () => {
    const channels = [
      { name: 'LFbrakeLinePress', unit: 'kPa', value: 40 },
      { name: 'RFbrakeLinePress', unit: 'bar', value: -1 },
      { name: 'LRbrakeLinePress', unit: 'bar', value: NaN },
      { name: 'RRbrakeLinePress', unit: 'bar', value: 0 },
      { name: 'BrakeABSCutPct', unit: '%', value: 25 },
      { name: 'LFtempL', unit: 'C', value: [70, 80] }, // a spatial array is not one temperature
      { name: 'RFtempL', unit: 'C', value: Infinity },
      { name: 'OilTemp', unit: '', value: 100 },
      { name: 'LFcoldPressure', unit: 'kPa', value: 180 },
      { name: 'LFtempCM', unit: 'C', value: 90 },
      { name: 'LFwearM', unit: '%', value: 0.9 },
    ];
    const ibt = await IBTFile.open(writeFixture('partial.ibt', { channels }));
    try {
      const { value: row } = await ibt.rows().next();
      expect(row.Physics_brake_pressure_rear_right).toBe(0);
      for (const field of ['Physics_brake_pressure_front_left', 'Physics_brake_pressure_front_right',
        'Physics_brake_pressure_rear_left', 'Physics_abs_cut', 'Physics_tyre_surface_temp_front_left_outer',
        'Physics_tyre_surface_temp_front_right_inner', 'Physics_oil_temp', 'Physics_wheel_pressure_front_left',
        'Physics_tyre_core_temp_front_left']) expect(row).not.toHaveProperty(field);
    } finally { await ibt.close(); }
  });

  it('uses the latest time-array measurement without treating it as a per-wheel array', async () => {
    const ibt = await IBTFile.open(writeFixture('times.ibt', { channels: [
      { name: 'LFbrakeLinePress', unit: 'bar', value: [10, 20, 30], countAsTime: true },
    ] }));
    try {
      expect((await ibt.rows().next()).value.Physics_brake_pressure_front_left).toBe(30);
    } finally { await ibt.close(); }
  });

  it.each([0, 4, 63])('imports geographic player position into standard XYZ with car id %i', async (playerIndex) => {
    const ibt = await IBTFile.open(writeFixture('position.ibt', {
      channels: positionChannels(), reference: trackReference(), playerIndex,
    }));
    try {
      const { value: row } = await ibt.rows().next();
      // Independent published PROJ topocentric example (GRS80 differs from
      // WGS84 by less than a millimeter here), reordered from ENU to EUN.
      // https://proj.org/en/stable/operations/conversions/topocentric.html
      const point = row.Graphics_car_coordinates[0];
      expect(point.x).toBeCloseTo(-189013.869, 3);
      expect(point.y).toBeCloseTo(-4220.171, 3);
      expect(point.z).toBeCloseTo(-128642.040, 3);
      expect(row.Graphics_player_car_id).toBe(playerIndex);
      expect(row.Graphics_car_id).toEqual([playerIndex, ...Array(59).fill(-1)]);
      expect(row.Graphics_car_coordinates.slice(1)).toEqual(Array(59).fill({ x: 0, y: 0, z: 0 }));
      expect(validateLiveTelemetryRow(row).ok).toBe(true);
      for (const raw of ['Lat', 'Lon', 'Alt']) expect(row).not.toHaveProperty(raw);
    } finally { await ibt.close(); }
  });

  it.each([
    [0, 0, 0, 'y', 0],
    [0, 0, 15, 'y', 15],
    [0, 0.001, 0, 'x', 111.319490788],
    [0.001, 0, 0, 'z', 110.574275816],
    [0, -0.001, 0, 'x', -111.319490788],
    [-0.001, 0, 0, 'z', -110.574275816],
  ])('maps (%s, %s, %s) to the correct meter-based %s axis', async (lat, lon, alt, axis, expected) => {
    const ibt = await IBTFile.open(writeFixture('axes.ibt', {
      channels: positionChannels(lat, lon, alt), reference: trackReference('0 deg', '0 deg', '0 m'),
    }));
    try {
      const point = (await ibt.rows().next()).value.Graphics_car_coordinates[0];
      expect(point[axis]).toBeCloseTo(expected, 6);
      if (lat === 0 && lon === 0) expect(point).toEqual({ x: 0, y: alt, z: 0 });
    } finally { await ibt.close(); }
  });

  it.each(['55 m', '55 deg', '55'])('accepts the native reference format %s and preserves it across file boundaries', async (lat) => {
    const options = { reference: trackReference(lat, lat.endsWith(' m') ? '5 m' : '5 deg'),
      channels: positionChannels(), speeds: [50] };
    const first = writeFixture('a.ibt', { ...options, channels: positionChannels(55, 5, 200) });
    const second = writeFixture('b.ibt', options);
    const outputPath = path.join(directory, 'positions.jsonl');
    await convertIBTFiles([first, second], outputPath, live);
    const rows = fs.readFileSync(outputPath, 'utf8').trim().split('\n').map(JSON.parse);
    expect(rows[0].Graphics_car_coordinates[0]).toEqual({ x: 0, y: 0, z: 0 });
    expect(rows[1].Graphics_car_coordinates[0].x).toBeCloseTo(-189013.869, 3);
    expect(rows.every((row) => validateLiveTelemetryRow(row).ok)).toBe(true);
  });

  it('keeps longitude wrap continuous at the date line', async () => {
    const ibt = await IBTFile.open(writeFixture('wrap.ibt', { channels: positionChannels(0, -179.999, 0),
      reference: trackReference('0 deg', '179.999 deg', '0 m') }));
    try {
      expect((await ibt.rows().next()).value.Graphics_car_coordinates[0].x).toBeCloseTo(222.638981541, 5);
    } finally { await ibt.close(); }
  });

  it.each([
    { channels: positionChannels().slice(0, 2) },
    { channels: positionChannels(NaN) },
    { channels: positionChannels(91) },
    { channels: positionChannels(0, 181) },
    { channels: positionChannels(0, 0, Infinity) },
    { channels: positionChannels([50, 51]) },
    { channels: positionChannels().map((channel) => ({ ...channel, unit: channel.name === 'Lat' ? 'rad' : channel.unit })) },
    { channels: positionChannels().map((channel) => ({ ...channel, unit: channel.name === 'Lon' ? 'm' : channel.unit })) },
    { channels: positionChannels().map((channel) => ({ ...channel, unit: channel.name === 'Alt' ? 'ft' : channel.unit })) },
    { reference: '' },
    { reference: trackReference('91 deg') },
    { reference: trackReference('55 deg', '181 deg') },
    { reference: trackReference('55 rad') },
    { reference: trackReference('55 deg', '5 deg', '200 ft') },
    { reference: trackReference('55 deg', '5 deg', 'null') },
    { playerIndex: -1 },
  ])('omits position when source or reference is unavailable/invalid: %o', async (options) => {
    const ibt = await IBTFile.open(writeFixture('invalid-position.ibt', {
      channels: positionChannels(), reference: trackReference(), ...options,
    }));
    try {
      const { value: row } = await ibt.rows().next();
      expect(row).not.toHaveProperty('Graphics_car_coordinates');
      expect(row).not.toHaveProperty('Graphics_car_id');
      expect(row.Physics_speed_kmh).toBe(180);
    } finally { await ibt.close(); }
  });

  it('does not reuse positions or track references after missing data, metadata changes or resets', () => {
    const channels = positionChannels();
    const adapter = new IRacingIBTAdapter(channels);
    const packet = { type: 'sample', tick: 1, values: { PlayerCarIdx: 63,
      ...Object.fromEntries(channels.map(({ name, value }) => [name, value])) } };
    adapter.updateSession(`WeekendInfo:\n${trackReference()}`);
    expect(adapter.adapt(packet).Graphics_car_id[0]).toBe(63);
    expect(adapter.adapt({ type: 'sample', tick: 2, values: {} })).not.toHaveProperty('Graphics_car_coordinates');
    expect(adapter.adapt(packet).Graphics_car_coordinates).toBeDefined();
    adapter.updateSession('WeekendInfo: null');
    expect(adapter.adapt(packet)).not.toHaveProperty('Graphics_car_coordinates');
    adapter.updateSession(`WeekendInfo:\n${trackReference()}`);
    adapter.reset();
    expect(adapter.adapt(packet)).not.toHaveProperty('Graphics_car_coordinates');
  });

  it.each([{ IsOnTrack: false }, { IsReplayPlaying: true }])('respects explicit cockpit/replay flags: %o', async (flags) => {
    const ibt = await IBTFile.open(writeFixture('flags.ibt', { channels: [
      ...diskChannels(), ...positionChannels(), ...Object.entries(flags).map(([name, value]) => ({ name, value, type: 1 })),
    ], reference: trackReference() }));
    try {
      const { value: row } = await ibt.rows().next();
      expect(row).not.toHaveProperty('Physics_speed_kmh');
      expect(row).not.toHaveProperty('Physics_brake_pressure_front_left');
      expect(row).not.toHaveProperty('Physics_tyre_surface_temp_front_left_inner');
      expect(row).not.toHaveProperty('Graphics_car_coordinates');
      expect(row).not.toHaveProperty('Graphics_car_id');
    } finally { await ibt.close(); }
  });

  it('keeps disk capabilities separate from live capture and never carries measurements into later rows', () => {
    expect(Object.keys(IRACING_IBT_FIELD_COVERAGE)).toEqual(LIVE_TELEMETRY_FIELDS);
    expect(Object.values(IRACING_IBT_FIELD_COVERAGE).filter(({ supported }) => supported)).toHaveLength(136);
    const channels = [...diskChannels(), ...positionChannels()];
    const values = Object.fromEntries(channels.map(({ name, value }) => [name, value]));
    const packet = { type: 'sample', tick: 1, values: { IsOnTrack: true, ...values } };
    const liveRow = new IRacingAdapter().adapt(packet);
    for (const field of LIVE_TELEMETRY_FIELDS.filter((field) => IRACING_IBT_FIELD_COVERAGE[field].supported && !IRACING_FIELD_COVERAGE[field].supported)) {
      const source = IRACING_IBT_FIELD_COVERAGE[field].source;
      if (IRACING_IBT_FIELD_COVERAGE[field].unit) {
        expect(IRACING_IBT_VARIABLES).toContain(source);
        expect(IRACING_VARIABLES).not.toContain(source);
      }
      expect(liveRow).not.toHaveProperty(field);
    }
    for (const source of ['Lat', 'Lon', 'Alt']) {
      expect(IRACING_IBT_VARIABLES).toContain(source);
      expect(IRACING_VARIABLES).not.toContain(source);
    }
    const adapter = new IRacingIBTAdapter(channels);
    expect(adapter.adapt(packet).Physics_brake_pressure_front_left).toBe(40);
    expect(adapter.adapt({ type: 'sample', tick: 2, values: {} })).not.toHaveProperty('Physics_brake_pressure_front_left');
    expect(() => adapter.adapt({ values: {} })).toThrow('Invalid');
  });

  it.each(['empty', 'truncated', 'variable bounds'])('rejects %s telemetry', async (kind) => {
    let buffer = fixture();
    if (kind === 'empty') buffer.writeInt32LE(0, 140);
    if (kind === 'truncated') buffer = buffer.subarray(0, buffer.length - 1);
    if (kind === 'variable bounds') buffer.writeInt32LE(1000, 148);
    const filePath = path.join(directory, 'invalid.ibt');
    fs.writeFileSync(filePath, buffer);
    await expect(IBTFile.open(filePath)).rejects.toThrow();
  });

  it('streams multiple native files into a separate JSONL recording without changing originals', async () => {
    const first = writeFixture('a.ibt'); const second = writeFixture('b.ibt', { speeds: [70] });
    const original = fs.readFileSync(first);
    const outputPath = path.join(directory, 'converted.jsonl');
    expect(await convertIBTFiles([first, second, first], outputPath, live)).toEqual({ filePath: outputPath, rowCount: 3 });
    expect(fs.readFileSync(outputPath, 'utf8').trim().split('\n').map(JSON.parse).map(row => row.Physics_speed_kmh)).toEqual([180, 216, 252]);
    expect(fs.readFileSync(first)).toEqual(original);
  });

  it('removes incomplete output when a selected file belongs to another car or session', async () => {
    const outputPath = path.join(directory, 'converted.jsonl');
    await expect(convertIBTFiles([writeFixture('wrong.ibt', { track: 'monza' })], outputPath, live)).rejects.toThrow('does not match');
    expect(fs.existsSync(outputPath)).toBe(false);
    expect(fs.existsSync(`${outputPath}.partial`)).toBe(false);
    await expect(convertIBTFiles([writeFixture('a.ibt'), writeFixture('b.ibt', { sessionId: 43 })], outputPath, live)).rejects.toThrow('same iRacing session');
  });

  it('rejects a file that changes while conversion is reading it', async () => {
    const filePath = writeFixture('growing.ibt');
    const ibt = await IBTFile.open(filePath);
    try {
      const rows = ibt.rows();
      await rows.next(); fs.appendFileSync(filePath, Buffer.alloc(64));
      await rows.next();
      await expect(rows.next()).rejects.toThrow('still writing');
    } finally { await ibt.close(); }
  });

  it('imports a standalone .ibt file without an app recording and preserves the source', async () => {
    const sourcePath = writeFixture('local.ibt', { channels: [...diskChannels(), ...positionChannels()], reference: trackReference() });
    const original = fs.readFileSync(sourcePath);
    const outputPath = path.join(directory, 'local.jsonl');
    const worker = new Worker(path.resolve(__dirname, '../../../electron/recording/readers/iracing/iracing-local-import-worker.js'), {
      workerData: { sourcePath, outputPath },
    });
    try {
      const result = await new Promise((resolve, reject) => { worker.once('message', resolve); worker.once('error', reject); });
      expect(result).toMatchObject({ type: 'complete', filePath: outputPath, rowCount: 2, track: 'spa', car: 'GT3' });
      const rows = fs.readFileSync(outputPath, 'utf8').trim().split('\n').map(JSON.parse);
      expect(rows.every((row) => validateLiveTelemetryRow(row).ok)).toBe(true);
      expect(rows[0].Graphics_car_coordinates[0].x).toBeCloseTo(-189013.869, 3);
      expect(rows[0].Physics_speed_kmh).toBe(180);
      expect(fs.readFileSync(sourcePath)).toEqual(original);
    } finally { await worker.terminate(); }
  });
});
