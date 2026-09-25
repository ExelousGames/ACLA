/** @jest-environment node */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { PassThrough } = require('stream');
const { IRacingAdapter, IRACING_FIELD_COVERAGE } = require('../../../electron/recording/readers/iracing/iracing-adapter');
const { IRacingReader } = require('../../../electron/recording/readers/iracing/iracing-reader');
const { DeliveryWindow, runIRacingReaderWorker } = require('../../../electron/recording/readers/iracing/iracing-reader-worker');
const { getReaderLaunchConfig } = require('../../../electron/recording/readers/iracing/iracing-reader-config');
const { LIVE_TELEMETRY_FIELDS } = require('../../data/live-telemetry-dataset');
const { validateSourceFrame } = require('../../../electron/recording/telemetry-contract');
const { RecordingWriter } = require('../../../electron/recording/workers/writer-worker');
const { RecordedFileReader } = require('../../../electron/recording/workers/recorded-file-reader-worker');

const sessionInfo = `
WeekendInfo:
 TrackName: spa
 TrackConfigName: endurance
DriverInfo:
 DriverCarIdx: 4
 DriverCarRedLine: 8500
 DriverCarFuelMaxLtr: 100.5
 Drivers:
 - CarIdx: 0
   CarIsPaceCar: 1
 - CarIdx: 2
   IsSpectator: 1
 - CarIdx: 4
   UserName: "Zoë Driver"
   AbbrevName: "Z Driver"
   CarScreenName: GT3
SessionInfo:
 Sessions:
 - SessionNum: 0
   SessionType: Practice
 - SessionNum: 1
   SessionType: Race
SplitTimeInfo:
 Sectors:
 - SectorNum: 0
   SectorStartPct: 0.0
 - SectorNum: 1
   SectorStartPct: 0.35
 - SectorNum: 2
   SectorStartPct: 0.7
`;
const packet = (values = {}, extra = {}) => ({ type: 'sample', tick: 120, values: {
  IsOnTrack: true, IsReplayPlaying: false, PlayerCarIdx: 4, SessionNum: 1, ...values,
}, ...extra });

describe('iRacing standard field adapter', () => {
  it('accounts for every field in the shared catalog', () => {
    expect(Object.keys(IRACING_FIELD_COVERAGE)).toEqual(LIVE_TELEMETRY_FIELDS);
    for (const field of Object.values(IRACING_FIELD_COVERAGE)) {
      expect(field.supported ? field.source : field.reason).toEqual(expect.any(String));
    }
  });

  it('converts units, gear, steering, lap times and flags before delivery', () => {
    const sample = new IRacingAdapter().adapt(packet({
      Speed: 50, Throttle: 0.75, BrakeRaw: 0.2, Clutch: 1, Gear: 3, RPM: 6500.25,
      SteeringWheelAngle: -Math.PI / 2, SteeringWheelAngleMax: 4 * Math.PI,
      FuelLevel: 24.5, dcBrakeBias: 54, BrakeABSactive: true, EngineWarnings: 0x10,
      LFpressure: 200, RFpressure: 210, LRpressure: 220, RRpressure: 230,
      LFshockDefl: 0.02, PitSvLFP: 190, LapCurrentLapTime: 62.1234,
      LapLastLapTime: 91.005, LapBestLapTime: 90, LapDeltaToSessionBestLap: -0.125,
      LapDeltaToSessionBestLap_OK: true, SessionFlags: 0x108, LapDistPct: 0.72,
      CarIdxTrackSurface: [1, -1, 3, 0], SessionTimeRemain: 300, OnPitRoad: false,
    }, { sessionInfo }));
    expect(sample).toMatchObject({
      Physics_speed_kmh: 180, Physics_gear: 4, Physics_rpm: 6500, Physics_steer_angle: -0.25,
      Physics_gas: 0.75, Physics_brake: 0.2, Physics_clutch: 1, Physics_fuel: 24.5,
      Physics_brake_bias: 0.54, Physics_abs: 1, Physics_pit_limiter_on: true,
      Physics_suspension_travel_front_left: 0.02,
      Graphics_current_time: 62123, Graphics_current_time_str: '1:02.123',
      Graphics_last_time: 91005, Graphics_last_time_str: '1:31.005',
      Graphics_delta_lap_time: -125, Graphics_delta_lap_time_str: '-0:00.125',
      Graphics_estimated_lap_time: 89875, Graphics_is_delta_positive: false,
      Graphics_flag: 2, Graphics_global_yellow: true, Graphics_global_green: false,
      Graphics_status: 2, Graphics_session_type: 2, Graphics_current_sector_index: 2,
      Graphics_active_cars: 3, Graphics_session_time_left: 300,
      Static_track: 'spa - endurance', Static_car_model: 'GT3', Static_player_name: 'Zoë Driver',
      Static_max_rpm: 8500, Static_max_fuel: 100.5, Static_num_cars: 1, Static_sector_count: 3,
    });
    expect(sample.Physics_wheel_pressure_front_left).toBeCloseTo(29.0075, 4);
    expect(sample.Physics_wheel_pressure_front_right).toBeCloseTo(30.4579, 4);
    expect(sample.Physics_wheel_pressure_rear_left).toBeCloseTo(31.9083, 4);
    expect(sample.Physics_wheel_pressure_rear_right).toBeCloseTo(33.3587, 4);
    expect(sample.Graphics_mfd_tyre_pressure_front_left).toBeCloseTo(27.5572, 4);
    expect(validateSourceFrame({ game: 'iracing', sample }, 'iracing').ok).toBe(true);
  });

  it('tracks the brake pedal while simulator-applied braking stays at full force', () => {
    const adapter = new IRacingAdapter();
    for (const brake of [0, 0.25, 1, 0.5, 0]) {
      const sample = adapter.adapt(packet({ Speed: 0, Brake: 1, BrakeRaw: brake, Throttle: 0.75 }));
      expect(sample.Physics_brake).toBe(brake);
      expect(sample.Physics_gas).toBe(0.75);
    }
  });

  it.each([undefined, null, NaN, Infinity, -0.1, 1.1, '0.5'])('omits unavailable or invalid brake pedal input: %p', (brake) => {
    const sample = new IRacingAdapter().adapt(packet({ Brake: 1, BrakeRaw: brake }));
    expect(sample).not.toHaveProperty('Physics_brake');
  });

  it.each([[-1, 0], [0, 1], [1, 2]])('converts iRacing gear %i to standard gear %i', (raw, expected) => {
    expect(new IRacingAdapter().adapt(packet({ Gear: raw })).Physics_gear).toBe(expected);
  });

  it('omits unavailable, nonfinite, sentinel and non-equivalent data', () => {
    const sample = new IRacingAdapter().adapt(packet({
      Speed: NaN, FuelLevel: -1, LapBestLapTime: -1, LapDistPct: -1,
      SessionTimeRemain: 604800, PlayerCarPosition: 0, RPM: Infinity,
      LapDeltaToSessionBestLap: 0.5, LapDeltaToSessionBestLap_OK: false,
      SteeringWheelAngle: 1, SteeringWheelAngleMax: 0, LFcoldPressure: 200,
      LFtempCM: 80, PlayerCarMyIncidentCount: 2, CarIdxLapDistPct: [0.2],
      VelocityX: NaN, SessionFlags: undefined,
    }));
    expect(sample).toEqual({ Physics_packed_id: 120, Graphics_packed_id: 120,
      Graphics_status: 2, Graphics_player_car_id: 4, Graphics_session_index: 1,
      Graphics_normalized_positions: { 0: 0.2 } });
  });

  it('caches YAML between updates and clears metadata on reconnect', () => {
    const adapter = new IRacingAdapter();
    adapter.adapt(packet({}, { sessionInfo }));
    const parse = jest.spyOn(adapter, 'updateSession');
    expect(adapter.adapt(packet()).Static_track).toBe('spa - endurance');
    expect(parse).not.toHaveBeenCalled();
    expect(adapter.adapt(packet({}, { sessionInfo: 'WeekendInfo:\n TrackName: monza\n' })).Static_track).toBe('monza');
    expect(adapter.adapt(packet())).not.toHaveProperty('Static_car_model');
    adapter.reset();
    expect(adapter.adapt(packet())).not.toHaveProperty('Static_track');
  });

  it('preserves native car indices, endpoints and pit cars in normalized positions', () => {
    const positions = Array(64).fill(-1);
    const surfaces = Array(64).fill(-1);
    for (const [carId, position, surface] of [[0, 0, 3], [4, 0.25, 2], [60, 0.75, 1], [63, 1, 3]]) {
      positions[carId] = position;
      surfaces[carId] = surface;
    }
    positions[2] = 0.5; // Not in the world, despite a stale lap fraction.
    const row = new IRacingAdapter().adapt(packet({ CarIdxLapDistPct: positions, CarIdxTrackSurface: surfaces, LapDistPct: 0.25 }));
    expect(row.Graphics_normalized_positions).toEqual({ 0: 0, 4: 0.25, 60: 0.75, 63: 1 });
    expect(row.Graphics_normalized_car_position).toBe(0.25);
    expect(IRACING_FIELD_COVERAGE.Graphics_normalized_positions.supported).toBe(true);
  });

  it('omits invalid positions and never carries them into another sample or replay', () => {
    const adapter = new IRacingAdapter();
    const values = { CarIdxLapDistPct: [0.5, -1, NaN, Infinity, null, '0.2', 1.01] };
    expect(adapter.adapt(packet(values)).Graphics_normalized_positions).toEqual({ 0: 0.5 });
    expect(adapter.adapt(packet({ CarIdxLapDistPct: [-1] })).Graphics_normalized_positions).toEqual({});
    expect(adapter.adapt(packet())).not.toHaveProperty('Graphics_normalized_positions');
    expect(adapter.adapt(packet({ ...values, IsReplayPlaying: true }))).not.toHaveProperty('Graphics_normalized_positions');
    expect(adapter.adapt(packet({ ...values, IsOnTrack: false }))).not.toHaveProperty('Graphics_normalized_positions');
  });

  it('never forwards stale cockpit values while spectating or replaying', () => {
    const adapter = new IRacingAdapter();
    expect(adapter.adapt(packet({ IsOnTrack: false, Speed: 40 })).Graphics_status).toBe(0);
    const replay = adapter.adapt(packet({ IsReplayPlaying: true, Speed: 40 }));
    expect(replay.Graphics_status).toBe(1);
    expect(replay).not.toHaveProperty('Physics_speed_kmh');
  });

  it('rejects malformed packets and unsafe or non-mapping YAML', () => {
    const adapter = new IRacingAdapter();
    expect(() => adapter.adapt({ values: {} })).toThrow('Invalid');
    expect(() => adapter.adapt(packet({}, { sessionInfo: '- entry' }))).toThrow('mapping');
    expect(() => adapter.adapt(packet({}, { sessionInfo: 'x: !!js/function function() {}' }))).toThrow();
  });
});

describe('iRacing process transport', () => {
  const options = { runtime: 'python', pythonExecutable: 'python.exe', scriptDirectory: os.tmpdir(), scriptName: 'iracing_sdk.py' };
  function childFixture() {
    const child = new EventEmitter();
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = jest.fn(() => child.emit('close', 0, null));
    child.stdin.on('finish', () => setImmediate(() => child.emit('close', 0, null)));
    return child;
  }

  it('allows one capture request at a time and waits for consumer capacity', async () => {
    const child = childFixture();
    const requests = [];
    child.stdin.on('data', (chunk) => requests.push(String(chunk)));
    const spawn = jest.fn(() => child);
    const reader = new IRacingReader(options, { spawn });
    let resume;
    const delivered = jest.fn(() => new Promise((resolve) => { resume = resolve; }));
    const started = reader.start(delivered);
    child.stdout.write('{"type":"rea');
    child.stdout.write('dy"}\n');
    await started;
    expect(requests).toEqual(['next\n']);
    child.stdout.write(`${JSON.stringify(packet({ Speed: 20 }))}\n`);
    expect(delivered).toHaveBeenCalledWith({ type: 'frame', frame: {
      game: 'iracing', sample: expect.objectContaining({ Physics_speed_kmh: 72 }),
    } });
    expect(requests).toHaveLength(1);
    resume();
    await Promise.resolve();
    expect(requests).toHaveLength(2);
    expect(spawn.mock.calls[0][2]).toMatchObject({ windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    await reader.stop();
    expect(reader.stop()).toBe(reader.stopPromise);
  });

  it('reports malformed capture output once and stops the child', async () => {
    const child = childFixture();
    const reader = new IRacingReader(options, { spawn: () => child });
    const events = jest.fn();
    const start = reader.start(events);
    child.stdout.write('{"type":"ready"}\n');
    await start;
    child.stdout.write('broken json\n');
    await reader.stop();
    expect(events).toHaveBeenCalledTimes(1);
    expect(events.mock.calls[0][0].type).toBe('fatal');
  });

  const pythonExecutable = process.env.ACLA_PYTHON_PATH
    || path.resolve(__dirname, '../../../.venv/py-scripts/Scripts/python.exe');
  (fs.existsSync(pythonExecutable) ? it : it.skip)('captures through a real subprocess and releases its pipes on stop', async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-iracing-process-'));
    const sourcePacket = packet({ Speed: 50, Gear: 2 }, { sessionInfo });
    fs.writeFileSync(path.join(directory, 'fixture.py'), [
      'import json, sys',
      `sample = json.loads(${JSON.stringify(JSON.stringify(sourcePacket))})`,
      'print(json.dumps({"type": "ready"}), flush=True)',
      'for command in sys.stdin:',
      '    print(json.dumps(sample), flush=True)',
    ].join('\n'));
    const reader = new IRacingReader({ ...options, pythonExecutable, scriptDirectory: directory, scriptName: 'fixture.py' });
    const frames = [];
    let resolveFrames;
    let rejectFrames;
    const received = new Promise((resolve, reject) => { resolveFrames = resolve; rejectFrames = reject; });
    try {
      await reader.start((event) => {
        if (event.type === 'fatal') { rejectFrames(new Error(event.error)); return; }
        frames.push(event.frame);
        if (frames.length === 10) resolveFrames();
      });
      await received;
      await reader.stop();
      expect(reader.child).toBeNull();
      expect(frames.length).toBeGreaterThanOrEqual(10);
      expect(frames[0]).toEqual({ game: 'iracing', sample: expect.objectContaining({
        Physics_speed_kmh: 180, Physics_gear: 3, Static_track: 'spa - endurance',
      }) });
    } finally {
      await reader.stop();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it('bounds deliveries until both disk and view acknowledge them', async () => {
    const window = new DeliveryWindow(2);
    expect(window.sentFrame()).toBeUndefined();
    let released = false;
    const pending = window.sentFrame().then(() => { released = true; });
    window.acknowledge('view', 2);
    await Promise.resolve();
    expect(released).toBe(false);
    window.acknowledge('writer', 1);
    await pending;
    expect(released).toBe(true);
    expect(() => window.acknowledge('writer', 3)).toThrow('acknowledgement');
  });

  it('reports a stalled consumer and releases waits on stop', async () => {
    const blocked = new DeliveryWindow(1, 5);
    await expect(blocked.sentFrame()).rejects.toThrow('not keeping up');
    const stopping = new DeliveryWindow(1);
    const pending = stopping.sentFrame();
    stopping.close();
    await expect(pending).resolves.toBeUndefined();
  });

  it('transfers only standard frames through worker ports and drains capture before end', async () => {
    const makePort = () => Object.assign(new EventEmitter(), { postMessage: jest.fn(), start: jest.fn(), close: jest.fn() });
    const parent = makePort();
    const ports = [makePort(), makePort()];
    let emit;
    let finishCapture;
    const start = jest.spyOn(IRacingReader.prototype, 'start').mockImplementation((callback) => { emit = callback; return Promise.resolve(); });
    const stop = jest.spyOn(IRacingReader.prototype, 'stop').mockImplementation(() => new Promise((resolve) => { finishCapture = resolve; }));
    try {
      runIRacingReaderWorker(parent);
      parent.emit('message', { data: { type: 'initialize', game: 'iracing', readerOptions: options,
        portRoles: ['frameToWriter', 'frameToView'] }, ports });
      await Promise.resolve();
      expect(parent.postMessage).toHaveBeenCalledWith({ type: 'ready', service: 'reader', game: 'iracing' });
      const frame = { game: 'iracing', sample: { Physics_speed_kmh: 180 } };
      emit({ type: 'frame', frame });
      for (const port of ports) {
        expect(port.postMessage).toHaveBeenCalledWith({ type: 'frame', frame });
        port.emit('message', { data: { type: 'ack', game: 'iracing', sequence: 1 } });
      }
      parent.emit('message', { data: { type: 'stop' } });
      expect(ports[0].postMessage).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'end' }));
      finishCapture();
      await new Promise(setImmediate);
      for (const port of ports) expect(port.postMessage).toHaveBeenLastCalledWith({ type: 'end', game: 'iracing' });
      expect(parent.postMessage).toHaveBeenLastCalledWith({ type: 'stopped', service: 'reader', game: 'iracing' });
    } finally { start.mockRestore(); stop.mockRestore(); }
  });

  it('rejects a worker descriptor for another game before launching capture', async () => {
    const parent = Object.assign(new EventEmitter(), { postMessage: jest.fn(), start: jest.fn() });
    const start = jest.spyOn(IRacingReader.prototype, 'start');
    try {
      runIRacingReaderWorker(parent);
      parent.emit('message', { data: { type: 'initialize', game: 'acc' }, ports: [] });
      await Promise.resolve();
      expect(start).not.toHaveBeenCalled();
      expect(parent.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'fatal' }));
    } finally { start.mockRestore(); }
  });
});

describe('iRacing recording integration', () => {
  it('resolves development and packaged capture resources without starting a process', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-iracing-config-'));
    try {
      for (const folder of ['src/py-scripts', '.venv/py-scripts/Scripts', 'py-scripts', 'python-env/Scripts']) {
        fs.mkdirSync(path.join(root, folder), { recursive: true });
      }
      for (const file of ['reader.js', 'src/py-scripts/iracing_sdk.py', 'py-scripts/iracing_sdk.py', '.venv/py-scripts/Scripts/python.exe', 'python-env/Scripts/python.exe']) {
        fs.writeFileSync(path.join(root, file), '');
      }
      const settings = { readerEntryPath: path.join(root, 'reader.js'), env: {}, platform: 'win32', resourcesPath: root };
      for (const isPackaged of [false, true]) {
        const result = getReaderLaunchConfig({ ...settings, app: { getAppPath: () => root, isPackaged } });
        expect(result.ok).toBe(true);
        expect(result.config.game).toBe('iracing');
        expect(fs.existsSync(path.join(result.config.readerOptions.scriptDirectory, result.config.readerOptions.scriptName))).toBe(true);
      }
      expect(getReaderLaunchConfig({ ...settings, platform: 'linux' }).ok).toBe(false);
    } finally { fs.rmSync(root, { recursive: true, force: true }); }
  });

  it('writes and reads adapted iRacing rows unchanged, with credits after disk writes', async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-iracing-recording-'));
    const committed = jest.fn();
    const writer = new RecordingWriter({ game: 'iracing', recordingDirectory: directory,
      progressPort: { postMessage: jest.fn() }, parentSend: jest.fn(), onCommitted: committed });
    try {
      const filePath = await writer.open();
      const adapter = new IRacingAdapter();
      const rows = Array.from({ length: 60 }, (_, tick) => adapter.adapt(packet({
        Speed: 30 + tick / 10, VelocityX: 30 + tick / 10, VelocityY: 2, VelocityZ: 0,
        LatAccel: 9.80665, VertAccel: 9.80665, LongAccel: -9.80665, Yaw: Math.PI / 2, Pitch: 0, Roll: 0,
        PitchRate: 0.1, YawRate: 0.2, RollRate: 0.3,
      }, { tick, ...(tick === 0 ? { sessionInfo } : {}) })));
      expect(rows[0]).toMatchObject({ Physics_local_velocity_z: 30, Physics_local_velocity_x: -2,
        Physics_g_force_x: -1, Physics_g_force_z: -1, Physics_local_angular_vel_y: -0.2 });
      expect(rows[0].Physics_velocity_x).toBeCloseTo(-30, 10);
      expect(rows[0].Physics_velocity_y).toBeCloseTo(0, 10);
      expect(rows[0].Physics_velocity_z).toBeCloseTo(-2, 10);
      // JSON stores signed zero as 0 (for example, the normalized zero roll).
      const serializedRows = JSON.parse(JSON.stringify(rows));
      rows.forEach((sample) => writer.acceptFrame({ game: 'iracing', sample }));
      expect(committed).not.toHaveBeenCalled();
      await writer.flush();
      expect(committed).toHaveBeenLastCalledWith(60);
      expect(fs.readFileSync(filePath, 'utf8').trim().split('\n').map(JSON.parse)).toEqual(serializedRows);
      const final = await writer.end();
      expect(final.writtenSamples).toBe(60);
      const output = [];
      let reader;
      reader = new RecordedFileReader({ readId: 'iracing-roundtrip', filePath, game: 'iracing', purpose: 'consume', recordingDirectory: directory,
        parentSend: jest.fn(), eventPort: { close: jest.fn(), postMessage: (message) => {
          if (message.type === 'chunk') { output.push(...message.rows); setImmediate(() => reader.acknowledgeChunk(message.chunkIndex)); }
        } } });
      await reader.start();
      expect(output).toEqual(serializedRows);
    } finally { fs.rmSync(directory, { recursive: true, force: true }); }
  });
});
