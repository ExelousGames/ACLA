import { ReadableStream as NodeReadableStream } from 'stream/web';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { parseRecordedDrivingSession, parseRecordedDrivingSessionStream, findRecordedGearShifts, findRecordedTelemetry, readRecordedDrivingObservation } from './recorded-driving-session';
import { cloneConfig } from './driving-model-config';
import { trainingFixture } from './training-fixtures';

const jsonl = (...rows: object[]) => rows.map((row) => JSON.stringify(row)).join('\n');
const sample = (time: number, overrides = {}) => ({ ...trainingFixture().live.currentTelemetry,
    Graphics_analyst_session_time: time, ...overrides });

it('reads desktop JSONL with BOM and blank lines, using recording seconds instead of wall time', () => {
    const session = parseRecordedDrivingSession(`\uFEFF${jsonl(sample(20), sample(20.1))}\r\n\r\n`, 'acc');
    expect(session.game).toBe('acc');
    expect(session.timestampField).toBe('Graphics_analyst_session_time');
    expect(session.startTimeSeconds).toBe(20);
    expect(session.durationSeconds).toBeCloseTo(0.1);
    expect(session.rows[0].timeSeconds).toBe(0);
});

it('uses game time for older recordings and infers the simulator from source frame envelopes', () => {
    const session = parseRecordedDrivingSession(jsonl({ game: 'iracing', sample: { Graphics_game_time: 4 } },
        { game: 'iracing', sample: { Graphics_game_time: 5 } }), 'acc');
    expect(session.game).toBe('iracing');
    expect(session.timestampField).toBe('Graphics_game_time');
    expect(session.durationSeconds).toBe(1);
});

const byteStream = (text: string) => {
    const bytes = new TextEncoder().encode(text).values();
    return new NodeReadableStream<Uint8Array>({ pull(controller) {
        const next = bytes.next();
        if (next.done) controller.close();
        else controller.enqueue(Uint8Array.of(next.value));
    } }) as unknown as ReadableStream<Uint8Array>;
};

it('imports split UTF-8, BOM, CRLF, blank lines and an unterminated final line from a stream', async () => {
    const car = 'Porschè 🏁';
    const row = (time: number, gear: number) => JSON.stringify({ game: 'acc',
        sample: sample(time, { Static_car_model: car, Physics_gear: gear }) });
    const stream = byteStream(`\uFEFF${row(20, 2)}\r\n\r\n${row(20.1, 1)}\r\n${row(20.2, 3)}`);
    const session = await parseRecordedDrivingSessionStream(stream, 'ac');
    expect(session).toMatchObject({ game: 'acc', timestampField: 'Graphics_analyst_session_time', startTimeSeconds: 20 });
    expect(session.durationSeconds).toBeCloseTo(0.2);
    expect(session.rows.map(({ sample: entry }) => [entry.Static_car_model, entry.Physics_gear])).toEqual([
        [car, 2], [car, 1], [car, 3],
    ]);
    expect(session.rows[1].neutralShift).toMatchObject({ gear: 2 });
    expect(stream.locked).toBe(false);
});

it.each([
    ['\uFEFF\r\n\t', /no samples/],
    ['{"Graphics_game_time":0}\r\n\r\n{bad}', /line 3: invalid JSON/],
    ['{"Graphics_game_time":0,"Physics_heading":"invalid"}', /line 1: Invalid value.*Physics_heading/],
    ['{"Graphics_game_time":1}\n{"Graphics_game_time":0}', /line 2: timestamps move backwards/],
])('keeps JSONL diagnostics for streamed input %s', async (value, error) => {
    const stream = byteStream(value);
    await expect(parseRecordedDrivingSessionStream(stream, 'acc')).rejects.toThrow(error);
    expect(stream.locked).toBe(false);
});

it('cancels a pending telemetry read and releases its stream reader', async () => {
    const cancellation = new AbortController();
    const cancel = jest.fn();
    let readStarted!: () => void;
    const reading = new Promise<void>((resolve) => { readStarted = resolve; });
    const stream = new NodeReadableStream<Uint8Array>({ pull: readStarted, cancel },
        { highWaterMark: 0 }) as unknown as ReadableStream<Uint8Array>;
    const result = parseRecordedDrivingSessionStream(stream, 'acc', cancellation.signal);
    await reading;
    cancellation.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(stream.locked).toBe(false);
});

it.each([
    ['', /no samples/],
    ['{bad}', /line 1: invalid JSON/],
    ['[]', /telemetry object/],
    [jsonl({ Physics_speed_kmh: 80 }), /timestamp in seconds is required/],
    [jsonl({ Graphics_game_time: -1 }), /non-negative/],
    [jsonl({ Graphics_game_time: 0, Physics_speed_kmh: '80' }), /Invalid value/],
    [jsonl({ Graphics_game_time: 0, random_field: 1 }), /Unknown live telemetry/],
    [jsonl(sample(1), sample(0)), /timestamps move backwards/],
    [jsonl(sample(1), { Graphics_game_time: 2 }), /Graphics_analyst_session_time/],
    [jsonl({ game: 'invalid', Graphics_game_time: 0 }), /unknown simulator/],
    [jsonl({ game: 'acc', Graphics_game_time: 0 }, { game: 'iracing', Graphics_game_time: 1 }), /simulator changed/],
])('rejects unusable recorded telemetry (%s)', (text, error) => {
    expect(() => parseRecordedDrivingSession(text, 'acc')).toThrow(error);
});

it('pairs causally, selects the latest duplicate, and rejects stale gaps and out-of-range times', () => {
    const session = parseRecordedDrivingSession(jsonl(sample(0), sample(0.2), sample(0.2, { Physics_speed_kmh: 90 }), sample(2)), 'acc');
    expect(findRecordedTelemetry(session, 0.19)?.sample.Physics_speed_kmh).toBe(108);
    expect(findRecordedTelemetry(session, 0.2)?.sample.Physics_speed_kmh).toBe(90);
    expect(findRecordedTelemetry(session, 0.7)).toBe(session.rows[2]);
    expect(findRecordedTelemetry(session, 0.701)).toBeNull();
    expect(findRecordedTelemetry(session, 1.9)).toBeNull();
    expect(findRecordedTelemetry(session, 3)).toBeNull();
    expect(findRecordedTelemetry(session, -1)).toBeNull();
    expect(findRecordedTelemetry(session, NaN)).toBeNull();
    expect(findRecordedTelemetry(session, 1, Infinity)).toBeNull();
});

it('detects single gear changes through brief neutral and ignores discontinuities', () => {
    const gear = (time: number, value: number, overrides = {}) => sample(time, { Physics_gear: value, ...overrides });
    const session = parseRecordedDrivingSession(jsonl(
        gear(0, 3), gear(0.0625, 1), gear(0.125, 4), gear(0.25, 3),
        gear(0.3125, 1), gear(0.375, 4), gear(0.375, 3), // A same-gear return at a duplicate timestamp.
        gear(0.4375, 6), // Multiple gears in one sample.
        gear(0.5, 5, { Static_track: 'spa' }), gear(1.125, 4, { Static_track: 'spa' }), // New segment, then a long gap.
        gear(1.1875, 4, { Static_track: 'spa', Graphics_status: 1 }), gear(1.25, 3, { Static_track: 'spa' }),
        gear(1.3125, 1, { Static_track: 'spa' }), gear(1.75, 1, { Static_track: 'spa' }), gear(1.875, 4, { Static_track: 'spa' }),
    ), 'acc');
    expect(findRecordedGearShifts(session, 0.5)).toEqual([
        { direction: 'up', before: 0, after: 0.125 },
        { direction: 'down', before: 0.125, after: 0.25 },
    ]);
});

it('matches a frame by telemetry clock time even when the video starts after the first row', () => {
    const session = parseRecordedDrivingSession(jsonl(sample(10), sample(11)), 'acc');
    expect(findRecordedTelemetry(session, 10)).toBe(session.rows[0]);
    expect(findRecordedTelemetry(session, 9.9)).toBeNull();
    expect(findRecordedTelemetry(session, 11)).toBe(session.rows[1]);
});

it('does not select the previous sample because of timestamp subtraction rounding', () => {
    const session = parseRecordedDrivingSession(jsonl(sample(100), sample(100.1)), 'acc');
    expect(findRecordedTelemetry(session, 100.05 + 0.05)).toBe(session.rows[1]);
    const fixture = trainingFixture();
    expect(readRecordedDrivingObservation(session, 100.1, fixture.observation.camera, fixture.referenceLap,
        50, cloneConfig(), fixture.circuitMap).observation?.measuredAt).toBe(100100);
});

it('builds the same observation features with streamed recording clocks and leaves live data untouched', async () => {
    const fixture = trainingFixture();
    const session = await parseRecordedDrivingSessionStream(byteStream(jsonl(sample(10), sample(10.2))), 'acc');
    const before = liveTelemetryStore.getSnapshot();
    const result = readRecordedDrivingObservation(session, 10.25, fixture.observation.camera,
        fixture.referenceLap, 50, cloneConfig(), fixture.circuitMap);
    expect(result.observation?.features).toEqual(fixture.observation.features);
    expect(result.observation?.capturedAt).toBe(10250);
    expect(result.observation?.measuredAt).toBe(10200);
    expect(result.observation?.session).toContain(session.id);
    expect(liveTelemetryStore.getSnapshot()).toBe(before);
    expect(readRecordedDrivingObservation(session, 1, fixture.observation.camera, fixture.referenceLap).error).toMatch(/No telemetry/);
});

it('separates imported sessions and context changes so replay learning cannot cross them', () => {
    const rows = jsonl(sample(0), sample(0.1, { Static_number_of_session: 2 }), sample(0.2, { Static_track: 'spa' }));
    const session = parseRecordedDrivingSession(rows, 'acc');
    expect(session.rows.map((row) => row.segment)).toEqual([0, 1, 2]);
    expect(parseRecordedDrivingSession(rows, 'acc').id).not.toBe(session.id);
    const fixture = trainingFixture();
    const first = readRecordedDrivingObservation(session, 0, fixture.observation.camera, fixture.referenceLap, 50, cloneConfig(), fixture.circuitMap);
    const second = readRecordedDrivingObservation(session, 0.1, fixture.observation.camera, fixture.referenceLap, 50, cloneConfig(), fixture.circuitMap);
    expect(first.observation?.session).not.toBe(second.observation?.session);
});

it('preserves paused, missing-data and mismatched-reference validation for recorded observations', () => {
    const fixture = trainingFixture();
    const paused = parseRecordedDrivingSession(jsonl(sample(0, { Graphics_status: 3 })), 'acc');
    expect(readRecordedDrivingObservation(paused, 0, fixture.observation.camera, fixture.referenceLap).error).toMatch(/not marked as running/);
    const missing = parseRecordedDrivingSession(jsonl({ Graphics_game_time: 0, Graphics_status: 2 }), 'acc');
    expect(readRecordedDrivingObservation(missing, 0, fixture.observation.camera, fixture.referenceLap).error).toMatch(/Game, car and map/);
    const valid = parseRecordedDrivingSession(jsonl(sample(0)), 'acc');
    expect(readRecordedDrivingObservation(valid, 0, fixture.observation.camera, { ...fixture.referenceLap, track: 'spa' }).error).toMatch(/does not match/);
});
