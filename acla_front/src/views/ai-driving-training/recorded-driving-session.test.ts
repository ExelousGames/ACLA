import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { parseRecordedDrivingSession, findRecordedTelemetry, readRecordedDrivingObservation } from './recorded-driving-session';
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

it('builds the same observation features with recorded clocks and leaves live data untouched', () => {
    const fixture = trainingFixture();
    const session = parseRecordedDrivingSession(jsonl(sample(10), sample(10.2)), 'acc');
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
