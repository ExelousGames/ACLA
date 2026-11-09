import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { PhraseEngine, PHRASE_DEFINITIONS } from './phrase-engine';
import { readReferenceGuidance, referencePositionAhead } from './phrase-reference-guidance';
import { circuitMap, steeringCornerFixture } from './test-fixtures';

// 1000 m square; each 0.01 of progress is exactly 10 m, including the lap seam.
const referenceLap = (): TopLapReferenceEntry => ({
    game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
    target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
        'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_steering'],
    x: Array.from({ length: 100 }, (_, i) => i / 100),
    y: Array.from({ length: 100 }, (_, i) => {
        const d = i * 10;
        const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
        return [...xy, 0, i >= 21 && i < 30 ? 0.8 : 0, i >= 61 && i < 90 ? 0.8 : 0, 0];
    }),
});
const sampleAt = (position: number, speed = 36): StandardTelemetrySample => ({
    Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: position,
    Physics_speed_kmh: speed, Physics_brake: 0, Physics_gas: 0,
});
const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});
function capture(lap: TopLapReferenceEntry, sample: StandardTelemetrySample, now: number, lateral = 0): TrackVisionDetection {
    return { capturedAt: now, width: 100, height: 100, detections: {}, reconstruction: null, geometry: null,
        telemetry: { ...createLiveTelemetryStore().getSnapshot(), game: 'acc', receivedAt: now, currentTelemetry: sample,
            telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0 },
        position: { coordinateFrame: 'car_local', inferenceMs: 0, referenceDelta: [lateral, 0, 0],
            normalizedPosition: sample.Graphics_normalized_car_position!, referenceLap: lap,
            references: { reference: { x: 0, y: 0, z: 0 } } } };
}

it.each([36, 72, 180])('enters Brake and speaks its child 500 ms early at %s km/h without an entry wait', (speed) => {
    const lap = referenceLap(), engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    const lead = speed / 3.6 * 0.5 / 1000;
    const before = sampleAt(0.2 - lead - 0.00001, speed);
    expect(engine.receiveTelemetry(frame(before), 1199).state.path).toEqual(['root']);
    // Waiting while still before the predicted boundary must not manufacture an entry.
    expect(engine.evaluate(1200, true).state.path).toEqual(['root']);
    const entry = sampleAt(0.2 - lead + 0.00001, speed);
    expect(engine.receiveTelemetry(frame(entry), 1201).state.path).toEqual(['root', 'Brake zone']);
    // Reference pressure reaches 10% at 201.25 m; the child previews that independently of zone entry.
    expect(engine.receiveTelemetry(frame(sampleAt(0.20125 - lead - 0.00001, speed)), 1202).events).toEqual([]);
    const speaking = sampleAt(0.20125 - lead + 0.00001, speed);
    expect(readReferenceGuidance(speaking, null, lap, 'acc', 1203).brake).toBeUndefined();
    const result = engine.receiveTelemetry(frame(speaking), 1203);
    expect(result.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(result.events).toEqual([expect.objectContaining({ sentence: 'Brake.', timestamp: 1203, expiresAt: 2703 })]);
    expect(engine.receiveTelemetry(frame(speaking), 1403).events).toEqual(result.events);
});

it.each([['Throttle', 0, 0.601875], ['More throttle', 0.3, 0.605625]] as const)(
    'previews the Throttle parent and %s child using the same position', (name, pedal, boundary) => {
        const lap = referenceLap(), engine = new PhraseEngine();
        engine.receiveMap(null, 0, lap);
        const at = (position: number, now: number) => {
            const sample = { ...sampleAt(position), Physics_gas: pedal };
            engine.receiveVision(capture(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        expect(at(0.59499, 1000).state.path).toEqual(['root']);
        expect(at(0.59501, 1001).state.path).toEqual(['root', 'Throttle zone']);
        expect(at(boundary - 0.00501, 1002).events).toEqual([]);
        const result = at(boundary - 0.00499, 1003);
        expect(result.state.path).toEqual(['root', 'Throttle zone', name, 'say phrase']);
        expect(result.events).toEqual([expect.objectContaining({ sentence: `${name}.`, timestamp: 1003 })]);
    });

it.each([
    ['brake', 'Physics_brake', 0.215, 'Brake'],
    ['throttle', 'Physics_gas', 0.65, 'Throttle'],
] as const)('uses intended %s pressure to enter its phrase before the measured pedal crosses the threshold', (cue, field, start, name) => {
    const lap = referenceLap(), engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === cue));
    engine.receiveMap(null, 0, lap);
    const at = (t: number) => {
        const now = 1000 + t * 1000;
        const sample = { ...sampleAt(start + 0.01 * t), [field]: 0.5 - 0.5 * t };
        engine.receiveVision(capture(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(0).events).toEqual([]);
    expect(at(0.2).events).toEqual([]);
    // Measured pressure is 30%, but it will be 5% at the shared 500 ms decision point.
    const measured = { ...sampleAt(start + 0.004), [field]: 0.3 };
    expect(readReferenceGuidance(measured, capture(lap, measured, 1400), lap, 'acc', 1400)[cue]).toBeUndefined();
    const result = at(0.4);
    expect(result.state.path).toEqual(['root', `${name} zone`, name, 'say phrase']);
    expect(result.events).toEqual([expect.objectContaining({ sentence: `${name}.`, timestamp: 1400 })]);
});

it('previews Steering and its direction child before the actual turn starts', () => {
    const { reference, map } = steeringCornerFixture();
    const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(map, 0, reference);
    engine.setDrivingAggression(100, 0);
    const at = (position: number, now: number) => {
        const sample = { ...sampleAt(position), Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0,
            Physics_local_velocity_x: 10, Physics_local_velocity_y: -4, Physics_local_velocity_z: 0 };
        engine.receiveVision(capture(reference, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(0.24499, 1000).events).toEqual([]);
    expect(at(0.24501, 1001).events).toEqual([expect.objectContaining({ sentence: 'Turn', timestamp: 1001 })]);
    expect(at(0.24502, 1002).events).toHaveLength(1);
    expect(at(0.24502, 1500).events).toHaveLength(1);
    const correction = at(0.24502, 1501);
    expect(correction.state.path).toEqual(['root', 'Steering', 'Stay right a bit', 'say phrase']);
    expect(correction.events.map(({ sentence }) => sentence)).toEqual(['Turn', 'Stay right a bit']);
});

it('previews the Slowly release brake child when reference deceleration is about to ease', () => {
    const { reference, map } = steeringCornerFixture();
    reference.y.forEach((row, i) => {
        row[4] = 0.8;
        row[6] = Math.sqrt(i <= 25 ? 400 : Math.max(100, 320 - 40 * (i - 26))) * 3.6;
    });
    const engine = new PhraseEngine();
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame({ ...sampleAt(position), Physics_brake: 0.8 }), now);
    expect(at(0.24501, 1000).events.map(({ sentence }) => sentence)).toEqual(['Turn']);
    expect(at(0.25499, 1001).events.map(({ sentence }) => sentence)).toEqual(['Turn']);
    expect(at(0.25501, 1002).events.map(({ sentence }) => sentence)).toEqual(['Turn', 'Slowly release brake']);
});

it.each([-4, 8])('previews straight membership for Turn to reference line and its children (offset %s)', (lateral) => {
    const lap = referenceLap(), map = circuitMap(), engine = new PhraseEngine();
    map.centerline_segments = [{ id: 'straight', tags: ['straight'], start_position: 0.4, end_position: 0.5 }];
    engine.receiveMap(map, 0, lap);
    const at = (position: number, now: number) => {
        const sample = sampleAt(position);
        engine.receiveVision(capture(lap, sample, now, lateral), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(0.39499, 1000).state.path).toEqual(['root']);
    const result = at(0.39501, 1001);
    expect(result.state.path).toEqual(['root', 'Turn to reference line', lateral < 0 ? 'Move to the left' : 'Move to the right',
        lateral < 0 ? 'Small movement' : 'Medium movement', 'say phrase']);
    expect(result.events[0].timestamp).toBe(1001);
    expect(result.mapContext.sectionId).toBeUndefined(); // Overtaking still sees the measured position.
});

it.each([false, true])('advances across start/finish (duplicate endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[3] = i >= 1 && i < 10 ? 0.8 : 0; row[4] = 0; });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    expect(referencePositionAhead(sampleAt(0.999), lap)).toBeCloseTo(0.004, 10);
    const engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sampleAt(0.99499)), 1000).state.path).toEqual(['root']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.99501)), 1001).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.9963)), 1002).events[0].sentence).toBe('Brake.');
});

it('uses physical XYZ distance on uneven normalized samples without changing telemetry or the reference', () => {
    const lap = referenceLap();
    // A 3-4-5 incline with uneven normalized spacing; at 14.4 km/h a 500 ms lead lands 2 m along it.
    lap.x = [0, 0.1, 0.7, 0.9];
    lap.y = [[0, 0, 0, 0, 0, 0], [0, 3, 4, 0, 0, 0], [20, 3, 4, 0, 0, 0], [20, 0, 0, 0, 0, 0]];
    const sample = sampleAt(0, 14.4), before = JSON.stringify([sample, lap]);
    expect(referencePositionAhead(sample, lap)).toBeCloseTo(0.04, 10);
    expect(JSON.stringify([sample, lap])).toBe(before);
    expect(referencePositionAhead(sampleAt(0.2, 0), lap)).toBe(0.2);
});

it.each([undefined, NaN, -1])('withholds projection with invalid live speed %s', (speed) => {
    const sample = { ...sampleAt(0.199), Physics_speed_kmh: speed }, lap = referenceLap(), engine = new PhraseEngine();
    expect(referencePositionAhead(sample, lap)).toBeUndefined();
    engine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sample), 1000).events).toEqual([]);
});

it('keeps capture validation on actual progress and real time while previewing reference conditions', () => {
    const lap = referenceLap(), sample = sampleAt(0.5999, 180), now = 1000;
    const view = capture(lap, sample, now);
    view.position!.normalizedPosition = 0.57; // 29.9 m traveled; allowed at this speed, but preview distance is 54.9 m.
    const engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    engine.receiveVision(view, now);
    expect(engine.receiveTelemetry(frame(sample), now).events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    const expired = new PhraseEngine();
    expired.receiveMap(null, 0, lap);
    expired.receiveVision(capture(lap, sample, now - 501), now);
    expect(expired.receiveTelemetry(frame(sample), now).events).toEqual([]);
    expect(engine.evaluate(now + 1501, true).telemetryReady).toBe(false);
});

it.each([['entry', 190], ['exit', 290]] as const)('uses the first progress derivative for brake-zone %s', (transition, start) => {
    const engine = new PhraseEngine(), lap = referenceLap();
    engine.receiveMap(null, 0, lap);
    const at = (t: number) => engine.receiveTelemetry(frame(sampleAt(
        (start + 2 * t + 10 * t ** 2) / 1000, (2 + 20 * t) * 3.6)), 1000 + t * 1000);
    at(0); at(0.2);
    const before = at(0.4);
    // The latest progress rate predicts 196.4 / 296.4 m at t=0.4.
    expect(before.state.path.includes('Brake zone')).toBe(transition === 'exit');
    expect(at(0.43).state.path.includes('Brake zone')).toBe(transition === 'exit');
    // At t=0.65 the linear preview reaches 200.775 / 300.775 m.
    const crossed = at(0.65);
    expect(crossed.state.path.includes('Brake zone')).toBe(transition === 'entry');
    if (transition === 'entry') expect(at(0.7).events.map(({ sentence }) => sentence)).toContain('Brake.');
});
