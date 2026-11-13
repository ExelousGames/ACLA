import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { getPhraseActions, PhraseEngine, PHRASE_DEFINITIONS } from './phrase-engine';
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

it.each([36, 72, 180])('speaks Brake one second early at %s km/h without a parent entry wait', (speed) => {
    const lap = referenceLap(), engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    const lead = speed / 3.6 / 1000;
    const before = sampleAt(0.2 - lead - 0.00001, speed);
    expect(engine.receiveTelemetry(frame(before), 1199).state.path).toEqual(['root']);
    // Waiting while still before the predicted boundary must not manufacture an entry.
    expect(engine.evaluate(1200, true).state.path).toEqual(['root']);
    const entry = sampleAt(0.2 - lead + 0.00001, speed);
    expect(engine.receiveTelemetry(frame(entry), 1201).state.path).toEqual(['root']);
    // Reference pressure reaches 10% at 201.25 m; the child previews that independently of zone entry.
    expect(engine.receiveTelemetry(frame(sampleAt(0.20125 - lead - 0.00001, speed)), 1202).events).toEqual([]);
    const speaking = sampleAt(0.20125 - lead + 0.00001, speed);
    expect(readReferenceGuidance(speaking, null, lap, 'acc', 1203).brake).toBeUndefined();
    const result = engine.receiveTelemetry(frame(speaking), 1203);
    expect(result.state.path).toEqual(['root', 'Track guide', 'Brake']);
    expect(result.events).toEqual([expect.objectContaining({ sentence: 'Brake.', timestamp: 1203, expiresAt: 2703 })]);
    expect(engine.receiveTelemetry(frame(speaking), 1403).events).toEqual(result.events);
});

it.each([200, 2000])('measures Brake independently using its %s ms clip duration', (durationMs) => {
    const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'brake')!;
    const measure = jest.spyOn(definition.conditions[0], 'evaluate');
    const instance = getPhraseActions(definition)[0].phrase.createInstance();
    try {
        const engine = new PhraseEngine([definition]);
        engine.setSpeechDurations(new Map([[instance.cacheKey, durationMs]]));
        engine.receiveMap(null, 0, referenceLap());
        const lead = durationMs / 100000;
        measure.mockClear();
        expect(engine.receiveTelemetry(frame(sampleAt(0.2 - lead - 0.00001)), 1000).state.path).toEqual(['root']);
        expect(measure).toHaveBeenCalled();
        expect(engine.receiveTelemetry(frame(sampleAt(0.2 - lead + 0.00001)), 1001).state.path).toEqual(['root']);
        expect(engine.receiveTelemetry(frame(sampleAt(0.20125 - lead - 0.00001)), 1002).events).toEqual([]);
        const spoken = engine.receiveTelemetry(frame(sampleAt(0.20125 - lead + 0.00001)), 1003);
        expect(spoken.state.path).toEqual(['root', 'Track guide', 'Brake']);
        expect(spoken.events).toEqual([expect.objectContaining({ timestamp: 1003,
            speech: { durationMs, preparationTimeMs: durationMs, reactionAllowanceMs: 0 } })]);
    } finally { measure.mockRestore(); }
});

it.each([['Throttle', 0, 0.601875, 'Squeeze gas.']] as const)(
    'previews %s directly from its own preparation window', (name, pedal, boundary, sentence) => {
        const lap = referenceLap(), engine = new PhraseEngine();
        engine.receiveMap(null, 0, lap);
        const at = (position: number, now: number) => {
            const sample = { ...sampleAt(position), Physics_gas: pedal };
            engine.receiveVision(capture(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        expect(at(0.58999, 1000).state.path).toEqual(['root']);
        expect(at(0.59001, 1001).state.path).toEqual(['root']);
        expect(at(boundary - 0.01001, 1002).events).toEqual([]);
        const result = at(boundary - 0.00999, 1003);
        expect(result.state.path).toEqual(['root', 'Track guide', name]);
        expect(result.events).toEqual([expect.objectContaining({ sentence, timestamp: 1003 })]);
    });

it.each([
    ['brake', 'Physics_brake', 0.215, 'Brake', 'Brake.'],
    ['throttle', 'Physics_gas', 0.65, 'Throttle', 'Squeeze gas.'],
] as const)('uses intended %s pressure to enter its phrase before the measured pedal crosses the threshold', (cue, field, start, name, sentence) => {
    const lap = referenceLap(), engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === cue));
    engine.receiveMap(null, 0, lap);
    const at = (t: number) => {
        const now = 1000 + t * 1000;
        const sample = { ...sampleAt(start + 0.01 * t), [field]: 0.5 - 0.5 * t };
        engine.receiveVision(capture(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(0).events).toEqual([]);
    // The derivative becomes available at 200 ms; its one-second preparation window reaches the pedal threshold.
    // Measured pressure is 40%, but the one-second preparation window reaches the pedal threshold.
    const measured = { ...sampleAt(start + 0.002), [field]: 0.4 };
    expect(readReferenceGuidance(measured, capture(lap, measured, 1400), lap, 'acc', 1400)[cue]).toBeUndefined();
    const result = at(0.2);
    expect(result.state.path).toEqual(['root', 'Track guide', name]);
    expect(result.events).toEqual([expect.objectContaining({ sentence, timestamp: 1200 })]);
});

it('previews Turn in without announcing a direction correction at the centerline corner start', () => {
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
    expect(at(0.25999, 1000).events).toEqual([]);
    expect(at(0.26001, 1001).events).toEqual([expect.objectContaining({ sentence: 'Turn in', timestamp: 1001 })]);
    expect(at(0.26002, 1002).events).toHaveLength(1);
    expect(at(0.26002, 1500).events).toHaveLength(1);
    const snapshot = at(0.26002, 1501);
    expect(snapshot.referenceSteeringDirection).toBe('right');
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
});

it('previews Ease brake when cornering still requires slowing to the reference', () => {
    const { reference, map } = steeringCornerFixture();
    reference.y.forEach((row, i) => {
        row[4] = 0.8;
        row[6] = Math.sqrt(i <= 27 ? 400 : Math.max(100, 320 - 40 * (i - 28))) * 3.6;
    });
    const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ id }) => id === 'reference-steering-release'));
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame({ ...sampleAt(position, 72), Physics_brake: 0.8 }), now);
    expect(at(0.24001, 1000).events).toEqual([]);
    expect(at(0.24999, 1001).events).toEqual([]);
    expect(at(0.25001, 1002).events.map(({ sentence }) => sentence)).toEqual(['Ease brake']);
});

it.each([-4, 8])('previews straight membership independently for movement cues (offset %s)', (lateral) => {
    const lap = referenceLap(), map = circuitMap(), engine = new PhraseEngine();
    engine.setDrivingAggression(100, 0);
    map.centerline_segments = [{ id: 'straight', tags: ['straight'], start_position: 0.4, end_position: 0.5 }];
    engine.receiveMap(map, 0, lap);
    const at = (position: number, now: number) => {
        const sample = sampleAt(position);
        engine.receiveVision(capture(lap, sample, now, lateral), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(0.38999, 1000).state.path).toEqual(['root']);
    const result = at(0.39001, 1001);
    expect(result.state.path).toEqual(['root', 'Track guide', lateral < 0 ? 'Small movement · left' : 'Medium movement · right']);
    expect(result.events[0].timestamp).toBe(1001);
    expect(result.mapContext.sectionId).toBeUndefined(); // Overtaking still sees the measured position.
});

it.each([false, true])('advances across start/finish (duplicate endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[3] = i >= 1 && i < 10 ? 0.8 : 0; row[4] = 0; });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    expect(referencePositionAhead(sampleAt(0.999), lap)).toBeCloseTo(0.029, 10);
    const engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sampleAt(0.98999)), 1000).state.path).toEqual(['root']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.99001)), 1001).state.path).toEqual(['root']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.9913)), 1002).events[0].sentence).toBe('Brake.');
});

it('uses physical XYZ distance on uneven normalized samples without changing telemetry or the reference', () => {
    const lap = referenceLap();
    // A 3-4-5 incline with uneven normalized spacing; at 14.4 km/h the three-second forecast travels 12 m.
    lap.x = [0, 0.1, 0.7, 0.9];
    lap.y = [[0, 0, 0, 0, 0, 0], [0, 3, 4, 0, 0, 0], [20, 3, 4, 0, 0, 0], [20, 0, 0, 0, 0, 0]];
    const sample = sampleAt(0, 14.4), before = JSON.stringify([sample, lap]);
    expect(referencePositionAhead(sample, lap)).toBeCloseTo(0.31, 10);
    expect(JSON.stringify([sample, lap])).toBe(before);
    expect(referencePositionAhead(sampleAt(0.2, 0), lap)).toBe(0.2);
});

it.each([undefined, NaN, -1])('withholds projection with invalid live speed %s', (speed) => {
    const sample = { ...sampleAt(0.199), Physics_speed_kmh: speed }, lap = referenceLap(), engine = new PhraseEngine();
    expect(referencePositionAhead(sample, lap)).toBeUndefined();
    engine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sample), 1000).events).toEqual([]);
});

it('previews throttle with stale position captures while still requiring fresh telemetry', () => {
    const lap = referenceLap(), sample = sampleAt(0.5999, 180), now = 1000;
    const view = capture(lap, sample, now);
    view.position!.normalizedPosition = 0.57; // 29.9 m traveled; allowed at this speed, but preview distance is 54.9 m.
    const engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    engine.receiveVision(view, now);
    expect(engine.receiveTelemetry(frame(sample), now).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    const expired = new PhraseEngine();
    expired.receiveMap(null, 0, lap);
    expired.receiveVision(capture(lap, sample, now - 501), now);
    expect(expired.receiveTelemetry(frame(sample), now).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    expect(engine.evaluate(now + 1501, true).telemetryReady).toBe(false);
});
