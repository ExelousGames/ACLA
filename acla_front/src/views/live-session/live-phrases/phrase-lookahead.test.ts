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

it.each([36, 72, 180])('enters Brake and speaks its child one second early at %s km/h without an entry wait', (speed) => {
    const lap = referenceLap(), engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    const lead = speed / 3.6 / 1000;
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

it('keeps the parent preparation at one second regardless of the child clip length', () => {
    const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'brake')!;
    const measure = jest.spyOn(definition.conditions[0], 'evaluate');
    const instance = getPhraseActions(definition)[0].phrases[0].createInstance();
    const short = new PhraseEngine([definition]), long = new PhraseEngine([definition]);
    short.setSpeechDurations(new Map([[instance.cacheKey, 200]]));
    long.setSpeechDurations(new Map([[instance.cacheKey, 2000]]));
    const lap = referenceLap();
    for (const engine of [short, long]) engine.receiveMap(null, 0, lap);
    measure.mockClear();
    for (const engine of [short, long]) {
        expect(engine.receiveTelemetry(frame(sampleAt(0.182)), 1000).state.path).toEqual(['root']);
    }
    expect(measure).not.toHaveBeenCalled();
    const waiting = short.receiveTelemetry(frame(sampleAt(0.1901)), 1100);
    expect(waiting.state.path).toEqual(['root', 'Brake zone']);
    const entered = long.receiveTelemetry(frame(sampleAt(0.1901)), 1100);
    expect(entered.state.path).toEqual(['root', 'Brake zone']);
    // Brake's own closure also has a one-second preparation window.
    const spoken = long.receiveTelemetry(frame(sampleAt(0.1913)), 1200);
    expect(spoken.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(spoken.events[0].speech).toMatchObject({ durationMs: 2000, preparationTimeMs: 2000 });
    expect(short.receiveTelemetry(frame(sampleAt(0.1913)), 1200).events).toHaveLength(0);
    expect(short.receiveTelemetry(frame(sampleAt(0.1993)), 1300).events).toHaveLength(1);
    measure.mockRestore();
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
        expect(at(0.58999, 1000).state.path).toEqual(['root']);
        expect(at(0.59001, 1001).state.path).toEqual(['root', 'Throttle zone']);
        expect(at(boundary - 0.01001, 1002).events).toEqual([]);
        const result = at(boundary - 0.00999, 1003);
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
    // The derivative becomes available at 200 ms; its one-second preparation window reaches the pedal threshold.
    // Measured pressure is 40%, but the one-second preparation window reaches the pedal threshold.
    const measured = { ...sampleAt(start + 0.002), [field]: 0.4 };
    expect(readReferenceGuidance(measured, capture(lap, measured, 1400), lap, 'acc', 1400)[cue]).toBeUndefined();
    const result = at(0.2);
    expect(result.state.path).toEqual(['root', `${name} zone`, name, 'say phrase']);
    expect(result.events).toEqual([expect.objectContaining({ sentence: `${name}.`, timestamp: 1200 })]);
});

it('previews Steering and its direction child at the centerline corner start', () => {
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
    expect(at(0.26001, 1001).events).toEqual([expect.objectContaining({ sentence: 'Turn', timestamp: 1001 })]);
    expect(at(0.26002, 1002).events).toHaveLength(1);
    expect(at(0.26002, 1500).events).toHaveLength(1);
    const correction = at(0.26002, 1501);
    expect(correction.state.path).toEqual(['root', 'Steering', 'Stay right a bit', 'say phrase']);
    expect(correction.events.map(({ sentence }) => sentence)).toEqual(['Turn', 'Stay right a bit']);
});

describe.each([0, 0.7])('Steering exit at the current position (lap offset %s)', (offset) => {
    it.each([0.35, 0.45])('waits for the captured range end with corner end %s', (cornerEnd) => {
        const { reference, map } = steeringCornerFixture();
        const positionAt = (position: number) => (position + offset) % 1;
        const points = reference.x.map((position, index) => ({ position: positionAt(position), row: reference.y[index] }))
            .sort((a, b) => a.position - b.position);
        reference.x = points.map(({ position }) => position);
        reference.y = points.map(({ row }) => row);
        map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point,
            normalized_position: positionAt(point.normalized_position) })).sort((a, b) => a.normalized_position - b.normalized_position);
        map.centerline_segments![0].start_position = positionAt(0.27);
        map.centerline_segments![0].end_position = positionAt(cornerEnd);
        const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, reference);
        const at = (position: number, now: number) => engine.receiveTelemetry(frame(sampleAt(positionAt(position))), now);

        expect(at(0.26001, 1000).state.path).toContain('Steering');
        expect(at(0.26001, 1001).state.path).toContain('Steering'); // Current position is still before the corner.
        expect(at(0.28, 1100).state.path).toContain('Steering');
        expect(at(0.2799, 1150).state.path).toContain('Steering'); // Small backward movement stays inside the range.
        if (offset) expect(at(0.30, 1200).state.path).toContain('Steering'); // Cross start/finish.
        const end = Math.min(cornerEnd, 0.41);
        expect(at(end - 0.005, 1300).state.path).toContain('Steering'); // Lookahead is already past the end.
        expect(engine.evaluate(3000).state.path).toContain('Steering'); // Stale telemetry cannot cause an exit.
        expect(at(end - 0.00001, 3100).state.path).toContain('Steering');
        expect(at(end + 1e-10, 3200).state.path).toEqual(['root']);
    });
});

it('exits an anticipated Steering range when current telemetry skips past the whole turn', () => {
    const { reference, map } = steeringCornerFixture();
    const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(map, 0, reference);
    expect(engine.receiveTelemetry(frame(sampleAt(0.26001)), 1000).state.path).toContain('Steering');
    expect(engine.receiveTelemetry(frame(sampleAt(0.42)), 1100).state.path).toEqual(['root']);
});

describe('Steering exit on positive reference acceleration', () => {
    it.each([36, 72, 180])('exits 200 ms before the reference gains speed at %s km/h', (speed) => {
        const { reference, map } = steeringCornerFixture();
        reference.y.forEach((row, i) => { row[6] = 36 + Math.max(0, i - 30); });
        const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'steering')!;
        const engine = new PhraseEngine([definition]);
        // A longer speech lead must not extend the exit's fixed 200 ms window.
        const instance = getPhraseActions(definition)[0].phrases[0].createInstance();
        engine.setSpeechDurations(new Map([[instance.cacheKey, 2000]]));
        engine.receiveMap(map, 0, reference);
        const at = (position: number, now: number) => engine.receiveTelemetry(frame(sampleAt(position, speed)), now);
        const lead = speed / 3.6 * 0.2 / 1000;

        const entered = at(0.27, 1000);
        expect(entered.state.path).toContain('Steering');
        expect(entered.events.map(({ sentence }) => sentence)).toEqual(['Turn']);
        engine.speechStarted(entered.events[0].id, 1000);
        const before = at(0.3 - lead - 0.00001, 1001);
        expect(before.state.path).toContain('Steering');
        const exit = before.root.children.find(({ id }) => id === 'reference-steering')!.children
            .find(({ name }) => name === 'exit to root')!;
        expect(exit.conditions).toContainEqual(expect.objectContaining({
            description: 'Reference acceleration is positive within the next 200 ms', connector: 'or', conditionFit: false,
        }));
        expect(at(0.3 - lead + 0.00001, 1002).state.path).toEqual(['root']);
        const stillAccelerating = at(0.301, 1003);
        expect(stillAccelerating.state.path).toEqual(['root']);
        expect(stillAccelerating.events.map(({ id }) => id)).toEqual([entered.events[0].id]);
    });

    it('checks inside the 200 ms window when acceleration has ended at its endpoint', () => {
        const { reference, map } = steeringCornerFixture();
        reference.y.forEach((row, i) => { row[6] = i <= 30 ? 36 : 40; });
        const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(0.27, 360)), 1000).state.path).toContain('Steering');
        // At 100 m/s, the 100 ms frame is inside the rise and the 200 ms frame is on the plateau.
        expect(readReferenceGuidance(sampleAt(0.315), null, reference, 'acc', 1001).acceleration).toBe(0);
        expect(engine.receiveTelemetry(frame(sampleAt(0.295, 360)), 1001).state.path).toEqual(['root']);
    });

    it('uses the shared predicted position when a measured progress rate is available', () => {
        const { reference, map } = steeringCornerFixture();
        reference.y.forEach((row, i) => { row[6] = 36 + Math.max(0, i - 30); });
        const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(0.293)), 1000).state.path).toContain('Steering');
        expect(engine.receiveTelemetry(frame(sampleAt(0.295)), 1100).state.path).toContain('Steering');
        // The measured rate reaches 0.301 in 200 ms; the speed-only fallback would stop at 0.299.
        expect(engine.receiveTelemetry(frame(sampleAt(0.297)), 1200).state.path).toEqual(['root']);
    });

    it('looks across start/finish for positive reference acceleration', () => {
        const { reference, map } = steeringCornerFixture();
        reference.y.forEach((row, i) => { row[6] = 36 + Math.max(0, i - 30); });
        const positionAt = (position: number) => (position + 0.701) % 1;
        const points = reference.x.map((position, index) => ({ position: positionAt(position), row: reference.y[index] }))
            .sort((a, b) => a.position - b.position);
        reference.x = points.map(({ position }) => position);
        reference.y = points.map(({ row }) => row);
        map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point,
            normalized_position: positionAt(point.normalized_position) })).sort((a, b) => a.normalized_position - b.normalized_position);
        map.centerline_segments![0].start_position = positionAt(0.27);
        map.centerline_segments![0].end_position = positionAt(0.45);
        const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(positionAt(0.27))), 1000).state.path).toContain('Steering');
        expect(engine.receiveTelemetry(frame(sampleAt(positionAt(0.2979))), 1001).state.path).toContain('Steering');
        expect(engine.receiveTelemetry(frame(sampleAt(positionAt(0.2985))), 1002).state.path).toEqual(['root']);
    });

    it.each(['flat', 'easing deceleration', 'missing', 'invalid'])('keeps Steering until range exit with %s reference speed', (profile) => {
        const { reference, map } = steeringCornerFixture();
        if (profile === 'easing deceleration') reference.y.forEach((row, i) => { row[6] = 36 + 100 / (i + 1); });
        if (profile === 'missing') {
            reference.target_features.pop();
            reference.y.forEach((row) => row.pop());
        }
        if (profile === 'invalid') reference.y[31][6] = NaN;
        const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(0.27)), 1000).state.path).toContain('Steering');
        expect(engine.receiveTelemetry(frame(sampleAt(0.299)), 1001).state.path).toContain('Steering');
        expect(engine.receiveTelemetry(frame(sampleAt(0.42)), 1002).state.path).toEqual(['root']);
    });
});

it('previews the Slowly release brake child when reference deceleration is about to ease', () => {
    const { reference, map } = steeringCornerFixture();
    reference.y.forEach((row, i) => {
        row[4] = 0.8;
        row[6] = Math.sqrt(i <= 27 ? 400 : Math.max(100, 320 - 40 * (i - 28))) * 3.6;
    });
    const engine = new PhraseEngine();
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame({ ...sampleAt(position), Physics_brake: 0.8 }), now);
    expect(at(0.26001, 1000).events.map(({ sentence }) => sentence)).toEqual(['Turn']);
    expect(at(0.26999, 1001).events.map(({ sentence }) => sentence)).toEqual(['Turn']);
    expect(at(0.27001, 1002).events.map(({ sentence }) => sentence)).toEqual(['Turn', 'Slowly release brake']);
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
    expect(at(0.38999, 1000).state.path).toEqual(['root']);
    const result = at(0.39001, 1001);
    expect(result.state.path).toEqual(['root', 'Turn to reference line', lateral < 0 ? 'Move to the left' : 'Move to the right',
        lateral < 0 ? 'Small movement' : 'Medium movement', 'say phrase']);
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
    expect(engine.receiveTelemetry(frame(sampleAt(0.99001)), 1001).state.path).toEqual(['root', 'Brake zone']);
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

it.each([['entry', 190], ['reference end', 290]] as const)('uses predicted brake-zone entry without a reference-end exit (%s)', (transition, start) => {
    const engine = new PhraseEngine(), lap = referenceLap();
    engine.receiveMap(null, 0, lap);
    const at = (t: number) => engine.receiveTelemetry(frame(sampleAt(
        (start + 2 * t + 10 * t ** 2) / 1000, (2 + 20 * t) * 3.6)), 1000 + t * 1000);
    at(0); at(0.2);
    const before = at(0.4);
    // The latest progress rate predicts 196.4 / 296.4 m at t=0.4.
    expect(before.state.path.includes('Brake zone')).toBe(transition !== 'entry');
    expect(at(0.43).state.path.includes('Brake zone')).toBe(transition !== 'entry');
    // At t=0.65 the linear preview reaches 200.775 / 300.775 m.
    const crossed = at(0.65);
    expect(crossed.state.path).toContain('Brake zone');
    if (transition === 'entry') expect(at(0.7).events.map(({ sentence }) => sentence)).toContain('Brake.');
    else {
        expect(at(0.9).state.path).toContain('Brake zone'); // Current position is still 299.9 m.
        expect(at(0.91).state.path).toContain('Brake zone'); // Crossing the 300 m reference end does not exit.
    }
});

it.each([false, true])('keeps early brake entry across the lap seam and reference end (skip interval: %s)', (skip) => {
    const lap = referenceLap(), engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'brake'));
    lap.y.forEach((row, i) => { row[3] = i >= 99 || i < 2 ? 0.8 : 0; row[4] = 0; });
    engine.receiveMap(null, 0, lap);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame({ ...sampleAt(position), Physics_brake: 0.8 }), now);
    expect(at(0.9701, 1000).state.path).toEqual(['root', 'Brake zone']);
    // Frozen measured progress must not cancel the anticipated entry or manufacture an exit.
    expect(at(0.9701, 1100).state.path).toEqual(['root', 'Brake zone']);
    if (!skip) expect(at(0.019999, 2000).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.02, 2100).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.9701, 3000).state.path).toEqual(['root', 'Brake zone']);
});
