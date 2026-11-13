import type { StandardTelemetrySample } from '../live-session-types';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

function setup(game: 'acc' | 'iracing' = 'acc', configure?: (reference: ReturnType<typeof steeringCornerFixture>['reference']) => void,
    aggression = 100) {
    const { map, reference } = steeringCornerFixture(game);
    const steering = [0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.35, 0.25, 0.15, 0.08, 0.02];
    reference.y.forEach((row, index) => { row[3] = steering[index - 27] ?? 0; });
    configure?.(reference);
    const engine = new PhraseEngine();
    engine.setDrivingAggression(aggression, 0);
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey, 200])));
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number, options: { errorDegrees?: number; vision?: boolean; brake?: number; speed?: number } = {}) => {
        // The fixture's centered tangent turns 12 degrees per reference sample.
        const angle = (12 * (position * 100 - 25.5) + (options.errorDegrees ?? 0)) * Math.PI / 180;
        const sample: StandardTelemetrySample = { Static_track: 'test', Static_car_model: 'car',
            Graphics_normalized_car_position: position, Physics_speed_kmh: options.speed ?? 36,
            Physics_brake: options.brake ?? 0, Physics_gas: 0, Physics_steer_angle: 0.3,
            Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0,
            Physics_local_velocity_x: 10 * Math.cos(angle),
            Physics_local_velocity_y: game === 'acc' ? 10 * Math.sin(angle) : 0,
            Physics_local_velocity_z: game === 'iracing' ? 10 * Math.sin(angle) : 0 };
        const vision: TrackVisionDetection = { capturedAt: now, width: 100, height: 100,
            detections: {}, reconstruction: null, geometry: null,
            telemetry: { ...createLiveTelemetryStore().getSnapshot(), game, receivedAt: now, currentTelemetry: sample,
                telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0 },
            position: { coordinateFrame: 'car_local', inferenceMs: 10, referenceDelta: [0, 0, 0],
                normalizedPosition: position, referenceLap: reference, references: { reference: { x: 0, y: 0, z: 0 } } } };
        engine.receiveVision(options.vision === false ? null : vision, now);
        const frame: LiveTelemetryFrameEvent = { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
            sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
            update: { type: 'frame', game, sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
        return engine.receiveTelemetry(frame, now);
    };
    return { engine, at };
}

it.each([0, 50, 99, 99.9, 100])('keeps the reference corner visible and gates directional speech at %s%%', (aggression) => {
    const { engine, at } = setup('acc', undefined, aggression);
    const entry = at(0.27, 1000);
    expect(entry.steeringCorner?.sectionId).toBe('turn-1');
    expect(entry.events.map(({ sentence }) => sentence)).toEqual(aggression === 100 ? ['Turn in'] : []);
    if (aggression === 100) engine.speechFinished(1, 1200);
    const middle = at(0.31, 1300);
    expect(middle.steeringCorner?.sectionId).toBe('turn-1');
    expect(middle.events.map(({ sentence }) => sentence)).toEqual(aggression === 100 ? ['Turn in', 'Hold line'] : []);
    if (aggression === 100) engine.speechFinished(2, 1500);
    at(0.36, 1600);
    const exit = at(0.36, 1601);
    expect(exit.events.map(({ sentence }) => sentence)).toEqual(aggression === 100 ? ['Turn in', 'Hold line', 'Unwind'] : []);
    if (aggression < 100) {
        expect(exit.steeringCorner?.sectionId).toBe('turn-1');
    }
});

it('clears pending direction speech when leaving 100% without losing the reference corner', () => {
    const { engine, at } = setup();
    expect(at(0.27, 1000).events).toHaveLength(1);
    expect(engine.setDrivingAggression(50, 1100).events).toEqual([]);
    const quiet = at(0.31, 1200);
    expect(quiet.steeringCorner?.sectionId).toBe('turn-1');
    expect(quiet.events).toEqual([]);
    engine.speechFinished(1, 1300);
    expect(at(0.32, 1400).events).toEqual([]);
    engine.setDrivingAggression(100, 1500);
    expect(at(0.32, 1600).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
});

it.each(['acc', 'iracing'] as const)('coaches turn-in, a loaded arc and coasting unwind once in %s', (game) => {
    const { engine, at } = setup(game);
    expect(at(0.27, 1000).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(1, 1200);
    const hold = at(0.31, 1300);
    expect(hold.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Hold line']);
    expect(at(0.31, 1400).events).toEqual(hold.events);
    engine.speechFinished(2, 1500);
    at(0.36, 1600);
    const exit = at(0.36, 1601);
    expect(exit.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Hold line', 'Unwind']);
    engine.speechFinished(3, 1801);
    expect(at(0.37, 1900).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Hold line', 'Unwind']);
});

it('keeps direction deviations silent after Hold line has spoken', () => {
    const { engine, at } = setup();
    engine.setDrivingAggression(100, 0);
    at(0.27, 1000);
    engine.speechFinished(1, 1200);
    expect(at(0.31, 1300).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Hold line']);
    engine.speechFinished(2, 1500);
    for (const now of [1600, 1800, 2000, 2100]) at(0.31, now, { errorDegrees: -40 });
    const snapshot = at(0.31, 2200, { errorDegrees: -40 });
    expect(snapshot.referenceSteeringDirection).toBe('right');
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Hold line']);
});

it.each([{ vision: false }, { errorDegrees: -40 }])('withholds Hold line without confirmed alignment: %j', (options) => {
    const { engine, at } = setup();
    at(0.27, 1000, options);
    engine.speechFinished(1, 1200);
    expect(at(0.31, 1300, options).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
});

it('skips a passed loaded arc while the short turn-in clip is still playing', () => {
    const { engine, at } = setup();
    at(0.27, 1000);
    expect(at(0.31, 1100).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    at(0.36, 1190);
    engine.speechFinished(1, 1200);
    const exit = at(0.36, 1210);
    expect(exit.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Unwind']);
});

it('keeps exit advice silent while the driver is still braking', () => {
    const { at } = setup();
    expect(at(0.36, 1000, { brake: 0.5 }).events.some(({ sentence }) => sentence === 'Unwind')).toBe(false);
    expect(at(0.36, 1100, { brake: 0.5 }).events.some(({ sentence }) => sentence === 'Unwind')).toBe(false);
});

it('does not repeat Unwind or Turn in after a plateau within the same exit', () => {
    const { engine, at } = setup('acc', (reference) => {
        const steering = [0.1, 0.2, 0.4, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.14, 0.08, 0.02];
        reference.y.forEach((row, index) => { row[3] = steering[index - 27] ?? 0; });
    });
    expect(at(0.325, 1000).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
    engine.speechFinished(1, 1200);
    expect(at(0.345, 1300).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
    expect(at(0.351, 1900).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
    expect(at(0.375, 2100).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
});

it('rearms after measured progress leaves the turn without delaying the next exit by two seconds', () => {
    const { engine, at } = setup();
    expect(at(0.36, 1000).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
    engine.speechFinished(1, 1200);
    at(0.43, 1300);
    expect(at(0.27, 1400).events.map(({ sentence }) => sentence)).toEqual(['Unwind', 'Turn in']);
    engine.speechFinished(2, 1600);
    at(0.36, 1700);
    expect(at(0.36, 1701).events.map(({ sentence }) => sentence)).toEqual(['Unwind', 'Turn in', 'Unwind']);
});

it('does not infer Unwind from acceleration on a constant-steering arc', () => {
    const { at } = setup('acc', (reference) => reference.y.forEach((row, index) => {
        row[3] = index >= 27 && index <= 40 ? 0.4 : 0;
        row[6] = index < 27 ? 36 : 36 + (index - 27);
    }));
    expect(at(0.31, 1000).events.some(({ sentence }) => sentence === 'Unwind')).toBe(false);
    expect(at(0.32, 1100).events.some(({ sentence }) => sentence === 'Unwind')).toBe(false);
});

it('prepares every corner cue as a short, fixed speech clip', () => {
    const cues = getPhraseSpeechInstances(PHRASE_DEFINITIONS.filter(({ referenceCue }) =>
        referenceCue === 'steering' || referenceCue === 'cornerExit' || referenceCue === 'easeBrake'));
    expect(cues.map(({ speech }) => speech.text)).toEqual(expect.arrayContaining([
        'Turn in', 'Ease brake', 'Hold line', 'Unwind', 'Set up left', 'Set up right', 'Set up center',
    ]));
    for (const sentence of ['More right', 'More left']) {
        expect(cues.map(({ speech }) => speech.text)).not.toContain(sentence);
    }
    expect(cues.map(({ speech }) => speech.text)).not.toContain('Set up next corner');
    expect(cues.every(({ speech, id }) => speech.text.split(/\s+/).length <= (id.startsWith('reference-consecutive-corners-') ? 3 : 2))).toBe(true);
});
