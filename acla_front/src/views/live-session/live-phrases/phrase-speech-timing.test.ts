import type { StandardTelemetrySample } from '../live-session-types';
import { getPhraseActions, PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine, type PhraseDefinition } from './phrase-engine';
import { steeringCornerFixture } from './test-fixtures';

const frame = (sample: StandardTelemetrySample) => ({
    type: 'frame' as const, sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame' as const, game: 'acc' as const, sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});
const phrase: PhraseDefinition = { id: 'timed', name: 'Timed instruction', description: 'Timing fixture.', holdMs: 0,
    sentence: 'A longer instruction.', conditions: [new PhraseCondition('speed', '>=', 30)] };
const prepare = (definition = phrase, durationMs = 2500) => {
    const engine = new PhraseEngine([definition]);
    const instance = getPhraseActions(definition)[0].phrases[0].createInstance();
    engine.setSpeechDurations(new Map([[instance.cacheKey, durationMs]]));
    return engine;
};

it('waits for playback to report the first and last spoken sounds', () => {
    const engine = prepare();
    const triggered = engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    expect(triggered.events[0].speech).toEqual({ durationMs: 2500, preparationTimeMs: 2500, reactionAllowanceMs: 0 });
    expect(engine.evaluate(1400).state.current).toBe('say phrase');
    engine.speechStarted(1, 1500);
    engine.speechStarted(1, 1600);
    expect(triggered.events[0].speech?.startedAt).toBeUndefined();
    expect(engine.evaluate(1700).state.current).toBe('say phrase');
    const released = engine.speechFinished(1, 1900);
    expect(released.state.path).toEqual(['root']);
    expect(released.events[0].speech).toMatchObject({ startedAt: 1500, releaseAt: 1900 });
    expect(engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 2000).events).toHaveLength(1);
});

it('releases on playback completion even when the clip is shorter than 200 ms', () => {
    const engine = prepare(phrase, 80);
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    engine.speechStarted(1, 1100);
    expect(engine.speechFinished(1, 1180).state.path).toEqual(['root']);
});

it.each([false, true])('releases missing, silent or rejected playback without waiting for onset (cancelled: %s)', (cancelled) => {
    const engine = prepare();
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    expect(engine.speechFinished(1, 1050, cancelled).state.path).toEqual(['root']);
    expect(engine.speechStarted(1, 1200).events[0].speech?.startedAt).toBeUndefined();
});

it('allows an explicit exit condition to release a phrase before audible onset', () => {
    const engine = prepare({ ...phrase, exitConditions: [new PhraseCondition('speed', '<', 30)] });
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    expect(engine.receiveTelemetry(frame({ Physics_speed_kmh: 10 }), 1050).state.path).toEqual(['root']);
});

it('waits for playback even before audible onset and ignores callbacks from a reset session', () => {
    const engine = prepare();
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    expect(engine.evaluate(9000).state.current).toBe('say phrase');
    engine.reset(9100);
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 9200);
    const current = engine.speechStarted(1, 9300);
    expect(current.events).toHaveLength(1);
    expect(current.events[0].id).toBe(2);
    expect(current.events[0].speech?.startedAt).toBeUndefined();
});

it('does not release audible speech when its scheduling deadline or estimated duration passes', () => {
    const engine = prepare(phrase, 100);
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 50 }), 1000);
    engine.speechStarted(1, 1100);
    expect(engine.evaluate(10000).state.current).toBe('say phrase');
    expect(engine.speechFinished(1, 10001).state.path).toEqual(['root']);
});

it('caps both closure and speech preparation at the fixed three-second forecast', () => {
    const definition = { ...PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'steering')!, preparationTimeMs: 8000 };
    const { map, reference } = steeringCornerFixture();
    const engine = prepare(definition, 8000);
    engine.receiveMap(map, 0, reference);
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36, Physics_brake: 0, Physics_gas: 0 };
    expect(engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.2399 }), 1000).events).toEqual([]);
    const entered = engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.2401 }), 1001);
    expect(entered.events[0]).toMatchObject({ sentence: 'Turn', speech: { preparationTimeMs: 8000 } });
});

it('waits for the one-second Steering parent before applying the child speech preparation time', () => {
    const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'steering')!;
    const { map, reference } = steeringCornerFixture();
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36,
        Physics_brake: 0, Physics_gas: 0, Graphics_normalized_car_position: 0.2601 };
    const short = prepare(definition, 400), long = prepare(definition, 2000);
    for (const engine of [short, long]) engine.receiveMap(map, 0, reference);
    expect(long.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.246 }), 500).events).toHaveLength(0);
    expect(long.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.251 }), 600).state.path).toEqual(['root']);
    expect(short.receiveTelemetry(frame(sample), 1000).events).toHaveLength(0);
    const scheduled = long.receiveTelemetry(frame(sample), 1000);
    expect(scheduled.events).toEqual([expect.objectContaining({ sentence: 'Turn', expiresAt: 3000,
        speech: { durationMs: 2000, preparationTimeMs: 2000, reactionAllowanceMs: 0 } })]);
    long.speechStarted(1, 1100);
    expect(long.evaluate(1300).state.current).toBe('say phrase');
    expect(long.speechFinished(1, 1400).state.path).toEqual(['root', 'Steering']);
});
