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

it.each([400, 2000])('uses the %s ms audio forecast for Steering entry and speaks immediately', (durationMs) => {
    const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'steering')!;
    const { map, reference } = steeringCornerFixture();
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36,
        Physics_brake: 0, Physics_gas: 0 };
    const engine = prepare(definition, durationMs);
    engine.receiveMap(map, 0, reference);
    const entryPosition = 0.27 - durationMs / 100000;
    const waiting = engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: entryPosition - 0.0001 }), 1000);
    expect(waiting.events).toEqual([]);
    expect(waiting.state.path).toEqual(['root']);
    const scheduled = engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: entryPosition + 0.0001 }), 1001);
    expect(scheduled.events).toEqual([expect.objectContaining({ sentence: 'Turn', timestamp: 1001,
        speech: { durationMs, preparationTimeMs: durationMs, reactionAllowanceMs: 0 } })]);
    expect(scheduled.state.path).toEqual(['root', 'Steering', 'say phrase']);
    engine.speechStarted(1, 1100);
    expect(engine.evaluate(1300).state.current).toBe('say phrase');
    expect(engine.speechFinished(1, 1400).state.path).toEqual(['root', 'Steering']);
});

it.each(['brake', 'throttle'] as const)('lets the %s parent admit a phrase using its own longer audio forecast', (pedal) => {
    const definition: PhraseDefinition = { ...phrase, id: `reference-${pedal}`, referenceCue: pedal,
        conditions: [new PhraseCondition('referencePedalZone', '=', pedal)] };
    const { reference } = steeringCornerFixture();
    reference.y.forEach((row, index) => { row[pedal === 'brake' ? 4 : 5] = index >= 21 && index < 40 ? 0.8 : 0; });
    const engine = prepare(definition, 2000);
    engine.receiveMap(null, 0, reference);
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36, Physics_brake: 0.8, Physics_gas: 0.8 };
    const waiting = engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.1799 }), 1000);
    expect(waiting.events).toEqual([]);
    const entered = engine.receiveTelemetry(frame({ ...sample, Graphics_normalized_car_position: 0.1801 }), 1001);
    expect(entered.events).toEqual([expect.objectContaining({ ruleId: definition.id, timestamp: 1001 })]);
    expect(entered.state.current).toBe('say phrase');
});

it('speaks on a brief throttle match inside the audio window before applying the active zone exit', () => {
    const definition: PhraseDefinition = { ...phrase, id: 'reference-throttle', referenceCue: 'throttle',
        conditions: [new PhraseCondition('referencePedalZone', '=', 'throttle')] };
    const { reference } = steeringCornerFixture();
    // Throttle spans 200–210 m; the 2 s forecast ends at 215 m, beyond this zone.
    reference.y.forEach((row, index) => {
        row[5] = index === 21 ? 0.8 : 0;
        row[4] = index >= 22 && index < 40 ? 0.8 : 0;
    });
    const engine = prepare(definition, 2000);
    engine.receiveMap(null, 0, reference);
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36,
        Physics_brake: 0, Physics_gas: 0, Graphics_normalized_car_position: 0.195 };
    const entered = engine.receiveTelemetry(frame(sample), 1000);
    expect(entered.events).toEqual([expect.objectContaining({ ruleId: definition.id, timestamp: 1000 })]);
    expect(entered.state.path).toEqual(['root', 'Throttle zone', definition.name, 'say phrase']);
    expect(engine.speechFinished(entered.events[0].id, 1001).state.path).toEqual(['root']);
});

it('uses only the current frame for entry until the phrase audio duration is prepared', () => {
    const definition = PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'steering')!;
    const { map, reference } = steeringCornerFixture();
    const engine = new PhraseEngine([definition]);
    engine.setSpeechDurations(new Map());
    engine.receiveMap(map, 0, reference);
    const sample = { Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36,
        Physics_brake: 0, Physics_gas: 0, Graphics_normalized_car_position: 0.2601 };
    expect(engine.receiveTelemetry(frame(sample), 1000).state.path).toEqual(['root']);
    const instance = getPhraseActions(definition)[0].phrases[0].createInstance();
    engine.setSpeechDurations(new Map([[instance.cacheKey, 2000]]));
    expect(engine.receiveTelemetry(frame(sample), 1001).events)
        .toEqual([expect.objectContaining({ sentence: 'Turn', timestamp: 1001 })]);
});
