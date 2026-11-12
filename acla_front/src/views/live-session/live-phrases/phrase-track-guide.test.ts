import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { getPhraseActions, PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine, type PhraseDefinition } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});
const sample = (position: number, patch: StandardTelemetrySample = {}): StandardTelemetrySample => ({
    Static_track: 'test', Static_car_model: 'car', Physics_speed_kmh: 36, Physics_brake: 0,
    Physics_gas: 0, Graphics_normalized_car_position: position, ...patch,
});
const definition = (id: string) => PHRASE_DEFINITIONS.find((phrase) => phrase.id === id)!;
const prepare = (engine: PhraseEngine, phrases: readonly PhraseDefinition[], durations: Record<string, number>) => {
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances(phrases).map((instance) => [instance.cacheKey,
        durations[instance.id.slice(0, instance.id.lastIndexOf(':say:'))] ?? 200])));
};

it('places every reference phrase directly inside the shared Track guide closure', () => {
    const snapshot = new PhraseEngine().evaluate(0);
    const guide = snapshot.root.children.find(({ id }) => id === 'track-guide')!;
    expect(guide.name).toBe('Track guide');
    expect(guide.children.filter(({ kind }) => kind === 'closure')).toEqual([]);
    expect(guide.children.filter(({ kind }) => kind === 'phrase'))
        .toEqual(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue).flatMap((phrase) =>
            getPhraseActions(phrase).map((action) => expect.objectContaining({
                id: action.id, name: phrase.name, description: action.sentence, children: [],
            }))));
    const names = [snapshot.root, ...snapshot.root.children, ...guide.children].map(({ name }) => name);
    for (const oldGroup of ['Brake zone', 'Throttle zone', 'Steering', 'Corner exit', 'Turn to reference line']) {
        expect(names).not.toContain(oldGroup);
    }
    for (const phrase of PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue)) {
        expect(getPhraseActions(phrase)).toHaveLength(1);
    }
});

it('keeps OR entry conditions grouped with additional phrase conditions in snapshots', () => {
    const phrase: PhraseDefinition = { ...definition('reference-throttle'), holdMs: 0,
        conditions: [new PhraseCondition('speed', '>=', 100), new PhraseCondition('speed', '<=', 40, undefined, 'or')],
        actionConditions: [new PhraseCondition('speed', '>=', 30)] };
    const engine = new PhraseEngine([phrase]);
    const unevaluated = engine.state.snapshot().children[0].children[0];
    expect(unevaluated).toMatchObject({ kind: 'phrase', children: [], conditions: [
        { connector: 'and', conditionFit: null, conditions: [
            { conditionFit: null }, { connector: 'or', conditionFit: null },
        ] },
        { connector: 'and', conditionFit: null, conditions: [{ conditionFit: null }] },
    ] });
    [20, 36].forEach((speed, index) => {
        const snapshot = engine.receiveTelemetry(frame(sample(0.15, { Physics_speed_kmh: speed })), 1000 + index * 100);
        const node = snapshot.root.children[0].children[0];
        expect(node.conditions).toMatchObject([
            { connector: 'and', conditionFit: true, conditions: [
                { conditionFit: false }, { connector: 'or', conditionFit: true },
            ] },
            { connector: 'and', conditionFit: speed >= 30, conditions: [{ conditionFit: speed >= 30 }] },
        ]);
        expect(snapshot.events).toHaveLength(index);
    });
    expect(unevaluated.conditions.every(({ conditionFit }) => conditionFit === null)).toBe(true);
});

it('uses the next cue audio forecast after Brake finishes without waiting for the old brake-zone exit', () => {
    const { map, reference } = steeringCornerFixture();
    reference.y.forEach((row, index) => { row[4] = index >= 10 && index < 20 ? 0.8 : 0; });
    const brake: PhraseDefinition = { ...definition('reference-brake'),
        conditions: [new PhraseCondition('referencePedalZone', '=', 'brake')] };
    const turn: PhraseDefinition = { ...definition('reference-steering'),
        conditions: [new PhraseCondition('referenceCue', '=', 'steering')] };
    const phrases = [brake, turn], engine = new PhraseEngine(phrases);
    prepare(engine, phrases, { [brake.id]: 100, [turn.id]: 2000 });
    engine.receiveMap(map, 0, reference);
    const braking = engine.receiveTelemetry(frame(sample(0.18)), 1000);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    engine.speechFinished(braking.events[0].id, 1100);
    // Speed stays exactly at the reference turn-in speed: the old strict-below exit never matches.
    const upcoming = engine.receiveTelemetry(frame(sample(0.2501)), 1200);
    expect(upcoming.events.map(({ sentence }) => sentence)).toEqual(['Brake.', 'Turn in']);
    expect(upcoming.events[1].speech?.preparationTimeMs).toBe(2000);
    expect(upcoming.state.path).toEqual(['root', 'Track guide', 'Turn in']);
});

it('uses the longer Ease brake clip before the Turn in clip can reach the corner', () => {
    const { map, reference } = steeringCornerFixture();
    map.centerline_segments![0].start_position = 0.25;
    let energy = 40 ** 2;
    reference.y.forEach((row, index) => {
        if (index >= 25 && index <= 30) energy -= 20 * (index <= 26 ? 4 : 3.8);
        row[6] = Math.sqrt(energy) * 3.6;
        row[4] = 0.8;
    });
    const phrases = [definition('reference-steering'), definition('reference-steering-release')];
    const engine = new PhraseEngine(phrases);
    prepare(engine, phrases, { 'reference-steering': 300, 'reference-steering-release': 2000 });
    engine.receiveMap(map, 0, reference);
    const upcoming = engine.receiveTelemetry(frame(sample(0.245, { Physics_brake: 0.8 })), 1000);
    expect(upcoming.events).toEqual([expect.objectContaining({ sentence: 'Ease brake', brakeRelease: true,
        speech: expect.objectContaining({ preparationTimeMs: 2000 }) })]);
    expect(upcoming.state.path).toEqual(['root', 'Track guide', 'Ease brake']);
});

it('announces an upcoming turn even while the current reference is accelerating', () => {
    const { map, reference } = steeringCornerFixture();
    reference.y.forEach((row, index) => { row[6] = index < 27 ? 20 + index * 0.5 : 36; });
    const turn = definition('reference-steering'), engine = new PhraseEngine([turn]);
    engine.setDrivingAggression(100, 0);
    prepare(engine, [turn], { [turn.id]: 2000 });
    engine.receiveMap(map, 0, reference);
    // The current 200 ms window is accelerating; the cue's 2 s window reaches the turn.
    const upcoming = engine.receiveTelemetry(frame(sample(0.2501)), 1000);
    expect(upcoming.events).toEqual([expect.objectContaining({ sentence: 'Turn in',
        speech: expect.objectContaining({ preparationTimeMs: 2000 }) })]);
});

it('rearms Brake after a single measured frame in its reference zone', () => {
    const { reference } = steeringCornerFixture();
    reference.y.forEach((row, index) => { row[4] = index >= 10 && index < 20 ? 0.8 : 0; });
    const brake: PhraseDefinition = { ...definition('reference-brake'),
        conditions: [new PhraseCondition('referencePedalZone', '=', 'brake')] };
    const engine = new PhraseEngine([brake]);
    prepare(engine, [brake], { [brake.id]: 200 });
    engine.receiveMap(null, 0, reference);
    const first = engine.receiveTelemetry(frame(sample(0.15)), 1000);
    expect(first.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    // Leave on the very next measurement before a playback callback can revisit the zone.
    engine.receiveTelemetry(frame(sample(0.25)), 1100);
    engine.speechFinished(first.events[0].id, 1200);
    const nextVisit = engine.receiveTelemetry(frame(sample(0.15)), 4000);
    expect(nextVisit.events.map(({ sentence }) => sentence)).toEqual(['Brake.', 'Brake.']);
});

it('disables active guide speech while other cues continue, then enables the guide again', () => {
    const { reference } = steeringCornerFixture();
    reference.y.forEach((row, index) => { row[4] = index >= 10 && index < 20 ? 0.8 : 0; });
    const brake: PhraseDefinition = { ...definition('reference-brake'),
        conditions: [new PhraseCondition('referencePedalZone', '=', 'brake')] };
    const other: PhraseDefinition = { id: 'other-guide', name: 'Other guide', description: 'Independent guide.',
        sentence: 'Other cue.', holdMs: 0, conditions: [new PhraseCondition('speed', '>=', 30)] };
    const phrases = [other, brake], engine = new PhraseEngine(phrases);
    prepare(engine, phrases, {});
    engine.receiveMap(null, 0, reference);
    const active = engine.receiveTelemetry(frame(sample(0.15)), 1000);
    expect(active.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const disabled = engine.setTrackGuideEnabled(false, 1050);
    expect(disabled.trackGuideEnabled).toBe(false);
    expect(disabled.state.path).not.toContain('Track guide');
    expect(disabled.steeringCorner).toBeUndefined();
    expect(disabled.events.filter(({ ruleId }) => ruleId === brake.id)
        .every(({ speech }) => speech?.releaseAt !== undefined)).toBe(true);
    const independent = engine.receiveTelemetry(frame(sample(0.15)), 1100);
    expect(independent.events[independent.events.length - 1].sentence).toBe('Other cue.');
    engine.speechFinished(independent.events[independent.events.length - 1].id, 1200);
    engine.setTrackGuideEnabled(true, 1300);
    const enabled = engine.receiveTelemetry(frame(sample(0.15)), 1400);
    expect(enabled.trackGuideEnabled).toBe(true);
    expect(enabled.events[enabled.events.length - 1].sentence).toBe('Brake.');
});
