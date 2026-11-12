import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { getPhraseActions, PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine, type PhraseDefinition } from './phrase-engine';

const frame = (speed = 50): LiveTelemetryFrameEvent => {
    const sample: StandardTelemetrySample = { Physics_speed_kmh: speed };
    return { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
        committedSampleCount: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
};
const phrase = (id: string, referenceCue: PhraseDefinition['referenceCue'], priority?: number): PhraseDefinition => ({
    id, referenceCue, priority, name: id, sentence: `${id}.`, description: 'Scheduling fixture.', holdMs: 0,
    conditions: [new PhraseCondition('speed', '>=', 30)],
});
const prepare = (phrases: PhraseDefinition[]) => {
    const engine = new PhraseEngine(phrases);
    engine.setSpeechDurations(new Map(phrases.flatMap(getPhraseActions).flatMap((action) =>
        action.phrases.map((source) => [source.createInstance().cacheKey, 200] as const))));
    return engine;
};
const reversed = () => [phrase('throttle', 'throttle'), phrase('turn', 'steering'), phrase('brake', 'brake')];

it('dispatches Brake, Turn, then Throttle regardless of declaration order and exposes the same schedule', () => {
    const engine = prepare(reversed());
    const first = engine.receiveTelemetry(frame(), 1000);
    expect(first.events.map(({ ruleId }) => ruleId)).toEqual(['brake']);
    expect(first.schedule?.items.map(({ id, status, priority }) => ({ id, status, priority }))).toEqual([
        { id: 'brake', status: 'active', priority: 3 },
        { id: 'turn', status: 'next', priority: 2 },
        { id: 'throttle', status: 'queued', priority: 1 },
    ]);
    const started = engine.speechStarted(first.events[0].id, 1050);
    expect(started.schedule?.items[0]).toMatchObject({ startAt: 1050, endAt: 1250, durationEstimated: false });
    expect(engine.receiveTelemetry(frame(), 1100).events).toHaveLength(1);
    engine.speechFinished(first.events[0].id, 1250);
    const second = engine.receiveTelemetry(frame(), 1300);
    expect(second.events.map(({ ruleId }) => ruleId)).toEqual(['brake', 'turn']);
    expect(second.schedule?.items.map(({ id }) => id)).toEqual(['turn', 'throttle']);
    engine.speechFinished(second.events[1].id, 1500);
    expect(engine.receiveTelemetry(frame(), 1550).events.map(({ ruleId }) => ruleId)).toEqual(['brake', 'turn', 'throttle']);
    expect(first.schedule?.items).toHaveLength(3);
});

it('removes predictions that stop matching while preserving the current playback', () => {
    const engine = prepare(reversed());
    engine.receiveTelemetry(frame(), 1000);
    const changed = engine.receiveTelemetry(frame(0), 1100);
    expect(changed.events).toHaveLength(1);
    expect(changed.schedule?.items.map(({ id }) => id)).toEqual(['brake']);
    engine.speechFinished(changed.events[0].id, 1200);
    const finished = engine.receiveTelemetry(frame(0), 1300);
    expect(finished.events).toHaveLength(1);
    expect(finished.schedule?.items).toEqual([]);
});

it('clears pending predictions with stale telemetry, guide disable, and reset', () => {
    const engine = prepare(reversed());
    engine.receiveTelemetry(frame(), 1000);
    expect(engine.evaluate(2501).schedule?.items.map(({ id }) => id)).toEqual(['brake']);
    expect(engine.setTrackGuideEnabled(false, 2600).schedule?.items).toEqual([]);
    engine.reset(2700);
    expect(engine.evaluate(2700).schedule?.items).toEqual([]);
});

it('keeps declaration order for equal priorities and honors explicit numeric overrides', () => {
    const engine = prepare([phrase('first', 'steering', 7), phrase('second', 'brake', 7), phrase('third', 'throttle', 5)]);
    const first = engine.receiveTelemetry(frame(), 1000);
    expect(first.events[0].ruleId).toBe('first');
    expect(first.schedule?.items.map(({ id }) => id)).toEqual(['first', 'second', 'third']);
    engine.speechFinished(first.events[0].id, 1200);
    expect(engine.receiveTelemetry(frame(), 1300).events[1].ruleId).toBe('second');
});

it('inherits action priorities while allowing an individual phrase action to take precedence', () => {
    const definition: PhraseDefinition = { ...phrase('custom', 'brake', 3), additionalActions: [
        { sentence: 'Inherited.' }, { sentence: 'Urgent.', priority: 8 },
    ] };
    const actions = getPhraseActions(definition);
    expect(actions.map(({ priority }) => priority)).toEqual([3, 3, 8]);
    expect(actions.map(({ phrases }) => phrases[0].createInstance().priority)).toEqual([3, 3, 8]);
    const snapshot = prepare([definition]).receiveTelemetry(frame(), 1000);
    expect(snapshot.events[0]).toMatchObject({ sentence: 'Urgent.', instance: { priority: 8 } });
});

it('lets explicit exit conditions take precedence over an eligible high-priority speech action', () => {
    const definition: PhraseDefinition = { ...phrase('custom', 'brake', 9),
        exitConditions: [new PhraseCondition('speed', '>=', 30)] };
    const snapshot = prepare([definition]).receiveTelemetry(frame(), 1000);
    expect(snapshot.events).toEqual([]);
    expect(snapshot.state.path).not.toContain('custom');
});

it('assigns the requested default priorities to the built-in driving instructions', () => {
    const priorities = Object.fromEntries(PHRASE_DEFINITIONS.map(({ id, priority }) => [id, priority]));
    expect(priorities).toMatchObject({
        'reference-brake': 3, 'reference-releaseBrake': 3, 'reference-steering-release': 3,
        'reference-steering': 2, 'reference-cornerExit': 2, 'reference-throttle': 1, 'reference-moreThrottle': 1,
    });
});

it('schedules competing non-reference closures and dispatches them in priority order after release', () => {
    const engine = prepare([phrase('low', undefined, 1), phrase('middle', undefined, 2), phrase('high', undefined, 3)]);
    const first = engine.receiveTelemetry(frame(), 1000);
    expect(first.events.map(({ ruleId }) => ruleId)).toEqual(['high']);
    expect(first.schedule?.items.map(({ id, status }) => ({ id, status }))).toEqual([
        { id: 'high', status: 'active' }, { id: 'middle', status: 'next' }, { id: 'low', status: 'queued' },
    ]);
    expect(engine.receiveTelemetry(frame(), 1100).schedule?.items.map(({ id }) => id)).toEqual(['high', 'middle', 'low']);
    engine.speechFinished(first.events[0].id, 1200);
    const second = engine.receiveTelemetry(frame(), 1300);
    expect(second.events.map(({ ruleId }) => ruleId)).toEqual(['high', 'middle']);
    engine.speechFinished(second.events[1].id, 1500);
    expect(engine.receiveTelemetry(frame(), 1600).events.map(({ ruleId }) => ruleId)).toEqual(['high', 'middle', 'low']);
});

it('compares a track group using its ready cues rather than an unmatched higher-priority cue', () => {
    const brake: PhraseDefinition = { ...phrase('brake', 'brake'), conditions: [new PhraseCondition('speed', '>=', 80)] };
    const engine = prepare([phrase('throttle', 'throttle'), brake, phrase('tactic', undefined, 2)]);
    const first = engine.receiveTelemetry(frame(), 1000);
    expect(first.events.map(({ ruleId }) => ruleId)).toEqual(['tactic']);
    expect(first.schedule?.items.map(({ id }) => id)).toEqual(['tactic', 'throttle']);
    engine.speechFinished(first.events[0].id, 1200);
    expect(engine.receiveTelemetry(frame(), 1300).events.map(({ ruleId }) => ruleId)).toEqual(['tactic', 'throttle']);
});

it('waits for confirmation before queueing a more important cue and does not interrupt active speech', () => {
    const brake: PhraseDefinition = { ...phrase('brake', 'brake'), holdMs: 500 };
    const engine = prepare([phrase('throttle', 'throttle'), brake]);
    const first = engine.receiveTelemetry(frame(), 1000);
    expect(first.events.map(({ ruleId }) => ruleId)).toEqual(['throttle']);
    expect(first.schedule?.items.map(({ id }) => id)).toEqual(['throttle']);
    expect(engine.receiveTelemetry(frame(), 1499).schedule?.items.map(({ id }) => id)).toEqual(['throttle']);
    const confirmed = engine.receiveTelemetry(frame(), 1500);
    expect(confirmed.events.map(({ ruleId }) => ruleId)).toEqual(['throttle']);
    expect(confirmed.schedule?.items.map(({ id, status }) => ({ id, status }))).toEqual([
        { id: 'throttle', status: 'active' }, { id: 'brake', status: 'next' },
    ]);
    engine.speechFinished(first.events[0].id, 1600);
    expect(engine.receiveTelemetry(frame(), 1700).events.map(({ ruleId }) => ruleId)).toEqual(['throttle', 'brake']);
});

it('keeps a rearmed cue out of the schedule until its cooldown ends', () => {
    const engine = prepare([phrase('throttle', 'throttle')]);
    const first = engine.receiveTelemetry(frame(), 1000);
    engine.speechFinished(first.events[0].id, 1100);
    engine.receiveTelemetry(frame(0), 1150);
    engine.receiveTelemetry(frame(0), 1650);
    const cooldown = engine.receiveTelemetry(frame(), 1700);
    expect(cooldown.events).toHaveLength(1);
    expect(cooldown.schedule?.items).toEqual([]);
});

it('ranks a track group by its ready cues and yields after playback to a more important closure', () => {
    const brake = { ...phrase('brake', 'brake'), conditions: [new PhraseCondition('speed', '>=', 100)] };
    const tactic = { ...phrase('tactic', undefined, 2), conditions: [new PhraseCondition('speed', '>=', 60)] };
    const engine = prepare([brake, phrase('throttle', 'throttle'), tactic]);
    const first = engine.receiveTelemetry(frame(50), 1000);
    expect(first.events[0].ruleId).toBe('throttle');
    const waiting = engine.receiveTelemetry(frame(70), 1100);
    expect(waiting.events).toHaveLength(1);
    expect(waiting.schedule?.items.map(({ id }) => id)).toEqual(['throttle', 'tactic']);
    engine.speechFinished(first.events[0].id, 1200);
    expect(engine.receiveTelemetry(frame(70), 1300).events[1].ruleId).toBe('tactic');
});
