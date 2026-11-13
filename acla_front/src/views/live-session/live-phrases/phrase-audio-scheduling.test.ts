import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { getPhraseActions, PhraseCondition, PhraseEngine, type PhraseDefinition } from './phrase-engine';
import brakeCurves from './curves/reference-brake.json';

const frame = (speed: number): LiveTelemetryFrameEvent => {
    const sample = { Physics_speed_kmh: speed };
    return { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
        committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
};
const definition: PhraseDefinition = {
    id: 'random', name: 'Random delivery', sentence: 'Brake.', description: 'Audio pool fixture.', holdMs: 0, priority: 0,
    amplifyCurves: brakeCurves,
    audioVariations: [{ inputs: { pitch: 0.5, timing: 0.2, energy: 0.3 } }, { speed: 0.8 }],
    conditions: [new PhraseCondition('speed', '>=', 45)],
    exitConditions: [new PhraseCondition('speed', '>=', 60)],
};
const prepare = (phrase = definition, durations = [40, 80, 500]) => {
    const engine = new PhraseEngine([phrase]);
    const action = getPhraseActions(phrase)[0];
    const instances = action.phrase.createAudioInstances([{}, ...(action.audioVariations ?? [])]);
    engine.setSpeechDurations(new Map(instances.map((instance, index) => [instance.cacheKey, durations[index]])));
    return { engine, instances };
};

afterEach(() => jest.restoreAllMocks());

it('filters prosody variants against the predicted phrase exit, then chooses once on entry', () => {
    const random = jest.spyOn(Math, 'random').mockReturnValue(0.999);
    const { engine, instances } = prepare();
    expect(engine.receiveTelemetry(frame(40), 1000).events).toEqual([]);
    expect(random).not.toHaveBeenCalled();
    const entered = engine.receiveTelemetry(frame(50), 1100);
    expect(entered.events).toHaveLength(1);
    expect(entered.events[0].instance?.cacheKey).toBe(instances[1].cacheKey);
    expect(entered.events[0].speech).toEqual({ durationMs: 80, preparationTimeMs: 500, reactionAllowanceMs: 0 });
    expect(entered.state.kind).toBe('phrase');
    expect(random).toHaveBeenCalledTimes(1);
    // Changed entry/timing predictions do not reselect or terminate active audio.
    expect(engine.receiveTelemetry(frame(20), 1150).events[0].speech?.releaseAt).toBeUndefined();
    expect(engine.evaluate(5000).state.kind).toBe('phrase');
    expect(random).toHaveBeenCalledTimes(1);
    expect(engine.speechFinished(entered.events[0].id, 5001).state.path).toEqual(['root']);
});

it('waits without consuming the phrase or choosing audio when no prepared variant fits', () => {
    const random = jest.spyOn(Math, 'random');
    const { engine } = prepare(definition, [500, 600, 700]);
    engine.receiveTelemetry(frame(40), 1000);
    const blocked = engine.receiveTelemetry(frame(50), 1100);
    expect(blocked.events).toEqual([]);
    expect(blocked.state.path).toEqual(['root']);
    expect(blocked.schedule?.items).toEqual([]);
    expect(random).not.toHaveBeenCalled();
    // Constant speed opens the window again; the same phrase remains armed.
    expect(engine.receiveTelemetry(frame(50), 2100).events).toHaveLength(1);
});

it('never selects an unprepared variant and does not randomize a singleton allowed pool', () => {
    const random = jest.spyOn(Math, 'random');
    const { engine, instances } = prepare(definition, [40, NaN, 500]);
    engine.receiveTelemetry(frame(40), 1000);
    const event = engine.receiveTelemetry(frame(50), 1100).events[0];
    expect(event.instance?.cacheKey).toBe(instances[0].cacheKey);
    expect(random).not.toHaveBeenCalled();
});

it('preserves a context-selected delivery instead of substituting a shorter random variant', () => {
    const { engine, instances } = prepare({ ...definition, instanceSettings: () => ({ speed: 0.8 }),
        preloadSettings: [{ speed: 0.8 }] });
    engine.receiveTelemetry(frame(40), 1000);
    const event = engine.receiveTelemetry(frame(50), 1100).events[0];
    expect(event.instance?.cacheKey).toBe(instances[2].cacheKey);
    expect(event.speech?.durationMs).toBe(500);
});

it('keeps equal-priority declaration order when filtering the phrase that is entering', () => {
    const fixed: PhraseDefinition = { ...definition, id: 'fixed', sentence: 'Fixed delivery.',
        audioVariations: undefined, exitConditions: undefined };
    const engine = new PhraseEngine([definition, fixed]);
    const instances = getPhraseActions(definition)[0].phrase.createAudioInstances([{}, ...definition.audioVariations!]);
    const fixedInstance = getPhraseActions(fixed)[0].phrase.createInstance();
    engine.setSpeechDurations(new Map([...instances.map((instance, index) => [instance.cacheKey, [40, 80, 500][index]] as const),
        [fixedInstance.cacheKey, 1000]]));
    engine.receiveTelemetry(frame(40), 1000);
    const entered = engine.receiveTelemetry(frame(50), 1100);
    expect(entered.events[0].ruleId).toBe(definition.id);
    expect(entered.schedule?.items.map(({ id }) => id)).toEqual([`${definition.id}:say:0`]);
});

it('lets another phrase enter when a higher-priority random pool cannot fit', () => {
    const fixed: PhraseDefinition = { ...definition, id: 'fixed', sentence: 'Fixed delivery.',
        priority: 1, audioVariations: undefined, exitConditions: undefined };
    const engine = new PhraseEngine([{ ...definition, priority: 3 }, fixed]);
    const instances = getPhraseActions(definition)[0].phrase.createAudioInstances([{}, ...definition.audioVariations!]);
    const fixedInstance = getPhraseActions(fixed)[0].phrase.createInstance();
    engine.setSpeechDurations(new Map([...instances.map((instance) => [instance.cacheKey, 500] as const),
        [fixedInstance.cacheKey, 1000]]));
    engine.receiveTelemetry(frame(40), 1000);
    const entered = engine.receiveTelemetry(frame(50), 1100);
    expect(entered.events.map(({ ruleId }) => ruleId)).toEqual([fixed.id]);
    expect(entered.schedule?.items.map(({ id }) => id)).toEqual([`${fixed.id}:say:0`]);
});

it.each([[0, 'Brake.'], [0.999999, 'Brake now.']] as const)(
    'randomly selects %s from wordings that fit the phrase window', (value, sentence) => {
        const random = jest.spyOn(Math, 'random').mockReturnValue(value);
        const { engine } = prepare({ ...definition, audioVariations: undefined,
            randomAlternatives: ['Brake now.', 'Apply the brakes.'] });
        expect(engine.receiveTelemetry(frame(40), 1000).events).toEqual([]);
        expect(random).not.toHaveBeenCalled();
        const entered = engine.receiveTelemetry(frame(50), 1100);
        expect(entered.events).toHaveLength(1);
        expect(entered.events[0].sentence).toBe(sentence);
        expect(entered.root.children[0].children.find(({ kind }) => kind === 'phrase')?.fields)
            .toMatchObject({ 'Allowed audio variants': 2, 'Phrase selection': 'Random (equal chance)' });
        expect(random).toHaveBeenCalledTimes(1);
        const exited = engine.receiveTelemetry(frame(65), 1150);
        expect(exited.events[0].speech?.releaseAt).toBe(1150);
        expect(random).toHaveBeenCalledTimes(1);
    });
