import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { COOLDOWN_MS, PhraseCondition, PhraseEngine, type PhraseDefinition } from './phrase-engine';

const frame = (speed: number): LiveTelemetryFrameEvent => {
    const sample = { Physics_speed_kmh: speed };
    return { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
        committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 } };
};
const phrase: PhraseDefinition = {
    id: 'variants', name: 'Variant guide', description: 'Pick one wording for an eligible action.', holdMs: 0,
    sentence: 'Move to the left.', sentenceAlternatives: ['Shift to the left.', 'Bring it to the left.'],
    conditions: [new PhraseCondition('speed', '>=', 20)],
    actionConditions: [new PhraseCondition('speed', '>=', 100)],
};

afterEach(() => jest.restoreAllMocks());

it.each(['primary', 'additional'])('chooses once per execution of a %s action, after its conditions match', (kind) => {
    const random = jest.spyOn(Math, 'random').mockReturnValueOnce(0).mockReturnValueOnce(1 / 3).mockReturnValueOnce(0.999999);
    const engine = new PhraseEngine([kind === 'primary' ? phrase : {
        ...phrase, sentence: 'Ineligible.', sentenceAlternatives: ['Also ineligible.'],
        actionConditions: [new PhraseCondition('speed', '>=', 200)], additionalActions: [phrase],
    }]);
    const expected: string[] = [];
    [phrase.sentence, ...phrase.sentenceAlternatives!].forEach((sentence, index) => {
        const now = index * (COOLDOWN_MS + 1000);
        engine.receiveTelemetry(frame(50), now);
        engine.evaluate(now + 50);
        expect(random).toHaveBeenCalledTimes(index);
        const spoken = engine.receiveTelemetry(frame(100), now + 100);
        expected.push(sentence);
        expect(spoken.events.map((event) => event.sentence)).toEqual(expected);
        expect(spoken.events[index]).toMatchObject({ ruleId: phrase.id, timestamp: now + 100 });
        expect(random).toHaveBeenCalledTimes(index + 1);
        engine.evaluate(now + 101);
        expect(engine.receiveTelemetry(frame(100), now + 150).events).toEqual(spoken.events);
        expect(random).toHaveBeenCalledTimes(index + 1);
        engine.receiveTelemetry(frame(10), now + 200);
        engine.evaluate(now + 700);
    });
});

it.each([undefined, []])('keeps the original sentence without alternatives (%s)', (sentenceAlternatives) => {
    const random = jest.spyOn(Math, 'random');
    const engine = new PhraseEngine([{ ...phrase, sentenceAlternatives }]);
    expect(engine.receiveTelemetry(frame(100), 0).events[0].sentence).toBe(phrase.sentence);
    expect(random).not.toHaveBeenCalled();
});

it('exposes alternatives when inspecting nested speech actions without choosing one', () => {
    const random = jest.spyOn(Math, 'random');
    const engine = new PhraseEngine([{ ...phrase, id: 'second-apex', additionalActions: [phrase, phrase] }]);
    const closure = engine.evaluate(0).root.children[0];
    const nestedAction = closure.children[0].children[1];
    expect(nestedAction).toMatchObject({ name: 'say phrase', description: phrase.sentence, fields: {
        'Phrase selection': 'Random (equal chance)', 'Alternative 1': 'Shift to the left.', 'Alternative 2': 'Bring it to the left.',
    } });
    expect(engine.evaluate(1).events).toEqual([]);
    expect(random).not.toHaveBeenCalled();
});
