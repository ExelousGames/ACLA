import { synthesizeProsodyTts, type TtsPack } from 'components/tts';
import { PhraseCondition, type PhraseDefinition } from './phrase-engine';
import { getPhraseSpeechInstances, loadAllPhraseSpeech } from './phrase-speech';
import brakeCurves from './curves/reference-brake.json';

jest.mock('components/tts', () => ({ synthesizeProsodyTts: jest.fn() }));

const synthesize = synthesizeProsodyTts as jest.MockedFunction<typeof synthesizeProsodyTts>;
const definition: PhraseDefinition = {
    id: 'prepared-variants', name: 'Prepared variants', description: 'Audio pool fixture.', holdMs: 0, priority: 0,
    sentence: 'Brake.', randomAlternatives: ['Apply the brake.'], amplifyCurves: brakeCurves,
    conditions: [new PhraseCondition('speed', '>=', 30)],
    preloadSettings: [{ inputs: { timing: 0.5 } }],
    audioVariations: [{ inputs: { timing: 0.5 } }, { inputs: { energy: 0.75, pitch: -0.25 } }],
};

beforeEach(() => jest.clearAllMocks());

it('prepares every wording and prosody choice once across random and contextual pools', () => {
    const instances = getPhraseSpeechInstances([definition, { ...definition, id: 'same-audio' }]);
    expect(instances).toHaveLength(6);
    for (const sentence of [definition.sentence, ...definition.randomAlternatives!]) {
        expect(instances.filter((instance) => instance.speech.text === sentence).map((instance) => instance.inputValues))
            .toEqual([[0, 0, 0], [0.5, 0, 0], [0, 0.75, -0.25]]);
    }
});

it('keeps each measured duration paired with its prepared waveform and reuses the complete pool', async () => {
    const instances = getPhraseSpeechInstances([definition]);
    const preparedPacks: TtsPack[] = instances.map((instance, index) => ({
        text: instance.speech.text, audioDataUrl: `data:audio/wav;base64,clip${index}`,
        durationMs: 400 + index * 100,
    }));
    preparedPacks.forEach((pack) => synthesize.mockResolvedValueOnce(pack));
    const packs = new Map<string, TtsPack>();
    const signal = new AbortController().signal;
    expect(await loadAllPhraseSpeech(instances, packs, signal)).toEqual({ complete: true, errors: [] });
    instances.forEach((instance, index) => {
        expect(packs.get(instance.cacheKey)).toBe(preparedPacks[index]);
        expect(synthesize).toHaveBeenNthCalledWith(index + 1,
            instance.curves, instance.inputValues, instance.speech, signal, true);
    });
    await loadAllPhraseSpeech(instances, packs, signal);
    expect(synthesize).toHaveBeenCalledTimes(instances.length);
});
