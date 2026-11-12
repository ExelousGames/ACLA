import { Phrase, PhraseInstance } from './phrase';
import brakeCurves from './curves/reference-brake.json';

it('creates independent runtime copies with per-instance float inputs and speech settings', () => {
    const phrase = new Phrase('brake', 'Brake.', brakeCurves, { voice: 'af_bella', inputs: { timing: 0.1 } });
    const settings = { speed: 1.4, inputs: { energy: 0.75, pitch: -0.25 } };
    const first = phrase.createInstance(settings);
    const second = phrase.createInstance({ inputs: { timing: 0.5 } });
    expect(first).toBeInstanceOf(PhraseInstance);
    expect(first.speech).toEqual({ text: 'Brake.', speed: 1.4, language: 'en-us', voice: 'af_bella' });
    expect(first.inputs).toEqual({ timing: 0.1, energy: 0.75, pitch: -0.25 });
    expect(second.inputs).toEqual({ timing: 0.5, energy: 0, pitch: 0 });
    expect(phrase.inputs).toEqual({ timing: 0.1, energy: 0, pitch: 0 });
    expect(first.amplifyCurves).toEqual(phrase.amplifyCurves);
    expect(first.amplifyCurves.timing[0]).not.toBe(phrase.amplifyCurves.timing[0]);
    settings.inputs.energy = 5;
    expect(first.inputs.energy).toBe(0.75);
    expect(Object.isFrozen(first.amplifyCurves.timing[0])).toBe(true);
    expect(Object.isFrozen(first)).toBe(true);
});

it('distinguishes settings and curves in audio identity while sharing identical speech across rules', () => {
    const phrase = new Phrase('brake', 'Brake.', brakeCurves);
    const key = phrase.createInstance().cacheKey;
    expect(phrase.createInstance().cacheKey).toBe(key);
    expect(new Phrase('other-rule', 'Brake.', brakeCurves).createInstance().cacheKey).toBe(key);
    expect(new Phrase('brake', 'Brake.').createInstance().cacheKey).not.toBe(key);
    for (const settings of [{ voice: 'af_bella' }, { language: 'en-gb' }, { speed: 1.5 }, { inputs: { pitch: 0.5 } }]) {
        expect(phrase.createInstance(settings).cacheKey).not.toBe(key);
    }
});

it('inherits and overrides scheduling priority without changing prepared audio identity', () => {
    const phrase = new Phrase('brake', 'Brake.', brakeCurves, { priority: 3 });
    const inherited = phrase.createInstance();
    const overridden = phrase.createInstance({ priority: 1 });
    expect(new Phrase('ordinary', 'Instruction.').priority).toBe(0);
    expect(phrase.priority).toBe(3);
    expect(inherited.priority).toBe(3);
    expect(inherited.createInstance().priority).toBe(3);
    expect(overridden.priority).toBe(1);
    expect(phrase.createInstance({ priority: 0 }).priority).toBe(0);
    expect(overridden.speech).toEqual(inherited.speech);
    expect(overridden.cacheKey).toBe(inherited.cacheKey);
    expect(new Phrase('other', 'Brake.', brakeCurves).createInstance().cacheKey).toBe(inherited.cacheKey);
});
