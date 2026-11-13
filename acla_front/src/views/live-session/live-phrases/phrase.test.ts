import { Phrase, PhraseInstance } from './phrase';
import brakeCurves from './curves/reference-brake.json';

afterEach(() => jest.restoreAllMocks());

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

it('builds a unique audio pool with independent timing, energy and pitch variations', () => {
    const phrase = new Phrase('brake', 'Brake.', brakeCurves);
    const variants = phrase.createAudioInstances([{}, { inputs: { timing: 0 } },
        { inputs: { timing: 0.5 } }, { inputs: { energy: 0.75 } }, { inputs: { pitch: -0.25 } },
        { inputs: { timing: 0.5 }, priority: 5 }]);
    expect(variants.map((variant) => variant.inputValues)).toEqual([
        [0, 0, 0], [0.5, 0, 0], [0, 0.75, 0], [0, 0, -0.25],
    ]);
    expect(phrase.createAudioInstances().map((variant) => variant.cacheKey)).toEqual([variants[0].cacheKey]);
    expect(phrase.createAudioInstances([])).toEqual([]);
});

it('owns immutable interchangeable wordings and prepares every wording with each audio setting', () => {
    const random = jest.spyOn(Math, 'random');
    const alternatives = [' Shift to the left. ', 'Move to the left.', 'Shift to the left.'];
    const phrase = new Phrase('left', ' Move to the left. ', brakeCurves, { randomAlternatives: alternatives, priority: 2 });
    alternatives.push('Unrelated instruction.');
    expect(phrase.randomAlternatives).toEqual(['Move to the left.', 'Shift to the left.']);
    expect(Object.isFrozen(phrase.randomAlternatives)).toBe(true);
    const instances = phrase.createAudioInstances([{}, { speed: 1.4 }, { speed: 1.4 }]);
    expect(instances.map(({ speech }) => [speech.text, speech.speed])).toEqual([
        ['Move to the left.', 1.2], ['Move to the left.', 1.4],
        ['Shift to the left.', 1.2], ['Shift to the left.', 1.4],
    ]);
    expect(new Set(instances.map(({ cacheKey }) => cacheKey)).size).toBe(4);
    instances.forEach((instance) => {
        expect(instance).toMatchObject({ id: 'left', priority: 2, randomAlternatives: [instance.speech.text] });
        expect(instance.createAudioInstances().map(({ cacheKey }) => cacheKey)).toEqual([instance.cacheKey]);
    });
    expect(phrase.createInstance().randomAlternatives).toEqual(['Move to the left.']);
    const options = { randomAlternatives: ['Another wording.'], speed: 1.4 };
    expect(phrase.createInstance(options).randomAlternatives).toEqual(['Move to the left.']);
    expect(random).not.toHaveBeenCalled();
});

it.each([[0, 0], [0.499999, 0], [0.5, 1], [0.999999, 1]])(
    'draws %s once from only the scheduler-approved pool', (value, index) => {
        const random = jest.spyOn(Math, 'random').mockReturnValue(value);
        const phrase = new Phrase('left', 'Move to the left.', undefined,
            { randomAlternatives: ['Shift to the left.', 'Bring it to the left.'] });
        const instances = phrase.createAudioInstances();
        const allowed = instances.slice(1);
        expect(random).not.toHaveBeenCalled();
        expect(phrase.selectAudioInstance(allowed)).toBe(allowed[index]);
        expect(random).toHaveBeenCalledTimes(1);
    });

it('does not draw for an empty or singleton approved pool', () => {
    const random = jest.spyOn(Math, 'random');
    const phrase = new Phrase('left', 'Move to the left.', undefined, { randomAlternatives: ['Shift to the left.'] });
    const allowed = phrase.createAudioInstances().slice(1);
    expect(phrase.selectAudioInstance([])).toBeUndefined();
    expect(phrase.selectAudioInstance(allowed)).toBe(allowed[0]);
    expect(random).not.toHaveBeenCalled();
});
