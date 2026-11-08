import apiService from 'services/api.service';
import { isTtsPack, synthesizeTts, synthesizeTtsPack } from './tts-service';
import { applyAmplifyCurves, synthesizeProsodyTts, type AmplifyCurve } from './tts-prosody';

jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn() } }));
const post = apiService.post as jest.Mock;

const wav = () => {
    const bytes = new Uint8Array(48);
    const view = new DataView(bytes.buffer);
    const tag = (offset: number, value: string) => bytes.set(Array.from(value, (c) => c.charCodeAt(0)), offset);
    tag(0, 'RIFF');
    view.setUint32(4, 40, true);
    tag(8, 'WAVE');
    tag(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 2, true);
    view.setUint32(28, 4, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    tag(36, 'data');
    view.setUint32(40, 4, true);
    return bytes.buffer;
};

beforeEach(() => post.mockReset());

it('requests backend WAV through the shared API and returns a serializable pack with its duration', async () => {
    post.mockResolvedValue({ data: wav() });
    const signal = new AbortController().signal;
    const pack = await synthesizeTts({ text: ' Brake smoothly. ', voice: 'af_bella' }, signal);
    expect(post).toHaveBeenCalledWith('/user-ai-model/voice-synthesize', {
        text: 'Brake smoothly.', voice: 'af_bella', speed: 1, language: 'en-us',
    }, { responseType: 'arraybuffer', timeout: 120_000, signal });
    expect(pack.durationMs).toBe(1000);
    expect(isTtsPack(JSON.parse(JSON.stringify(pack)))).toBe(true);
    expect(atob(pack.audioDataUrl.split(',')[1]).slice(0, 4)).toBe('RIFF');
});

it('preserves batch order and stops preparing when cancelled', async () => {
    const controller = new AbortController();
    post.mockImplementationOnce(async () => {
        controller.abort();
        return { data: wav() };
    });
    await expect(synthesizeTtsPack([{ text: 'First' }, { text: 'Second' }], controller.signal))
        .rejects.toMatchObject({ name: 'AbortError' });
    expect(post).toHaveBeenCalledTimes(1);
    post.mockResolvedValue({ data: wav() });
    expect((await synthesizeTtsPack([{ text: 'First' }, { text: 'Second' }])).map((pack) => pack.text))
        .toEqual(['First', 'Second']);
});

it('rejects invalid audio, blank text, and failed synthesis without creating a pack', async () => {
    await expect(synthesizeTts({ text: ' ' })).rejects.toThrow('1 to 4000');
    expect(post).not.toHaveBeenCalled();
    post.mockResolvedValue({ data: new ArrayBuffer(0) });
    await expect(synthesizeTts({ text: 'Brake' })).rejects.toThrow('WAV');
    post.mockRejectedValue(new Error('Speech unavailable'));
    await expect(synthesizeTts({ text: 'Brake' })).rejects.toThrow('Speech unavailable');
    expect(isTtsPack({ text: 'Brake', audioDataUrl: 'https://example.com/voice.wav', durationMs: 1000 })).toBe(false);
});

const curves: AmplifyCurve[] = [
    { setting: 'timing', points: [{ position: 0, value: -0.2 }, { position: 1, value: 0.4 }] },
    { setting: 'energy', points: [{ position: 0, value: 2 }, { position: 0.5, value: 4 }, { position: 1, value: 0 }] },
    { setting: 'pitch', points: [{ position: 0, value: 1 }, { position: 1, value: -2 }] },
];
const preview = {
    phonemes: 'aˈɪ b!',
    tokens: Array.from('aˈɪ b!', (symbol, index) => ({ index, symbol, token_id: index + 1 })),
    sample_rate: 24000, max_phonemes: 510,
};

it('previews the exact speech then sends scaled additive controls over the indexed phoneme range', async () => {
    const original = JSON.stringify(curves);
    post.mockResolvedValueOnce({ data: preview }).mockResolvedValueOnce({ data: wav() });
    const signal = new AbortController().signal;
    const result = await synthesizeProsodyTts(curves, [0.5, 1.5, -0.25], {
        text: ' Brake! ', voice: 'af_bella', speed: 1.2, language: 'en-gb',
    }, signal);
    expect(post).toHaveBeenNthCalledWith(1, '/voice/phonemes', { text: 'Brake!', language: 'en-gb' }, { timeout: 120_000, signal });
    expect(post).toHaveBeenNthCalledWith(2, '/voice/synthesize/prosody', {
        text: 'Brake!', voice: 'af_bella', speed: 1.2, language: 'en-gb',
        prosody: { expected_phonemes: preview.phonemes, curves: [{ start: 0, end: 6,
            timing: [{ position: 0, value: 0.9 }, { position: 1, value: 1.2 }],
            energy: [{ position: 0, value: 3 }, { position: 0.5, value: 6 }, { position: 1, value: 0 }],
            pitch: [{ position: 0, value: -0.25 }, { position: 1, value: 0.5 }],
        }] },
    }, { responseType: 'arraybuffer', timeout: 120_000, signal });
    expect(result.text).toBe('Brake!');
    expect(result.durationMs).toBe(1000);
    expect(isTtsPack(result)).toBe(true);
    expect(JSON.stringify(curves)).toBe(original);
});

it('uses neutral timing, energy and pitch when inputs are zero', () => {
    const applied = applyAmplifyCurves(curves, [0, 0, 0]);
    expect(applied.timing!.every((point) => point.value === 1)).toBe(true);
    expect(applied.energy!.every((point) => point.value === 0)).toBe(true);
    expect(applied.pitch!.every((point) => point.value === 0)).toBe(true);
});

it.each([
    { curves, inputs: [1], message: 'one input' },
    { curves, inputs: [1, NaN, 1], message: 'finite' },
    { curves, inputs: [Infinity, 1, 1], message: 'finite' },
    { curves, inputs: [10, 1, 1], message: 'timing' },
    { curves, inputs: [1, 4, 1], message: 'energy' },
    { curves, inputs: [1, 1, 8], message: 'pitch' },
    { curves: [curves[0], curves[0]], inputs: [1, 1], message: 'at most one' },
    { curves: [{ setting: 'pitch', points: [{ position: 0.1, value: 1 }, { position: 1, value: 1 }] }], inputs: [1], message: 'endpoints' },
    { curves: [{ setting: 'pitch', points: [{ position: 0, value: 1 }, { position: 0, value: 1 }, { position: 1, value: 1 }] }], inputs: [1], message: 'increasing' },
    { curves: [{ setting: 'pitch', points: [{ position: 0, value: NaN }, { position: 1, value: 1 }] }], inputs: [0], message: 'finite' },
])('rejects invalid amplification before requesting speech: $message', async ({ curves, inputs, message }) => {
    await expect(synthesizeProsodyTts(curves as AmplifyCurve[], inputs, { text: 'Brake.' })).rejects.toThrow(message);
    expect(post).not.toHaveBeenCalled();
});

it.each(['before preview', 'after preview', 'after synthesis'])('honors cancellation %s', async (stage) => {
    const controller = new AbortController();
    if (stage === 'before preview') controller.abort();
    post.mockImplementationOnce(async () => {
        if (stage === 'after preview') controller.abort();
        return { data: preview };
    }).mockImplementationOnce(async () => {
        controller.abort();
        return { data: wav() };
    });
    await expect(synthesizeProsodyTts(curves, [0, 0, 0], { text: 'Brake.' }, controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(post).toHaveBeenCalledTimes(stage === 'before preview' ? 0 : stage === 'after preview' ? 1 : 2);
});

it('rejects changed or missing phoneme indices and propagates backend and WAV errors', async () => {
    post.mockResolvedValueOnce({ data: { ...preview, tokens: preview.tokens.slice(1) } });
    await expect(synthesizeProsodyTts(curves, [0, 0, 0], { text: 'Brake.' })).rejects.toThrow('indexed phoneme');
    expect(post).toHaveBeenCalledTimes(1);
    post.mockResolvedValueOnce({ data: preview }).mockRejectedValueOnce(new Error('Phoneme sequence changed'));
    await expect(synthesizeProsodyTts(curves, [0, 0, 0], { text: 'Brake.' })).rejects.toThrow('Phoneme sequence changed');
    post.mockResolvedValueOnce({ data: preview }).mockResolvedValueOnce({ data: new ArrayBuffer(0) });
    await expect(synthesizeProsodyTts(curves, [0, 0, 0], { text: 'Brake.' })).rejects.toThrow('WAV');
});
