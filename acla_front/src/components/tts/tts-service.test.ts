import apiService from 'services/api.service';
import { createTtsPack, isTtsPack, synthesizeTts, synthesizeTtsPack } from './tts-service';
import { applyAmplifyCurves, synthesizeProsodyTts, type AmplifyCurve } from './tts-prosody';

jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn() } }));
const post = apiService.post as jest.Mock;

const wav = (samples = [0, 0], channels = 1, sampleRate = 2) => {
    const bytes = new Uint8Array(44 + samples.length * 2);
    const view = new DataView(bytes.buffer);
    const tag = (offset: number, value: string) => bytes.set(Array.from(value, (c) => c.charCodeAt(0)), offset);
    tag(0, 'RIFF');
    view.setUint32(4, bytes.length - 8, true);
    tag(8, 'WAVE');
    tag(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channels * 2, true);
    view.setUint16(32, channels * 2, true);
    view.setUint16(34, 16, true);
    tag(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, index) => view.setInt16(44 + index * 2, sample, true));
    return bytes.buffer;
};

beforeEach(() => post.mockReset());

it('scans speech forward from the start and backward from the end, retaining internal silence', () => {
    const pack = createTtsPack('Release the brake.', wav([0, 200, 8000, 0, 0, 0, -400, 300, 0], 1, 1000));
    expect(pack.speechBounds).toEqual({ startMs: 2, endMs: 7 });
    expect(pack.durationMs).toBe(9);
    expect(isTtsPack(JSON.parse(JSON.stringify(pack)))).toBe(true);
});

it('includes the first and final sample when speech reaches the file edges', () => {
    const pack = createTtsPack('Brake.', wav([500, 0, -500], 1, 24000));
    expect(pack.speechBounds).toEqual({ startMs: 0, endMs: pack.durationMs });
    expect(isTtsPack(pack)).toBe(true);
});

it.each([1, 2])('trims only outer silence from %s-channel WAVs and updates the playable duration', (channels) => {
    const frames = [0, 200, 8000, 0, 0, 0, -400, 300, 0];
    const samples = frames.flatMap((value) => channels === 1 ? [value] : [0, value]);
    const original = wav(samples, channels, 24000);
    const pack = createTtsPack('Release the brake.', original, true);
    const bytes = Uint8Array.from(atob(pack.audioDataUrl.split(',')[1]), (char) => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    expect(pack.durationMs).toBe(5 * 1000 / 24000);
    expect(pack.speechBounds).toEqual({ startMs: 0, endMs: pack.durationMs });
    expect(isTtsPack(pack)).toBe(true);
    expect(view.getUint32(4, true)).toBe(bytes.length - 8);
    expect(view.getUint32(40, true)).toBe(5 * channels * 2);
    expect(Array.from({ length: 5 * channels }, (_, i) => view.getInt16(44 + i * 2, true)))
        .toEqual(samples.slice(2 * channels, 7 * channels));
    expect(new Uint8Array(original).byteLength).toBe(44 + samples.length * 2);
    expect(createTtsPack(pack.text, bytes.buffer).speechBounds).toEqual(pack.speechBounds);
});

it('keeps natural-completion audio when a WAV is silent or cannot be scanned', () => {
    const silent = wav([0, 0, 0], 1, 1000);
    expect(createTtsPack('Silent.', silent, true)).toEqual(createTtsPack('Silent.', silent));
    const unscanned = wav([8000, 0, 0], 1, 1000);
    new DataView(unscanned).setUint16(20, 3, true);
    expect(createTtsPack('Unscanned.', unscanned, true)).toEqual(createTtsPack('Unscanned.', unscanned));
});

it('trims across WAV data chunks and ignores padded metadata between them', () => {
    const first = new Uint8Array(wav([0, 8000, 0], 1, 1000));
    const second = new Uint8Array(wav([0, -8000, 0], 1, 1000)).subarray(36);
    const metadata = new Uint8Array([74, 85, 78, 75, 1, 0, 0, 0, 23, 0]); // JUNK, one byte plus padding.
    const combined = new Uint8Array(first.length + metadata.length + second.length);
    combined.set(first);
    combined.set(metadata, first.length);
    combined.set(second, first.length + metadata.length);
    new DataView(combined.buffer).setUint32(4, combined.length - 8, true);
    const pack = createTtsPack('Hold brake pressure.', combined.buffer, true);
    const bytes = Uint8Array.from(atob(pack.audioDataUrl.split(',')[1]), (char) => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    expect(pack.durationMs).toBe(4);
    expect(pack.speechBounds).toEqual({ startMs: 0, endMs: 4 });
    expect(Array.from({ length: 4 }, (_, i) => view.getInt16(44 + i * 2, true))).toEqual([8000, 0, 0, -8000]);
});

it('prepares live phrase prosody WAVs with their audible duration', async () => {
    post.mockResolvedValueOnce({ data: preview })
        .mockResolvedValueOnce({ data: wav([0, 8000, 0, -8000, 0], 1, 1000) });
    const pack = await synthesizeProsodyTts([], [], { text: 'Hold brake pressure.' }, undefined, true);
    expect(pack.durationMs).toBe(3);
    expect(pack.speechBounds).toEqual({ startMs: 0, endMs: 3 });
});

it('checks every channel and represents entirely silent audio without a speech onset', () => {
    expect(createTtsPack('Brake.', wav([0, 0, 0, -500, 0, 0, 10000, 0, 0, 0], 2, 1000)).speechBounds)
        .toEqual({ startMs: 1, endMs: 4 });
    const silent = createTtsPack('Silent.', wav([0, 327, -327], 1, 1000));
    expect(silent.speechBounds).toBeNull();
    expect(isTtsPack(silent)).toBe(true);
    expect(isTtsPack({ ...silent, speechBounds: { startMs: 2, endMs: 1 } })).toBe(false);
    expect(isTtsPack({ ...silent, speechBounds: { startMs: 0, endMs: 4 } })).toBe(false);
});

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
