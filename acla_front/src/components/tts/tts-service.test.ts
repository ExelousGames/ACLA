import apiService from 'services/api.service';
import { isTtsPack, synthesizeTts, synthesizeTtsPack } from './tts-service';

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
