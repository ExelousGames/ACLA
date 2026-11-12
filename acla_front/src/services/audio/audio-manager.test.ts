import { audioManager, type AudioPlayer, type AudioType, type PlaybackHandle, type StreamOptions } from './index';
import { installAudioDoubles, TestAudioContext } from './test-audio';

const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const wav = btoa('test wav bytes');
const pcm = new Int16Array([0, 32767, -32768]).buffer;
let audio: ReturnType<typeof installAudioDoubles>;
let handles: PlaybackHandle[];
let players: AudioPlayer[];
const createPlayer = () => {
    const player = audioManager.createPlayer('voice');
    players.push(player);
    return player;
};
const play = (type: AudioType = 'voice', priority = 50, volume = 0.8) => {
    const handle = audioManager.play({ type, priority, volume, url: 'data:audio/wav;base64,c2F2ZWQ=' });
    handles.push(handle);
    return handle;
};
const stream = (options: Partial<StreamOptions> = {}) => {
    const handle = audioManager.createStream({ type: 'voice', priority: 50, format: 'wav', ...options });
    handles.push(handle);
    return handle;
};

beforeEach(() => { audio = installAudioDoubles(); handles = []; players = []; });
afterEach(async () => {
    players.forEach((player) => player.dispose());
    handles.forEach((handle) => handle.stop());
    await flush();
    audio.restore();
});

it('prepares one player graph and reuses it after completion, cancellation and playback failure', async () => {
    const player = createPlayer();
    await player.prepare();
    await player.prepare();
    expect(audio.media).toHaveLength(1);
    expect(audio.play).not.toHaveBeenCalled();
    const context = TestAudioContext.instances[0];
    for (const status of ['completed', 'cancelled', 'failed'] as const) {
        const handle = player.play({ url: `data:audio/wav;base64,${wav}`, priority: 25, onLevel: jest.fn() });
        if (status === 'completed') audio.media[0].dispatchEvent(new Event('ended'));
        else if (status === 'cancelled') handle.stop();
        else audio.media[0].dispatchEvent(new Event('error'));
        expect((await handle.finished).status).toBe(status);
        expect(audio.media[0].hasAttribute('src')).toBe(false);
        expect(context.close).not.toHaveBeenCalled();
    }
    expect(audio.play).toHaveBeenCalledTimes(3);
    expect(audio.media).toHaveLength(1);
    expect(TestAudioContext.instances).toHaveLength(1);
    expect(context.createMediaElementSource).toHaveBeenCalledTimes(1);
    expect(context.createAnalyser).toHaveBeenCalledTimes(1);
    const active = player.play({ url: 'data:audio/wav;base64,bmV4dA==', priority: 25 });
    player.dispose();
    player.dispose();
    expect(active.outcome).toEqual({ status: 'cancelled' });
    expect(context.close).toHaveBeenCalledTimes(1);
    expect(context.analyser.disconnect).toHaveBeenCalledTimes(1);
    expect(context.createMediaElementSource.mock.results[0].value.disconnect).toHaveBeenCalledTimes(1);
    expect(player.play({ url: 'unused', priority: 25 }).outcome).toEqual({ status: 'cancelled' });
    await expect(player.prepare()).rejects.toThrow('disposed');
});

it('keeps player priority arbitration and does not cancel the chat that interrupted it on disposal', async () => {
    const player = createPlayer();
    await player.prepare();
    const first = player.play({ url: 'data:audio/wav;base64,b25l', priority: 25 });
    expect(player.play({ url: 'unused', priority: 24 }).outcome).toEqual({ status: 'rejected' });
    expect(first.outcome).toBeUndefined();
    const next = player.play({ url: 'data:audio/wav;base64,dHdv', priority: 25 });
    expect(first.outcome).toEqual({ status: 'replaced' });
    const chat = play();
    expect(next.outcome).toEqual({ status: 'replaced' });
    expect(player.play({ url: 'unused', priority: 25 }).outcome).toEqual({ status: 'rejected' });
    player.dispose();
    expect(chat.outcome).toBeUndefined();
    expect(audio.media[1].hasAttribute('src')).toBe(true);
});

it('ignores callbacks and play rejection from a replaced clip on the same element', async () => {
    const player = createPlayer();
    await player.prepare();
    let reject!: (error: Error) => void;
    audio.play.mockImplementationOnce(() => new Promise<void>((_, fail) => { reject = fail; }));
    const onStart = jest.fn();
    const onComplete = jest.fn();
    const old = player.play({ url: 'data:audio/wav;base64,b25l', priority: 25, onStart, onComplete });
    const media = audio.media[0];
    const ended = media.onended!;
    const error = media.onerror!;
    const next = player.play({ url: 'data:audio/wav;base64,dHdv', priority: 25 });
    ended.call(media, new Event('ended'));
    error.call(media, new Event('error'));
    reject(new Error('stale failure'));
    await flush();
    expect(old.outcome).toEqual({ status: 'replaced' });
    expect(onStart).not.toHaveBeenCalled();
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(next.isActive()).toBe(true);
    expect(audio.media).toHaveLength(1);
    media.dispatchEvent(new Event('ended'));
    await expect(next.finished).resolves.toEqual({ status: 'completed' });
});

it('resumes a prepared context again if it is suspended between clips', async () => {
    const player = createPlayer();
    await player.prepare();
    const context = TestAudioContext.instances[0];
    context.state = 'suspended';
    const first = player.play({ url: 'data:audio/wav;base64,b25l', priority: 25 });
    const next = player.play({ url: 'data:audio/wav;base64,dHdv', priority: 25 });
    expect(context.resume).toHaveBeenCalledTimes(1);
    context.resolveResume();
    await flush();
    expect(first.outcome).toEqual({ status: 'replaced' });
    expect(next.isActive()).toBe(true);
    expect(TestAudioContext.instances).toHaveLength(1);
});

it('stops metering between clips and delivers new levels only to the current player request', async () => {
    jest.useFakeTimers();
    try {
        const player = createPlayer();
        await player.prepare();
        const oldLevel = jest.fn();
        player.play({ url: 'data:audio/wav;base64,b25l', priority: 25, volume: 1, onLevel: oldLevel });
        const context = TestAudioContext.instances[0];
        const media = audio.media[0];
        Object.defineProperty(media, 'paused', { configurable: true, value: false });
        context.level = 0.25;
        jest.advanceTimersByTime(16);
        await flush();
        expect(oldLevel).toHaveBeenCalledWith(0.25, 0);
        media.dispatchEvent(new Event('ended'));
        expect(jest.getTimerCount()).toBe(0);
        const onLevel = jest.fn();
        player.play({ url: 'data:audio/wav;base64,dHdv', priority: 25, volume: 0.5, onLevel });
        media.currentTime = 0.2;
        jest.advanceTimersByTime(16);
        await flush();
        expect(onLevel).toHaveBeenCalledWith(0.125, 200);
        expect(oldLevel).toHaveBeenCalledTimes(1);
        player.dispose();
        expect(jest.getTimerCount()).toBe(0);
    } finally { jest.useRealTimers(); }
});

it('reports measured output levels after playback starts and disposes the meter with playback', async () => {
    jest.useFakeTimers();
    try {
        const onLevel = jest.fn();
        const handle = audioManager.play({ type: 'voice', priority: 25, volume: 1, url: 'data:audio/wav;base64,c2F2ZWQ=', onLevel });
        handles.push(handle);
        Object.defineProperty(audio.media[0], 'paused', { configurable: true, value: false });
        await flush();
        const meter = TestAudioContext.instances[0];
        jest.advanceTimersByTime(16);
        await flush();
        expect(onLevel).toHaveBeenLastCalledWith(0, 0);
        meter.level = 0.25;
        audio.media[0].currentTime = 0.75;
        jest.advanceTimersByTime(16);
        await flush();
        expect(onLevel).toHaveBeenLastCalledWith(0.25, 750);
        audio.media[0].dispatchEvent(new Event('waiting'));
        await flush();
        expect(onLevel).toHaveBeenLastCalledWith(0, 750);
        handle.stop();
        expect(meter.close).toHaveBeenCalledTimes(1);
        expect(jest.getTimerCount()).toBe(0);
        const count = onLevel.mock.calls.length;
        jest.advanceTimersByTime(100);
        await flush();
        expect(onLevel).toHaveBeenCalledTimes(count);
    } finally { jest.useRealTimers(); }
});

it.each<AudioType>(['voice', 'music', 'alert'])('arbitrates replacement, rejection and newer ties within %s', async (type) => {
    const first = play(type, 50);
    const lower = play(type, 49);
    expect(lower.outcome).toEqual({ status: 'rejected' });
    expect(first.outcome).toBeUndefined();
    expect(audio.media).toHaveLength(1);
    const higher = play(type, 51);
    expect(first.outcome).toEqual({ status: 'replaced' });
    expect(audio.media[0].hasAttribute('src')).toBe(false);
    const tie = play(type, 51);
    expect(higher.outcome).toEqual({ status: 'replaced' });
    audio.media[2].dispatchEvent(new Event('ended'));
    await expect(tie.finished).resolves.toEqual({ status: 'completed' });
});

it('overlaps types and ducks only audible voice to 20% of configured music volume', async () => {
    const music = play('music', 100, 0.65);
    const voice = stream();
    const alert = play('alert', 200);
    expect(audio.media[0].volume).toBe(0.65);
    voice.enqueueBase64Wav(wav);
    await flush();
    expect(audio.media[0].volume).toBeCloseTo(0.13);
    expect([music.outcome, voice.outcome, alert.outcome]).toEqual([undefined, undefined, undefined]);
    audio.media[2].dispatchEvent(new Event('waiting'));
    expect(audio.media[0].volume).toBe(0.65);
    audio.media[2].dispatchEvent(new Event('playing'));
    expect(audio.media[0].volume).toBeCloseTo(0.13);
    audio.media[2].dispatchEvent(new Event('ended'));
    expect(audio.media[0].volume).toBe(0.65);
    expect(voice.outcome).toBeUndefined();
    voice.enqueueBase64Wav(wav);
    await flush();
    expect(audio.media[0].volume).toBeCloseTo(0.13);
    voice.stop();
    expect(audio.media[0].volume).toBe(0.65);
    expect(music.outcome).toBeUndefined();
});

it('does not duck for muted or blocked voice and restores on failure', async () => {
    play('music');
    play('voice', 50, 0);
    await flush();
    expect(audio.media[0].volume).toBe(0.8);
    audio.play.mockRejectedValueOnce(new Error('blocked'));
    const blocked = play();
    await flush();
    expect(blocked.outcome?.status).toBe('failed');
    expect(audio.media[0].volume).toBe(0.8);
    play();
    await flush();
    expect(audio.media[0].volume).toBeCloseTo(0.16);
    audio.media[3].dispatchEvent(new Event('error'));
    expect(audio.media[0].volume).toBe(0.8);
});

it('admits on the first chunk, keeps WAV order, and finishes only after draining', async () => {
    const current = play();
    const pending = stream();
    expect(current.outcome).toBeUndefined();
    pending.enqueueBase64Wav(btoa('first'));
    pending.enqueueBase64Wav(btoa('second'));
    pending.enqueueBase64Wav(btoa('third'));
    expect(current.outcome?.status).toBe('replaced');
    expect(audio.createObjectURL).toHaveBeenCalledTimes(1);
    pending.finish();
    expect(pending.enqueueBase64Wav(wav)).toBe(false);
    for (let i = 1; i <= 3; i++) {
        expect(audio.media[i].src).toBe(`blob:test-${i}`);
        expect(pending.outcome).toBeUndefined();
        audio.media[i].dispatchEvent(new Event('ended'));
    }
    await expect(pending.finished).resolves.toEqual({ status: 'completed' });
    expect(audio.revokeObjectURL.mock.calls).toEqual([['blob:test-1'], ['blob:test-2'], ['blob:test-3']]);
});

it('permanently discards a stream, including during silence; retained assets require explicit replay', async () => {
    const saved = Object.freeze({ audioDataUrl: 'data:audio/wav;base64,c2F2ZWQ=' });
    const voice = stream();
    voice.enqueueBase64Wav(wav);
    audio.media[0].dispatchEvent(new Event('ended'));
    expect(voice.isActive()).toBe(false);
    expect(play('voice', 49).outcome?.status).toBe('rejected');
    const replacement = play();
    replacement.stop();
    expect(voice.enqueueBase64Wav(wav)).toBe(false);
    expect(voice.outcome?.status).toBe('replaced');
    const replay = audioManager.play({ type: 'voice', priority: 50, url: saved.audioDataUrl });
    handles.push(replay);
    expect(audio.media[2].src).toBe(saved.audioDataUrl);
    expect(replay.outcome).toBeUndefined();
});

it('stops current and pending WAV chunks and ignores stale play results and DOM events', async () => {
    let resolve!: () => void;
    audio.play.mockImplementationOnce(() => new Promise<void>((done) => { resolve = done; }));
    const onStart = jest.fn();
    const onComplete = jest.fn();
    const voice = stream({ onStart, onComplete });
    voice.enqueueBase64Wav(wav);
    voice.enqueueBase64Wav(wav);
    const ended = audio.media[0].onended!;
    voice.stop();
    voice.stop();
    resolve();
    ended.call(audio.media[0], new Event('ended'));
    await flush();
    expect(onStart).not.toHaveBeenCalled();
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith({ status: 'cancelled' });
    expect(audio.media).toHaveLength(1);
    expect(audio.revokeObjectURL).toHaveBeenCalledTimes(1);
    expect(voice.enqueueBase64Wav(wav)).toBe(false);
});

it('ignores a stale play rejection without failing its replacement', async () => {
    let reject!: (error: Error) => void;
    audio.play.mockImplementationOnce(() => new Promise<void>((_, fail) => { reject = fail; }));
    const old = play();
    const next = play();
    reject(new Error('late failure'));
    await flush();
    expect(old.outcome?.status).toBe('replaced');
    expect(next.outcome).toBeUndefined();
});

it('schedules PCM chunks in order and releases every current and future node on discard', async () => {
    const voice = stream({ format: 'pcm16', sampleRate: 3 });
    voice.enqueuePcm16(pcm);
    voice.enqueuePcm16(pcm);
    const context = TestAudioContext.instances[0];
    expect(context.nodes.map((node) => node.start.mock.calls)).toEqual([[[0]], [[1]]]);
    expect(context.createBuffer.mock.results[0].value.copyToChannel.mock.calls[0][0])
        .toEqual(new Float32Array([0, 1, -1]));
    const staleEnded = context.nodes[0].onended!;
    const next = play();
    context.nodes.forEach((node) => {
        expect(node.stop).toHaveBeenCalledTimes(1);
        expect(node.disconnect).toHaveBeenCalledTimes(1);
        expect(node.buffer).toBeNull();
    });
    expect(context.gain.disconnect).toHaveBeenCalledTimes(1);
    expect(context.close).toHaveBeenCalledTimes(1);
    staleEnded();
    expect(voice.enqueuePcm16(pcm)).toBe(false);
    expect(next.outcome).toBeUndefined();
});

it('preserves PCM order across asynchronous resume and waits for all nodes to end', async () => {
    TestAudioContext.initialState = 'suspended';
    const voice = stream({ format: 'pcm16', sampleRate: 3 });
    voice.enqueuePcm16(pcm);
    voice.enqueuePcm16(pcm);
    voice.finish();
    const context = TestAudioContext.instances[0];
    expect(context.nodes).toHaveLength(0);
    context.resolveResume();
    await flush();
    expect(context.nodes.map((node) => node.start.mock.calls)).toEqual([[[0]], [[1]]]);
    context.nodes[0].onended!();
    expect(voice.outcome).toBeUndefined();
    context.nodes[1].onended!();
    await expect(voice.finished).resolves.toEqual({ status: 'completed' });
    expect(context.close).toHaveBeenCalledTimes(1);
});

it.each(['resolve', 'reject'] as const)('ignores PCM resume %s after cancellation', async (result) => {
    TestAudioContext.initialState = 'suspended';
    const voice = stream({ format: 'pcm16' });
    voice.enqueuePcm16(pcm);
    const context = TestAudioContext.instances[0];
    voice.stop();
    if (result === 'resolve') context.resolveResume();
    else context.rejectResume(new Error('late resume'));
    await flush();
    expect(context.nodes).toHaveLength(0);
    expect(voice.outcome?.status).toBe('cancelled');
    expect(context.close).toHaveBeenCalledTimes(1);
});

it('reports invalid chunks as terminal failures and permits an empty stream to finish', async () => {
    const invalidWav = stream();
    invalidWav.enqueueBase64Wav('not base64!');
    expect(invalidWav.outcome?.status).toBe('failed');
    const invalidPcm = stream({ format: 'pcm16' });
    invalidPcm.enqueuePcm16(new ArrayBuffer(1));
    expect(invalidPcm.outcome?.status).toBe('failed');
    const empty = stream();
    empty.finish();
    await expect(empty.finished).resolves.toEqual({ status: 'completed' });
    expect(TestAudioContext.instances).toHaveLength(0);
});

it('applies ducking to scheduled PCM music through its gain node', async () => {
    const music = stream({ type: 'music', format: 'pcm16', volume: 0.6 });
    music.enqueuePcm16(pcm);
    const context = TestAudioContext.instances[0];
    const voice = play();
    await flush();
    expect(context.gain.gain.value).toBeCloseTo(0.12);
    voice.stop();
    expect(context.gain.gain.value).toBe(0.6);
    expect(music.outcome).toBeUndefined();
});

it('keeps notification callbacks outside arbitration and isolates callback failures', async () => {
    const log = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    const onComplete = jest.fn(() => {
        play('alert');
        throw new Error('caller failure');
    });
    const voice = stream({ onComplete });
    voice.enqueueBase64Wav(wav);
    const next = play();
    expect(voice.outcome?.status).toBe('replaced');
    await flush();
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(next.outcome).toBeUndefined();
    expect(log).toHaveBeenCalledTimes(1);
});
