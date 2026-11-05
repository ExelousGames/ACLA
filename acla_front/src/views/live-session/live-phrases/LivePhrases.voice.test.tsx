import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { synthesizeTts, type TtsPack } from 'components/tts';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { audioManager, type PlaybackHandle } from 'services/audio';
import { installAudioDoubles } from 'services/audio/test-audio';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import LivePhrases from './LivePhrases';
import { PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';

jest.mock('components/tts', () => ({ synthesizeTts: jest.fn() }));
const synthesize = synthesizeTts as jest.MockedFunction<typeof synthesizeTts>;
const sentence = PHRASE_DEFINITIONS.find((rule) => rule.id === 'pressure-feint')!.sentence;
const sentences = Array.from(new Set([...PHRASE_DEFINITIONS.filter((rule) => rule.referenceCue),
    ...PHRASE_DEFINITIONS.filter((rule) => !rule.referenceCue)].flatMap((rule) => [rule, ...(rule.additionalActions ?? [])]
        .flatMap((action) => [action.sentence, ...(action.sentenceAlternatives ?? [])]))));
const pack = (text: string): TtsPack => ({ text, audioDataUrl: `data:audio/wav;base64,${btoa(text)}`, durationMs: 3000 });
let audio: ReturnType<typeof installAudioDoubles>;
let otherPlayback: PlaybackHandle | undefined;

const mount = () => {
    const telemetry = createLiveTelemetryStore();
    const map = circuitMap();
    let detection = vision(Date.now(), { player: 'middle' });
    let notifyVision: () => void = () => undefined;
    let sequence = 0;
    const root = {
        getComponentName: () => 'live-session',
        subscribeTelemetry: telemetry.subscribeEvents,
        getTrackVisionDetection: () => detection,
        subscribeTrackVision: (listener: () => void) => { notifyVision = listener; return () => undefined; },
        getLiveCircuitMap: () => map,
        getLiveReferenceLap: () => null,
        getLiveCornerSummaries: () => [],
        subscribeLiveCircuitMap: () => () => undefined,
    };
    const Root = () => {
        useRegisterOperationComponentRef(React.useRef(root));
        return null;
    };
    const view = render(<React.StrictMode><OperationComponentRefProvider>
        <Root /><LivePhrases name="live-phrases" />
    </OperationComponentRefProvider></React.StrictMode>);
    const frame = (speed = 100) => {
        detection = vision(Date.now(), { player: 'middle' });
        notifyVision();
        telemetry.publishFrame({
            type: 'frame', game: 'acc', sequence: ++sequence, committedCount: 0, committedSequence: 0,
            sample: { Graphics_status: 2, Physics_speed_kmh: speed, Graphics_normalized_car_position: 0.11 },
        });
    };
    return {
        ...view, telemetry, frame,
        enable: async () => { await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Enable detection' })); }); },
        trigger: async () => { await act(async () => { frame(); jest.advanceTimersByTime(800); frame(); }); },
    };
};

beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(10_000);
    audio = installAudioDoubles();
    otherPlayback = undefined;
    synthesize.mockReset().mockImplementation(async ({ text }) => pack(text));
});

afterEach(async () => {
    cleanup();
    otherPlayback?.stop();
    await act(async () => undefined);
    audio.restore();
    jest.useRealTimers();
});

it('prepares on enable, speaks each trigger once without an overlay, and reuses clips after toggling', async () => {
    const play = jest.spyOn(audioManager, 'play');
    const view = mount();
    expect(synthesize).not.toHaveBeenCalled();
    expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
    await view.enable();
    expect(synthesize.mock.calls.map(([request]) => request.text)).toEqual(sentences);
    expect(sentences).toEqual(expect.arrayContaining(['Ease the brake.', 'Throttle.', 'More throttle.', 'Less throttle.']));
    expect(sentences).not.toContain('Less brake.');
    expect(sentences).not.toContain('Release the throttle.');
    expect(sentences).not.toContain('Throttle soon.');
    expect(sentences).not.toContain('Steering.');
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    expect(play).not.toHaveBeenCalled();
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(1);
    expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(sentence).audioDataUrl, type: 'voice', priority: 25 }));
    await act(async () => { view.frame(); jest.advanceTimersByTime(250); view.frame(); });
    expect(play).toHaveBeenCalledTimes(1);
    const signal = synthesize.mock.calls[0][1]!;
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
    expect(signal.aborted).toBe(true);
    expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(2);
    view.unmount();
    expect(play.mock.results[1].value.outcome).toEqual({ status: 'cancelled' });
});

it('plays the additional chicane action using its own prepared sentence', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.closures.find((rule) => rule.id === 'second-apex')!.status = 'Active';
    snapshot.events = [{ id: 1, ruleId: 'second-apex', sentence: 'brake early, Hold inside', timestamp: Date.now() }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const play = jest.spyOn(audioManager, 'play');
    try {
        const view = mount();
        await view.enable();
        expect(synthesize).toHaveBeenCalledWith({ text: 'brake early, Hold inside', speed: 1.2 }, expect.any(AbortSignal));
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({
            url: pack('brake early, Hold inside').audioDataUrl, type: 'voice', priority: 25,
        }));
        expect(screen.getByRole('region', { name: 'Triggered sentences' })).toHaveTextContent('brake early, Hold inside');
    } finally {
        receive.mockRestore();
    }
});

it('changes driving aggression without restarting detection or speech preparation and cancels old cues', async () => {
    const play = jest.spyOn(audioManager, 'play');
    const receiveMap = jest.spyOn(PhraseEngine.prototype, 'receiveMap');
    const aggression = jest.spyOn(PhraseEngine.prototype, 'setDrivingAggression');
    try {
        const view = mount();
        const slider = screen.getByRole('slider', { name: 'Driving aggression' });
        expect(slider).toHaveValue('0');
        expect(slider).toHaveAttribute('min', '0');
        expect(slider).toHaveAttribute('max', '50');
        expect(slider).toHaveAttribute('aria-valuetext', '0% caution (Reference)');
        fireEvent.change(slider, { target: { value: '25' } });
        await view.enable();
        expect(aggression).toHaveBeenLastCalledWith(25, Date.now());
        const mapCalls = receiveMap.mock.calls.length;
        await view.trigger();
        expect(play).toHaveBeenCalledTimes(1);
        const signal = synthesize.mock.calls[0][1]!;
        fireEvent.change(slider, { target: { value: '50' } });
        expect(slider).toHaveAttribute('aria-valuetext', '50% caution relative to reference');
        expect(screen.getByText('50% caution')).toBeVisible();
        expect(aggression).toHaveBeenLastCalledWith(50, Date.now());
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        expect(receiveMap).toHaveBeenCalledTimes(mapCalls);
        expect(signal.aborted).toBe(false);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(screen.getByText('Speech: Ready')).toBeInTheDocument();
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Triggered sentences' })).not.toHaveTextContent(sentence);
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        await view.enable();
        expect(slider).toHaveValue('50');
        expect(aggression).toHaveBeenLastCalledWith(50, Date.now());
    } finally {
        play.mockRestore();
        receiveMap.mockRestore();
        aggression.mockRestore();
    }
});

it('shows speech as ready only after the final TTS clip is received', async () => {
    const lastSentence = sentences[sentences.length - 1];
    let resolve!: (value: TtsPack) => void;
    const pending = new Promise<TtsPack>((done) => { resolve = done; });
    synthesize.mockImplementation(async ({ text }) => text === lastSentence ? pending : pack(text));
    const view = mount();
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    expect(screen.getByText('Speech: Preparing')).toHaveAttribute('data-ready', 'false');
    expect(screen.queryByText('Speech: Ready')).not.toBeInTheDocument();
    await act(async () => { resolve(pack(lastSentence)); });
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
});

it.each(['left', 'right'])('prepares every variant and plays the chosen %s movement wording once', async (direction) => {
    const chosen = `Bring it to the ${direction}.`;
    synthesize.mockImplementation(async ({ text }) => ({ ...pack(text), durationMs: 500 }));
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    const rule = snapshot.closures.find((item) => item.id === `reference-move-${direction}-small`)!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 36)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence: chosen, timestamp: Date.now(), expiresAt: Date.now() + 1500 }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    const play = jest.spyOn(audioManager, 'play');
    try {
        const view = mount();
        await view.enable();
        expect(synthesize.mock.calls.map(([request]) => request.text)).toEqual(sentences);
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(chosen).audioDataUrl }));
        expect(screen.getByRole('region', { name: 'Triggered sentences' })).toHaveTextContent(chosen);
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it.each(['on time', 'too late', 'clip too long', 'condition cleared'])(
    'only speaks a prepared reference cue when it can finish before the action: %s', async (state) => {
        const snapshot = new PhraseEngine().evaluate(Date.now());
        snapshot.telemetryReady = true;
        const rule = snapshot.closures.find((item) => item.id === 'reference-brake')!;
        rule.conditions = [new PhraseCondition('speed', '>=', 20, 36)];
        snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Brake.', timestamp: Date.now(), expiresAt: Date.now() + 1200 }];
        const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
        let resolve!: (value: TtsPack) => void;
        const pending = new Promise<TtsPack>((done) => { resolve = done; });
        synthesize.mockImplementation(async ({ text }) => text === 'Brake.' ? pending : pack(text));
        try {
            const view = mount();
            await view.enable();
            await act(async () => { view.frame(); });
            expect(audio.play).not.toHaveBeenCalled();
            if (state === 'too late') jest.setSystemTime(Date.now() + 800);
            if (state === 'condition cleared') {
                rule.conditions = [new PhraseCondition('speed', '>=', 20, 0)];
                await act(async () => { view.frame(); });
            }
            await act(async () => { resolve({ ...pack('Brake.'), durationMs: state === 'clip too long' ? 1300 : 500 }); });
            expect(audio.play).toHaveBeenCalledTimes(state === 'on time' ? 1 : 0);
        } finally { receive.mockRestore(); }
    });

it('lets Turn interrupt an earlier phrase', async () => {
    synthesize.mockImplementation(async ({ text }) => ({ ...pack(text), durationMs: text === 'Turn' ? 400 : 3000 }));
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = 'prepare';
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 80)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence, timestamp: Date.now(), expiresAt: Date.now() + 5000 }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    const play = jest.spyOn(audioManager, 'play');
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(sentence).audioDataUrl }));
        jest.setSystemTime(Date.now() + 1100);
        snapshot.referenceSteeringPhase = 'turn';
        snapshot.events = [...snapshot.events, { id: 2, ruleId: rule.id, sentence: 'Turn', timestamp: Date.now(),
            expiresAt: Date.now() + 1500, steeringPhase: 'turn' }];
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(2);
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack('Turn').audioDataUrl }));
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it('discards an unprepared Turn cue when the reference leaves turn-in', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = 'turn';
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 80)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Turn', timestamp: Date.now(),
        expiresAt: Date.now() + 1500, steeringPhase: 'turn' }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    let resolve!: (value: TtsPack) => void;
    const pending = new Promise<TtsPack>((done) => { resolve = done; });
    synthesize.mockImplementation(async ({ text }) => text === 'Turn' ? pending : pack(text));
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        snapshot.referenceSteeringPhase = 'prepare';
        await act(async () => { view.frame(); resolve({ ...pack('Turn'), durationMs: 400 }); });
        expect(audio.play).not.toHaveBeenCalled();
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it('lets live chat interrupt and suppress phrases without replaying discarded events', async () => {
    const play = jest.spyOn(audioManager, 'play');
    const view = mount();
    await view.enable();
    await view.trigger();
    const chat = audioManager.createStream({ type: 'voice', priority: 50, format: 'wav' });
    otherPlayback = chat;
    chat.enqueueBase64Wav(btoa('chat'));
    expect(play.mock.results[0].value.outcome).toEqual({ status: 'replaced' });
    // Chat retains voice priority even between spoken chunks.
    audio.media[1].dispatchEvent(new Event('ended'));
    act(() => view.telemetry.resetSession());
    await view.trigger();
    expect(play.mock.results[1].value.outcome).toEqual({ status: 'rejected' });
    expect(chat.outcome).toBeUndefined();
    chat.stop();
    await act(async () => { view.frame(); jest.advanceTimersByTime(250); view.frame(); });
    expect(play).toHaveBeenCalledTimes(2);
    act(() => view.telemetry.resetSession());
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(3);
    expect(play.mock.results[2].value.outcome).toBeUndefined();
    act(() => view.telemetry.resetSession());
    expect(play.mock.results[2].value.outcome).toEqual({ status: 'cancelled' });
});

it.each(['ready', 'disable', 'reset', 'expired', 'condition cleared', 'unmount'])(
    'handles a trigger during TTS preparation when %s', async (state) => {
        let resolve!: (value: TtsPack) => void;
        const pending = new Promise<TtsPack>((done) => { resolve = done; });
        synthesize.mockImplementation(async ({ text }) => text === sentence ? pending : pack(text));
        const view = mount();
        await view.enable();
        // Preparation is sequential; later sentences wait for this request.
        expect(synthesize).toHaveBeenCalledTimes(sentences.indexOf(sentence) + 1);
        await view.trigger();
        expect(audio.play).not.toHaveBeenCalled();
        if (state === 'disable') fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        if (state === 'reset') act(() => view.telemetry.resetSession());
        if (state === 'expired') jest.setSystemTime(Date.now() + 8000);
        if (state === 'condition cleared') act(() => view.frame(0));
        if (state === 'unmount') view.unmount();
        await act(async () => { resolve(pack(sentence)); });
        expect(audio.play).toHaveBeenCalledTimes(state === 'ready' ? 1 : 0);
        if (state === 'disable') expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
        if (state === 'ready') {
            await act(async () => { view.frame(); });
            expect(audio.play).toHaveBeenCalledTimes(1);
        }
        if (state === 'disable' || state === 'unmount') {
            expect(synthesize.mock.calls.every(([, signal]) => signal?.aborted)).toBe(true);
        }
    },
);

it('keeps preparing other sentences after a synthesis failure and reports playback failures', async () => {
    synthesize.mockRejectedValueOnce(new Error('Speech service unavailable'));
    const view = mount();
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    expect(screen.getByRole('alert')).toHaveTextContent('Speech service unavailable');
    expect(screen.getByText('Speech: Incomplete')).toHaveAttribute('data-ready', 'false');
    audio.play.mockRejectedValueOnce(new Error('Audio blocked'));
    await view.trigger();
    expect(screen.getByRole('alert')).toHaveTextContent('Audio blocked');
    expect(screen.getByRole('region', { name: 'Triggered sentences' })).toHaveTextContent(sentence);
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length + 1);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
});
