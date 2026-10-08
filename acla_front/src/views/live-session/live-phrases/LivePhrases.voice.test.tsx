import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { synthesizeTts, type TtsPack } from 'components/tts';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { audioManager, type PlaybackHandle } from 'services/audio';
import { installAudioDoubles } from 'services/audio/test-audio';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import LivePhrases from './LivePhrases';
import { PHRASE_RULES } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';

jest.mock('components/tts', () => ({ synthesizeTts: jest.fn() }));
const synthesize = synthesizeTts as jest.MockedFunction<typeof synthesizeTts>;
const sentence = PHRASE_RULES.find((rule) => rule.id === 'inside-outbraking')!.sentence;
const pack = (text: string): TtsPack => ({ text, audioDataUrl: `data:audio/wav;base64,${btoa(text)}`, durationMs: 3000 });
let audio: ReturnType<typeof installAudioDoubles>;
let otherPlayback: PlaybackHandle | undefined;

const mount = () => {
    const telemetry = createLiveTelemetryStore();
    const map = circuitMap();
    let detection = vision(Date.now());
    let notifyVision: () => void = () => undefined;
    let sequence = 0;
    const root = {
        getComponentName: () => 'live-session',
        subscribeTelemetry: telemetry.subscribeEvents,
        getTrackVisionDetection: () => detection,
        subscribeTrackVision: (listener: () => void) => { notifyVision = listener; return () => undefined; },
        getLiveCircuitMap: () => map,
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
        detection = vision(Date.now());
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
    expect(synthesize.mock.calls.map(([request]) => request.text)).toEqual(PHRASE_RULES.map((rule) => rule.sentence));
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
    expect(synthesize).toHaveBeenCalledTimes(PHRASE_RULES.length);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(2);
    view.unmount();
    expect(play.mock.results[1].value.outcome).toEqual({ status: 'cancelled' });
});

it('shows speech as ready only after the final TTS clip is received', async () => {
    const lastSentence = PHRASE_RULES[PHRASE_RULES.length - 1].sentence;
    let resolve!: (value: TtsPack) => void;
    const pending = new Promise<TtsPack>((done) => { resolve = done; });
    synthesize.mockImplementation(async ({ text }) => text === lastSentence ? pending : pack(text));
    const view = mount();
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(PHRASE_RULES.length);
    expect(screen.getByText('Speech: Preparing')).toHaveAttribute('data-ready', 'false');
    expect(screen.queryByText('Speech: Ready')).not.toBeInTheDocument();
    await act(async () => { resolve(pack(lastSentence)); });
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
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
        expect(synthesize).toHaveBeenCalledTimes(PHRASE_RULES.findIndex((rule) => rule.sentence === sentence) + 1);
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
    expect(synthesize).toHaveBeenCalledTimes(PHRASE_RULES.length);
    expect(screen.getByRole('alert')).toHaveTextContent('Speech service unavailable');
    expect(screen.getByText('Speech: Incomplete')).toHaveAttribute('data-ready', 'false');
    audio.play.mockRejectedValueOnce(new Error('Audio blocked'));
    await view.trigger();
    expect(screen.getByRole('alert')).toHaveTextContent('Audio blocked');
    expect(screen.getByRole('region', { name: 'Triggered sentences' })).toHaveTextContent(sentence);
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(PHRASE_RULES.length + 1);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
});
