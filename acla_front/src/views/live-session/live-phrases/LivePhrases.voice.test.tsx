import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { synthesizeProsodyTts, type TtsPack } from 'components/tts';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { audioManager, type PlaybackHandle } from 'services/audio';
import { installAudioDoubles } from 'services/audio/test-audio';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import LivePhrases from './LivePhrases';
import { PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';
import { Phrase } from './phrase';
import brakeCurves from './curves/reference-brake.json';
import { getPhraseSpeechInstances, loadAllPhraseSpeech } from './phrase-speech';
import { brakeReleaseSettings, SLOWLY_RELEASE_BRAKE } from './phrase-brake-release';

jest.mock('components/tts', () => ({ synthesizeProsodyTts: jest.fn() }));
const synthesize = synthesizeProsodyTts as jest.MockedFunction<typeof synthesizeProsodyTts>;
const sentence = PHRASE_DEFINITIONS.find((rule) => rule.id === 'pressure-feint')!.sentence;
const instances = getPhraseSpeechInstances();
const sentences = instances.map((instance) => instance.speech.text);
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
    synthesize.mockReset().mockImplementation(async (_curves, _inputs, { text }) => pack(text));
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
    expect(synthesize.mock.calls.map(([, , request]) => request.text)).toEqual(sentences);
    expect(sentences).toEqual(expect.arrayContaining(['Release brake slower.', 'Release brake faster.', 'Hold brake pressure.', 'Throttle.', 'More throttle.', 'Less throttle.']));
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
    const signal = synthesize.mock.calls[0][3]!;
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
        expect(synthesize).toHaveBeenCalledWith(expect.any(Array), [0, 0, 0], { text: 'brake early, Hold inside', speed: 1.2, language: 'en-us' }, expect.any(AbortSignal));
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
        expect(slider).toHaveAttribute('max', '100');
        expect(slider).toHaveAttribute('aria-valuetext', '0% earlier braking (Reference)');
        fireEvent.change(slider, { target: { value: '25' } });
        await view.enable();
        expect(aggression).toHaveBeenLastCalledWith(25, Date.now());
        const mapCalls = receiveMap.mock.calls.length;
        await view.trigger();
        expect(play).toHaveBeenCalledTimes(1);
        const signal = synthesize.mock.calls[0][3]!;
        fireEvent.change(slider, { target: { value: '100' } });
        expect(slider).toHaveAttribute('aria-valuetext', '100% earlier braking (Direction guide)');
        expect(screen.getByText('100% · Direction guide')).toBeVisible();
        expect(screen.getByText('100% earlier braking')).toBeVisible();
        expect(aggression).toHaveBeenLastCalledWith(100, Date.now());
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        expect(receiveMap).toHaveBeenCalledTimes(mapCalls);
        expect(signal.aborted).toBe(false);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(screen.getByText('Speech: Ready')).toBeInTheDocument();
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Triggered sentences' })).not.toHaveTextContent(sentence);
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        await view.enable();
        expect(slider).toHaveValue('100');
        expect(aggression).toHaveBeenLastCalledWith(100, Date.now());
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
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === lastSentence ? pending : pack(text));
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
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({ ...pack(text), durationMs: 500 }));
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
        expect(synthesize.mock.calls.map(([, , request]) => request.text)).toEqual(sentences);
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
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === 'Brake.' ? pending : pack(text));
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
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({ ...pack(text), durationMs: text === 'Turn' ? 400 : 3000 }));
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = 'prepare';
    snapshot.state.path = ['root', 'Steering'];
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
        snapshot.events = [...snapshot.events, { id: 2, ruleId: rule.id, sentence: 'Turn', timestamp: Date.now(),
            expiresAt: Date.now() + 1500, steeringPhase: 'prepare' }];
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(2);
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack('Turn').audioDataUrl }));
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it.each(['reference starts turn-in', 'reference leaves turn-in', 'Steering exits'])('checks a queued Turn when %s', async (state) => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = state === 'reference starts turn-in' ? 'prepare' : 'turn';
    snapshot.state.path = ['root', 'Steering', 'say phrase'];
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 80)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Turn', timestamp: Date.now(),
        expiresAt: Date.now() + 1500, steeringPhase: snapshot.referenceSteeringPhase }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    let resolve!: (value: TtsPack) => void;
    const pending = new Promise<TtsPack>((done) => { resolve = done; });
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === 'Turn' ? pending : pack(text));
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        if (state === 'Steering exits') snapshot.state.path = ['root'];
        else snapshot.referenceSteeringPhase = state === 'reference starts turn-in' ? 'turn' : 'prepare';
        await act(async () => { view.frame(); resolve({ ...pack('Turn'), durationMs: 400 }); });
        expect(audio.play).toHaveBeenCalledTimes(state === 'Steering exits' ? 0 : 1);
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it.each(['still wide', 'aligned', 'opposite', 'missing direction', 'exited Steering', 'stale telemetry'])(
    'checks a queued direction correction when %s, independently of Steering entry conditions', async (state) => {
        const snapshot = new PhraseEngine().evaluate(Date.now());
        snapshot.telemetryReady = true;
        snapshot.referenceSteeringDirection = 'tooWide';
        snapshot.state.path = ['root', 'Steering', 'Too wide', 'say phrase'];
        const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
        rule.conditions = [new PhraseCondition('referenceCue', '=', 'steering', 'none')];
        snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Too wide', timestamp: Date.now(),
            expiresAt: Date.now() + 1500, steeringDirection: 'tooWide' }];
        const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
        const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
        let resolve!: (value: TtsPack) => void;
        const pending = new Promise<TtsPack>((done) => { resolve = done; });
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === 'Too wide' ? pending : pack(text));
        try {
            const view = mount();
            await view.enable();
            receiveVision.mockReturnValue(snapshot);
            await act(async () => { view.frame(); });
            if (state === 'aligned') snapshot.referenceSteeringDirection = 'aligned';
            if (state === 'opposite') snapshot.referenceSteeringDirection = 'tooTight';
            if (state === 'missing direction') snapshot.referenceSteeringDirection = undefined;
            if (state === 'exited Steering') snapshot.state.path = ['root'];
            if (state === 'stale telemetry') snapshot.telemetryReady = false;
            await act(async () => { view.frame(); resolve({ ...pack('Too wide'), durationMs: 400 }); });
            expect(audio.play).toHaveBeenCalledTimes(state === 'still wide' ? 1 : 0);
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
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === sentence ? pending : pack(text));
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
            expect(synthesize.mock.calls.every(([, , , signal]) => signal?.aborted)).toBe(true);
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

it('preloads every deceleration delivery and only reads cached instances during playback', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.state.path = ['root', 'Steering'];
    snapshot.referenceBrakeReleaseRate = 0.25;
    // Once entered, the release action uses Steering scope and its own condition, not entry conditions.
    snapshot.closures.find((rule) => rule.id === 'reference-steering')!.conditions = [new PhraseCondition('referenceCue', '=', 'steering', 'none')];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const play = jest.spyOn(audioManager, 'play');
    synthesize.mockImplementation(async (_curves, inputs, speech) => pack(`${speech.text}:${inputs[0]}`));
    try {
        const view = mount();
        await view.enable();
        const preparedCount = synthesize.mock.calls.length;
        expect(synthesize.mock.calls.filter(([, , speech]) => speech.text === SLOWLY_RELEASE_BRAKE)).toHaveLength(4);
        for (const [index, rate] of Array.from([0.25, 0.75, 1.5, 3, 0.25].entries())) {
            const instance = instances.find((item) => item.speech.text === SLOWLY_RELEASE_BRAKE
                && item.inputs.timing === brakeReleaseSettings(rate).inputs!.timing)!;
            snapshot.referenceBrakeReleaseRate = rate;
            snapshot.events = [{ id: index + 1, ruleId: 'reference-steering', sentence: SLOWLY_RELEASE_BRAKE,
                instance, timestamp: Date.now(), expiresAt: Date.now() + 4000, steeringPhase: 'prepare', brakeRelease: true }];
            await act(async () => { view.frame(); });
            expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(`${SLOWLY_RELEASE_BRAKE}:${instance.inputs.timing}`).audioDataUrl }));
        }
        expect(synthesize).toHaveBeenCalledTimes(preparedCount);
        expect(play).toHaveBeenCalledTimes(5);
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        await view.enable();
        await act(async () => { view.frame(); });
        expect(synthesize).toHaveBeenCalledTimes(preparedCount);
    } finally { receive.mockRestore(); play.mockRestore(); }
});

it('skips an unprepared instance without synthesizing or substituting default audio', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.closures.find((rule) => rule.id === 'pressure-feint')!.status = 'Active';
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    try {
        const view = mount();
        await view.enable();
        const instance = new Phrase('pressure-feint:say:0', sentence, brakeCurves).createInstance({ inputs: { energy: 0.5 } });
        snapshot.events = [{ id: 1, ruleId: 'pressure-feint', sentence, instance, timestamp: Date.now() }];
        await act(async () => { view.frame(); view.frame(); });
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(audio.play).not.toHaveBeenCalled();
        snapshot.events = [{ id: 2, ruleId: 'pressure-feint', sentence, timestamp: Date.now() }];
        await act(async () => { view.frame(); });
        expect(audio.play).toHaveBeenCalledTimes(1);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    } finally { receive.mockRestore(); }
});

it('loads an explicit batch once, preserving different settings and reusing identical instances', async () => {
    const phrase = new Phrase('custom', sentence, brakeCurves);
    const variants = [phrase.createInstance({ inputs: { energy: 0.25 } }), phrase.createInstance({ inputs: { energy: 0.75 } })];
    const packs = new Map<string, TtsPack>();
    const signal = new AbortController().signal;
    expect(await loadAllPhraseSpeech([...variants, variants[0]], packs, signal)).toEqual({ complete: true, errors: [] });
    await loadAllPhraseSpeech(variants, packs, signal);
    expect(synthesize).toHaveBeenCalledTimes(2);
    expect(packs.size).toBe(2);
    expect(synthesize).toHaveBeenCalledWith(variants[1].curves, [0, 0.75, 0], variants[1].speech, signal);
});

it.each(['ready', 'release ended', 'exited Steering', 'stale telemetry'])(
    'waits for the complete startup batch and checks pending brake-release validity: %s', async (state) => {
        const snapshot = new PhraseEngine().evaluate(Date.now());
        snapshot.telemetryReady = true;
        snapshot.state.path = ['root', 'Steering'];
        snapshot.referenceBrakeReleaseRate = 0.25;
        const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
        rule.conditions = [new PhraseCondition('speed', '>=', 20, 36)];
        const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
        const instance = instances.find((item) => item.speech.text === SLOWLY_RELEASE_BRAKE && item.inputs.timing === 1)!;
        let resolve!: (value: TtsPack) => void;
        const pending = new Promise<TtsPack>((done) => { resolve = done; });
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === sentences[sentences.length - 1] ? pending : pack(text));
        try {
            const view = mount();
            await view.enable();
            snapshot.events = [{ id: 1, ruleId: rule.id, sentence: SLOWLY_RELEASE_BRAKE, instance, timestamp: Date.now(),
                expiresAt: Date.now() + 4000, steeringPhase: 'prepare', brakeRelease: true }];
            await act(async () => { view.frame(); view.frame(); });
            expect(synthesize).toHaveBeenCalledTimes(sentences.length);
            expect(audio.play).not.toHaveBeenCalled();
            if (state === 'release ended') snapshot.referenceBrakeReleaseRate = undefined;
            if (state === 'exited Steering') snapshot.state.path = ['root'];
            if (state === 'stale telemetry') snapshot.telemetryReady = false;
            await act(async () => { view.frame(); resolve(pack(sentences[sentences.length - 1])); });
            expect(audio.play).toHaveBeenCalledTimes(state === 'ready' ? 1 : 0);
            expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        } finally { receive.mockRestore(); }
    },
);
