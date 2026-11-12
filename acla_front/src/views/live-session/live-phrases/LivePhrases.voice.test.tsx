import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { synthesizeProsodyTts, type TtsPack } from 'components/tts';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { audioManager, type AudioPlayer, type PlaybackHandle } from 'services/audio';
import { installAudioDoubles, TestAudioContext } from 'services/audio/test-audio';
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

const spyOnPlayer = () => {
    const play = jest.fn<PlaybackHandle, Parameters<AudioPlayer['play']>>();
    const createPlayer = audioManager.createPlayer.bind(audioManager);
    jest.spyOn(audioManager, 'createPlayer').mockImplementation((type) => {
        const player = createPlayer(type);
        const originalPlay = player.play;
        player.play = (options) => {
            play.mockImplementationOnce(originalPlay);
            return play(options);
        };
        return player;
    });
    return play;
};

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
    const play = spyOnPlayer();
    const view = mount();
    expect(synthesize).not.toHaveBeenCalled();
    expect(audio.media).toHaveLength(0);
    expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
    await view.enable();
    expect(audio.media).toHaveLength(1);
    expect(TestAudioContext.instances).toHaveLength(1);
    expect(audio.play).not.toHaveBeenCalled();
    expect(synthesize.mock.calls.map(([, , request]) => request.text)).toEqual(sentences);
    expect(sentences).not.toContain('More left');
    expect(sentences).not.toContain('More right');
    expect(sentences).toEqual(expect.arrayContaining(['Hold brake.', 'Squeeze gas.', 'More gas.', 'Ease gas.']));
    expect(sentences).not.toContain('Release brake slower.');
    expect(sentences).not.toContain('Release brake faster.');
    expect(sentences).not.toContain('Less brake.');
    expect(sentences).not.toContain('Release the throttle.');
    expect(sentences).not.toContain('Throttle soon.');
    expect(sentences).not.toContain('Steering.');
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    expect(play).not.toHaveBeenCalled();
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(1);
    expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(sentence).audioDataUrl, priority: 25 }));
    await act(async () => { view.frame(); jest.advanceTimersByTime(250); view.frame(); });
    expect(play).toHaveBeenCalledTimes(1);
    const signal = synthesize.mock.calls[0][3]!;
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
    expect(signal.aborted).toBe(true);
    expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
    expect(TestAudioContext.instances[0].close).toHaveBeenCalledTimes(1);
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(2);
    expect(audio.media).toHaveLength(2);
    expect(TestAudioContext.instances).toHaveLength(2);
    view.unmount();
    expect(play.mock.results[1].value.outcome).toEqual({ status: 'cancelled' });
    expect(TestAudioContext.instances[1].close).toHaveBeenCalledTimes(1);
});

it.each(['ready', 'failed', 'disabled'])('waits for player preparation when the audio context is %s', async (state) => {
    TestAudioContext.initialState = 'suspended';
    const view = mount();
    await view.enable();
    const context = TestAudioContext.instances[0];
    expect(context.resume).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Speech: Preparing')).toBeInTheDocument();
    await view.trigger();
    expect(audio.play).not.toHaveBeenCalled();
    if (state === 'disabled') fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    await act(async () => {
        if (state === 'failed') context.rejectResume(new Error('Audio blocked'));
        else context.resolveResume();
    });
    if (state === 'ready') {
        expect(screen.getByText('Speech: Ready')).toBeInTheDocument();
        await view.trigger();
        expect(audio.play).toHaveBeenCalledTimes(1);
        expect(TestAudioContext.instances).toHaveLength(1);
    } else {
        expect(audio.play).not.toHaveBeenCalled();
        expect(screen.getByText(state === 'failed' ? 'Speech: Incomplete' : 'Speech: Disabled')).toBeInTheDocument();
        if (state === 'failed') expect(screen.getByRole('alert')).toHaveTextContent('Audio blocked');
        else expect(context.close).toHaveBeenCalledTimes(1);
    }
});

it('shows phrase response from trigger through delayed playback and audio completion', async () => {
    let resolvePlay!: () => void;
    audio.play.mockImplementationOnce(() => new Promise<void>((resolve) => { resolvePlay = resolve; }));
    const view = mount();
    await view.enable();
    await view.trigger();
    const timeline = within(screen.getByRole('list', { name: 'Phrase response timeline' }));
    const stage = (label: string) => within(timeline.getByText(label).closest('li')!);
    const time = (label: string) => stage(label).getByText(/\d+:\d+:\d+/).getAttribute('datetime');
    expect(timeline.getAllByRole('listitem').map((item) => item.querySelector('strong')?.textContent))
        .toEqual(['Trigger', 'Play request', 'Playback start', 'Audio end', 'Phrase release']);
    expect(stage('Trigger').getByText('+0 ms')).toBeInTheDocument();
    expect(stage('Play request').getByText('+0 ms')).toBeInTheDocument();
    expect(stage('Playback start').getByText('Pending')).toBeInTheDocument();
    const triggeredAt = Date.parse(time('Trigger')!);
    await act(async () => { jest.advanceTimersByTime(120); resolvePlay(); });
    const startedAt = time('Playback start');
    expect(Date.parse(startedAt!) - triggeredAt).toBe(120);
    expect(stage('Playback start').getByText('+120 ms')).toBeInTheDocument();
    const media = audio.media[0];
    await act(async () => {
        media.dispatchEvent(new Event('waiting'));
        jest.advanceTimersByTime(200);
        media.dispatchEvent(new Event('playing'));
    });
    expect(time('Playback start')).toBe(startedAt);
    expect(stage('Audio end').getByText('Pending')).toBeInTheDocument();
    expect(stage('Phrase release').getByText('Pending')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(100); media.dispatchEvent(new Event('ended')); });
    expect(stage('Audio end').getByText('+420 ms')).toBeInTheDocument();
    expect(stage('Audio end').getByText('Completed')).toBeInTheDocument();
    expect(stage('Phrase release').getByText('+420 ms')).toBeInTheDocument();
    expect(time('Audio end')).toBe(time('Phrase release'));
});

it('updates history for phrase state changes without reformatting it on telemetry or timer updates', async () => {
    const view = mount();
    await view.enable();
    await view.trigger();
    const history = screen.getByRole('region', { name: 'Phrase history' });
    const initialHistory = history.innerHTML;
    const formatTimestamp = jest.spyOn(Date.prototype, 'toISOString');

    for (let frame = 0; frame < 10; frame++) {
        await act(async () => { jest.advanceTimersByTime(50); view.frame(100 + frame); });
    }
    expect(history.innerHTML).toBe(initialHistory);
    expect(formatTimestamp).not.toHaveBeenCalled();
    expect(audio.play).toHaveBeenCalledTimes(1);

    await act(async () => { audio.media[0].dispatchEvent(new Event('ended')); });
    expect(within(history).getByText('Completed')).toBeInTheDocument();
    expect(formatTimestamp).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    expect(within(history).queryByRole('list', { name: 'Phrase response timeline' })).not.toBeInTheDocument();
});

it.each(['rejected', 'failed', 'replaced'] as const)('shows %s playback in phrase history', async (outcome) => {
    const view = mount();
    await view.enable();
    if (outcome === 'rejected') otherPlayback = audioManager.play({ url: 'chat.wav', type: 'voice', priority: 50 });
    if (outcome === 'failed') audio.play.mockRejectedValueOnce(new Error('Audio blocked'));
    await view.trigger();
    if (outcome === 'replaced') await act(async () => {
        jest.advanceTimersByTime(100);
        otherPlayback = audioManager.play({ url: 'chat.wav', type: 'voice', priority: 50 });
    });
    const timeline = within(screen.getByRole('list', { name: 'Phrase response timeline' }));
    const stage = (label: string) => within(timeline.getByText(label).closest('li')!);
    expect(stage('Audio end').getByText({ rejected: 'Rejected', failed: 'Failed', replaced: 'Interrupted' }[outcome])).toBeInTheDocument();
    expect(stage('Phrase release').queryByText('Pending')).not.toBeInTheDocument();
    if (outcome !== 'replaced') expect(stage('Playback start').getByText('Not reached')).toBeInTheDocument();
    else expect(stage('Playback start').queryByText('Not reached')).not.toBeInTheDocument();
});

it('plays the additional chicane action using its own prepared sentence', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.closures.find((rule) => rule.id === 'second-apex')!.status = 'Active';
    snapshot.events = [{ id: 1, ruleId: 'second-apex', sentence: 'Brake early. Hold inside.', timestamp: Date.now() }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const play = spyOnPlayer();
    try {
        const view = mount();
        await view.enable();
        expect(synthesize).toHaveBeenCalledWith(expect.any(Array), [0, 0, 0], { text: 'Brake early. Hold inside.', speed: 1.2, language: 'en-us' }, expect.any(AbortSignal), true);
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({
            url: pack('Brake early. Hold inside.').audioDataUrl, priority: 25,
        }));
        expect(screen.getByRole('region', { name: 'Phrase history' })).toHaveTextContent('Brake early. Hold inside.');
    } finally {
        receive.mockRestore();
    }
});

it('uses the trimmed duration for entry forecasts and waits through pauses and buffering for playback completion', async () => {
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({
        ...pack(text), durationMs: 1400, speechBounds: { startMs: 0, endMs: 1400 },
    }));
    const durations = jest.spyOn(PhraseEngine.prototype, 'setSpeechDurations');
    const started = jest.spyOn(PhraseEngine.prototype, 'speechStarted');
    const finished = jest.spyOn(PhraseEngine.prototype, 'speechFinished');
    const play = spyOnPlayer();
    const view = mount();
    await view.enable();
    expect(Array.from(durations.mock.calls[durations.mock.calls.length - 1][0].values())).toEqual(instances.map(() => 1400));
    await view.trigger();
    const media = audio.media[0];
    const meter = TestAudioContext.instances[0];
    await act(async () => { jest.advanceTimersByTime(500); });
    expect(screen.getByLabelText('Closure state')).toHaveTextContent('say phrase');
    expect(started).not.toHaveBeenCalled();
    Object.defineProperty(media, 'paused', { configurable: true, value: false });
    media.currentTime = 0.001;
    meter.level = 0.2;
    await act(async () => { jest.advanceTimersByTime(16); });
    expect(started).toHaveBeenCalledTimes(1);
    // A gap 506 ms into the clip must not release the phrase for a sibling cue.
    media.currentTime = 0.506;
    meter.level = 0;
    await act(async () => { jest.advanceTimersByTime(16); });
    expect(screen.getByLabelText('Closure state')).toHaveTextContent('say phrase');
    await act(async () => { media.dispatchEvent(new Event('waiting')); jest.advanceTimersByTime(3500); view.frame(); });
    expect(finished).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Closure state')).toHaveTextContent('say phrase');
    media.currentTime = 1.399;
    meter.level = 0.2;
    await act(async () => { jest.advanceTimersByTime(16); });
    expect(finished).not.toHaveBeenCalled();
    media.currentTime = 1.4;
    meter.level = 0;
    await act(async () => { jest.advanceTimersByTime(16); });
    expect(finished).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Closure state')).toHaveTextContent('say phrase');
    expect(play.mock.results[0].value.isActive()).toBe(true);
    expect(screen.getByRole('region', { name: 'Phrase scheduler' })).toHaveTextContent(sentence);
    expect(audio.pause).not.toHaveBeenCalled();
    await act(async () => { media.dispatchEvent(new Event('ended')); });
    expect(finished).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText('Closure state')).toHaveTextContent(/^State: root$/);
    expect(screen.getByRole('region', { name: 'Phrase scheduler' })).not.toHaveTextContent(sentence);
    expect(play.mock.results[0].value.outcome).toEqual({ status: 'completed' });
    view.unmount();
    expect(meter.close).toHaveBeenCalledTimes(1);
    const reads = meter.analyser.getFloatTimeDomainData.mock.calls.length;
    await act(async () => { jest.advanceTimersByTime(1000); });
    expect(meter.analyser.getFloatTimeDomainData).toHaveBeenCalledTimes(reads);
});

it.each([undefined, null])('waits for natural completion when speech bounds are %s', async (speechBounds) => {
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({ ...pack(text), speechBounds }));
    const started = jest.spyOn(PhraseEngine.prototype, 'speechStarted');
    const finished = jest.spyOn(PhraseEngine.prototype, 'speechFinished');
    const view = mount();
    await view.enable();
    await view.trigger();
    const media = audio.media[0];
    Object.defineProperty(media, 'paused', { configurable: true, value: false });
    const meter = TestAudioContext.instances[0];
    if (speechBounds === undefined) meter.level = 0.2;
    await act(async () => { jest.advanceTimersByTime(16); });
    expect(started).toHaveBeenCalledTimes(speechBounds === undefined ? 1 : 0);
    meter.level = 0;
    await act(async () => { jest.advanceTimersByTime(500); });
    expect(finished).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Closure state')).toHaveTextContent('say phrase');
    await act(async () => { media.dispatchEvent(new Event('ended')); });
    expect(screen.getByLabelText('Closure state')).toHaveTextContent(/^State: root$/);
});

it('changes instructor aggression without restarting detection or speech preparation and cancels old cues', async () => {
    const play = spyOnPlayer();
    const receiveMap = jest.spyOn(PhraseEngine.prototype, 'receiveMap');
    const aggression = jest.spyOn(PhraseEngine.prototype, 'setDrivingAggression');
    try {
        const view = mount();
        const slider = screen.getByRole('slider', { name: 'Instructor aggression' });
        expect(slider).toHaveValue('0');
        expect(slider).toHaveAttribute('min', '0');
        expect(slider).toHaveAttribute('max', '100');
        expect(slider).toHaveAttribute('aria-valuetext', '0% braking adjustment (Reference)');
        fireEvent.change(slider, { target: { value: '25' } });
        await view.enable();
        expect(aggression).toHaveBeenLastCalledWith(25, Date.now());
        const mapCalls = receiveMap.mock.calls.length;
        await view.trigger();
        expect(play).toHaveBeenCalledTimes(1);
        const signal = synthesize.mock.calls[0][3]!;
        fireEvent.change(slider, { target: { value: '100' } });
        expect(slider).toHaveAttribute('aria-valuetext', '100% braking adjustment (Direction guide)');
        expect(screen.getByText('100% · Direction guide')).toBeVisible();
        expect(screen.getByText('100% braking adjustment')).toBeVisible();
        expect(aggression).toHaveBeenLastCalledWith(100, Date.now());
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        expect(receiveMap).toHaveBeenCalledTimes(mapCalls);
        expect(signal.aborted).toBe(false);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(screen.getByText('Speech: Ready')).toBeInTheDocument();
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Phrase history' })).not.toHaveTextContent(sentence);
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

it('toggles track guidance independently without interrupting overtaking audio or restarting detection', async () => {
    const play = spyOnPlayer();
    const receiveMap = jest.spyOn(PhraseEngine.prototype, 'receiveMap');
    const trackGuide = jest.spyOn(PhraseEngine.prototype, 'setTrackGuideEnabled');
    try {
        const view = mount();
        const toggle = screen.getByRole('switch', { name: 'Track guide' });
        expect(toggle).toBeChecked();
        fireEvent.click(toggle);
        await view.enable();
        expect(trackGuide).toHaveBeenLastCalledWith(false, Date.now());
        const mapCalls = receiveMap.mock.calls.length;
        await view.trigger();
        expect(play).toHaveBeenCalledTimes(1);
        const signal = synthesize.mock.calls[0][3]!;
        fireEvent.click(toggle);
        expect(trackGuide).toHaveBeenLastCalledWith(true, Date.now());
        fireEvent.click(toggle);
        expect(trackGuide).toHaveBeenLastCalledWith(false, Date.now());
        expect(play.mock.results[0].value.isActive()).toBe(true);
        expect(receiveMap).toHaveBeenCalledTimes(mapCalls);
        expect(signal.aborted).toBe(false);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        await view.enable();
        expect(toggle).not.toBeChecked();
        expect(trackGuide).toHaveBeenLastCalledWith(false, Date.now());
    } finally {
        play.mockRestore();
        receiveMap.mockRestore();
        trackGuide.mockRestore();
    }
});

it.each(['preparing', 'playing'])('honors disabling track guidance while speech is %s', async (state) => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.events = [{ id: 1, ruleId: 'reference-brake', sentence: 'Brake.', timestamp: Date.now() }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    const toggleGuide = jest.spyOn(PhraseEngine.prototype, 'setTrackGuideEnabled');
    const play = spyOnPlayer();
    let resolve!: (value: TtsPack) => void;
    const pending = new Promise<TtsPack>((done) => { resolve = done; });
    synthesize.mockImplementation(async (_curves, _inputs, { text }) =>
        state === 'preparing' && text === 'Brake.' ? pending : pack(text));
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        toggleGuide.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(state === 'preparing' ? 0 : 1);
        fireEvent.click(screen.getByRole('switch', { name: 'Track guide' }));
        if (state === 'playing') expect(play.mock.results[0].value.outcome).toEqual({ status: 'cancelled' });
        await act(async () => { resolve(pack('Brake.')); view.frame(); });
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(state === 'preparing' ? 0 : 1);
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('switch', { name: 'Track guide' }));
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(state === 'preparing' ? 0 : 1);
    } finally {
        receive.mockRestore();
        receiveVision.mockRestore();
        toggleGuide.mockRestore();
        play.mockRestore();
    }
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
    const play = spyOnPlayer();
    try {
        const view = mount();
        await view.enable();
        expect(synthesize.mock.calls.map(([, , request]) => request.text)).toEqual(sentences);
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(chosen).audioDataUrl }));
        expect(screen.getByRole('region', { name: 'Phrase history' })).toHaveTextContent(chosen);
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it.each(['one millisecond late', 'expired', 'clip too long', 'condition cleared', 'action released'])(
    'plays a triggered reference cue without rechecking its duration, deadline or conditions: %s', async (state) => {
        const snapshot = new PhraseEngine().evaluate(Date.now());
        snapshot.telemetryReady = true;
        const rule = snapshot.closures.find((item) => item.id === 'reference-brake')!;
        rule.conditions = [new PhraseCondition('speed', '>=', 20, 36)];
        snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Brake.', timestamp: Date.now(), expiresAt: Date.now() + 1200 }];
        const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({
            ...pack(text), durationMs: state === 'clip too long' ? 1300 : 1200,
        }));
        try {
            const view = mount();
            await view.enable();
            expect(audio.play).not.toHaveBeenCalled();
            if (state === 'one millisecond late') jest.setSystemTime(Date.now() + 1);
            if (state === 'expired') jest.setSystemTime(Date.now() + 2000);
            if (state === 'condition cleared') {
                rule.conditions = [new PhraseCondition('speed', '>=', 20, 0)];
            }
            if (state === 'action released') {
                snapshot.events[0].speech = { durationMs: 1200, reactionAllowanceMs: 0, releaseAt: Date.now() };
            }
            await act(async () => { view.frame(); });
            expect(audio.play).toHaveBeenCalledTimes(1);
        } finally { receive.mockRestore(); }
    });

it('immediately submits the next triggered cue and lets the audio manager replace the earlier phrase', async () => {
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({ ...pack(text), durationMs: text === 'Turn in' ? 400 : 3000 }));
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = 'prepare';
    snapshot.state.path = ['root', 'Track guide', 'Turn in'];
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 80)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence, timestamp: Date.now(), expiresAt: Date.now() + 5000 }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    const play = spyOnPlayer();
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack(sentence).audioDataUrl }));
        jest.setSystemTime(Date.now() + 1100);
        const staleEnded = audio.media[0].onended!;
        snapshot.events = [...snapshot.events, { id: 2, ruleId: rule.id, sentence: 'Turn in', timestamp: Date.now(),
            expiresAt: Date.now() + 1500, steeringPhase: 'prepare' }];
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(2);
        expect(play.mock.results[0].value.outcome).toEqual({ status: 'replaced' });
        expect(play.mock.results[1].value.isActive()).toBe(true);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack('Turn in').audioDataUrl }));
        expect(audio.media).toHaveLength(1);
        expect(TestAudioContext.instances).toHaveLength(1);
        await act(async () => { staleEnded.call(audio.media[0], new Event('ended')); view.frame(); });
        expect(play).toHaveBeenCalledTimes(2);
        expect(play.mock.results[1].value.isActive()).toBe(true);
        view.unmount();
        expect(play.mock.results[1].value.outcome).toEqual({ status: 'cancelled' });
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it.each(['reference starts turn-in', 'reference leaves turn-in', 'Steering exits'])('plays an entered Turn without revalidation when %s', async (state) => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.referenceSteeringPhase = state === 'reference starts turn-in' ? 'prepare' : 'turn';
    snapshot.state.path = ['root', 'Track guide', 'Turn in', 'say phrase'];
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 80)];
    snapshot.events = [{ id: 1, ruleId: rule.id, sentence: 'Turn in', timestamp: Date.now(),
        expiresAt: Date.now() + 1500, steeringPhase: snapshot.referenceSteeringPhase }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    synthesize.mockImplementation(async (_curves, _inputs, { text }) => ({ ...pack(text), durationMs: 400 }));
    try {
        const view = mount();
        await view.enable();
        receiveVision.mockReturnValue(snapshot);
        if (state === 'Steering exits') snapshot.state.path = ['root'];
        else snapshot.referenceSteeringPhase = state === 'reference starts turn-in' ? 'turn' : 'prepare';
        await act(async () => { view.frame(); });
        expect(audio.play).toHaveBeenCalledTimes(1);
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});

it('lets live chat interrupt and suppress phrases without replaying discarded events', async () => {
    const play = spyOnPlayer();
    const view = mount();
    await view.enable();
    await view.trigger();
    const chat = audioManager.createStream({ type: 'voice', priority: 50, format: 'wav' });
    otherPlayback = chat;
    await act(async () => { chat.enqueueBase64Wav(btoa('chat')); });
    expect(play.mock.results[0].value.outcome).toEqual({ status: 'replaced' });
    // Chat retains voice priority even between spoken chunks.
    audio.media[1].dispatchEvent(new Event('ended'));
    await act(async () => { view.telemetry.resetSession(); });
    await view.trigger();
    expect(play.mock.results[1].value.outcome).toEqual({ status: 'rejected' });
    expect(chat.outcome).toBeUndefined();
    chat.stop();
    await act(async () => { view.frame(); jest.advanceTimersByTime(250); view.frame(); });
    expect(play).toHaveBeenCalledTimes(2);
    await act(async () => { view.telemetry.resetSession(); });
    await view.trigger();
    expect(play).toHaveBeenCalledTimes(3);
    expect(play.mock.results[2].value.outcome).toBeUndefined();
    expect(audio.media).toHaveLength(2); // The phrase player and chat's separate media element.
    expect(TestAudioContext.instances).toHaveLength(1);
    await act(async () => { view.telemetry.resetSession(); });
    expect(play.mock.results[2].value.outcome).toEqual({ status: 'cancelled' });
});

it.each(['ready', 'disable', 'reset', 'delayed', 'condition cleared', 'unmount'])(
    'ignores driving events during speech preparation when %s', async (state) => {
        let resolve!: (value: TtsPack) => void;
        const pending = new Promise<TtsPack>((done) => { resolve = done; });
        synthesize.mockImplementation(async (_curves, _inputs, { text }) => text === sentence ? pending : pack(text));
        const view = mount();
        await view.enable();
        // Preparation is sequential; later sentences wait for this request.
        expect(synthesize).toHaveBeenCalledTimes(sentences.indexOf(sentence) + 1);
        expect(screen.getByText('Detection: Waiting for speech')).toHaveAttribute('data-ready', 'false');
        await view.trigger();
        expect(audio.play).not.toHaveBeenCalled();
        expect(screen.getByRole('region', { name: 'Phrase history' })).not.toHaveTextContent(sentence);
        if (state === 'disable') fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        if (state === 'reset') act(() => view.telemetry.resetSession());
        if (state === 'delayed') {
            act(() => { for (let i = 0; i < 32; i++) { jest.advanceTimersByTime(250); view.frame(); } });
        }
        if (state === 'condition cleared') act(() => view.frame(0));
        if (state === 'unmount') view.unmount();
        await act(async () => { resolve(pack(sentence)); });
        expect(audio.play).not.toHaveBeenCalled();
        if (state === 'disable') expect(screen.getByText('Speech: Disabled')).toHaveAttribute('data-ready', 'false');
        if (state === 'disable' || state === 'unmount') {
            expect(synthesize.mock.calls.every(([, , , signal]) => signal?.aborted)).toBe(true);
        } else {
            expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
            expect(screen.getByText('Detection: Enabled')).toHaveAttribute('data-ready', 'true');
            expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
            expect(screen.getByRole('region', { name: 'Phrase history' })).not.toHaveTextContent(sentence);
            await view.trigger();
            expect(audio.play).toHaveBeenCalledTimes(1);
        }
    },
);

it('waits after a synthesis failure, retries only missing clips, and reports later playback failures', async () => {
    synthesize.mockRejectedValueOnce(new Error('Speech service unavailable'));
    const view = mount();
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    expect(screen.getByRole('alert')).toHaveTextContent('Speech service unavailable');
    expect(screen.getByText('Speech: Incomplete')).toHaveAttribute('data-ready', 'false');
    expect(screen.getByText('Detection: Waiting for speech')).toHaveAttribute('data-ready', 'false');
    await view.trigger();
    expect(audio.play).not.toHaveBeenCalled();
    expect(screen.getByRole('region', { name: 'Phrase history' })).not.toHaveTextContent(sentence);
    expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
    await view.enable();
    expect(synthesize).toHaveBeenCalledTimes(sentences.length + 1);
    expect(synthesize.mock.calls[sentences.length][2].text).toBe(sentences[0]);
    expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    audio.play.mockRejectedValueOnce(new Error('Audio blocked'));
    await view.trigger();
    expect(screen.getByRole('alert')).toHaveTextContent('Audio blocked');
    expect(screen.getByRole('region', { name: 'Phrase history' })).toHaveTextContent(sentence);
});

it('does not restart a ready run when cancelled preparation resolves after re-enabling', async () => {
    let resolveOld!: (value: TtsPack) => void;
    synthesize.mockImplementationOnce(() => new Promise<TtsPack>((resolve) => { resolveOld = resolve; }));
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry');
    const receiveMap = jest.spyOn(PhraseEngine.prototype, 'receiveMap');
    try {
        const view = mount();
        await view.enable();
        const oldSignal = synthesize.mock.calls[0][3]!;
        await view.trigger();
        expect(receive).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        expect(oldSignal.aborted).toBe(true);
        await view.enable();
        expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
        const mapCalls = receiveMap.mock.calls.length;
        const preparedCount = synthesize.mock.calls.length;
        await act(async () => { resolveOld(pack('cancelled clip')); });
        expect(synthesize).toHaveBeenCalledTimes(preparedCount);
        expect(receiveMap).toHaveBeenCalledTimes(mapCalls);
        expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(audio.play).not.toHaveBeenCalled();
        await act(async () => { view.frame(); });
        expect(receive).toHaveBeenCalledTimes(1);
        await act(async () => { jest.advanceTimersByTime(800); view.frame(); });
        expect(audio.play).toHaveBeenCalledTimes(1);
    } finally { receive.mockRestore(); receiveMap.mockRestore(); }
});

it('preloads every easing-duration delivery and only reads cached instances during playback', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.state.path = ['root', 'Track guide', 'Ease brake'];
    snapshot.referenceBrakeReleaseRate = 0.25;
    // Entered release speech plays its prepared instance without rechecking entry conditions.
    snapshot.closures.find((rule) => rule.id === 'reference-steering-release')!.conditions = [new PhraseCondition('referenceCue', '=', 'steering', 'none')];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const play = spyOnPlayer();
    synthesize.mockImplementation(async (_curves, inputs, speech) => pack(`${speech.text}:${inputs[0]}`));
    try {
        const view = mount();
        await view.enable();
        const preparedCount = synthesize.mock.calls.length;
        expect(synthesize.mock.calls.filter(([, , speech]) => speech.text === SLOWLY_RELEASE_BRAKE)).toHaveLength(4);
        for (const [index, duration] of Array.from([900, 800, 700, 600, 900].entries())) {
            const instance = instances.find((item) => item.speech.text === SLOWLY_RELEASE_BRAKE
                && item.inputs.timing === brakeReleaseSettings(duration,
                    (settings) => 600 + settings.inputs!.timing! * 300).inputs!.timing)!;
            snapshot.referenceBrakeReleaseDurationMs = duration;
            snapshot.events = [{ id: index + 1, ruleId: 'reference-steering-release', sentence: SLOWLY_RELEASE_BRAKE,
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
    const cachedCounts: number[] = [];
    const prepared = jest.fn(() => { cachedCounts.push(packs.size); });
    expect(await loadAllPhraseSpeech([...variants, variants[0]], packs, signal, prepared)).toEqual({ complete: true, errors: [] });
    await loadAllPhraseSpeech(variants, packs, signal, prepared);
    expect(synthesize).toHaveBeenCalledTimes(2);
    expect(prepared).toHaveBeenCalledTimes(2);
    expect(cachedCounts).toEqual([1, 2]);
    expect(packs.size).toBe(2);
    expect(synthesize).toHaveBeenCalledWith(variants[1].curves, [0, 0.75, 0], variants[1].speech, signal, true);
});

it('waits for the entire catalog even after the matching clip is prepared and starts with fresh telemetry', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.events = [{ id: 1, ruleId: 'reference-brake', sentence: 'Brake.', timestamp: Date.now() }];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
    let resolveBrake!: (value: TtsPack) => void;
    let resolveLast!: (value: TtsPack) => void;
    const brake = new Promise<TtsPack>((resolve) => { resolveBrake = resolve; });
    const last = new Promise<TtsPack>((resolve) => { resolveLast = resolve; });
    const lastSentence = sentences[sentences.length - 1];
    synthesize.mockImplementation(async (_curves, _inputs, { text }) =>
        text === 'Brake.' ? brake : text === lastSentence ? last : pack(text));
    const play = spyOnPlayer();
    const durations = jest.spyOn(PhraseEngine.prototype, 'setSpeechDurations');
    try {
        const view = mount();
        await view.enable();
        await act(async () => { view.frame(); });
        expect(play).not.toHaveBeenCalled();
        expect(receive).not.toHaveBeenCalled();
        expect(receiveVision).not.toHaveBeenCalled();
        await act(async () => { resolveBrake(pack('Brake.')); });
        await act(async () => { view.frame(); jest.advanceTimersByTime(800); view.frame(); });
        expect(play).not.toHaveBeenCalled();
        expect(receive).not.toHaveBeenCalled();
        expect(receiveVision).not.toHaveBeenCalled();
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
        expect(screen.getByText('Speech: Preparing')).toHaveAttribute('data-ready', 'false');
        expect(screen.queryByText('Speech: Ready')).not.toBeInTheDocument();
        expect(screen.getByText('Detection: Waiting for speech')).toHaveAttribute('data-ready', 'false');
        expect(screen.getByRole('region', { name: 'Phrase history' })).not.toHaveTextContent('Brake.');
        await act(async () => { view.frame(); resolveLast(pack(lastSentence)); });
        expect(screen.getByText('Speech: Ready')).toHaveAttribute('data-ready', 'true');
        expect(screen.getByText('Detection: Enabled')).toHaveAttribute('data-ready', 'true');
        expect(play).not.toHaveBeenCalled();
        expect(receive).not.toHaveBeenCalled();
        const preparedDurations = durations.mock.calls[durations.mock.calls.length - 1][0];
        expect(Array.from(preparedDurations.keys())).toEqual(instances.map((instance) => instance.cacheKey));
        expect(Array.from(preparedDurations.values())).toEqual(instances.map(() => 3000));
        await act(async () => { view.frame(); });
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ url: pack('Brake.').audioDataUrl }));
    } finally { receive.mockRestore(); receiveVision.mockRestore(); durations.mockRestore(); }
});

it('waits for the complete catalog before playing a prepared brake-release variant', async () => {
    const snapshot = new PhraseEngine().evaluate(Date.now());
    snapshot.telemetryReady = true;
    snapshot.state.path = ['root', 'Track guide', 'Ease brake'];
    snapshot.referenceBrakeReleaseRate = 0.25;
    const rule = snapshot.closures.find((item) => item.id === 'reference-steering-release')!;
    rule.conditions = [new PhraseCondition('speed', '>=', 20, 36)];
    const receive = jest.spyOn(PhraseEngine.prototype, 'receiveTelemetry').mockReturnValue(snapshot);
    const receiveVision = jest.spyOn(PhraseEngine.prototype, 'receiveVision');
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
        expect(screen.getByText('Speech: Preparing')).toBeInTheDocument();
        expect(audio.play).not.toHaveBeenCalled();
        expect(receive).not.toHaveBeenCalled();
        await act(async () => { view.frame(); resolve(pack(sentences[sentences.length - 1])); });
        expect(audio.play).not.toHaveBeenCalled();
        expect(receive).not.toHaveBeenCalled();
        receiveVision.mockReturnValue(snapshot);
        await act(async () => { view.frame(); });
        expect(audio.play).toHaveBeenCalledTimes(1);
        await act(async () => { view.frame(); });
        expect(audio.play).toHaveBeenCalledTimes(1);
        expect(synthesize).toHaveBeenCalledTimes(sentences.length);
    } finally { receive.mockRestore(); receiveVision.mockRestore(); }
});
