import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import {
    createOperationComponentRefDirectory, OperationComponentRefProvider, useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import { AiOverlayManagerController } from 'views/floating-chat/AiOverlayManager';
import { isAiOverlayComponentHandle } from 'views/floating-chat/ai-overlay-types';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import { LivePhraseOverlay, LIVE_PHRASE_DISPLAY_MS } from './LivePhraseOverlay';
import { livePhraseOverlayRenderer } from './graphs/LivePhraseDisplay';
import { PHRASE_DEFINITIONS, PhraseEngine, type PhraseEvent, type PhraseSnapshot } from './phrase-engine';
import LivePhrases from './LivePhrases';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import { circuitMap, steeringCornerFixture, vision } from './test-fixtures';
import { synthesizeProsodyTts } from 'components/tts';
import { installAudioDoubles } from 'services/audio/test-audio';

jest.mock('components/tts', () => ({ synthesizeProsodyTts: jest.fn(() => new Promise(() => undefined)) }));

const session = {
    presentationId: 'phrases-presentation', aiSessionId: 'live-session', mode: 'live' as const,
    displayIdentity: { name: 'Kestrel' },
};
const phraseEvent = (index = 0, id = index + 1): PhraseEvent => ({
    id, ruleId: PHRASE_DEFINITIONS[index].id, sentence: PHRASE_DEFINITIONS[index].sentence, timestamp: Date.now(),
});
const snapshot = (...events: PhraseEvent[]) => ({ ...new PhraseEngine().evaluate(Date.now()), events });
const turnInEvent = (id = 1) => phraseEvent(PHRASE_DEFINITIONS.findIndex((rule) => rule.id === 'reference-steering'), id);

describe('Live phrase overlay addons', () => {
    let manager: AiOverlayManagerController;
    let directory: ReturnType<typeof createOperationComponentRefDirectory>;
    let overlay: LivePhraseOverlay;
    let unsubscribe: () => void;
    let audio: ReturnType<typeof installAudioDoubles>;
    const send = jest.fn();
    const onError = jest.fn();
    const createSession = jest.fn();
    const destroySession = jest.fn();
    const setEnabled = jest.fn();

    beforeEach(async () => {
        (synthesizeProsodyTts as jest.Mock).mockImplementation(() => new Promise(() => undefined));
        jest.useFakeTimers();
        jest.setSystemTime(10_000);
        audio = installAudioDoubles();
        send.mockReset();
        onError.mockReset();
        createSession.mockReset().mockResolvedValue({ success: true, presentation: session });
        destroySession.mockReset().mockResolvedValue({ success: true, ended: true });
        setEnabled.mockReset();
        (window as any).electronAPI = {
            createOverlaySession: createSession,
            destroyOverlaySession: destroySession,
            setOverlayEnabled: setEnabled,
        };
        await overlaySessionClient.destroy();
        destroySession.mockClear();
        manager = new AiOverlayManagerController({
            send, now: Date.now,
            setTimer: (callback, delay) => setTimeout(callback, delay),
            clearTimer: clearTimeout,
        });
        directory = createOperationComponentRefDirectory(() => manager.syncReferences(directory.getComponentRefs()));
        unsubscribe = overlaySessionClient.subscribe((next) => manager.setPresentation(next));
        overlay = new LivePhraseOverlay(directory, 'live-phrases', onError);
    });

    afterEach(async () => {
        cleanup();
        overlay.dispose();
        unsubscribe();
        manager.dispose();
        await overlaySessionClient.destroy();
        delete (window as any).electronAPI;
        audio.restore();
        jest.useRealTimers();
    });

    it('keeps triggered phrases out of the overlay and registers only the steering addon', async () => {
        await overlaySessionClient.create(session);
        expect(directory.getComponentNames()).toEqual(['live-phrases:steering-corner']);
        expect(directory.getComponentRefs().every((ref) => isAiOverlayComponentHandle(ref.current))).toBe(true);
        PHRASE_DEFINITIONS.forEach((_rule, index) => overlay.update(snapshot(phraseEvent(index))));
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(setEnabled).not.toHaveBeenCalled();
    });

    it('does not create a presentation for triggered phrases', async () => {
        await act(async () => {
            overlay.update(snapshot(phraseEvent()));
            overlay.update(snapshot(phraseEvent(1)));
        });
        expect(createSession).not.toHaveBeenCalled();
        expect(manager.getPresentationSnapshot()).toBeNull();
        expect(setEnabled).not.toHaveBeenCalled();
    });

    it('creates one local presentation for the steering corner and preserves overlay visibility', async () => {
        const { map, reference } = steeringCornerFixture();
        await act(async () => {
            overlay.update(steeringSnapshot(), map, reference);
            overlay.update(steeringSnapshot(), map, reference);
        });
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(createSession).toHaveBeenCalledWith(expect.objectContaining({ aiSessionId: 'live-phrases', mode: 'live' }));
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        expect(setEnabled).not.toHaveBeenCalled();
        await act(async () => overlay.dispose());
        expect(destroySession).toHaveBeenCalledWith(session.presentationId);
    });

    it('does not publish live guidance into another mode or replace its presentation', async () => {
        createSession.mockResolvedValue({ success: true, presentation: { ...session, mode: 'recorded' } });
        await overlaySessionClient.create({ ...session, mode: 'recorded' });
        const { map, reference } = steeringCornerFixture();
        overlay.update(steeringSnapshot(), map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        expect(createSession).toHaveBeenCalledTimes(1);
    });

    it('keeps steering local in the browser and reports a failed desktop presentation', async () => {
        const { map, reference } = steeringCornerFixture();
        delete (window as any).electronAPI.createOverlaySession;
        overlay.update(steeringSnapshot(), map, reference);
        expect(createSession).not.toHaveBeenCalled();
        (window as any).electronAPI.createOverlaySession = createSession;
        createSession.mockRejectedValue(new Error('Window unavailable'));
        overlay.update(snapshot());
        await act(async () => overlay.update(steeringSnapshot(), map, reference));
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'Window unavailable' }));
    });

    it('rejects invalid display data before rendering', () => {
        expect(livePhraseOverlayRenderer.validateSnapshot(null)).toBe(false);
        expect(livePhraseOverlayRenderer.validateSnapshot({ ...phraseEvent(), name: 'Test', eventId: 1, timestamp: 1e20 })).toBe(false);
        expect(livePhraseOverlayRenderer.validateSnapshot({ ...phraseEvent(), name: 'Test', eventId: 1, sentence: '' })).toBe(false);
    });

    const steeringSnapshot = (id = 1): PhraseSnapshot => ({
        ...snapshot(turnInEvent(id)), telemetryReady: true, mapReady: true, referenceLoaded: true, trackGuideEnabled: true,
        steeringCorner: { sectionId: 'turn-1', cornerStartPosition: 0.27, cornerEndPosition: 0.45 },
        state: { current: 'Track guide', description: '', kind: 'closure', path: ['root', 'Track guide'] },
    });

    it.each([0, 50, 99.9])('does not show an eligible reference corner without Turn in at %s%% aggression', async (aggression) => {
        await overlaySessionClient.create(session);
        const { map, reference } = steeringCornerFixture();
        const engine = new PhraseEngine();
        engine.setDrivingAggression(aggression, 0);
        engine.receiveMap(map, 0, reference);
        const sample = { Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: 0.28,
            Physics_speed_kmh: 36, Physics_brake: 0, Physics_gas: 0 };
        const active = engine.receiveTelemetry({ type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
            sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
            update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } }, 1000);
        expect(active.trackGuideEnabled).toBe(true);
        expect(active.steeringCorner?.sectionId).toBe('turn-1');
        expect(active.events).toEqual([]);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
    });

    it('captures the reference line once and expires exactly two seconds after Turn in', async () => {
        await overlaySessionClient.create(session);
        const { map, reference } = steeringCornerFixture();
        const active = steeringSnapshot();
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([expect.objectContaining({
            componentName: 'live-phrases:steering-corner', componentType: 'live_steering_corner', status: 'focus',
            snapshot: expect.objectContaining({ title: 'Steering · Turn 1' }),
        })]);
        const card = manager.getPresentationSnapshot()!.cards[0];
        const calls = send.mock.calls.length;
        jest.advanceTimersByTime(1000);
        reference.y[30][0] += 100;
        overlay.update({ ...active, steeringCorner: { ...active.steeringCorner, cornerEndPosition: 0.4 } },
            { ...map, circuit_name: 'Updated map' }, { ...reference });
        // Leaving the corner or losing input does not change a captured card.
        overlay.update({ ...active, steeringCorner: undefined, telemetryReady: false, mapReady: false }, null, null);
        expect(send).toHaveBeenCalledTimes(calls);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([card]);
        jest.advanceTimersByTime(999);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        jest.advanceTimersByTime(1);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        const source = directory.getComponentRefs()[0].current;
        expect(isAiOverlayComponentHandle(source) && source.getSnapshot()).toBeNull();
        overlay.dispose();
        expect(directory.getComponentNames()).toEqual([]);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        // This presentation belongs to the Assistant, not to Live Phrases.
        expect(destroySession).not.toHaveBeenCalled();
    });

    it('replaces the current card immediately and gives the new card its own two seconds', async () => {
        await overlaySessionClient.create(session);
        const { map, reference } = steeringCornerFixture();
        overlay.update(steeringSnapshot(), map, reference);
        const first = manager.getPresentationSnapshot()!.cards[0];
        jest.advanceTimersByTime(1000);
        const calls = send.mock.calls.length;
        overlay.update({ ...steeringSnapshot(2), steeringCorner: {
            sectionId: 'turn-2', cornerStartPosition: 0.3, cornerEndPosition: 0.4,
        } }, map, reference);
        const cards = manager.getPresentationSnapshot()!.cards;
        expect(cards).toHaveLength(1);
        expect(cards[0]).toMatchObject({ componentName: first.componentName, revision: first.revision + 1,
            snapshot: { title: 'Steering · turn-2' } });
        expect(cards[0].snapshot).not.toEqual(first.snapshot);
        expect(send).toHaveBeenCalledTimes(calls + 1);
        jest.advanceTimersByTime(1000); // The old card's deadline cannot remove the replacement.
        expect(manager.getPresentationSnapshot()!.cards).toEqual(cards);
        jest.advanceTimersByTime(1000);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
    });

    it('keeps only the newest captured line when presentation creation is delayed', async () => {
        let finish!: (value: unknown) => void;
        createSession.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
        const { map, reference } = steeringCornerFixture();
        overlay.update(steeringSnapshot(), map, reference);
        jest.advanceTimersByTime(1000);
        const active = { ...steeringSnapshot(2), steeringCorner: {
            sectionId: 'turn-2', cornerStartPosition: 0.3, cornerEndPosition: 0.4,
        } };
        overlay.update(active, map, reference);
        // Later input changes must not affect the snapshot waiting for the presentation.
        const expectedLine = reference.y[30][0];
        reference.y[30][0] += 100;
        jest.advanceTimersByTime(500);
        await act(async () => finish({ success: true, presentation: session }));
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([expect.objectContaining({
            snapshot: expect.objectContaining({ title: 'Steering · turn-2',
                referenceLine: expect.arrayContaining([expect.objectContaining({ x: expectedLine })]) }),
        })]);
        jest.advanceTimersByTime(1499);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        jest.advanceTimersByTime(1);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
    });

    it('allows a new trigger for the same corner after reset without replaying old events', async () => {
        await overlaySessionClient.create(session);
        const { map, reference } = steeringCornerFixture();
        const expired = steeringSnapshot();
        jest.advanceTimersByTime(LIVE_PHRASE_DISPLAY_MS);
        overlay.update(expired, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        const active = steeringSnapshot(2);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        overlay.update({ ...active, trackGuideEnabled: false }, map, reference);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        overlay.update(snapshot(), map, reference);
        overlay.update(steeringSnapshot(), map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
    });

    it.each(['reset', 'guide disabled', 'expired', 'unmount'] as const)('cancels a pending corner on %s', async (action) => {
        let finish!: (value: unknown) => void;
        createSession.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
        const { map, reference } = steeringCornerFixture();
        const active = steeringSnapshot();
        overlay.update(active, map, reference);
        if (action === 'reset') overlay.update(snapshot(), map, reference);
        if (action === 'guide disabled') overlay.update({ ...active, trackGuideEnabled: false }, map, reference);
        if (action === 'expired') jest.advanceTimersByTime(LIVE_PHRASE_DISPLAY_MS);
        if (action === 'unmount') overlay.dispose();
        await act(async () => finish({ success: true, presentation: session }));
        expect(manager.getPresentationSnapshot()?.cards ?? []).toEqual([]);
    });

    it('shows only the corner when a phrase triggers', async () => {
        const { map, reference } = steeringCornerFixture();
        const active = { ...steeringSnapshot(), events: [phraseEvent(PHRASE_DEFINITIONS.findIndex((rule) => rule.id === 'reference-steering'))] };
        await act(async () => overlay.update(active, map, reference));
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        expect(manager.getPresentationSnapshot()!.cards[0].componentType).toBe('live_steering_corner');
        expect(setEnabled).not.toHaveBeenCalled();
    });

    it.each(['acc', 'iracing'] as const)('plays Turn in and publishes its reference line together in %s once speech is ready', async (game) => {
        await overlaySessionClient.create(session);
        const submit = jest.fn(async (presentation) => ({
            presentationId: presentation.presentationId, presentationRevision: presentation.presentationRevision, accepted: true,
        }));
        (window as any).electronAPI.sendOverlayPresentation = submit;
        const telemetry = createLiveTelemetryStore();
        const { map, reference } = steeringCornerFixture(game);
        const root = {
            getComponentName: () => 'live-session', subscribeTelemetry: telemetry.subscribeEvents,
            getTrackVisionDetection: () => null, subscribeTrackVision: () => () => undefined,
            getLiveCircuitMap: () => map, getLiveReferenceLap: () => reference,
            getLiveCornerSummaries: () => [], subscribeLiveCircuitMap: () => () => undefined,
        };
        const Root = () => { useRegisterOperationComponentRef(React.useRef(root)); return null; };
        const view = render(<React.StrictMode><OperationComponentRefProvider>
            <Root /><LivePhrases name="live-phrases" />
        </OperationComponentRefProvider></React.StrictMode>);
        fireEvent.change(screen.getByRole('slider', { name: 'Instructor aggression' }), { target: { value: '100' } });
        fireEvent.click(screen.getByRole('button', { name: 'Enable detection' }));
        await act(async () => {
            telemetry.publishFrame({ type: 'frame', game, sequence: 1, committedCount: 0, committedSequence: 0,
                sample: { Static_track: 'test', Static_car_model: 'car', Graphics_status: 2,
                    Physics_speed_kmh: 36, Graphics_normalized_car_position: 0.25 } });
        });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards
            .some((card: any) => card.componentType === 'live_steering_corner')).toBe(false);
        expect(audio.play).not.toHaveBeenCalled();
        // Retry preparation with ready clips; no frames from preparation are replayed.
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        (synthesizeProsodyTts as jest.Mock).mockImplementation(async (_curves, _inputs, { text }) => ({
            text, audioDataUrl: `data:audio/wav;base64,${btoa(text)}`, durationMs: 200,
        }));
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Enable detection' })); });
        expect(audio.play).not.toHaveBeenCalled();
        await act(async () => {
            telemetry.publishFrame({ type: 'frame', game, sequence: 2, committedCount: 0, committedSequence: 0,
                sample: { Static_track: 'test', Static_car_model: 'car', Graphics_status: 2,
                    Physics_speed_kmh: 36, Graphics_normalized_car_position: 0.27 } });
        });
        const cards = submit.mock.calls[submit.mock.calls.length - 1][0].cards;
        const corner = cards.find((card: any) => card.componentType === 'live_steering_corner');
        expect(corner).toMatchObject({ status: 'focus', snapshot: { title: 'Steering · Turn 1' } });
        expect(corner.snapshot.referenceLine.length).toBeGreaterThan(2);
        expect(corner.snapshot.leftBoundary.length).toBeGreaterThan(2);
        expect(corner.snapshot.rightBoundary.length).toBeGreaterThan(2);
        expect(audio.play).toHaveBeenCalledTimes(1);
        expect(audio.media[audio.media.length - 1].src).toBe(`data:audio/wav;base64,${btoa('Turn in')}`);
        // The card retains its full lifetime even after telemetry goes stale.
        await act(async () => { jest.advanceTimersByTime(1999); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards).toEqual(cards);
        await act(async () => { jest.advanceTimersByTime(1); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards
            .some((card: any) => card.componentType === 'live_steering_corner')).toBe(false);
        view.unmount();
    });

    it('keeps a real rule trigger in the local history without an overlay card under StrictMode', async () => {
        (synthesizeProsodyTts as jest.Mock).mockImplementation(async (_curves, _inputs, { text }) => ({
            text, audioDataUrl: `data:audio/wav;base64,${btoa(text)}`, durationMs: 200,
        }));
        await overlaySessionClient.create(session);
        const submit = jest.fn(async (presentation) => ({
            presentationId: presentation.presentationId,
            presentationRevision: presentation.presentationRevision,
            accepted: true,
        }));
        (window as any).electronAPI.sendOverlayPresentation = submit;
        const telemetry = createLiveTelemetryStore();
        const map = circuitMap();
        const detection = vision(Date.now(), { player: 'middle' });
        const root = {
            getComponentName: () => 'live-session',
            subscribeTelemetry: telemetry.subscribeEvents,
            getTrackVisionDetection: () => detection,
            subscribeTrackVision: () => () => undefined,
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
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Enable detection' })); });
        const publish = (sequence: number) => telemetry.publishFrame({
            type: 'frame', game: 'acc', sequence, committedCount: 0, committedSequence: 0,
            sample: { Graphics_status: 2, Physics_speed_kmh: 100, Graphics_normalized_car_position: 0.11 },
        });
        await act(async () => {
            publish(1);
            jest.advanceTimersByTime(800);
            publish(2);
        });
        const cards = submit.mock.calls[submit.mock.calls.length - 1][0].cards;
        expect(cards).toEqual([]);
        expect(within(screen.getByRole('region', { name: 'Phrase history' })).getByText(
            PHRASE_DEFINITIONS.find((rule) => rule.id === 'pressure-feint')!.sentence,
        )).toBeVisible();
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Disable detection' })); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards).toEqual([]);
        view.unmount();
        jest.runAllTicks();
        expect(jest.getTimerCount()).toBe(0);
    });
});
