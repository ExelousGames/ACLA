import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
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

jest.mock('components/tts', () => ({ synthesizeProsodyTts: jest.fn(() => new Promise(() => undefined)) }));

const session = {
    presentationId: 'phrases-presentation', aiSessionId: 'live-session', mode: 'live' as const,
    displayIdentity: { name: 'Kestrel' },
};
const phraseEvent = (index = 0, id = index + 1): PhraseEvent => ({
    id, ruleId: PHRASE_DEFINITIONS[index].id, sentence: PHRASE_DEFINITIONS[index].sentence, timestamp: Date.now(),
});
const snapshot = (...events: PhraseEvent[]) => ({ ...new PhraseEngine().evaluate(Date.now()), events });

describe('Live phrase overlay addons', () => {
    let manager: AiOverlayManagerController;
    let directory: ReturnType<typeof createOperationComponentRefDirectory>;
    let overlay: LivePhraseOverlay;
    let unsubscribe: () => void;
    const send = jest.fn();
    const onError = jest.fn();
    const createSession = jest.fn();
    const destroySession = jest.fn();
    const setEnabled = jest.fn();

    beforeEach(async () => {
        (synthesizeProsodyTts as jest.Mock).mockImplementation(() => new Promise(() => undefined));
        jest.useFakeTimers();
        jest.setSystemTime(10_000);
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
        overlay.dispose();
        unsubscribe();
        manager.dispose();
        await overlaySessionClient.destroy();
        delete (window as any).electronAPI;
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

    const steeringSnapshot = (): PhraseSnapshot => ({
        ...snapshot(), telemetryReady: true, mapReady: true, referenceLoaded: true,
        steeringCorner: { sectionId: 'turn-1', cornerStartPosition: 0.27, cornerEndPosition: 0.45 },
        state: { current: 'Steering', description: '', kind: 'closure', path: ['root', 'Steering'] },
    });

    it('inserts the corner immediately on closure entry without a speech event and retains it until exit', async () => {
        await overlaySessionClient.create(session);
        const { map, reference } = steeringCornerFixture();
        const active = steeringSnapshot();
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([expect.objectContaining({
            componentName: 'live-phrases:steering-corner', componentType: 'live_steering_corner', status: 'focus',
            snapshot: expect.objectContaining({ title: 'Steering · Turn 1' }),
        })]);
        const calls = send.mock.calls.length;
        jest.advanceTimersByTime(LIVE_PHRASE_DISPLAY_MS + 1000);
        overlay.update(active, map, reference);
        expect(send).toHaveBeenCalledTimes(calls);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        overlay.update(snapshot(), map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        overlay.update(active, map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        overlay.dispose();
        expect(directory.getComponentNames()).toEqual([]);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        // This presentation belongs to the Assistant, not to Live Phrases.
        expect(destroySession).not.toHaveBeenCalled();
    });

    it.each(['exit', 'stale', 'map removed', 'unmount'] as const)('cancels a pending corner on %s', async (action) => {
        let finish!: (value: unknown) => void;
        createSession.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
        const { map, reference } = steeringCornerFixture();
        const active = steeringSnapshot();
        overlay.update(active, map, reference);
        if (action === 'exit') overlay.update(snapshot(), map, reference);
        if (action === 'stale') overlay.update({ ...active, telemetryReady: false }, map, reference);
        if (action === 'map removed') overlay.update(active, null, reference);
        if (action === 'unmount') overlay.dispose();
        await act(async () => finish({ success: true, presentation: session }));
        expect(manager.getPresentationSnapshot()?.cards ?? []).toEqual([]);
    });

    it('shows only the corner when a phrase triggers and does not insert it into recorded mode', async () => {
        const { map, reference } = steeringCornerFixture();
        const active = { ...steeringSnapshot(), events: [phraseEvent(PHRASE_DEFINITIONS.findIndex((rule) => rule.name === 'Steering'))] };
        await act(async () => overlay.update(active, map, reference));
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        expect(manager.getPresentationSnapshot()!.cards[0].componentType).toBe('live_steering_corner');
        expect(setEnabled).not.toHaveBeenCalled();
        overlay.update(snapshot());
        createSession.mockResolvedValue({ success: true, presentation: { ...session, mode: 'recorded' } });
        await overlaySessionClient.create({ ...session, mode: 'recorded' });
        overlay.update(steeringSnapshot(), map, reference);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        expect(createSession).toHaveBeenCalledTimes(2);
    });

    it.each(['acc', 'iracing'] as const)('publishes a Steering corner at centerline entry in %s before speech is ready', async (game) => {
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
        fireEvent.click(screen.getByRole('button', { name: 'Enable detection' }));
        await act(async () => {
            telemetry.publishFrame({ type: 'frame', game, sequence: 1, committedCount: 0, committedSequence: 0,
                sample: { Static_track: 'test', Static_car_model: 'car', Graphics_status: 2,
                    Physics_speed_kmh: 36, Graphics_normalized_car_position: 0.25 } });
        });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards
            .some((card: any) => card.componentType === 'live_steering_corner')).toBe(false);
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
        // Expire live input while retained speech history still exists.
        await act(async () => { jest.advanceTimersByTime(1750); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards
            .some((card: any) => card.componentType === 'live_steering_corner')).toBe(false);
        view.unmount();
    });

    it('keeps a real rule trigger in the local history without an overlay card under StrictMode', async () => {
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
        fireEvent.click(screen.getByRole('button', { name: 'Enable detection' }));
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
        expect(within(screen.getByRole('region', { name: 'Triggered sentences' })).getByText(
            PHRASE_DEFINITIONS.find((rule) => rule.id === 'pressure-feint')!.sentence,
        )).toBeVisible();
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Disable detection' })); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards).toEqual([]);
        view.unmount();
        jest.runAllTicks();
        expect(jest.getTimerCount()).toBe(0);
    });
});
