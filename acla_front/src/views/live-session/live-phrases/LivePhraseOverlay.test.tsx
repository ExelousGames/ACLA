import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import {
    createOperationComponentRefDirectory, OperationComponentRefProvider, useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import { AiOverlayManagerController } from 'views/floating-chat/AiOverlayManager';
import { isAiOverlayComponentHandle, isJsonSafe } from 'views/floating-chat/ai-overlay-types';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import { LivePhraseOverlay, LIVE_PHRASE_DISPLAY_MS, type LivePhraseOverlaySnapshot } from './LivePhraseOverlay';
import { livePhraseOverlayRenderer } from './graphs/LivePhraseDisplay';
import { PHRASE_DEFINITIONS, PhraseEngine, type PhraseEvent } from './phrase-engine';
import LivePhrases from './LivePhrases';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import { circuitMap, vision } from './test-fixtures';

jest.mock('components/tts', () => ({ synthesizeTts: jest.fn(() => new Promise(() => undefined)) }));

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

    it('registers every phrase as an addon and renders its JSON-safe snapshot in all supported states', async () => {
        await overlaySessionClient.create(session);
        expect(directory.getComponentNames()).toHaveLength(PHRASE_DEFINITIONS.length);
        expect(directory.getComponentRefs().every((ref) => isAiOverlayComponentHandle(ref.current))).toBe(true);
        PHRASE_DEFINITIONS.forEach((_rule, index) => overlay.update(snapshot(phraseEvent(index))));
        const cards = manager.getPresentationSnapshot()!.cards;
        expect(cards).toHaveLength(PHRASE_DEFINITIONS.length);
        cards.forEach((card) => {
            expect(card.componentName).toBe(`live-phrases:${(card.snapshot as PhraseEvent).ruleId}`);
            expect(card.componentType).toBe('live_phrase');
            expect(isJsonSafe(card.snapshot)).toBe(true);
            expect(livePhraseOverlayRenderer.validateSnapshot(card.snapshot)).toBe(true);
            const data = card.snapshot as LivePhraseOverlaySnapshot;
            for (const status of ['expanded', 'focus', 'folded'] as const) {
                const view = render(<>{livePhraseOverlayRenderer.renderOverlay(data, status, {
                    componentName: card.componentName, revision: card.revision, emitRendererEvent: jest.fn(),
                })}</>);
                expect(screen.getByText(data.name)).toBeVisible();
                expect(Boolean(screen.queryByText(data.sentence))).toBe(status !== 'folded');
                view.unmount();
            }
        });
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(setEnabled).not.toHaveBeenCalled();
    });

    it('does not republish retained events or extend their lifetime on telemetry updates', async () => {
        await overlaySessionClient.create(session);
        const event = phraseEvent();
        overlay.update(snapshot(event));
        const calls = send.mock.calls.length;
        jest.advanceTimersByTime(1_000);
        overlay.update({ ...snapshot(event), telemetryReady: true });
        expect(send).toHaveBeenCalledTimes(calls);
        jest.advanceTimersByTime(LIVE_PHRASE_DISPLAY_MS - 1_000);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        manager.setPresentation({ ...session, presentationId: 'next-presentation' });
        manager.setPresentation(session);
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
    });

    it('clears cards on reset, accepts a fresh trigger and releases all sources on unmount', async () => {
        await overlaySessionClient.create(session);
        overlay.update(snapshot(phraseEvent()));
        overlay.update(snapshot());
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        overlay.update(snapshot(phraseEvent(0, 2)));
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(1);
        overlay.dispose();
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        expect(directory.getComponentNames()).toEqual([]);
        expect(jest.getTimerCount()).toBe(0);
        // This presentation belongs to the Assistant, not to Live Phrases.
        expect(destroySession).not.toHaveBeenCalled();
    });

    it('creates one local presentation for pending phrases and preserves overlay visibility', async () => {
        await act(async () => {
            overlay.update(snapshot(phraseEvent()));
            overlay.update(snapshot(phraseEvent(1)));
        });
        expect(createSession).toHaveBeenCalledTimes(1);
        expect(createSession).toHaveBeenCalledWith(expect.objectContaining({ aiSessionId: 'live-phrases', mode: 'live' }));
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(2);
        expect(setEnabled).not.toHaveBeenCalled();
        await act(async () => overlay.dispose());
        expect(destroySession).toHaveBeenCalledWith(session.presentationId);
    });

    it.each(['reset', 'unmount', 'expiry'] as const)('drops pending phrases after %s', async (action) => {
        let finish!: (value: unknown) => void;
        createSession.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
        overlay.update(snapshot(phraseEvent()));
        if (action === 'reset') overlay.update(snapshot());
        if (action === 'unmount') overlay.dispose();
        if (action === 'expiry') jest.advanceTimersByTime(LIVE_PHRASE_DISPLAY_MS);
        await act(async () => finish({ success: true, presentation: session }));
        expect(manager.getPresentationSnapshot()?.cards ?? []).toEqual([]);
        expect(destroySession.mock.calls).toEqual(action === 'unmount' ? [[session.presentationId]] : []);
    });

    it('does not publish live guidance into another mode or replace its presentation', async () => {
        createSession.mockResolvedValue({ success: true, presentation: { ...session, mode: 'recorded' } });
        await overlaySessionClient.create({ ...session, mode: 'recorded' });
        overlay.update(snapshot(phraseEvent()));
        expect(manager.getPresentationSnapshot()!.cards).toEqual([]);
        expect(createSession).toHaveBeenCalledTimes(1);
    });

    it('keeps phrases local in the browser and reports a failed desktop presentation', async () => {
        delete (window as any).electronAPI.createOverlaySession;
        overlay.update(snapshot(phraseEvent()));
        expect(createSession).not.toHaveBeenCalled();
        (window as any).electronAPI.createOverlaySession = createSession;
        createSession.mockRejectedValue(new Error('Window unavailable'));
        await act(async () => overlay.update(snapshot(phraseEvent(1))));
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'Window unavailable' }));
    });

    it('rejects invalid display data before rendering', () => {
        expect(livePhraseOverlayRenderer.validateSnapshot(null)).toBe(false);
        expect(livePhraseOverlayRenderer.validateSnapshot({ ...phraseEvent(), name: 'Test', eventId: 1, timestamp: 1e20 })).toBe(false);
        expect(livePhraseOverlayRenderer.validateSnapshot({ ...phraseEvent(), name: 'Test', eventId: 1, sentence: '' })).toBe(false);
    });

    it('automatically sends a real rule trigger through the registered manager under StrictMode', async () => {
        await overlaySessionClient.create(session);
        const submit = jest.fn(async (presentation) => ({
            presentationId: presentation.presentationId,
            presentationRevision: presentation.presentationRevision,
            accepted: true,
        }));
        (window as any).electronAPI.sendOverlayPresentation = submit;
        const telemetry = createLiveTelemetryStore();
        const map = circuitMap();
        const detection = vision(Date.now());
        const root = {
            getComponentName: () => 'live-session',
            subscribeTelemetry: telemetry.subscribeEvents,
            getTrackVisionDetection: () => detection,
            subscribeTrackVision: () => () => undefined,
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
        expect(cards).toHaveLength(1);
        expect(cards[0]).toMatchObject({
            componentName: 'live-phrases:inside-outbraking', componentType: 'live_phrase', status: 'focus',
            snapshot: { ruleId: 'inside-outbraking', sentence: PHRASE_DEFINITIONS.find((rule) => rule.id === 'inside-outbraking')!.sentence },
        });
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Disable detection' })); });
        expect(submit.mock.calls[submit.mock.calls.length - 1][0].cards).toEqual([]);
        view.unmount();
        jest.runAllTicks();
        expect(jest.getTimerCount()).toBe(0);
    });
});
