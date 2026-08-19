import { AiOverlayManagerController, type AiOverlayManagerTransport } from './AiOverlayManager';
import { MutableAiOverlayComponent } from './MutableAiOverlayComponent';
import type { AiOverlayPresentationSnapshot } from './ai-overlay-types';

const session = {
    presentationId: 'presentation-1',
    aiSessionId: 'ai-1',
    mode: 'live' as const,
    displayIdentity: { name: 'Kestrel' },
};

describe('AiOverlayManagerController', () => {
    let now: number;
    let sent: AiOverlayPresentationSnapshot[];
    let transport: AiOverlayManagerTransport;
    let manager: AiOverlayManagerController;

    beforeEach(() => {
        jest.useFakeTimers();
        now = 1_000;
        sent = [];
        transport = {
            send: (presentation) => { sent.push(presentation); },
            now: () => now,
            setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
            clearTimer: (timer) => clearTimeout(timer),
        };
        manager = new AiOverlayManagerController(transport);
        manager.setPresentation(session);
    });

    afterEach(() => {
        manager.dispose();
        jest.useRealTimers();
    });

    const component = (
        name: string,
        componentType = 'test-card',
        placement: 'pinned' | 'flow' = 'flow',
    ) => new MutableAiOverlayComponent<{ value: string }>(
        name,
        componentType,
        (_snapshot, publication) => ({
            placement,
            requestedStatus: publication.requestedStatus ?? 'expanded',
            presentationId: publication.presentationId,
        }),
    );

    it('uses componentName as identity and lets multiple references share a componentType', () => {
        const first = component('tool:run-1', 'tool_status');
        const second = component('tool:run-2', 'tool_status');
        const firstRef = { current: first };
        const secondRef = { current: second };
        manager.syncReferences([firstRef, secondRef]);

        first.publish({ value: 'started' });
        second.publish({ value: 'started' });
        first.publish({ value: 'complete' });

        const cards = manager.getPresentationSnapshot()!.cards;
        expect(cards.map((card) => card.componentName)).toEqual(['tool:run-1', 'tool:run-2']);
        expect(cards.every((card) => card.componentType === 'tool_status')).toBe(true);
        expect(cards.find((card) => card.componentName === 'tool:run-1')).toMatchObject({
            snapshot: { value: 'complete' },
            revision: 2,
        });
    });

    it('orders pinned cards first and flow cards by most recent publication', () => {
        const olderFlow = component('flow:older');
        const pinned = component('pinned', 'test-card', 'pinned');
        const newerFlow = component('flow:newer');
        manager.syncReferences([
            { current: olderFlow },
            { current: pinned },
            { current: newerFlow },
        ]);

        olderFlow.publish({ value: 'one' });
        pinned.publish({ value: 'two' });
        newerFlow.publish({ value: 'three' });

        expect(manager.getPresentationSnapshot()!.cards.map((card) => card.componentName))
            .toEqual(['pinned', 'flow:newer', 'flow:older']);
    });

    it('executes component-provided transient removal and fold deadlines', () => {
        const transient = new MutableAiOverlayComponent<{ value: string }>(
            'transient',
            'tool_status',
            () => ({ placement: 'flow', requestedStatus: 'expanded', transientDurationMs: 3_800 }),
        );
        const folding = new MutableAiOverlayComponent<{ value: string }>(
            'folding',
            'baseline_progress',
            () => ({ placement: 'flow', requestedStatus: 'expanded', foldAfterMs: 3_800 }),
        );
        manager.syncReferences([{ current: transient }, { current: folding }]);
        transient.publish({ value: 'running' });
        folding.publish({ value: '25%' });

        now += 3_800;
        jest.advanceTimersByTime(3_800);

        expect(manager.getPresentationSnapshot()!.cards).toEqual([
            expect.objectContaining({ componentName: 'folding', status: 'folded' }),
        ]);

        folding.publish({ value: '50%' });
        expect(manager.getPresentationSnapshot()!.cards[0]).toMatchObject({
            componentName: 'folding',
            status: 'expanded',
            revision: 2,
        });
    });

    it('arbitrates full-size requests by recency and falls back when a source clears', () => {
        const first = component('comparison:first', 'driver_expert_comparison');
        const second = component('comparison:second', 'driver_expert_comparison');
        manager.syncReferences([{ current: first }, { current: second }]);
        first.publish({ value: 'first' }, { requestedStatus: 'full_size' });
        second.publish({ value: 'second' }, { requestedStatus: 'full_size' });

        expect(manager.getPresentationSnapshot()!.cards).toEqual([
            expect.objectContaining({ componentName: 'comparison:second', status: 'full_size' }),
            expect.objectContaining({ componentName: 'comparison:first', status: 'expanded' }),
        ]);

        second.clear();
        expect(manager.getPresentationSnapshot()!.cards).toEqual([
            expect.objectContaining({ componentName: 'comparison:first', status: 'full_size' }),
        ]);
    });

    it('ignores stale renderer events and invokes the retained creator reference', () => {
        const onRendererEvent = jest.fn(() => ({ removeAfterMs: 500 }));
        const source = new MutableAiOverlayComponent<{ value: string }>(
            'message',
            'ai_message',
            () => ({ placement: 'flow', requestedStatus: 'expanded' }),
            onRendererEvent,
        );
        manager.syncReferences([{ current: source }]);
        source.publish({ value: 'hello' });
        source.publish({ value: 'updated' });

        manager.handleRendererEvent({
            presentationId: session.presentationId,
            componentName: 'message',
            revision: 1,
            event: 'visual_complete',
        });
        expect(onRendererEvent).not.toHaveBeenCalled();

        manager.handleRendererEvent({
            presentationId: session.presentationId,
            componentName: 'message',
            revision: 2,
            event: 'visual_complete',
        });
        expect(onRendererEvent).toHaveBeenCalledTimes(1);
        now += 500;
        jest.advanceTimersByTime(500);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(0);
    });

    it('removes unregistered sources and drops presentation-scoped cards on replacement', () => {
        const global = component('goal', 'goal');
        const scoped = component('message', 'ai_message');
        const globalRef = { current: global };
        const scopedRef = { current: scoped };
        manager.syncReferences([globalRef, scopedRef]);
        global.publish({ value: 'active' });
        scoped.publish({ value: 'old session' }, { presentationId: session.presentationId });

        manager.setPresentation({ ...session, presentationId: 'presentation-2' });
        expect(manager.getPresentationSnapshot()!.cards.map((card) => card.componentName)).toEqual(['goal']);

        manager.syncReferences([]);
        expect(manager.getPresentationSnapshot()!.cards).toHaveLength(0);
        expect(sent.at(-1)?.cards).toHaveLength(0);
    });

    it('reports overlay transmission errors to the console', async () => {
        const error = new Error('transport unavailable');
        const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        transport.send = () => Promise.reject(error);

        manager.setPresentation(session);
        await Promise.resolve();

        expect(consoleError).toHaveBeenCalledWith(
            'Failed to transmit overlay presentation.',
            error,
        );
        consoleError.mockRestore();
    });
});
