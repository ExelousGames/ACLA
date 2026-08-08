import { overlayDisplayClient, overlaySessionClient } from './overlay-display-client';
import type { OverlayLifecycleEvent } from './overlay-display-types';

describe('overlay display client', () => {
    const descriptor = {
        aiSessionId: 'ai-live-1',
        mode: 'live' as const,
        displayIdentity: { name: 'Kestrel', agentTags: ['Live'] },
    };
    const presentation = { ...descriptor, presentationId: 'presentation-live-1' };
    let lifecycleListener: ((event: OverlayLifecycleEvent) => void) | null;
    const sendOverlayDisplayRequest = jest.fn();
    const removeLifecycleListener = jest.fn();

    beforeEach(() => {
        lifecycleListener = null;
        sendOverlayDisplayRequest.mockReset().mockImplementation(async (request) => ({
            presentationId: request.presentationId,
            requestId: request.requestId,
            accepted: true,
            instanceId: 'map:singleton',
        }));
        removeLifecycleListener.mockReset();
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                sendOverlayDisplayRequest,
                onOverlayLifecycle: (listener: (event: OverlayLifecycleEvent) => void) => {
                    lifecycleListener = listener;
                    return () => {
                        lifecycleListener = null;
                        removeLifecycleListener();
                    };
                },
                createOverlaySession: jest.fn(async () => ({ success: true, presentation })),
                destroyOverlaySession: jest.fn(async () => ({ success: true, ended: true })),
                setOverlayEnabled: jest.fn(async (enabled: boolean) => ({ success: true, enabled })),
                isOverlayEnabled: jest.fn(async () => true),
            },
        });
    });

    it('returns acknowledged stable instance IDs and sends complete JSON snapshots', async () => {
        const instanceId = await overlayDisplayClient.forPresentation(presentation.presentationId).upsert('map', {
            status: 'unavailable',
            title: 'Circuit map',
            reason: 'No track selected',
        }, { metadata: { name: 'Kestrel', agentTags: ['Track Guide'] } });

        expect(instanceId).toBe('map:singleton');
        expect(sendOverlayDisplayRequest).toHaveBeenCalledWith(expect.objectContaining({
            presentationId: presentation.presentationId,
            command: {
                operation: 'upsert',
                type: 'map',
                snapshot: {
                    status: 'unavailable',
                    title: 'Circuit map',
                    reason: 'No track selected',
                },
                options: { metadata: { name: 'Kestrel', agentTags: ['Track Guide'] } },
            },
        }));
    });

    it('correlates lifecycle listeners and removes the Electron subscription when idle', () => {
        const first = jest.fn();
        const second = jest.fn();
        const scoped = overlayDisplayClient.forPresentation(presentation.presentationId);
        const unsubscribeFirst = scoped.subscribeLifecycle('map:singleton', first);
        const unsubscribeSecond = scoped.subscribeLifecycle('other:singleton', second);
        lifecycleListener?.({
            eventId: 'event-1',
            presentationId: presentation.presentationId,
            instanceId: 'map:singleton',
            type: 'map',
            kind: 'updated',
            at: 1,
        });
        expect(first).toHaveBeenCalledTimes(1);
        expect(second).not.toHaveBeenCalled();

        unsubscribeFirst();
        expect(removeLifecycleListener).not.toHaveBeenCalled();
        unsubscribeSecond();
        expect(removeLifecycleListener).toHaveBeenCalledTimes(1);
    });

    it('supports abortable lifecycle waits without leaking subscriptions', async () => {
        const controller = new AbortController();
        const waiting = overlayDisplayClient.forPresentation(presentation.presentationId).waitForLifecycle(
            'map:singleton',
            'exited',
            { signal: controller.signal },
        );
        controller.abort();
        await expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
        expect(removeLifecycleListener).toHaveBeenCalledTimes(1);
    });

    it('creates typed presentations while keeping the global visibility setting independent', async () => {
        expect(overlaySessionClient.available()).toBe(true);
        await expect(overlaySessionClient.create(descriptor)).resolves.toEqual(presentation);
        expect(overlaySessionClient.current()).toEqual(presentation);
        await expect(overlaySessionClient.setEnabled(true)).resolves.toBeUndefined();
        await expect(overlaySessionClient.isEnabled()).resolves.toBe(true);
        await expect(overlaySessionClient.destroy(presentation.presentationId)).resolves.toBeUndefined();
        expect(overlaySessionClient.current()).toBeNull();
    });

    it('isolates lifecycle listeners for identical instance IDs in different presentations', () => {
        const current = jest.fn();
        const stale = jest.fn();
        const removeCurrent = overlayDisplayClient
            .forPresentation(presentation.presentationId)
            .subscribeLifecycle('ai_message:singleton', current);
        const removeStale = overlayDisplayClient
            .forPresentation('presentation-old')
            .subscribeLifecycle('ai_message:singleton', stale);

        lifecycleListener?.({
            eventId: 'event-current',
            presentationId: presentation.presentationId,
            instanceId: 'ai_message:singleton',
            type: 'ai_message',
            kind: 'updated',
            at: 1,
        });
        expect(current).toHaveBeenCalledTimes(1);
        expect(stale).not.toHaveBeenCalled();
        removeCurrent();
        removeStale();
    });
});
