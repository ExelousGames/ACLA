import {
    overlaySessionClient,
    sendOverlayPresentation,
    subscribeOverlayRendererEvents,
} from './overlay-display-client';
import type {
    AiOverlayPresentationSnapshot,
    AiOverlayRendererEvent,
} from './ai-overlay-types';

const session = {
    presentationId: 'presentation-1',
    aiSessionId: 'ai-1',
    mode: 'live' as const,
    displayIdentity: { name: 'Kestrel' },
};

const snapshot = (): AiOverlayPresentationSnapshot => ({
    presentationId: session.presentationId,
    presentationRevision: 1,
    session,
    cards: [],
});

describe('overlay renderer transport', () => {
    const sendOverlayPresentationMock = jest.fn();
    let rendererEventListener: ((event: AiOverlayRendererEvent) => void) | null = null;

    beforeEach(async () => {
        sendOverlayPresentationMock.mockReset().mockResolvedValue({
            presentationId: session.presentationId,
            presentationRevision: 1,
            accepted: true,
        });
        rendererEventListener = null;
        (window as any).electronAPI = {
            sendOverlayPresentation: sendOverlayPresentationMock,
            onOverlayRendererEvent: (listener: typeof rendererEventListener) => {
                rendererEventListener = listener;
                return () => { rendererEventListener = null; };
            },
            createOverlaySession: jest.fn().mockResolvedValue({ success: true, presentation: session }),
            destroyOverlaySession: jest.fn().mockResolvedValue({ success: true, ended: true }),
            setOverlayEnabled: jest.fn().mockResolvedValue({ success: true, enabled: true }),
            isOverlayEnabled: jest.fn().mockResolvedValue(true),
        };
        await overlaySessionClient.destroy().catch(() => undefined);
    });

    it('sends one JSON presentation snapshot and validates its acknowledgement', async () => {
        const presentation = snapshot();
        await expect(sendOverlayPresentation(presentation)).resolves.toMatchObject({ accepted: true });
        expect(sendOverlayPresentationMock).toHaveBeenCalledWith(presentation);
        expect(sendOverlayPresentationMock.mock.calls[0][0]).not.toBe(presentation);
    });

    it('rejects non-serializable presentation data before IPC', async () => {
        const presentation = snapshot();
        presentation.cards = [{ snapshot: () => undefined } as any];
        await expect(sendOverlayPresentation(presentation)).rejects.toThrow('JSON-safe');
        expect(sendOverlayPresentationMock).not.toHaveBeenCalled();
    });

    it('forwards renderer events to manager subscribers', () => {
        const listener = jest.fn();
        const unsubscribe = subscribeOverlayRendererEvents(listener);
        const event: AiOverlayRendererEvent = {
            presentationId: session.presentationId,
            componentName: 'message:one',
            revision: 2,
            event: 'visual_complete',
        };
        rendererEventListener?.(event);
        expect(listener).toHaveBeenCalledWith(event);
        unsubscribe();
        expect(rendererEventListener).toBeNull();
    });

    it('publishes session replacement and teardown to subscribers', async () => {
        const listener = jest.fn();
        const unsubscribe = overlaySessionClient.subscribe(listener);
        await overlaySessionClient.create({
            aiSessionId: session.aiSessionId,
            mode: session.mode,
            displayIdentity: session.displayIdentity,
        });
        expect(listener).toHaveBeenLastCalledWith(session);
        await overlaySessionClient.destroy(session.presentationId);
        expect(listener).toHaveBeenLastCalledWith(null);
        unsubscribe();
    });
});
