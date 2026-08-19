import type {
    AiOverlayPresentationAcknowledgement,
    AiOverlayPresentationSession,
    AiOverlayPresentationSnapshot,
    AiOverlayRendererEvent,
    AiOverlaySessionDescriptor,
} from './ai-overlay-types';
import { cloneJsonSnapshot, isJsonSafe } from './ai-overlay-types';

interface OverlaySessionResult {
    success: boolean;
    presentation?: AiOverlayPresentationSession;
    ended?: boolean;
    error?: string;
}

interface ElectronOverlayApi {
    createOverlaySession?: (descriptor: AiOverlaySessionDescriptor) => Promise<OverlaySessionResult>;
    destroyOverlaySession?: (presentationId: string) => Promise<OverlaySessionResult>;
    setOverlayEnabled?: (enabled: boolean) => Promise<{ success: boolean; enabled?: boolean; error?: string }>;
    isOverlayEnabled?: () => Promise<boolean>;
    sendOverlayPresentation?: (
        presentation: AiOverlayPresentationSnapshot,
    ) => Promise<AiOverlayPresentationAcknowledgement>;
    onOverlayRendererEvent?: (
        listener: (event: AiOverlayRendererEvent) => void,
    ) => (() => void);
}

const getApi = (): ElectronOverlayApi | undefined => (
    typeof window === 'undefined'
        ? undefined
        : (window as unknown as { electronAPI?: ElectronOverlayApi }).electronAPI
);

export const sendOverlayPresentation = async (
    presentation: AiOverlayPresentationSnapshot,
): Promise<AiOverlayPresentationAcknowledgement> => {
    if (!isJsonSafe(presentation)) {
        throw new Error('Overlay presentations must contain only JSON-safe values.');
    }
    const api = getApi();
    if (!api?.sendOverlayPresentation) throw new Error('Electron overlay is unavailable.');
    const serializable = cloneJsonSnapshot(presentation);
    const acknowledgement = await api.sendOverlayPresentation(serializable);
    if (acknowledgement.presentationId !== presentation.presentationId
        || acknowledgement.presentationRevision !== presentation.presentationRevision) {
        throw new Error('Overlay acknowledgement belongs to another presentation revision.');
    }
    if (!acknowledgement.accepted) {
        throw new Error(acknowledgement.error || 'Overlay presentation was rejected.');
    }
    return acknowledgement;
};

export const subscribeOverlayRendererEvents = (
    listener: (event: AiOverlayRendererEvent) => void,
): (() => void) => getApi()?.onOverlayRendererEvent?.(listener) ?? (() => undefined);

let currentPresentation: AiOverlayPresentationSession | null = null;
let sessionRequestSequence = 0;
const sessionListeners = new Set<(presentation: AiOverlayPresentationSession | null) => void>();

const notifySessionListeners = () => {
    sessionListeners.forEach((listener) => listener(currentPresentation));
};

const assertSessionResult = (result: OverlaySessionResult | undefined) => {
    if (!result?.success) throw new Error(result?.error || 'Electron overlay session request failed.');
};

export const overlaySessionClient = {
    available(): boolean {
        return Boolean(getApi()?.createOverlaySession && getApi()?.setOverlayEnabled);
    },

    current(): AiOverlayPresentationSession | null {
        return currentPresentation;
    },

    subscribe(listener: (presentation: AiOverlayPresentationSession | null) => void): () => void {
        sessionListeners.add(listener);
        listener(currentPresentation);
        return () => sessionListeners.delete(listener);
    },

    async create(descriptor: AiOverlaySessionDescriptor): Promise<AiOverlayPresentationSession> {
        const api = getApi();
        if (!api?.createOverlaySession) throw new Error('Electron overlay is unavailable.');
        const sequence = ++sessionRequestSequence;
        const result = await api.createOverlaySession(descriptor);
        assertSessionResult(result);
        if (!result.presentation) throw new Error('Electron did not return an overlay presentation.');
        if (sequence === sessionRequestSequence) {
            currentPresentation = result.presentation;
            notifySessionListeners();
        }
        return result.presentation;
    },

    async destroy(presentationId = currentPresentation?.presentationId): Promise<void> {
        if (!presentationId) return;
        const api = getApi();
        if (!api?.destroyOverlaySession) return;
        const result = await api.destroyOverlaySession(presentationId);
        assertSessionResult(result);
        if (result.ended !== false && currentPresentation?.presentationId === presentationId) {
            currentPresentation = null;
            notifySessionListeners();
        }
    },

    async setEnabled(enabled: boolean): Promise<void> {
        const api = getApi();
        if (!api?.setOverlayEnabled) throw new Error('Electron overlay is unavailable.');
        const result = await api.setOverlayEnabled(enabled);
        if (!result?.success) throw new Error(result?.error || 'Electron overlay visibility request failed.');
    },

    async isEnabled(): Promise<boolean> {
        return Boolean(await getApi()?.isOverlayEnabled?.());
    },
};
