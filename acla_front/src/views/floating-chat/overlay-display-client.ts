import {
    isJsonSafe,
    type OverlayDisplayAcknowledgement,
    type OverlayDisplayCommand,
    type OverlayDisplayRequest,
    type OverlayDisplayType,
    type OverlayExitReason,
    type OverlayLifecycleEvent,
    type OverlayPolicy,
    type OverlayPresentationSession,
    type OverlaySessionDescriptor,
    type OverlaySnapshotByType,
    type OverlayTarget,
    type OverlayUpsertOptions,
} from './overlay-display-types';

interface OverlaySessionResult {
    success: boolean;
    presentation?: OverlayPresentationSession;
    ended?: boolean;
    error?: string;
}

interface ElectronOverlayApi {
    createOverlaySession?: (descriptor: OverlaySessionDescriptor) => Promise<OverlaySessionResult>;
    destroyOverlaySession?: (presentationId: string) => Promise<OverlaySessionResult>;
    setOverlayEnabled?: (enabled: boolean) => Promise<{ success: boolean; enabled?: boolean; error?: string }>;
    isOverlayEnabled?: () => Promise<boolean>;
    sendOverlayDisplayRequest?: (request: OverlayDisplayRequest) => Promise<OverlayDisplayAcknowledgement>;
    onOverlayLifecycle?: (listener: (event: OverlayLifecycleEvent) => void) => (() => void);
}

const getApi = (): ElectronOverlayApi | undefined => (
    typeof window === 'undefined'
        ? undefined
        : (window as unknown as { electronAPI?: ElectronOverlayApi }).electronAPI
);

let requestSequence = 0;
const createRequestId = (): string => (
    `overlay-request-${Date.now().toString(36)}-${(++requestSequence).toString(36)}`
);

const sendRequest = async (
    presentationId: string,
    command: OverlayDisplayCommand,
): Promise<OverlayDisplayAcknowledgement> => {
    const request: OverlayDisplayRequest = {
        presentationId,
        requestId: createRequestId(),
        command,
    };
    let serializableRequest: OverlayDisplayRequest;
    try {
        serializableRequest = JSON.parse(JSON.stringify(request)) as OverlayDisplayRequest;
    } catch {
        throw new Error('Overlay requests must be JSON-safe.');
    }
    if (!isJsonSafe(serializableRequest)) throw new Error('Overlay requests must be JSON-safe.');
    const api = getApi();
    if (!api?.sendOverlayDisplayRequest) throw new Error('Electron overlay is unavailable.');
    const acknowledgement = await api.sendOverlayDisplayRequest(serializableRequest);
    if (acknowledgement?.presentationId !== presentationId) {
        throw new Error('Overlay acknowledgement belongs to another presentation.');
    }
    if (!acknowledgement.accepted) {
        throw new Error(acknowledgement.error || 'Overlay request was rejected.');
    }
    return acknowledgement;
};

type LifecycleListener = (event: OverlayLifecycleEvent) => void;
const lifecycleKey = (presentationId: string, instanceId: string) => `${presentationId}\u0000${instanceId}`;
const lifecycleListeners = new Map<string, Set<LifecycleListener>>();
let removeElectronListener: (() => void) | null = null;

const ensureLifecycleSubscription = () => {
    if (removeElectronListener) return;
    const api = getApi();
    if (!api?.onOverlayLifecycle) return;
    removeElectronListener = api.onOverlayLifecycle((event) => {
        lifecycleListeners
            .get(lifecycleKey(event.presentationId, event.instanceId))
            ?.forEach((listener) => listener(event));
    });
};

const releaseLifecycleSubscriptionIfIdle = () => {
    if (lifecycleListeners.size > 0 || !removeElectronListener) return;
    removeElectronListener();
    removeElectronListener = null;
};

export interface ScopedOverlayDisplayClient {
    readonly presentationId: string;
    upsert<T extends OverlayDisplayType>(
        type: T,
        fullSnapshot: OverlaySnapshotByType[T],
        options?: OverlayUpsertOptions,
    ): Promise<string>;
    setPolicy(target: OverlayTarget, policy: OverlayPolicy): Promise<void>;
    exit(target: OverlayTarget, reason?: OverlayExitReason): Promise<void>;
    subscribeLifecycle(instanceId: string, listener: LifecycleListener): () => void;
    waitForLifecycle(
        instanceId: string,
        predicate?: OverlayLifecycleEvent['kind'] | ((event: OverlayLifecycleEvent) => boolean),
        options?: { signal?: AbortSignal },
    ): Promise<OverlayLifecycleEvent>;
}

const createScopedClient = (presentationId: string): ScopedOverlayDisplayClient => ({
    presentationId,

    async upsert<T extends OverlayDisplayType>(
        type: T,
        fullSnapshot: OverlaySnapshotByType[T],
        options?: OverlayUpsertOptions,
    ): Promise<string> {
        const acknowledgement = await sendRequest(presentationId, {
            operation: 'upsert',
            type,
            snapshot: fullSnapshot,
            options,
        });
        if (!acknowledgement.instanceId) throw new Error('Overlay did not acknowledge an instance ID.');
        return acknowledgement.instanceId;
    },

    async setPolicy(target: OverlayTarget, policy: OverlayPolicy): Promise<void> {
        await sendRequest(presentationId, { operation: 'set_policy', target, policy });
    },

    async exit(target: OverlayTarget, reason: OverlayExitReason = 'producer_exit'): Promise<void> {
        await sendRequest(presentationId, { operation: 'exit', target, reason });
    },

    subscribeLifecycle(instanceId: string, listener: LifecycleListener): () => void {
        const key = lifecycleKey(presentationId, instanceId);
        const listeners = lifecycleListeners.get(key) ?? new Set<LifecycleListener>();
        listeners.add(listener);
        lifecycleListeners.set(key, listeners);
        ensureLifecycleSubscription();
        return () => {
            const current = lifecycleListeners.get(key);
            current?.delete(listener);
            if (current?.size === 0) lifecycleListeners.delete(key);
            releaseLifecycleSubscriptionIfIdle();
        };
    },

    waitForLifecycle(
        instanceId: string,
        predicate: OverlayLifecycleEvent['kind'] | ((event: OverlayLifecycleEvent) => boolean) = 'exited',
        options: { signal?: AbortSignal } = {},
    ): Promise<OverlayLifecycleEvent> {
        return new Promise((resolve, reject) => {
            const matches = typeof predicate === 'function'
                ? predicate
                : (event: OverlayLifecycleEvent) => event.kind === predicate;
            let unsubscribe: () => void = () => undefined;
            const onAbort = () => {
                unsubscribe();
                reject(new DOMException('Overlay lifecycle wait was aborted.', 'AbortError'));
            };
            if (options.signal?.aborted) {
                onAbort();
                return;
            }
            unsubscribe = createScopedClient(presentationId).subscribeLifecycle(instanceId, (event) => {
                if (!matches(event)) return;
                options.signal?.removeEventListener('abort', onAbort);
                unsubscribe();
                resolve(event);
            });
            options.signal?.addEventListener('abort', onAbort, { once: true });
        });
    },
});

let currentPresentation: OverlayPresentationSession | null = null;
let sessionRequestSequence = 0;

const requireCurrentPresentation = (): OverlayPresentationSession => {
    if (!currentPresentation) throw new Error('No active overlay presentation.');
    return currentPresentation;
};

/** Convenience facade for producers that publish synchronously into the current presentation. */
export const overlayDisplayClient = {
    forPresentation: createScopedClient,
    currentPresentation: (): OverlayPresentationSession | null => currentPresentation,

    upsert<T extends OverlayDisplayType>(
        type: T,
        fullSnapshot: OverlaySnapshotByType[T],
        options?: OverlayUpsertOptions,
    ): Promise<string> {
        return createScopedClient(requireCurrentPresentation().presentationId)
            .upsert(type, fullSnapshot, options);
    },

    setPolicy(target: OverlayTarget, policy: OverlayPolicy): Promise<void> {
        return createScopedClient(requireCurrentPresentation().presentationId).setPolicy(target, policy);
    },

    exit(target: OverlayTarget, reason: OverlayExitReason = 'producer_exit'): Promise<void> {
        return createScopedClient(requireCurrentPresentation().presentationId).exit(target, reason);
    },

    subscribeLifecycle(instanceId: string, listener: LifecycleListener): () => void {
        return createScopedClient(requireCurrentPresentation().presentationId)
            .subscribeLifecycle(instanceId, listener);
    },

    waitForLifecycle(
        instanceId: string,
        predicate: OverlayLifecycleEvent['kind'] | ((event: OverlayLifecycleEvent) => boolean) = 'exited',
        options: { signal?: AbortSignal } = {},
    ): Promise<OverlayLifecycleEvent> {
        return createScopedClient(requireCurrentPresentation().presentationId)
            .waitForLifecycle(instanceId, predicate, options);
    },
};

const assertSessionResult = (result: OverlaySessionResult | undefined) => {
    if (!result?.success) throw new Error(result?.error || 'Electron overlay session request failed.');
};

export const overlaySessionClient = {
    available(): boolean {
        return Boolean(getApi()?.createOverlaySession && getApi()?.setOverlayEnabled);
    },

    current(): OverlayPresentationSession | null {
        return currentPresentation;
    },

    async create(descriptor: OverlaySessionDescriptor): Promise<OverlayPresentationSession> {
        const api = getApi();
        if (!api?.createOverlaySession) throw new Error('Electron overlay is unavailable.');
        const sequence = ++sessionRequestSequence;
        const result = await api.createOverlaySession(descriptor);
        assertSessionResult(result);
        if (!result.presentation) throw new Error('Electron did not return an overlay presentation.');
        if (sequence === sessionRequestSequence) currentPresentation = result.presentation;
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
