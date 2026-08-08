import type { OverlayDisplayDefinition } from './overlay-display-registry';
import { overlayDisplayRegistry } from './overlay-display-registry';
import {
    cloneJsonSnapshot,
    isJsonSafe,
    type OverlayComponentEvent,
    type OverlayDisplayAcknowledgement,
    type OverlayDisplayRequest,
    type OverlayDisplayType,
    type OverlayExitReason,
    type OverlayLifecycleEvent,
    type OverlayPolicy,
    type OverlayPresentationSession,
    type OverlayShellMetadata,
    type OverlayTarget,
} from './overlay-display-types';

export interface OverlayDisplayInstance {
    instanceId: string;
    type: OverlayDisplayType;
    key?: string;
    snapshot: unknown;
    metadata: OverlayShellMetadata;
    policy: OverlayPolicy;
    folded: boolean;
    shown: boolean;
    createdAt: number;
    updatedAt: number;
    revision: number;
    exitAt: number | null;
    foldAt: number | null;
}

export interface OverlayPresentationState {
    presentation: OverlayPresentationSession | null;
    instances: OverlayDisplayInstance[];
    enabled: boolean;
    eventSequence: number;
}

export interface OverlayTransitionResult {
    state: OverlayPresentationState;
    events: OverlayLifecycleEvent[];
    acknowledgement?: OverlayDisplayAcknowledgement;
}

export type OverlayPresentationAction =
    | { type: 'replace'; state: OverlayPresentationState }
    | { type: 'clear' };

export const initialOverlayPresentationState: OverlayPresentationState = {
    presentation: null,
    instances: [],
    enabled: false,
    eventSequence: 0,
};

export const overlayPresentationReducer = (
    state: OverlayPresentationState,
    action: OverlayPresentationAction,
): OverlayPresentationState => {
    if (action.type === 'replace') return action.state;
    if (action.type === 'clear') return { ...state, presentation: null, instances: [] };
    return state;
};

const definitionFor = (type: unknown): OverlayDisplayDefinition | null => {
    if (typeof type !== 'string' || !(type in overlayDisplayRegistry)) return null;
    return overlayDisplayRegistry[type as OverlayDisplayType] as OverlayDisplayDefinition;
};

const makeEmitter = (
    state: OverlayPresentationState,
    now: number,
    events: OverlayLifecycleEvent[],
    presentationId = state.presentation?.presentationId ?? 'unscoped',
) => (
    instanceId: string,
    kind: OverlayLifecycleEvent['kind'],
    details: Partial<OverlayLifecycleEvent> = {},
) => {
    state.eventSequence += 1;
    events.push({
        eventId: `overlay-event-${state.eventSequence}`,
        presentationId,
        instanceId,
        kind,
        at: now,
        ...details,
    });
};

const rejectRequest = (
    current: OverlayPresentationState,
    requestId: string,
    now: number,
    error: string,
    presentationId = current.presentation?.presentationId ?? 'unscoped',
): OverlayTransitionResult => {
    const state = { ...current, instances: [...current.instances] };
    const events: OverlayLifecycleEvent[] = [];
    makeEmitter(state, now, events, presentationId)(`rejected:${requestId}`, 'rejected', {
        requestId,
        message: error,
    });
    return {
        state,
        events,
        acknowledgement: { presentationId, requestId, accepted: false, error },
    };
};

const canonicalInstanceId = (
    definition: OverlayDisplayDefinition,
    request: OverlayDisplayRequest,
): { instanceId?: string; key?: string; error?: string } => {
    if (request.command.operation !== 'upsert') return { error: 'Not an upsert request.' };
    const { options, snapshot } = request.command;
    if (definition.cardinality === 'singleton') {
        return { instanceId: `${definition.type}:singleton` };
    }
    if (definition.cardinality === 'keyed') {
        const key = definition.resolveKey
            ? definition.resolveKey(snapshot, options)
            : options?.key ?? null;
        if (!key || !key.trim()) return { error: `${definition.type} requires a stable key.` };
        return { instanceId: `${definition.type}:key:${encodeURIComponent(key)}`, key };
    }
    const requestedId = options?.instanceId?.trim();
    return { instanceId: requestedId || `${definition.type}:multiple:${request.requestId}` };
};

const lifecycleDeadlines = (
    definition: OverlayDisplayDefinition,
    snapshot: unknown,
    policy: OverlayPolicy,
    now: number,
): Pick<OverlayDisplayInstance, 'exitAt' | 'foldAt' | 'folded'> => {
    if (policy === 'transient') {
        const duration = definition.transientDurationMs?.(snapshot) ?? null;
        return {
            exitAt: duration === null ? null : now + Math.max(0, duration),
            foldAt: null,
            folded: false,
        };
    }
    if (policy === 'fold_until_update') {
        return {
            exitAt: null,
            foldAt: now + Math.max(0, definition.pulseDurationMs ?? 0),
            folded: false,
        };
    }
    return { exitAt: null, foldAt: null, folded: false };
};

const resolveTargetIndex = (
    state: OverlayPresentationState,
    target: OverlayTarget,
): { index?: number; error?: string } => {
    if ('instanceId' in target) {
        const index = state.instances.findIndex((instance) => instance.instanceId === target.instanceId);
        return index >= 0 ? { index } : { error: `Overlay instance '${target.instanceId}' was not found.` };
    }
    const definition = definitionFor(target.type);
    if (!definition) return { error: `Unknown overlay display type '${target.type}'.` };
    if (definition.cardinality === 'multiple') {
        return { error: `Multiple display '${target.type}' must be targeted by instanceId.` };
    }
    if (definition.cardinality === 'keyed' && !target.key?.trim()) {
        return { error: `Keyed display '${target.type}' requires a key.` };
    }
    const instanceId = definition.cardinality === 'singleton'
        ? `${target.type}:singleton`
        : `${target.type}:key:${encodeURIComponent(target.key!)}`;
    const index = state.instances.findIndex((instance) => instance.instanceId === instanceId);
    return index >= 0 ? { index } : { error: `Overlay instance '${instanceId}' was not found.` };
};

export const applyOverlayDisplayRequest = (
    current: OverlayPresentationState,
    request: OverlayDisplayRequest,
    now = Date.now(),
): OverlayTransitionResult => {
    if (!request || typeof request.presentationId !== 'string' || !request.presentationId.trim()) {
        return rejectRequest(current, request?.requestId || 'unknown', now, 'Overlay request requires a presentationId.');
    }
    if (!request || typeof request.requestId !== 'string' || !request.requestId.trim()) {
        return rejectRequest(current, 'unknown', now, 'Overlay request requires a requestId.', request.presentationId);
    }
    if (request.presentationId !== current.presentation?.presentationId) {
        return rejectRequest(
            current,
            request.requestId,
            now,
            `Overlay presentation '${request.presentationId}' is no longer active.`,
            request.presentationId,
        );
    }
    const command = request.command;
    if (!command || typeof command !== 'object') {
        return rejectRequest(current, request.requestId, now, 'Overlay request requires a command.');
    }
    const state: OverlayPresentationState = {
        ...current,
        instances: current.instances.map((instance) => ({ ...instance })),
    };
    const events: OverlayLifecycleEvent[] = [];
    const emit = makeEmitter(state, now, events);

    if (command.operation === 'upsert') {
        const definition = definitionFor(command.type);
        if (!definition) {
            return rejectRequest(current, request.requestId, now, `Unknown overlay display type '${command.type}'.`);
        }
        if (!isJsonSafe(command.snapshot) || !definition.validateSnapshot(command.snapshot)) {
            return rejectRequest(current, request.requestId, now, `Invalid snapshot for '${command.type}'.`);
        }
        if (command.options?.metadata !== undefined && !isJsonSafe(command.options.metadata)) {
            return rejectRequest(current, request.requestId, now, 'Overlay metadata must be JSON-safe.');
        }
        const identity = canonicalInstanceId(definition, request);
        if (!identity.instanceId) {
            return rejectRequest(current, request.requestId, now, identity.error || 'Invalid overlay identity.');
        }
        const existingIndex = state.instances.findIndex((instance) => instance.instanceId === identity.instanceId);
        if (existingIndex >= 0 && state.instances[existingIndex].type !== command.type) {
            return rejectRequest(current, request.requestId, now, 'The requested instanceId belongs to another display type.');
        }
        const snapshot = cloneJsonSnapshot(command.snapshot);
        const metadata = cloneJsonSnapshot(command.options?.metadata ?? {});
        if (existingIndex >= 0) {
            const existing = state.instances[existingIndex];
            const deadlines = lifecycleDeadlines(definition, snapshot, existing.policy, now);
            state.instances[existingIndex] = {
                ...existing,
                snapshot,
                metadata,
                updatedAt: now,
                revision: existing.revision + 1,
                ...deadlines,
            };
            emit(existing.instanceId, 'updated', {
                type: existing.type,
                policy: existing.policy,
                requestId: request.requestId,
            });
            if (state.enabled && !existing.shown) {
                state.instances[existingIndex].shown = true;
                emit(existing.instanceId, 'shown', { type: existing.type, policy: existing.policy });
            }
        } else {
            const deadlines = lifecycleDeadlines(definition, snapshot, definition.initialPolicy, now);
            const instance: OverlayDisplayInstance = {
                instanceId: identity.instanceId,
                type: command.type,
                key: identity.key,
                snapshot,
                metadata,
                policy: definition.initialPolicy,
                shown: state.enabled,
                createdAt: now,
                updatedAt: now,
                revision: 0,
                ...deadlines,
            };
            state.instances.push(instance);
            emit(instance.instanceId, 'accepted', {
                type: instance.type,
                policy: instance.policy,
                requestId: request.requestId,
            });
            if (state.enabled) emit(instance.instanceId, 'shown', { type: instance.type, policy: instance.policy });
        }
        return {
            state,
            events,
            acknowledgement: {
                presentationId: request.presentationId,
                requestId: request.requestId,
                accepted: true,
                instanceId: identity.instanceId,
            },
        };
    }

    if (command.operation !== 'set_policy' && command.operation !== 'exit') {
        return rejectRequest(current, request.requestId, now, 'Unknown overlay operation.');
    }
    const resolved = resolveTargetIndex(state, command.target);
    if (resolved.index === undefined) {
        return rejectRequest(current, request.requestId, now, resolved.error || 'Overlay target was not found.');
    }
    const instance = state.instances[resolved.index];
    const definition = definitionFor(instance.type)!;

    if (command.operation === 'exit') {
        state.instances.splice(resolved.index, 1);
        emit(instance.instanceId, 'exited', {
            type: instance.type,
            policy: instance.policy,
            reason: command.reason,
            requestId: request.requestId,
        });
        return {
            state,
            events,
            acknowledgement: {
                presentationId: request.presentationId,
                requestId: request.requestId,
                accepted: true,
                instanceId: instance.instanceId,
            },
        };
    }

    if (!definition.permittedTransitions[instance.policy].includes(command.policy)) {
        return rejectRequest(current, request.requestId, now, `Policy transition ${instance.policy} -> ${command.policy} is not permitted.`);
    }
    const deadlines = lifecycleDeadlines(definition, instance.snapshot, command.policy, now);
    state.instances[resolved.index] = { ...instance, policy: command.policy, updatedAt: now, ...deadlines };
    emit(instance.instanceId, 'policy_changed', {
        type: instance.type,
        policy: command.policy,
        requestId: request.requestId,
    });
    return {
        state,
        events,
        acknowledgement: {
            presentationId: request.presentationId,
            requestId: request.requestId,
            accepted: true,
            instanceId: instance.instanceId,
        },
    };
};

export const advanceOverlayTimers = (
    current: OverlayPresentationState,
    now = Date.now(),
): OverlayTransitionResult => {
    const state: OverlayPresentationState = {
        ...current,
        instances: current.instances.map((instance) => ({ ...instance })),
    };
    const events: OverlayLifecycleEvent[] = [];
    const emit = makeEmitter(state, now, events);
    state.instances = state.instances.filter((instance) => {
        if (instance.exitAt !== null && instance.exitAt <= now) {
            emit(instance.instanceId, 'exited', {
                type: instance.type,
                policy: instance.policy,
                reason: 'transient_complete',
            });
            return false;
        }
        if (!instance.folded && instance.foldAt !== null && instance.foldAt <= now) {
            instance.folded = true;
            instance.foldAt = null;
            emit(instance.instanceId, 'folded', { type: instance.type, policy: instance.policy });
        }
        return true;
    });
    return { state, events };
};

export const applyOverlayComponentEvent = (
    current: OverlayPresentationState,
    instanceId: string,
    event: OverlayComponentEvent,
    now = Date.now(),
): OverlayTransitionResult => {
    const state: OverlayPresentationState = {
        ...current,
        instances: current.instances.map((instance) => ({ ...instance })),
    };
    const index = state.instances.findIndex((instance) => instance.instanceId === instanceId);
    if (index < 0) return { state: current, events: [] };
    const instance = state.instances[index];
    const definition = definitionFor(instance.type)!;
    const directive = definition.lifecycleReducer?.(instance.snapshot, event);
    if (!directive) return { state: current, events: [] };
    state.instances[index] = {
        ...instance,
        exitAt: directive.exitAfterMs === undefined ? instance.exitAt : now + directive.exitAfterMs,
        foldAt: directive.foldAfterMs === undefined ? instance.foldAt : now + directive.foldAfterMs,
    };
    return { state, events: [] };
};

export const setOverlayEnabled = (
    current: OverlayPresentationState,
    enabled: boolean,
    now = Date.now(),
): OverlayTransitionResult => {
    const state: OverlayPresentationState = {
        ...current,
        enabled,
        instances: current.instances.map((instance) => ({ ...instance, shown: enabled })),
    };
    const events: OverlayLifecycleEvent[] = [];
    if (enabled) {
        const emit = makeEmitter(state, now, events);
        current.instances.forEach((instance) => {
            if (!instance.shown) emit(instance.instanceId, 'shown', { type: instance.type, policy: instance.policy });
        });
    }
    return { state, events };
};

export const manuallyDismissOverlayInstance = (
    current: OverlayPresentationState,
    instanceId: string,
    now = Date.now(),
): OverlayTransitionResult => {
    const instance = current.instances.find((candidate) => candidate.instanceId === instanceId);
    if (!instance) return { state: current, events: [] };
    const definition = definitionFor(instance.type)!;
    if (!definition.manualDismiss) return { state: current, events: [] };
    const state = {
        ...current,
        instances: current.instances.filter((candidate) => candidate.instanceId !== instanceId),
    };
    const events: OverlayLifecycleEvent[] = [];
    makeEmitter(state, now, events)(instance.instanceId, 'exited', {
        type: instance.type,
        policy: instance.policy,
        reason: 'manual_dismiss',
    });
    return { state, events };
};

export const toggleOverlayFold = (
    current: OverlayPresentationState,
    instanceId: string,
    now = Date.now(),
): OverlayTransitionResult => {
    const index = current.instances.findIndex((instance) => instance.instanceId === instanceId);
    if (index < 0 || current.instances[index].policy === 'pinned_top') return { state: current, events: [] };
    const state = { ...current, instances: current.instances.map((instance) => ({ ...instance })) };
    const instance = state.instances[index];
    instance.folded = !instance.folded;
    instance.foldAt = null;
    if (instance.folded && instance.exitAt === null) {
        const definition = definitionFor(instance.type)!;
        const directive = definition.lifecycleReducer?.(instance.snapshot, 'visual_complete');
        if (directive?.exitAfterMs !== undefined) instance.exitAt = now + directive.exitAfterMs;
    }
    const events: OverlayLifecycleEvent[] = [];
    const emit = makeEmitter(state, now, events);
    emit(instance.instanceId, instance.folded ? 'folded' : 'shown', {
        type: instance.type,
        policy: instance.policy,
    });
    return { state, events };
};

export const orderOverlayInstances = (
    instances: readonly OverlayDisplayInstance[],
): OverlayDisplayInstance[] => [...instances].sort((left, right) => {
    const leftPinned = left.policy === 'pinned_top';
    const rightPinned = right.policy === 'pinned_top';
    if (leftPinned !== rightPinned) return leftPinned ? -1 : 1;
    return right.updatedAt - left.updatedAt || left.instanceId.localeCompare(right.instanceId);
});

export const getNextOverlayDeadline = (
    instances: readonly OverlayDisplayInstance[],
): number | null => instances.reduce<number | null>((next, instance) => {
    const deadlines = [instance.exitAt, instance.foldAt].filter((value): value is number => value !== null);
    if (!deadlines.length) return next;
    const candidate = Math.min(...deadlines);
    return next === null ? candidate : Math.min(next, candidate);
}, null);

export const beginOverlayPresentation = (
    current: OverlayPresentationState,
    presentation: OverlayPresentationSession,
    now = Date.now(),
): OverlayTransitionResult => {
    if (current.presentation?.presentationId === presentation.presentationId) {
        return {
            state: {
                ...current,
                presentation: cloneJsonSnapshot(presentation),
            },
            events: [],
        };
    }

    const previousPresentationId = current.presentation?.presentationId;
    const state: OverlayPresentationState = {
        ...current,
        presentation: cloneJsonSnapshot(presentation),
        instances: [],
    };
    const events: OverlayLifecycleEvent[] = [];
    if (previousPresentationId) {
        const emit = makeEmitter(state, now, events, previousPresentationId);
        current.instances.forEach((instance) => emit(instance.instanceId, 'exited', {
            type: instance.type,
            policy: instance.policy,
            reason: 'replaced',
        }));
    }
    return { state, events };
};

export const endOverlayPresentation = (
    current: OverlayPresentationState,
    presentationId: string,
    now = Date.now(),
): OverlayTransitionResult => {
    if (current.presentation?.presentationId !== presentationId) {
        return { state: current, events: [] };
    }
    const state: OverlayPresentationState = {
        ...current,
        presentation: null,
        instances: [],
    };
    const events: OverlayLifecycleEvent[] = [];
    const emit = makeEmitter(state, now, events, presentationId);
    current.instances.forEach((instance) => emit(instance.instanceId, 'exited', {
        type: instance.type,
        policy: instance.policy,
        reason: 'session_ended',
    }));
    return { state, events };
};

export const exitAllOverlayInstances = (
    current: OverlayPresentationState,
    reason: OverlayExitReason,
    now = Date.now(),
): OverlayTransitionResult => {
    const state = { ...current, instances: [] };
    const events: OverlayLifecycleEvent[] = [];
    const emit = makeEmitter(state, now, events);
    current.instances.forEach((instance) => emit(instance.instanceId, 'exited', {
        type: instance.type,
        policy: instance.policy,
        reason,
    }));
    return { state, events };
};
