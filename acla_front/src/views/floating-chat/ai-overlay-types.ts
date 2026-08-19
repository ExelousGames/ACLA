import type React from 'react';

export const OVERLAY_HOLD_MS = 3_800;
export const OVERLAY_COMPARISON_COMPLETION_PAUSE_MS = 800;

export type AiOverlayPlacement = 'pinned' | 'flow';
export type AiOverlayDisplayStatus = 'expanded' | 'folded' | 'full_size';
export type AiOverlayShellSlot = 'speech' | 'card';

export interface AiOverlayShellMetadata {
    name?: string;
    emotion?: string;
    agentTags?: string[];
}

export type AiOverlaySessionMode =
    | 'front_desk'
    | 'live'
    | 'recorded'
    | 'user_summary'
    | 'agent';

export interface AiOverlaySessionDescriptor {
    aiSessionId: string;
    mode: AiOverlaySessionMode;
    displayIdentity: AiOverlayShellMetadata;
}

export interface AiOverlayPresentationSession extends AiOverlaySessionDescriptor {
    presentationId: string;
}

export interface AiOverlayComponentBehavior {
    placement: AiOverlayPlacement;
    requestedStatus: AiOverlayDisplayStatus;
    shellSlot?: AiOverlayShellSlot;
    transientDurationMs?: number | null;
    foldAfterMs?: number | null;
    remove?: boolean;
    /** Restricts an ephemeral source to the presentation that created it. */
    presentationId?: string;
}

export interface AiOverlayRendererEvent {
    presentationId: string;
    componentName: string;
    revision: number;
    event: string;
}

export interface AiOverlayRendererEventDirective {
    remove?: boolean;
    removeAfterMs?: number | null;
    foldAfterMs?: number | null;
    requestedStatus?: AiOverlayDisplayStatus;
}

export type AiOverlaySnapshotListener<TSnapshot = unknown> = (snapshot: TSnapshot) => void;

export interface AiOverlayComponentHandle<TSnapshot = unknown> {
    getComponentName(): string;
    getComponentType(): string;
    getSnapshot(): TSnapshot;
    subscribe(listener: AiOverlaySnapshotListener<TSnapshot>): () => void;
    getOverlayBehavior(snapshot: TSnapshot): AiOverlayComponentBehavior;
    getOverlayMetadata(): AiOverlayShellMetadata;
    handleOverlayRendererEvent(
        event: AiOverlayRendererEvent,
    ): AiOverlayRendererEventDirective | void;
}

export interface AiOverlayPresentationCard {
    componentName: string;
    componentType: string;
    snapshot: unknown;
    revision: number;
    metadata: AiOverlayShellMetadata;
    status: AiOverlayDisplayStatus;
    placement: AiOverlayPlacement;
    shellSlot: AiOverlayShellSlot;
}

export interface AiOverlayPresentationSnapshot {
    presentationId: string;
    presentationRevision: number;
    session: AiOverlayPresentationSession;
    cards: AiOverlayPresentationCard[];
}

export interface AiOverlayPresentationAcknowledgement {
    presentationId: string;
    presentationRevision: number;
    accepted: boolean;
    error?: string;
}

export interface AiOverlayDimensions {
    width: number;
    height: number;
}

export interface AiOverlayRenderContext {
    componentName: string;
    revision: number;
    emitRendererEvent(event: string): void;
}

export interface AiOverlayRenderer<TSnapshot = unknown> {
    componentType: string;
    validateSnapshot(snapshot: unknown): snapshot is TSnapshot;
    renderOverlay(
        snapshot: TSnapshot,
        status: AiOverlayDisplayStatus,
        context: AiOverlayRenderContext,
    ): React.ReactNode;
    dimensions: Partial<Record<AiOverlayDisplayStatus, AiOverlayDimensions>>;
}

export const isJsonSafe = (value: unknown, seen = new Set<unknown>()): boolean => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (Array.isArray(value)) {
        if (seen.has(value)) return false;
        seen.add(value);
        const valid = value.every((item) => isJsonSafe(item, seen));
        seen.delete(value);
        return valid;
    }
    if (typeof value !== 'object' || value === undefined) return false;
    if (Object.getPrototypeOf(value) !== Object.prototype) return false;
    if (seen.has(value)) return false;
    seen.add(value);
    const valid = Object.values(value as Record<string, unknown>)
        .every((item) => isJsonSafe(item, seen));
    seen.delete(value);
    return valid;
};

export const cloneJsonSnapshot = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export const isAiOverlayComponentHandle = (
    value: unknown,
): value is AiOverlayComponentHandle<any> => {
    if (!value || typeof value !== 'object') return false;
    const handle = value as Partial<AiOverlayComponentHandle>;
    return typeof handle.getComponentName === 'function'
        && typeof handle.getComponentType === 'function'
        && typeof handle.getSnapshot === 'function'
        && typeof handle.subscribe === 'function'
        && typeof handle.getOverlayBehavior === 'function'
        && typeof handle.getOverlayMetadata === 'function'
        && typeof handle.handleOverlayRendererEvent === 'function';
};
