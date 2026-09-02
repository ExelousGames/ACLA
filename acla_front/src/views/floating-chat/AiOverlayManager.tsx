import React from 'react';
import type {
    AiToolComponentRef,
    AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    cloneJsonSnapshot,
    isAiOverlayComponentHandle,
    isJsonSafe,
    type AiOverlayComponentBehavior,
    type AiOverlayComponentHandle,
    type AiOverlayDisplayStatus,
    type AiOverlayPresentationCard,
    type AiOverlayPresentationSession,
    type AiOverlayPresentationSnapshot,
    type AiOverlayRendererEvent,
} from './ai-overlay-types';
import {
    overlaySessionClient,
    sendOverlayPresentation,
    subscribeOverlayRendererEvents,
} from './overlay-display-client';

type OverlaySource = {
    ref: AiToolComponentRef;
    handle: AiOverlayComponentHandle<any>;
    unsubscribe: () => void;
};

type ManagedOverlayCard = AiOverlayPresentationCard & {
    source: OverlaySource;
    behavior: AiOverlayComponentBehavior;
    requestedStatus: AiOverlayDisplayStatus;
    updatedAt: number;
    removeAt: number | null;
    foldAt: number | null;
    focusRequestOrder: number | null;
};

export interface AiOverlayManagerTransport {
    send(presentation: AiOverlayPresentationSnapshot): Promise<unknown> | unknown;
    now(): number;
    setTimer(callback: () => void, delayMs: number): ReturnType<typeof setTimeout>;
    clearTimer(timer: ReturnType<typeof setTimeout>): void;
}

const defaultTransport: AiOverlayManagerTransport = {
    send: sendOverlayPresentation,
    now: Date.now,
    setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
    clearTimer: (timer) => clearTimeout(timer),
};

const deadlineFrom = (duration: number | null | undefined, now: number): number | null => (
    typeof duration === 'number' && Number.isFinite(duration) && duration >= 0
        ? now + duration
        : null
);

export class AiOverlayManagerController {
    private readonly sources = new Map<AiToolComponentRef, OverlaySource>();
    private readonly cards = new Map<string, ManagedOverlayCard>();
    private presentation: AiOverlayPresentationSession | null = null;
    private presentationRevision = 0;
    private updateOrder = 0;
    private focusRequestOrder = 0;
    private timer: ReturnType<typeof setTimeout> | null = null;
    private disposed = false;

    constructor(private readonly transport: AiOverlayManagerTransport = defaultTransport) {}

    syncReferences(refs: readonly AiToolComponentRef[]): void {
        if (this.disposed) return;
        const retained = new Set(refs);
        Array.from(this.sources.entries()).forEach(([ref, source]) => {
            const current = ref.current;
            if (!retained.has(ref) || current !== source.handle) this.detachSource(ref);
        });
        refs.forEach((ref) => {
            if (this.sources.has(ref)) return;
            const handle = ref.current;
            if (!isAiOverlayComponentHandle(handle)) return;
            const source: OverlaySource = {
                ref,
                handle,
                unsubscribe: () => undefined,
            };
            source.unsubscribe = handle.subscribe((snapshot) => this.consume(source, snapshot));
            this.sources.set(ref, source);
            this.consume(source, handle.getSnapshot());
        });
    }

    setPresentation(presentation: AiOverlayPresentationSession | null): void {
        if (this.disposed) return;
        const changed = this.presentation?.presentationId !== presentation?.presentationId;
        this.presentation = presentation ? cloneJsonSnapshot(presentation) : null;
        if (!changed) {
            this.publish();
            return;
        }
        this.cards.forEach((card, componentName) => {
            const scopedPresentation = card.behavior.presentationId;
            if (scopedPresentation && scopedPresentation !== presentation?.presentationId) {
                this.cards.delete(componentName);
            }
        });
        this.sources.forEach((source) => this.consume(source, source.handle.getSnapshot(), false));
        this.changed();
    }

    handleRendererEvent(event: AiOverlayRendererEvent): void {
        if (this.disposed
            || !event.event.trim()
            || event.presentationId !== this.presentation?.presentationId) return;
        const card = this.cards.get(event.componentName);
        if (!card || card.revision !== event.revision) return;
        const currentHandle = card.source.ref.current;
        if (!isAiOverlayComponentHandle(currentHandle)
            || currentHandle.getComponentName() !== event.componentName) return;
        const directive = currentHandle.handleOverlayRendererEvent(event);
        if (!directive) return;
        const now = this.transport.now();
        if (directive.remove) {
            this.cards.delete(card.componentName);
        } else {
            if (directive.requestedStatus) {
                card.requestedStatus = directive.requestedStatus;
                if (directive.requestedStatus === 'focus') {
                    card.focusRequestOrder = ++this.focusRequestOrder;
                }
            }
            if (directive.removeAfterMs !== undefined) {
                card.removeAt = deadlineFrom(directive.removeAfterMs, now);
            }
            if (directive.foldAfterMs !== undefined) {
                card.foldAt = deadlineFrom(directive.foldAfterMs, now);
            }
        }
        this.changed();
    }

    getPresentationSnapshot(): AiOverlayPresentationSnapshot | null {
        if (!this.presentation) return null;
        const cards = this.orderedCards().map((card) => {
            const { source, behavior, requestedStatus, updatedAt, removeAt, foldAt, focusRequestOrder, ...serializable } = card;
            return cloneJsonSnapshot(serializable);
        });
        return {
            presentationId: this.presentation.presentationId,
            presentationRevision: this.presentationRevision,
            session: cloneJsonSnapshot(this.presentation),
            cards,
        };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        if (this.timer) this.transport.clearTimer(this.timer);
        this.timer = null;
        this.sources.forEach((source) => source.unsubscribe());
        this.sources.clear();
        this.cards.clear();
    }

    private consume(source: OverlaySource, suppliedSnapshot: unknown, notify = true): void {
        if (this.disposed || source.ref.current !== source.handle) return;
        const componentName = source.handle.getComponentName();
        const snapshot = suppliedSnapshot;
        const behavior = source.handle.getOverlayBehavior(snapshot);
        if (snapshot === null || behavior.remove) {
            const removed = this.cards.delete(componentName);
            if (removed && notify) this.changed();
            return;
        }
        if (behavior.presentationId
            && behavior.presentationId !== this.presentation?.presentationId) {
            const removed = this.cards.delete(componentName);
            if (removed && notify) this.changed();
            return;
        }
        if (!isJsonSafe(snapshot)) {
            throw new Error(`Overlay component '${componentName}' published a non-serializable snapshot.`);
        }
        const componentType = source.handle.getComponentType();
        if (!componentType.trim()) {
            throw new Error(`Overlay component '${componentName}' returned an empty componentType.`);
        }
        const now = this.transport.now();
        const existing = this.cards.get(componentName);
        const requestedStatus = behavior.requestedStatus;
        const card: ManagedOverlayCard = {
            componentName,
            componentType,
            snapshot: cloneJsonSnapshot(snapshot),
            revision: (existing?.revision ?? 0) + 1,
            metadata: cloneJsonSnapshot(source.handle.getOverlayMetadata()),
            status: requestedStatus,
            placement: behavior.placement,
            shellSlot: behavior.shellSlot ?? 'card',
            source,
            behavior,
            requestedStatus,
            updatedAt: ++this.updateOrder,
            removeAt: deadlineFrom(behavior.transientDurationMs, now),
            foldAt: deadlineFrom(behavior.foldAfterMs, now),
            focusRequestOrder: requestedStatus === 'focus'
                ? ++this.focusRequestOrder
                : existing?.focusRequestOrder ?? null,
        };
        this.cards.set(componentName, card);
        if (notify) this.changed();
    }

    private detachSource(ref: AiToolComponentRef): void {
        const source = this.sources.get(ref);
        if (!source) return;
        source.unsubscribe();
        this.sources.delete(ref);
        const componentName = source.handle.getComponentName();
        if (this.cards.get(componentName)?.source === source) {
            this.cards.delete(componentName);
            this.changed();
        }
    }

    private orderedCards(): ManagedOverlayCard[] {
        const cards = Array.from(this.cards.values());
        const focused = cards.reduce<ManagedOverlayCard | null>((active, card) => (
            card.requestedStatus === 'focus'
            && card.focusRequestOrder !== null
            && (!active || card.focusRequestOrder > active.focusRequestOrder!)
                ? card
                : active
        ), null);
        cards.forEach((card) => {
            if (card === focused) {
                card.status = 'focus';
            } else if (focused && card.shellSlot !== 'speech') {
                card.status = 'folded';
            } else {
                card.status = card.requestedStatus;
            }
        });
        return cards.sort((left, right) => {
            if (left.placement !== right.placement) return left.placement === 'pinned' ? -1 : 1;
            return right.updatedAt - left.updatedAt || left.componentName.localeCompare(right.componentName);
        });
    }

    private advanceTimers(): void {
        this.timer = null;
        const now = this.transport.now();
        let changed = false;
        this.cards.forEach((card, componentName) => {
            if (card.removeAt !== null && card.removeAt <= now) {
                this.cards.delete(componentName);
                changed = true;
                return;
            }
            if (card.foldAt !== null && card.foldAt <= now) {
                card.foldAt = null;
                card.requestedStatus = 'folded';
                changed = true;
            }
        });
        if (changed) this.publish();
        this.scheduleTimer();
    }

    private scheduleTimer(): void {
        if (this.timer) this.transport.clearTimer(this.timer);
        this.timer = null;
        const deadlines = Array.from(this.cards.values())
            .flatMap((card) => [card.removeAt, card.foldAt])
            .filter((value): value is number => value !== null);
        if (!deadlines.length) return;
        const delay = Math.max(0, Math.min(...deadlines) - this.transport.now());
        this.timer = this.transport.setTimer(() => this.advanceTimers(), delay);
    }

    private changed(): void {
        this.publish();
        this.scheduleTimer();
    }

    private publish(): void {
        if (!this.presentation) return;
        this.presentationRevision += 1;
        const snapshot = this.getPresentationSnapshot();
        if (!snapshot) return;
        void Promise.resolve(this.transport.send(snapshot)).catch((error) => {
            console.error('Failed to transmit overlay presentation.', error);
        });
    }
}

const AiOverlayManager: React.FC<{
    directory: AiToolComponentRefDirectory;
    directoryRevision: number;
}> = ({ directory, directoryRevision }) => {
    const controllerRef = React.useRef<AiOverlayManagerController | null>(null);
    if (!controllerRef.current) controllerRef.current = new AiOverlayManagerController();

    React.useLayoutEffect(() => {
        controllerRef.current?.syncReferences(directory.getComponentRefs());
    }, [directory, directoryRevision]);

    React.useEffect(() => {
        let controller = controllerRef.current;
        if (!controller) {
            controller = new AiOverlayManagerController();
            controllerRef.current = controller;
            controller.syncReferences(directory.getComponentRefs());
        }
        const activeController = controller;
        const unsubscribeSession = overlaySessionClient.subscribe((presentation) => {
            activeController.setPresentation(presentation);
        });
        const unsubscribeEvents = subscribeOverlayRendererEvents((event) => {
            activeController.handleRendererEvent(event);
        });
        return () => {
            unsubscribeSession();
            unsubscribeEvents();
            activeController.dispose();
            if (controllerRef.current === activeController) controllerRef.current = null;
        };
    }, [directory]);

    return null;
};

export default AiOverlayManager;
