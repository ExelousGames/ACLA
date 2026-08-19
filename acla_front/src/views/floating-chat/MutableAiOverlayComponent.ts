import type {
    AiOverlayComponentBehavior,
    AiOverlayComponentHandle,
    AiOverlayRendererEvent,
    AiOverlayRendererEventDirective,
    AiOverlayShellMetadata,
    AiOverlaySnapshotListener,
} from './ai-overlay-types';
import { cloneJsonSnapshot } from './ai-overlay-types';

export interface MutableAiOverlayPublication {
    metadata?: AiOverlayShellMetadata;
    presentationId?: string;
    requestedStatus?: AiOverlayComponentBehavior['requestedStatus'];
}

export class MutableAiOverlayComponent<TSnapshot>
implements AiOverlayComponentHandle {
    private snapshot: TSnapshot | null = null;
    private metadata: AiOverlayShellMetadata = {};
    private presentationId: string | undefined;
    private requestedStatus: AiOverlayComponentBehavior['requestedStatus'] | undefined;
    private readonly listeners = new Set<AiOverlaySnapshotListener<TSnapshot | null>>();

    constructor(
        private readonly componentName: string,
        private readonly componentType: string,
        private readonly behavior: (
            snapshot: TSnapshot | null,
            publication: MutableAiOverlayPublication,
        ) => AiOverlayComponentBehavior,
        private readonly rendererEventHandler: (
            event: AiOverlayRendererEvent,
        ) => AiOverlayRendererEventDirective | void = () => undefined,
    ) {}

    getComponentName(): string {
        return this.componentName;
    }

    getComponentType(): string {
        return this.componentType;
    }

    getSnapshot(): TSnapshot | null {
        return this.snapshot === null ? null : cloneJsonSnapshot(this.snapshot);
    }

    getOverlayMetadata(): AiOverlayShellMetadata {
        return cloneJsonSnapshot(this.metadata);
    }

    getOverlayBehavior(snapshot: TSnapshot | null): AiOverlayComponentBehavior {
        return this.behavior(snapshot, {
            metadata: this.metadata,
            presentationId: this.presentationId,
            requestedStatus: this.requestedStatus,
        });
    }

    handleOverlayRendererEvent(
        event: AiOverlayRendererEvent,
    ): AiOverlayRendererEventDirective | void {
        return this.rendererEventHandler(event);
    }

    subscribe(listener: AiOverlaySnapshotListener<TSnapshot | null>): () => void {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    publish(snapshot: TSnapshot, publication: MutableAiOverlayPublication = {}): void {
        this.snapshot = cloneJsonSnapshot(snapshot);
        this.metadata = cloneJsonSnapshot(publication.metadata ?? {});
        this.presentationId = publication.presentationId;
        this.requestedStatus = publication.requestedStatus;
        const next = this.getSnapshot();
        this.listeners.forEach((listener) => listener(next));
    }

    clear(): void {
        if (this.snapshot === null) return;
        this.snapshot = null;
        this.listeners.forEach((listener) => listener(null));
    }
}
