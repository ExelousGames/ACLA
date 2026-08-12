import type { MutableRefObject } from 'react';
import type {
    AiToolComponentRefDirectory,
    AiToolComponentRefOwner,
    NamedAiToolComponentHandle,
} from 'contexts/AiToolComponentRefContext';

export type AiToolComponentSnapshotListener<TSnapshot> = (
    snapshot: TSnapshot,
) => void;

/**
 * Shared lifecycle for the mutually-exclusive AI workflow runners.
 *
 * Runners own their directory registration rather than relying on a mounted
 * React component. This lets a tool command create the runtime atomically and
 * lets completion remove it immediately, while the host remains a pure
 * observer of whichever runner is active.
 */
export abstract class AiToolComponentBase<TSnapshot>
implements NamedAiToolComponentHandle {
    private readonly componentRefOwner: AiToolComponentRefOwner;
    private componentRefDirectory: AiToolComponentRefDirectory | null = null;
    private readonly snapshotListeners = new Set<AiToolComponentSnapshotListener<TSnapshot>>();
    private snapshot: TSnapshot;
    private disposed = false;

    protected constructor(
        private readonly componentName: string,
        initialSnapshot: TSnapshot,
    ) {
        this.snapshot = initialSnapshot;
        this.componentRefOwner = Symbol(`ai-tool-component:${componentName}`);
    }

    getComponentName(): string {
        return this.componentName;
    }

    getSnapshot(): TSnapshot {
        return this.cloneSnapshot(this.snapshot);
    }

    subscribe(listener: AiToolComponentSnapshotListener<TSnapshot>): () => void {
        this.snapshotListeners.add(listener);
        return () => {
            this.snapshotListeners.delete(listener);
        };
    }

    addComponentRef(
        directory: AiToolComponentRefDirectory,
    ): MutableRefObject<this | null> {
        const ref = directory.createComponentRef(
            this.componentName,
            this.componentRefOwner,
            this,
        ) as MutableRefObject<this | null>;
        this.componentRefDirectory = directory;
        return ref;
    }

    deleteComponentRef(): boolean {
        const directory = this.componentRefDirectory;
        if (!directory) return false;
        const released = directory.releaseComponentRef(
            this.componentName,
            this.componentRefOwner,
        );
        this.componentRefDirectory = null;
        return released;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.onDispose();
        this.deleteComponentRef();
        this.snapshotListeners.clear();
    }

    protected publishSnapshot(snapshot: TSnapshot): void {
        if (this.disposed) return;
        this.snapshot = this.cloneSnapshot(snapshot);
        const published = this.getSnapshot();
        this.snapshotListeners.forEach((listener) => listener(published));
    }

    protected cloneSnapshot(snapshot: TSnapshot): TSnapshot {
        return snapshot;
    }

    protected isDisposed(): boolean {
        return this.disposed;
    }

    protected onDispose(): void {
        // Runners override this to abort their own work.
    }
}

export default AiToolComponentBase;
