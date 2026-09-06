import type { MutableRefObject } from 'react';
import type {
    OperationComponentRefDirectory,
    NamedOperationComponentHandle,
} from 'contexts/OperationComponentRefContext';

export type OperationComponentSnapshotListener<TSnapshot> = (
    snapshot: TSnapshot,
) => void;

/**
 * Shared registration and snapshot lifecycle for operation components.
 *
 * Runners own their directory registration rather than relying on a mounted
 * React component. This lets a command create the runtime atomically and
 * lets completion remove it immediately, while the host remains a pure
 * observer of whichever runner is active.
 */
export abstract class OperationComponentBase<TSnapshot>
implements NamedOperationComponentHandle {
    private readonly componentRef: MutableRefObject<this | null> = { current: this };
    private componentRefDirectory: OperationComponentRefDirectory | null = null;
    private readonly snapshotListeners = new Set<OperationComponentSnapshotListener<TSnapshot>>();
    private snapshot: TSnapshot;
    private disposed = false;

    protected constructor(
        private readonly componentName: string,
        initialSnapshot: TSnapshot,
    ) {
        this.snapshot = initialSnapshot;
    }

    getComponentName(): string {
        return this.componentName;
    }

    getSnapshot(): TSnapshot {
        return this.cloneSnapshot(this.snapshot);
    }

    subscribe(listener: OperationComponentSnapshotListener<TSnapshot>): () => void {
        this.snapshotListeners.add(listener);
        return () => {
            this.snapshotListeners.delete(listener);
        };
    }

    addComponentRef(
        directory: OperationComponentRefDirectory,
    ): MutableRefObject<this | null> {
        this.componentRef.current = this;
        directory.registerComponentRef(this.componentRef);
        this.componentRefDirectory = directory;
        return this.componentRef;
    }

    deleteComponentRef(): boolean {
        const directory = this.componentRefDirectory;
        if (!directory) return false;
        const released = directory.unregisterComponentRef(this.componentRef);
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

export default OperationComponentBase;
