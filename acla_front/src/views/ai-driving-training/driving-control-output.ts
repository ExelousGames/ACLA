import { DrivingAction, NEUTRAL } from './driving-observation';
import type { ControlSource } from './driving-control';
import type { VirtualController } from './virtual-controller';

export type OutputOwner = ControlSource | 'test';
export interface ControlLease {
    /** False means ownership was revoked before the command was acknowledged. */
    apply(action: DrivingAction): Promise<boolean>;
}

/** The only vJoy writer. Each handoff revokes the previous driver's lease. */
export class DrivingControlOutput {
    private revision = 0;
    private appliedOwner: OutputOwner | null = null;
    private pending: Promise<unknown> = Promise.resolve();

    constructor(private controller: VirtualController) {}

    acquire(owner: OutputOwner): ControlLease {
        const revision = ++this.revision;
        return { apply: (action) => {
            const snapshot = { ...action };
            return this.enqueue(async () => {
                if (revision !== this.revision) return false;
                // Release the outgoing driver's pedals and paddles before a handoff.
                if (this.appliedOwner !== null && this.appliedOwner !== owner) {
                    await this.controller.apply(NEUTRAL);
                    this.appliedOwner = null;
                    if (revision !== this.revision) return false;
                }
                await this.controller.apply(snapshot);
                this.appliedOwner = owner;
                return revision === this.revision;
            });
        } };
    }

    async release(): Promise<void> {
        this.revision++;
        await this.enqueue(async () => {
            await this.controller.apply(NEUTRAL);
            this.appliedOwner = null;
        });
    }

    private enqueue<T>(write: () => Promise<T>): Promise<T> {
        const result = this.pending.then(write);
        // A failed command must not prevent the queued neutral release.
        this.pending = result.catch(() => undefined);
        return result;
    }
}
