import type { DrivingPolicyInput } from './driving-memory';
import type { DrivingRolloutInterval } from './driving-rollout';

export interface DrivingConsequenceBatch {
    sequence: DrivingRolloutInterval[];
    input: DrivingPolicyInput;
}

/** CPU collection continues while the consumer trains or infers. Taking a batch
 * transfers a complete time window without touching the next window. */
export class DrivingConsequenceBuffer {
    private pending: DrivingRolloutInterval[] = [];
    private input?: DrivingPolicyInput;
    private ready: DrivingConsequenceBatch[] = [];

    constructor(private seconds: number) {}

    get hasReady() { return this.ready.length > 0; }

    reset() {
        this.pending = [];
        this.input = undefined;
        this.ready = [];
    }

    add(interval: DrivingRolloutInterval, input: DrivingPolicyInput) {
        this.pending.push(interval);
        this.input = input;
        if (interval.endedAt - this.pending[0].startedAt < this.seconds * 1000) return;
        this.ready.push({ sequence: this.pending, input });
        this.pending = [];
        this.input = undefined;
        // Bound a stalled consumer's backlog without shortening training windows.
        if (this.ready.length > 2) this.ready.shift();
    }

    take() { return this.ready.shift(); }

    /** Close an ownership segment, preserving its final partial window. */
    finish() {
        const batches = this.ready;
        if (this.pending.length && this.input) batches.push({ sequence: this.pending, input: this.input });
        this.reset();
        return batches;
    }
}
