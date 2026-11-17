import type { DrivingPolicyInput } from './driving-memory';
import type { DrivingRolloutInterval } from './driving-rollout';

export interface DrivingConsequenceBatch {
    sequence: DrivingRolloutInterval[];
    input: DrivingPolicyInput;
}

/** Delay each acknowledged action's learning by its rolling consequence horizon.
 * Retain adjacent transition states while later intervals mature the queue. */
export class DrivingConsequenceBuffer {
    private pending: { interval: DrivingRolloutInterval; input: DrivingPolicyInput }[] = [];
    private ready: DrivingConsequenceBatch[] = [];

    constructor(private seconds: number) {}

    get hasReady() { return this.ready.length > 0; }

    reset() {
        this.pending = [];
        this.ready = [];
    }

    add(interval: DrivingRolloutInterval, input: DrivingPolicyInput) {
        this.pending.push({ interval: { ...interval, samples: interval.samples.slice() }, input });
        const horizon = this.seconds * 1000;
        this.pending.forEach((origin, index) => {
            const groups = new Map<number, DrivingRolloutInterval['samples']>();
            const remaining: DrivingRolloutInterval['samples'] = [];
            for (const sample of origin.interval.samples) {
                const deadline = (sample.startedAt ?? origin.interval.startedAt) + horizon;
                if (interval.endedAt < deadline) { remaining.push(sample); continue; }
                let end = this.pending.length - 1;
                while (end > index && this.pending[end].interval.endedAt > deadline) end--;
                // One observed interval is the minimum measurable return when
                // telemetry is slower than the configured horizon.
                groups.set(end, [...(groups.get(end) ?? []), sample]);
            }
            groups.forEach((samples, end) => {
                const endpoint = this.pending[end], previous = this.ready[this.ready.length - 1];
                // Actions sharing a maturity endpoint can be submitted together.
                const matching = previous && previous.input === endpoint.input
                    && previous.sequence[previous.sequence.length - 1].endedAt === endpoint.interval.endedAt
                    ? previous.sequence.findIndex((sample) => sample.startedAt === origin.interval.startedAt) : -1;
                if (previous && matching >= 0) previous.sequence[matching] = {
                    ...origin.interval, samples: [...previous.sequence[matching].samples, ...samples],
                };
                else this.ready.push({
                    sequence: this.pending.slice(index, end + 1).map(({ interval: item }, i) => ({ ...item, samples: i ? [] : samples })),
                    input: endpoint.input,
                });
            });
            origin.interval = { ...origin.interval, samples: remaining };
        });
        while (this.pending.length && !this.pending[0].interval.samples.length) this.pending.shift();
        // Keep at most one extra horizon of matured work if learning falls behind.
        while (this.ready.length && (this.ready.length > 256
            || this.ready[0].sequence[this.ready[0].sequence.length - 1].endedAt < interval.endedAt - horizon)) this.ready.shift();
    }

    take() { return this.ready.shift(); }

    /** Close an ownership segment, scoring its still-untrained partial suffix. */
    finish() {
        const batches = this.ready;
        if (this.pending.length) batches.push({
            sequence: this.pending.map(({ interval }) => interval),
            input: this.pending[this.pending.length - 1].input,
        });
        this.reset();
        return batches;
    }
}
