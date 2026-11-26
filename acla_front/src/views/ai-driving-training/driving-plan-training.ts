import { emptyRecordedSamplingWeights, UNLABELED_RECORDED_SAMPLING_CATEGORY,
    type RecordedSamplingCategory, type RecordedSamplingWeights } from './driving-model-config';
import type { DrivingPolicy, PlannedActionTarget, RecordedPlanTarget } from './driving-policy';
import type { DrivingPolicyInput } from './driving-memory';

type ObservedAction = Omit<PlannedActionTarget, 'step'>;
type PendingPlan = { features: DrivingPolicyInput; appliedAt: number; nextStep: number;
    targets: PlannedActionTarget[]; category: RecordedSamplingCategory };
type TrainingPlan = RecordedPlanTarget & { category: RecordedSamplingCategory };
const BATCH_SIZE = 8;
export const RECORDED_PLAN_WINDOW = 256;

function shuffle<T>(items: T[], random: () => number) {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
}

/** Pick from each shuffled label pool without replacement, then shuffle the selected batch. */
function prepareBatches(plans: TrainingPlan[], weights: RecordedSamplingWeights, random: () => number) {
    const byCategory = new Map<RecordedSamplingCategory, TrainingPlan[]>();
    for (const plan of plans) {
        const pool = byCategory.get(plan.category);
        if (pool) pool.push(plan);
        else byCategory.set(plan.category, [plan]);
    }
    const pools = Array.from(byCategory, ([category, candidates]) => {
        shuffle(candidates, random);
        return { candidates, weight: weights[category], credit: 0 };
    });

    const batches: TrainingPlan[][] = [];
    for (let remaining = plans.length; remaining > 0;) {
        const batch: TrainingPlan[] = [];
        const size = Math.min(remaining, BATCH_SIZE);
        for (let i = 0; i < size; i++) {
            // Carry fractional quotas across batches, and redistribute slots as pools run out.
            let selected = pools[0], totalWeight = 0;
            for (const pool of pools) {
                totalWeight += pool.weight;
                pool.credit += pool.weight;
                if (pool.credit > selected.credit) selected = pool;
            }
            selected.credit -= totalWeight;
            batch.push(selected.candidates.pop()!);
            if (!selected.candidates.length) pools.splice(pools.indexOf(selected), 1);
        }
        shuffle(batch, random);
        batches.push(batch);
        remaining -= size;
    }
    return batches;
}

/** Align targets in time before shuffling; keep only one training window of decoded inputs ready. */
export class DrivingPlanTraining {
    private pending: PendingPlan[] = [];
    private ready: TrainingPlan[] = [];
    private batches: TrainingPlan[][] = [];
    private readonly counts = emptyRecordedSamplingWeights();

    constructor(private readonly weights: RecordedSamplingWeights = {
        ...emptyRecordedSamplingWeights(), [UNLABELED_RECORDED_SAMPLING_CATEGORY]: 100,
    }, private readonly random = Math.random) {}

    get hasPending() { return Boolean(this.pending.length || this.ready.length || this.batches.length); }
    get trained() { return { ...this.counts }; }

    async recordAsync(policy: DrivingPolicy, features: DrivingPolicyInput | undefined, target: ObservedAction,
        appliedAt: number, active = () => true, category: RecordedSamplingCategory = UNLABELED_RECORDED_SAMPLING_CATEGORY) {
        if (!active()) return;
        for (const plan of this.pending) {
            while (plan.nextStep < policy.planTimes.length) {
                const due = plan.appliedAt + policy.planTimes[plan.nextStep] * 1000;
                const delay = appliedAt - due;
                // Frame and slot timestamps reach the same time through different arithmetic.
                const roundoff = 4 * Number.EPSILON * Math.max(Math.abs(appliedAt), Math.abs(due));
                if (delay < -roundoff) break;
                if (delay <= policy.configuration.learning.planTargetToleranceMs + roundoff) {
                    plan.targets.push({ step: plan.nextStep, ...target });
                }
                plan.nextStep++;
            }
        }
        if (features && this.weights[category] > 0) {
            const cameras = features.cameras.slice();
            // Recorded plans already own the consecutive history images used for training.
            if (features.recordedHistory) cameras.fill(null, 0, cameras.length - 1);
            this.pending.push({ features: { ...features, features: features.features.slice(), cameras },
                appliedAt, nextStep: 1, targets: [{ step: 0, ...target }], category });
        }
        this.completePlans(policy.planTimes.length);
        await this.trainWindow(policy, active, false);
    }

    private completePlans(slots: number) {
        while (this.pending.length) {
            const { features, targets, nextStep, category } = this.pending[0];
            if (nextStep < slots) break;
            this.ready.push({ input: features, targets, category });
            this.pending.shift();
        }
    }

    /** Close a consecutive range. Return whether another window is ready to submit. */
    async finishAsync(policy: DrivingPolicy, active = () => true, final = true) {
        if (!active()) return false;
        this.completePlans(0);
        return this.trainWindow(policy, active, final);
    }

    private windowReady(final: boolean) {
        return final ? this.ready.length > 0 : this.ready.length >= RECORDED_PLAN_WINDOW;
    }

    private async trainWindow(policy: DrivingPolicy, active: () => boolean, final: boolean) {
        if (!this.batches.length && active() && this.windowReady(final)) {
            this.batches = prepareBatches(this.ready.splice(0, RECORDED_PLAN_WINDOW), this.weights, this.random);
        }
        while (this.batches.length && active()) {
            const batch = this.batches[0];
            if (!await policy.learnRecordedBatchAsync(batch, active)) return false;
            // A cancelled optimizer batch is rolled back; leave those examples queued for a graceful stop.
            this.batches.shift();
            for (const { category } of batch) this.counts[category]++;
        }
        return this.windowReady(final);
    }
}
