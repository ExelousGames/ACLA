import { DEFAULT_DRIVING_CONFIG, emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, UNLABELED_RECORDED_SAMPLING_CATEGORY,
    validateRecordedBatchLabelCount, type RecordedSamplingCategory, type RecordedSamplingWeights } from './driving-model-config';
import type { DrivingPolicy, PlannedActionTarget, RecordedPlanTarget } from './driving-policy';
import type { DrivingPolicyInput } from './driving-memory';

type ObservedAction = Omit<PlannedActionTarget, 'step'>;
type PendingPlan = { features: DrivingPolicyInput; appliedAt: number; nextStep: number;
    targets: PlannedActionTarget[]; pool: SamplePool };
type SamplePool = { plans: RecordedPlanTarget[]; seen: number; capacity: number };
type PoolEntry = [RecordedSamplingCategory, SamplePool];
export const RECORDED_PLAN_WINDOW = 256;

function labelQuotas(categories: RecordedSamplingCategory[], weights: RecordedSamplingWeights, slots: number) {
    const quotas = emptyRecordedSamplingWeights();
    let totalWeight = categories.reduce((sum, category) => sum + weights[category], 0);
    const byWeight = [...categories].sort((a, b) => weights[a] - weights[b]);
    // Reserve one for labels below one slot, then rescale the remaining weights.
    for (const category of byWeight) {
        const weight = weights[category];
        if (weight / totalWeight * slots >= 1) break;
        quotas[category] = 1;
        totalWeight -= weight;
        slots--;
    }
    for (const category of categories) {
        if (!quotas[category]) quotas[category] = Math.max(1, weights[category] / totalWeight * slots);
    }
    return quotas;
}

/** Align plan labels in time, then keep every eligible label represented in each optimizer batch. */
export class DrivingPlanTraining {
    private pending: PendingPlan[] = [];
    private pools = new Map<RecordedSamplingCategory, SamplePool>();
    private sampling?: { available: PoolEntry[]; quotas: RecordedSamplingWeights };
    private credit = emptyRecordedSamplingWeights();
    private counts = emptyRecordedSamplingWeights();
    private batches = 0;

    constructor(private weights: RecordedSamplingWeights = {
        ...emptyRecordedSamplingWeights(), [UNLABELED_RECORDED_SAMPLING_CATEGORY]: 100,
    },
        private batchSize = DEFAULT_DRIVING_CONFIG.recordedBatchSize,
        private batchCount = DEFAULT_DRIVING_CONFIG.recordedBatchCount,
        private random = Math.random) {
        const categories = RECORDED_SAMPLING_CATEGORIES.filter((category) => weights[category] > 0);
        const slots = Math.max(batchSize, categories.length);
        const quotas = labelQuotas(categories, weights, slots);
        for (const category of categories) {
            const capacity = Math.ceil(RECORDED_PLAN_WINDOW * quotas[category] / slots);
            this.pools.set(category, { plans: [], seen: 0, capacity });
        }
    }

    get sampled() { return { ...this.counts }; }
    get completedBatches() { return this.batches; }

    record(policy: DrivingPolicy, features: DrivingPolicyInput | undefined, target: ObservedAction,
        appliedAt: number, category: RecordedSamplingCategory = UNLABELED_RECORDED_SAMPLING_CATEGORY) {
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
        const pool = this.pools.get(category);
        if (features && pool) {
            const cameras = features.cameras.slice();
            // Recorded plans already own the consecutive history images used for training.
            if (features.recordedHistory) cameras.fill(null, 0, cameras.length - 1);
            this.pending.push({ features: { ...features, features: features.features.slice(), cameras },
                appliedAt, nextStep: 1, targets: [{ step: 0, ...target }], pool });
        }
        this.completePlans(policy.planTimes.length);
    }

    completeRange() { this.completePlans(0); }

    private completePlans(slots: number) {
        while (this.pending.length) {
            const { features, targets, nextStep, pool } = this.pending[0];
            if (nextStep < slots) break;
            // Reservoir sampling retains a cross-section even when a category is first seen late.
            const index = pool.seen < pool.capacity ? pool.seen : Math.floor(this.random() * (pool.seen + 1));
            if (index < pool.capacity) pool.plans[index] = { input: features, targets };
            pool.seen++;
            this.pending.shift();
        }
    }

    private prepareSampling() {
        const available = Array.from(this.pools).filter(([, pool]) => pool.plans.length > 0);
        validateRecordedBatchLabelCount(this.batchSize, available.length);
        for (const [, { plans }] of available) {
            // A short run should also sample across each category's full recording.
            for (let i = plans.length - 1; i > 0; i--) {
                const j = Math.floor(this.random() * (i + 1));
                [plans[i], plans[j]] = [plans[j], plans[i]];
            }
        }
        const quotas = labelQuotas(available.map(([category]) => category), this.weights, this.batchSize);
        return { available, quotas };
    }

    async finishAsync(policy: DrivingPolicy, active = () => true, maxBatches = Infinity) {
        if (!active() || this.batches >= this.batchCount) return;
        this.completeRange();
        const { available, quotas } = this.sampling ??= this.prepareSampling();
        if (!available.length) return;
        const batches = Math.min(this.batchCount - this.batches, maxBatches);
        for (let i = 0; i < batches && active(); i++) {
            const credit = { ...this.credit }, counts = { ...this.counts };
            const batch: RecordedPlanTarget[] = [];
            const draw = ([category, { plans }]: PoolEntry) => {
                batch.push(plans[counts[category] % plans.length]);
                counts[category]++;
            };
            for (const entry of available) {
                const [category] = entry;
                draw(entry);
                credit[category] += quotas[category] - 1;
            }
            // Allocate unreserved slots by weighted quota and carry unused fractions forward.
            while (batch.length < this.batchSize) {
                const selected = available.reduce((best, entry) => credit[entry[0]] > credit[best[0]] ? entry : best);
                draw(selected);
                credit[selected[0]]--;
            }
            if (!await policy.learnRecordedBatchAsync(batch, active)) break;
            this.credit = credit;
            this.counts = counts;
            this.batches++;
        }
    }
}
