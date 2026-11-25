import { emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, UNLABELED_RECORDED_SAMPLING_CATEGORY,
    type RecordedSamplingCategory, type RecordedSamplingWeights } from './driving-model-config';
import type { DrivingPolicy, PlannedActionTarget, RecordedPlanTarget } from './driving-policy';
import type { DrivingPolicyInput } from './driving-memory';

type ObservedAction = Omit<PlannedActionTarget, 'step'>;
type PendingPlan = { features: DrivingPolicyInput; appliedAt: number; nextStep: number;
    targets: PlannedActionTarget[]; pool: SamplePool };
type SamplePool = { plans: RecordedPlanTarget[]; seen: number; capacity: number };
const BATCH_SIZE = 8;
export const RECORDED_PLAN_WINDOW = 256;

/** Align plan labels in time, then draw optimizer batches from fixed category proportions. */
export class DrivingPlanTraining {
    private pending: PendingPlan[] = [];
    private pools = new Map<RecordedSamplingCategory, SamplePool>();
    private readyDraws = 0;
    private credit = emptyRecordedSamplingWeights();
    private counts = emptyRecordedSamplingWeights();

    constructor(private weights: RecordedSamplingWeights = {
        ...emptyRecordedSamplingWeights(), [UNLABELED_RECORDED_SAMPLING_CATEGORY]: 100,
    },
        private random = Math.random) {
        const categories = RECORDED_SAMPLING_CATEGORIES.filter((category) => weights[category] > 0);
        const totalWeight = categories.reduce((sum, category) => sum + weights[category], 0);
        for (const category of categories) {
            const capacity = Math.ceil(RECORDED_PLAN_WINDOW * weights[category] / totalWeight);
            this.pools.set(category, { plans: [], seen: 0, capacity });
        }
    }

    get hasPending() { return this.pending.length > 0 || this.readyDraws > 0; }
    get sampled() { return { ...this.counts }; }

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
        const pool = this.pools.get(category);
        if (features && pool) {
            const cameras = features.cameras.slice();
            // Recorded plans already own the consecutive history images used for training.
            if (features.recordedHistory) cameras.fill(null, 0, cameras.length - 1);
            this.pending.push({ features: { ...features, features: features.features.slice(), cameras },
                appliedAt, nextStep: 1, targets: [{ step: 0, ...target }], pool });
        }
        this.completePlans(policy.planTimes.length);
        await this.submit(policy, false, active, RECORDED_PLAN_WINDOW);
    }

    private completePlans(slots: number) {
        while (this.pending.length) {
            const { features, targets, nextStep, pool } = this.pending[0];
            if (nextStep < slots) break;
            // Reservoir sampling retains a cross-section even when a category is first seen late.
            const index = pool.seen < pool.capacity ? pool.seen : Math.floor(this.random() * (pool.seen + 1));
            if (index < pool.capacity) pool.plans[index] = { input: features, targets };
            pool.seen++;
            this.readyDraws++;
            this.pending.shift();
        }
    }

    async finishAsync(policy: DrivingPolicy, active = () => true, final = true, maxDraws = Infinity) {
        if (!active()) return false;
        this.completePlans(0);
        return this.submit(policy, final, active, maxDraws);
    }

    private async submit(policy: DrivingPolicy, final: boolean, active: () => boolean, maxDraws: number) {
        if (!this.readyDraws || (!final && this.readyDraws < RECORDED_PLAN_WINDOW)) return false;
        const available = Array.from(this.pools).filter(([, pool]) => pool.plans.length > 0);
        if (!available.length || (!final && available.length < this.pools.size)) return false;
        const total = available.reduce((sum, [category]) => sum + this.weights[category], 0);
        const budget = Math.min(this.readyDraws, maxDraws);
        let remaining = budget;
        while (remaining >= (final ? 1 : BATCH_SIZE) && active()) {
            const credit = { ...this.credit }, counts = { ...this.counts };
            const batch: RecordedPlanTarget[] = [];
            const size = Math.min(remaining, BATCH_SIZE);
            for (let i = 0; i < size; i++) {
                // Carry fractional quotas across batches, including shares smaller than one slot.
                let selected = available[0];
                for (const entry of available) {
                    const [category] = entry;
                    credit[category] += this.weights[category];
                    if (credit[category] > credit[selected[0]]) selected = entry;
                }
                const [next, { plans }] = selected;
                credit[next] -= total;
                batch.push(plans[counts[next] % plans.length]);
                counts[next]++;
            }
            if (!await policy.learnRecordedBatchAsync(batch, active)) break;
            this.credit = credit;
            this.counts = counts;
            this.readyDraws -= size;
            remaining -= size;
        }
        return remaining < budget && this.readyDraws > 0;
    }
}
