import { DrivingAction } from './driving-observation';
import { DrivingPolicy, PlannedActionTarget, ShiftAvailability } from './driving-policy';
import type { DrivingPolicyInput } from './driving-memory';

/** Match each plan's future slots to later acknowledged actions in the same episode.
 * A target uses the first sample within the model's saved timing tolerance. Missing
 * slots are skipped, never filled by repeating one action across several times. */
export class DrivingPlanTraining {
    private pending: { features: DrivingPolicyInput; appliedAt: number; nextStep: number }[] = [];

    reset() { this.pending = []; }

    async recordAsync(policy: DrivingPolicy, features: DrivingPolicyInput, action: DrivingAction, shifts: ShiftAvailability,
        appliedAt: number, active = () => true, onReplaySubmitted?: () => void) {
        for (const sample of this.collect(policy, features, action, shifts, appliedAt)) {
            if (!active()) return;
            await policy.learnPlannedActionsAsync(sample.features, sample.targets);
            onReplaySubmitted?.();
        }
    }

    private collect(policy: DrivingPolicy, features: DrivingPolicyInput, action: DrivingAction, shifts: ShiftAvailability, appliedAt: number) {
        const samples: { features: DrivingPolicyInput; targets: PlannedActionTarget[] }[] = [];
        for (const plan of this.pending) {
            const targets: PlannedActionTarget[] = [];
            while (plan.nextStep < policy.planTimes.length) {
                const due = plan.appliedAt + policy.planTimes[plan.nextStep] * 1000;
                if (appliedAt < due) break;
                if (appliedAt - due <= policy.configuration.learning.planTargetToleranceMs) targets.push({ step: plan.nextStep, action, shifts });
                plan.nextStep++;
            }
            if (targets.length) samples.push({ features: plan.features, targets });
        }
        this.pending = this.pending.filter((plan) => plan.nextStep < policy.planTimes.length);
        this.pending.push({ features: { features: features.features.slice(), cameras: features.cameras.slice() }, appliedAt, nextStep: 1 });
        return samples;
    }
}
