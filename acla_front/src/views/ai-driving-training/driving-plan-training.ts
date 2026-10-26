import { DrivingAction } from './driving-observation';
import { DRIVING_PLAN_TIMES, DrivingPolicy, PlannedActionTarget, ShiftAvailability } from './driving-policy';

/** Match each plan's future slots to later acknowledged actions in the same episode.
 * A target may use the first sample up to 250 ms after its scheduled time. Missing
 * slots are skipped, never filled by repeating one action across several times. */
export class DrivingPlanTraining {
    private pending: { features: number[]; appliedAt: number; nextStep: number }[] = [];

    reset() { this.pending = []; }

    record(policy: DrivingPolicy, features: number[], action: DrivingAction, shifts: ShiftAvailability, appliedAt: number) {
        for (const plan of this.pending) {
            const targets: PlannedActionTarget[] = [];
            while (plan.nextStep < DRIVING_PLAN_TIMES.length) {
                const due = plan.appliedAt + DRIVING_PLAN_TIMES[plan.nextStep] * 1000;
                if (appliedAt < due) break;
                if (appliedAt - due <= 250) targets.push({ step: plan.nextStep, action, shifts });
                plan.nextStep++;
            }
            if (targets.length) policy.learnPlannedActions(plan.features, targets);
        }
        this.pending = this.pending.filter((plan) => plan.nextStep < DRIVING_PLAN_TIMES.length);
        this.pending.push({ features: features.slice(), appliedAt, nextStep: 1 });
    }
}
