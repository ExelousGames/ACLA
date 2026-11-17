import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingPolicyInput } from './driving-memory';
import type { PlanStepCredit, PolicyDecision, ShiftAvailability } from './driving-policy';

export interface DrivingRolloutInterval {
    startedAt: number;
    endedAt: number;
    reward: number;
    /** Adjacent observed states, retained independently of the action's policy version. */
    input?: DrivingPolicyInput;
    next?: DrivingPolicyInput;
    nextShifts?: ShiftAvailability;
    terminal?: boolean;
    samples: { decision: PolicyDecision; credits: PlanStepCredit[]; rewardShare: number; startedAt?: number }[];
}

/** Delayed scheduling does not extend the SAC transition: each target retains
 * its adjacent observations and immediate reward. Future behavior-policy rewards
 * are learned through the critic's one-step bootstrap, never mixed into this target. */
export function drivingSequenceTargets(sequence: DrivingRolloutInterval[], _learning: DrivingModelConfig['learning']) {
    return sequence.flatMap((interval) => interval.samples.map(({ decision, credits }) => ({
        decision, credits, input: interval.input ?? decision.input, next: interval.next,
        nextShifts: interval.nextShifts, terminal: interval.terminal ?? false,
        // Every command is a proxy for this same complete observed transition.
        // Splitting its Bellman reward while retaining its full elapsed discount
        // would change the learned value merely by acknowledging controls more often.
        reward: interval.reward,
        elapsedSeconds: (interval.endedAt - interval.startedAt) / 1000,
    })));
}
