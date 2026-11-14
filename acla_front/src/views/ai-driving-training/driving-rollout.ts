import type { DrivingModelConfig } from './driving-model-config';
import type { PlanStepCredit, PolicyDecision } from './driving-policy';

export interface DrivingRolloutInterval {
    startedAt: number;
    endedAt: number;
    reward: number;
    samples: { decision: PolicyDecision; credits: PlanStepCredit[]; rewardShare: number }[];
}

/** Discount by elapsed time, not frame count. The GPU adds the frozen endpoint
 * value using elapsedSeconds; a truncated sequence is not an episode terminal. */
export function drivingSequenceTargets(sequence: DrivingRolloutInterval[], learning: DrivingModelConfig['learning']) {
    let laterRewards = 0;
    const end = sequence[sequence.length - 1]?.endedAt;
    const targets = sequence.map(() => [] as {
        decision: PolicyDecision; credits: PlanStepCredit[]; reward: number; elapsedSeconds: number;
    }[]);
    for (let i = sequence.length - 1; i >= 0; i--) {
        const interval = sequence[i];
        const discount = Math.pow(learning.discount, (interval.endedAt - interval.startedAt) / 1000 / learning.discountIntervalSeconds);
        const future = discount * laterRewards;
        targets[i] = interval.samples.map(({ decision, credits, rewardShare }) => ({
            decision, credits, reward: interval.reward * rewardShare + future,
            elapsedSeconds: (end - interval.startedAt) / 1000,
        }));
        laterRewards = interval.reward + future;
    }
    return targets.flat();
}
