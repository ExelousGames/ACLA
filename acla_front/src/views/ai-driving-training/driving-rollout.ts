import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingPolicyInput } from './driving-memory';
import type { PlanStepCredit, PolicyDecision, GearAvailability } from './driving-policy';

export interface DrivingRolloutInterval {
    startedAt: number;
    endedAt: number;
    reward: number;
    /** Adjacent observed states, retained independently of the action's policy version. */
    input?: DrivingPolicyInput;
    next?: DrivingPolicyInput;
    nextGears?: GearAvailability;
    terminal?: boolean;
    samples: { decision: PolicyDecision; credits: PlanStepCredit[]; rewardShare: number; startedAt?: number }[];
}

/** Sum the contiguous observed rewards for each action. The SAC critic adds
 * its entropy-regularized bootstrap at the end of the window. */
export function drivingSequenceTargets(sequence: DrivingRolloutInterval[], learning: DrivingModelConfig['learning'],
    finalInput?: DrivingPolicyInput) {
    const returns: { reward: number; end: DrivingRolloutInterval }[] = new Array(sequence.length);
    for (let index = sequence.length - 1; index >= 0; index--) {
        const interval = sequence[index];
        const following = !interval.terminal && interval.endedAt === sequence[index + 1]?.startedAt
            ? returns[index + 1] : undefined;
        const elapsedSeconds = (interval.endedAt - interval.startedAt) / 1000;
        const discount = Math.pow(learning.discount, elapsedSeconds / learning.discountIntervalSeconds);
        returns[index] = { reward: interval.reward + discount * (following?.reward ?? 0), end: following?.end ?? interval };
    }

    const last = sequence[sequence.length - 1];
    return sequence.flatMap((interval, index) => {
        const { reward, end } = returns[index];
        const next = end.next ?? (end === last ? finalInput : undefined);
        // Commands in the same observed interval share its full return. Splitting
        // it would make the learned value depend on controller polling frequency.
        return interval.samples.map(({ decision, credits }) => ({
            decision, credits, input: interval.input ?? decision.input, next,
            nextGears: end.nextGears, terminal: end.terminal ?? false,
            reward, elapsedSeconds: (end.endedAt - interval.startedAt) / 1000,
        }));
    });
}
