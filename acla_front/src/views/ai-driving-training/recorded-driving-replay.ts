import { DrivingMemory, DrivingPolicyInput } from './driving-memory';
import { DrivingObservation, encodeDrivingAction } from './driving-observation';
import { DrivingPlanTraining, RECORDED_PLAN_WINDOW } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';
import type { DrivingModelConfig } from './driving-model-config';
import type { RecordedDrivingBalance } from './recorded-driving-balance';
import type { RecordedObservationResult } from './recorded-driving-session';

export interface RecordedReplayProgress {
    samples: number;
    skipped: number;
    /** Mean absolute error of the signed pedal command (0–2). */
    pedalError: number;
    steeringError: number;
    lastError?: string;
}

export function isConsecutiveRecordedObservation(previous: DrivingObservation, current: DrivingObservation,
    { learning, runtime }: DrivingModelConfig): boolean {
    const maxTransitionMs = runtime.maxTransitionSeconds * 1000;
    const maxCaptureGapMs = Math.min(maxTransitionMs, Math.max(100, runtime.controlIntervalMs) + learning.planTargetToleranceMs);
    const elapsed = current.capturedAt - previous.capturedAt;
    const telemetryGap = current.measuredAt - previous.measuredAt;
    const roundoff = 4 * Number.EPSILON * Math.max(Math.abs(current.capturedAt), Math.abs(previous.capturedAt));
    const from = previous.normalizedPosition, to = current.normalizedPosition;
    const movement = from !== null && to !== null ? Math.abs(to - from) : undefined;
    return elapsed > 0 && elapsed <= maxCaptureGapMs + roundoff && telemetryGap > 0 && telemetryGap <= maxTransitionMs
        && current.measuredAt > previous.capturedAt && current.session === previous.session && current.reference === previous.reference
        && (movement === undefined || Math.min(movement, 1 - movement) <= runtime.maxPositionJump);
}

/** Offline demonstration learning/evaluation. This path has no controller dependency. */
export class RecordedDrivingReplay {
    readonly progress: RecordedReplayProgress = { samples: 0, skipped: 0, pedalError: 0, steeringError: 0 };
    private memory: DrivingMemory;
    private plans: DrivingPlanTraining;
    private previous?: { observation: DrivingObservation; input: DrivingPolicyInput; shiftGear: number; clipLabel?: string };

    constructor(private policy: DrivingPolicy, private train: boolean, private balance?: RecordedDrivingBalance) {
        this.memory = new DrivingMemory(policy.configuration, train);
        this.plans = new DrivingPlanTraining(balance?.weights);
    }

    get sampled() { return this.plans.sampled; }

    async startRange(active = () => true) {
        await this.finish(active, false);
        if (!active()) return;
        this.memory.reset();
        this.previous = undefined;
    }

    async finish(active = () => true, final = true) {
        while (this.plans.hasPending && active()) {
            const result = { hasMoreDraws: false };
            await this.policy.trainBatchAsync(async () => {
                result.hasMoreDraws = await this.plans.finishAsync(this.policy, active, final, RECORDED_PLAN_WINDOW);
            });
            if (!final || !result.hasMoreDraws) break;
        }
    }

    async consume(result: RecordedObservationResult, active = () => true, clipLabel?: string): Promise<void> {
        if (!active()) return;
        const observation = result.observation;
        if (!observation || !this.policy.matches(observation)) {
            this.progress.skipped++;
            this.progress.lastError = result.error || 'Recorded inputs do not match this model.';
            await this.startRange(active);
            return;
        }
        if (this.previous && !isConsecutiveRecordedObservation(this.previous.observation, observation, this.policy.configuration)) {
            await this.startRange(active);
            if (!active()) return;
        }
        const input = this.memory.encode(observation);
        const pending = this.previous;
        const shiftGear = result.shiftGear ?? observation.gear;
        this.previous = { observation, input, shiftGear, clipLabel };
        if (!pending) return;

        // Telemetry records held axes, not paddle presses. A single subsequent gear
        // change supplies the shift target; ambiguous multi-gear jumps are masked.
        const gearDelta = shiftGear - pending.shiftGear;
        const knownShift = Number.isInteger(gearDelta) && Math.abs(gearDelta) <= 1;
        const shifts = { up: knownShift, down: knownShift };
        // Keep the earlier observation as the plan origin and supervise it with
        // the next sample's axes, which were unavailable at that origin.
        const action = encodeDrivingAction(observation.controls);
        if (this.train) {
            const target = { ...action, shiftUp: knownShift && gearDelta === 1, shiftDown: knownShift && gearDelta === -1 };
            await this.policy.trainBatchAsync(async () => {
                if (!active()) return;
                this.policy.observeDemonstration(pending.input);
                if (active()) await this.plans.recordAsync(this.policy, pending.input, { action: target, shifts },
                    pending.observation.capturedAt, active, this.balance?.select(pending.clipLabel));
            });
        } else {
            const decision = await this.policy.decideAsync(pending.input, false, shifts);
            try {
                if (!active()) return;
                const predicted = decision.plan[0].action;
                const count = this.progress.samples + 1;
                const pedalError = Math.abs(predicted.pedal - action.pedal);
                this.progress.pedalError += (pedalError - this.progress.pedalError) / count;
                this.progress.steeringError += (Math.abs(predicted.steering - action.steering) - this.progress.steeringError) / count;
            } finally { this.policy.releaseDecision(decision); }
        }
        if (active()) this.progress.samples++;
    }
}
