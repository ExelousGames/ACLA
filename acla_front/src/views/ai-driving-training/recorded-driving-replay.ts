import { DrivingMemory, DrivingPolicyInput } from './driving-memory';
import { DrivingObservation, encodeDrivingAction } from './driving-observation';
import { DrivingPlanTraining, RECORDED_PLAN_WINDOW } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';
import { isDrivingGear, type DrivingModelConfig } from './driving-model-config';
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
    private previous?: { observation: DrivingObservation; input: DrivingPolicyInput; clipLabel?: string; clipId?: string };

    constructor(private policy: DrivingPolicy, private train: boolean, private balance?: RecordedDrivingBalance) {
        this.memory = new DrivingMemory(policy.configuration, train);
        const { recordedBatchSize, recordedBatchCount } = policy.configuration;
        this.plans = new DrivingPlanTraining(balance?.weights, recordedBatchSize, recordedBatchCount);
    }

    get sampled() { return this.plans.sampled; }
    get completedBatches() { return this.plans.completedBatches; }

    startRange(active = () => true) {
        if (!active()) return;
        this.plans.completeRange();
        this.memory.reset();
        this.previous = undefined;
    }

    async finish(active = () => true, onProgress?: (completedBatches: number) => void | Promise<void>) {
        if (!this.train) return;
        const { recordedBatchCount, recordedBatchSize, runtime } = this.policy.configuration;
        const batchesPerChunk = Math.min(Math.max(1, Math.floor(RECORDED_PLAN_WINDOW / recordedBatchSize)),
            Math.ceil(runtime.saveEverySteps / recordedBatchSize));
        while (this.completedBatches < recordedBatchCount && active()) {
            const previous = this.completedBatches;
            await this.policy.trainBatchAsync(() => this.plans.finishAsync(this.policy, active, batchesPerChunk));
            if (this.completedBatches === previous) break;
            await onProgress?.(this.completedBatches);
        }
    }

    async consume(result: RecordedObservationResult, active = () => true, clipLabel?: string, clipId?: string): Promise<void> {
        if (!active()) return;
        const observation = result.observation;
        if (!observation || !this.policy.matches(observation)) {
            this.progress.skipped++;
            this.progress.lastError = result.error || 'Recorded inputs do not match this model.';
            this.startRange(active);
            return;
        }
        if (this.previous && !isConsecutiveRecordedObservation(this.previous.observation, observation, this.policy.configuration)) {
            this.startRange(active);
        }
        const input = this.memory.encode(observation);
        const pending = this.previous;
        this.previous = { observation, input, clipLabel, clipId };
        if (!pending) return;

        // Keep the earlier observation as the plan origin and supervise it with
        // the next sample's controls and measured gear, including neutral and reverse.
        const knownGear = observation.gearAvailable !== false && isDrivingGear(observation.gear);
        const action = encodeDrivingAction(observation.controls, knownGear ? observation.gear : 1);
        if (this.train) {
            await this.policy.trainBatchAsync(async () => {
                if (!active()) return;
                this.policy.observeDemonstration(pending.input);
                this.plans.record(this.policy, pending.input, { action, gearWeight: knownGear ? 1 : 0 }, pending.observation.capturedAt,
                    this.balance?.select(pending.clipLabel), pending.clipId);
            });
        } else {
            const decision = await this.policy.decideAsync(pending.input, false);
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
