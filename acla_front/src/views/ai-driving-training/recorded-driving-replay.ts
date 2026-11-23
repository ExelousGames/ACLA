import { DrivingMemory, DrivingPolicyInput } from './driving-memory';
import { DrivingAction, DrivingObservation, encodeDrivingAction } from './driving-observation';
import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';
import type { RecordedObservationResult } from './recorded-driving-session';

const RECORDED_FRAME_GAP_ALLOWANCE_MS = 100;

export interface RecordedReplayProgress {
    samples: number;
    skipped: number;
    /** Mean absolute error of the signed pedal command (0–2). */
    pedalError: number;
    steeringError: number;
    lastError?: string;
}

/** Offline demonstration learning/evaluation. This path has no controller dependency. */
export class RecordedDrivingReplay {
    readonly progress: RecordedReplayProgress = { samples: 0, skipped: 0, pedalError: 0, steeringError: 0 };
    private memory: DrivingMemory;
    private plans = new DrivingPlanTraining();
    private previous?: { observation: DrivingObservation; input: DrivingPolicyInput; action: DrivingAction; shiftGear: number };

    constructor(private policy: DrivingPolicy, private train: boolean) {
        this.memory = new DrivingMemory(policy.configuration, RECORDED_FRAME_GAP_ALLOWANCE_MS);
    }

    startRange() {
        this.memory.reset();
        this.plans.reset();
        this.previous = undefined;
    }

    async consume(result: RecordedObservationResult, active = () => true): Promise<void> {
        if (!active()) return;
        const observation = result.observation;
        if (!observation || !this.policy.matches(observation)) {
            this.progress.skipped++;
            this.progress.lastError = result.error || 'Recorded inputs do not match this model.';
            this.startRange();
            return;
        }
        const previous = this.previous;
        if (previous) {
            const { runtime } = this.policy.configuration;
            const frameGapMs = observation.capturedAt - previous.observation.capturedAt;
            const maxFrameGapMs = runtime.maxTransitionSeconds * 1000 + RECORDED_FRAME_GAP_ALLOWANCE_MS;
            const telemetryGapMs = observation.measuredAt - previous.observation.measuredAt;
            let movement = observation.normalizedPosition - previous.observation.normalizedPosition;
            if (movement < -0.5) movement += 1;
            if (movement > 0.5) movement -= 1;
            if (frameGapMs <= 0 || frameGapMs > maxFrameGapMs || telemetryGapMs < 0
                || observation.session !== previous.observation.session || observation.reference !== previous.observation.reference
                || Math.abs(movement) > runtime.maxPositionJump) {
                this.startRange();
            } else if (telemetryGapMs === 0) {
                // Wait for a new reading while retaining the pending observation and plan targets.
                return;
            }
        }
        const input = this.memory.encode(observation);
        const pending = this.previous;
        const shiftGear = result.shiftGear ?? observation.gear;
        this.previous = { observation, input, action: encodeDrivingAction(observation.controls), shiftGear };
        if (!pending) return;

        // Telemetry records held axes, not paddle presses. A single subsequent gear
        // change supplies the shift target; ambiguous multi-gear jumps are masked.
        const gearDelta = shiftGear - pending.shiftGear;
        const knownShift = Number.isInteger(gearDelta) && Math.abs(gearDelta) <= 1;
        const shifts = { up: knownShift, down: knownShift };
        const action = { ...pending.action, shiftUp: knownShift && gearDelta === 1,
            shiftDown: knownShift && gearDelta === -1 };
        if (this.train) {
            await this.policy.trainBatchAsync(async () => {
                if (!active()) return;
                await this.policy.learnDemonstrationAsync(pending.input, action, shifts);
                if (active()) await this.plans.recordAsync(this.policy, pending.input, action, shifts,
                    pending.observation.capturedAt, active);
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
