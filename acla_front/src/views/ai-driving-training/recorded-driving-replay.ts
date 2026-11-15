import { DrivingMemory, DrivingPolicyInput } from './driving-memory';
import { DrivingObservation, ObservationResult } from './driving-observation';
import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';

export interface RecordedReplayProgress {
    samples: number;
    skipped: number;
    pedalError: number;
    steeringError: number;
    lastError?: string;
}

/** Offline demonstration learning/evaluation. This path has no controller dependency. */
export class RecordedDrivingReplay {
    readonly progress: RecordedReplayProgress = { samples: 0, skipped: 0, pedalError: 0, steeringError: 0 };
    private memory: DrivingMemory;
    private plans = new DrivingPlanTraining();
    private previous?: { observation: DrivingObservation; input: DrivingPolicyInput };

    constructor(private policy: DrivingPolicy, private train: boolean) {
        this.memory = new DrivingMemory(policy.configuration);
    }

    private reset() {
        this.memory.reset();
        this.plans.reset();
        this.previous = undefined;
    }

    async consume(result: ObservationResult, active = () => true): Promise<void> {
        if (!active()) return;
        const observation = result.observation;
        if (!observation || !this.policy.matches(observation)) {
            this.progress.skipped++;
            this.progress.lastError = result.error || 'Recorded inputs do not match this model.';
            this.reset();
            return;
        }
        const previous = this.previous;
        if (previous) {
            const elapsed = observation.capturedAt - previous.observation.capturedAt;
            let movement = observation.normalizedPosition - previous.observation.normalizedPosition;
            if (movement < -0.5) movement += 1;
            if (movement > 0.5) movement -= 1;
            if (elapsed <= 0 || elapsed > this.policy.configuration.runtime.maxTransitionSeconds * 1000
                || observation.measuredAt <= previous.observation.measuredAt
                || observation.session !== previous.observation.session || observation.reference !== previous.observation.reference
                || Math.abs(movement) > this.policy.configuration.runtime.maxPositionJump) this.reset();
        }
        const input = this.memory.encode(observation);
        const pending = this.previous;
        this.previous = { observation, input };
        if (!pending) return;

        // Telemetry records held axes, not paddle presses. A single subsequent gear
        // change supplies the shift target; ambiguous multi-gear jumps are masked.
        const gearDelta = observation.gear - pending.observation.gear;
        const knownShift = Number.isInteger(gearDelta) && Math.abs(gearDelta) <= 1;
        const shifts = { up: knownShift, down: knownShift };
        const action = { ...pending.observation.controls, shiftUp: knownShift && gearDelta === 1,
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
                this.progress.pedalError += (Math.abs((predicted.throttle - predicted.brake)
                    - (action.throttle - action.brake)) - this.progress.pedalError) / count;
                this.progress.steeringError += (Math.abs(predicted.steering - action.steering) - this.progress.steeringError) / count;
            } finally { this.policy.releaseDecision(decision); }
        }
        if (active()) this.progress.samples++;
    }
}
