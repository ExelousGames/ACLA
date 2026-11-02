import { DrivingAction, DrivingObservation, DrivingReward, drivingReward,
    NEUTRAL, ObservationResult } from './driving-observation';
import { DrivingPolicy, PlannedAction, PolicyDecision, ShiftAvailability } from './driving-policy';
import { ControlSource, DrivingControlSelector, ManualControls } from './driving-control';
import { ControlLease, DrivingControlOutput } from './driving-control-output';
import { DrivingMemory, DrivingPolicyInput } from './driving-memory';
import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingModelTimer } from './driving-model-timing';

import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';

export interface TrainingState {
    mode: 'idle' | 'manual' | 'training' | 'evaluation';
    status: string;
    action: DrivingAction;
    plan: PlannedAction[] | null;
    source: ControlSource | null;
    steps: number;
    episodeReward: number;
    reward: DrivingReward | null;
}
export const initialTrainingState: TrainingState = {
    mode: 'idle', status: 'Connect vJoy, then choose human driving or prepare the AI inputs.', action: NEUTRAL,
    source: null, steps: 0, episodeReward: 0, reward: null, plan: null,
};

interface TrainingSample {
    observation: DrivingObservation;
    input: DrivingPolicyInput;
    decision?: PolicyDecision;
    action: DrivingAction;
    shifts: ShiftAvailability;
    source: ControlSource;
    appliedAt: number;
}

export class DrivingTrainingLoop {
    state = { ...initialTrainingState };
    private get config() { return this.policy?.configuration ?? DEFAULT_DRIVING_CONFIG; }
    private previous?: TrainingSample;
    private boundary?: DrivingObservation;
    private selector = new DrivingControlSelector();
    private controlChangedAt = -Infinity;
    private lastFrame = -Infinity;
    private generation = 0;
    private controlBusy = false;
    private modelLease?: ControlLease;
    private modelRevision = 0;
    private starting = false;
    private timer?: ReturnType<typeof setInterval>;
    private shiftReleaseAt = 0;
    private nextShiftAt = 0;
    private lease?: ControlLease;
    private policy?: DrivingPolicy;
    private memory = new DrivingMemory();
    private planTraining = new DrivingPlanTraining();
    private modelTiming?: DrivingModelTimer;

    constructor(private output: DrivingControlOutput, private read: () => ObservationResult,
        private publish: (state: TrainingState) => void, private now = Date.now,
        private readManual?: () => Promise<ManualControls | null>) {}

    getModelTimingSnapshot() { return this.modelTiming?.snapshot(); }

    setControlSource(source: ControlSource) {
        if (this.state.mode === 'idle') return;
        if (source === 'ai') {
            if (!this.policy) throw new Error('Start training or run a trained model before handing control to AI.');
            const input = this.read();
            const error = this.observationError(input);
            if (error) throw new Error(error);
        }
        this.changeSource(source);
    }

    async start(policy: DrivingPolicy, train: boolean) {
        if (this.starting || (this.state.mode !== 'idle' && this.state.mode !== 'manual')) throw new Error('Stop the current session before starting another.');
        this.starting = true;
        const generation = this.generation;
        try {
            await policy.initializeGpu();
            if (generation !== this.generation) return;
            const input = this.read();
            if (!input.observation) throw new Error(input.error);
            if (!policy.matches(input.observation)) throw new Error('Model inputs differ from this game, car, map or observation configuration. Create a new model.');
            this.begin(policy, train, input.observation);
        } finally { this.starting = false; }
    }

    startManual() { this.begin(); }

    private begin(policy?: DrivingPolicy, train = false, boundary?: DrivingObservation) {
        const humanDriving = this.state.mode === 'manual';
        if ((!humanDriving && (this.state.mode !== 'idle' || this.controlBusy || this.modelLease)) || (humanDriving && !policy)) {
            throw new Error('Stop the current session before starting another.');
        }
        clearInterval(this.timer);
        this.discardPrevious();
        this.policy = policy;
        this.memory = new DrivingMemory(this.config);
        this.planTraining.reset();
        this.modelTiming = undefined;
        this.boundary = boundary;
        if (!humanDriving) this.selector = new DrivingControlSelector();
        this.selector.source = !policy || humanDriving ? 'manual' : 'ai';
        this.lease = this.output.acquire(this.selector.source);
        this.controlChangedAt = -Infinity;
        this.lastFrame = -Infinity;
        this.shiftReleaseAt = 0;
        if (!humanDriving) this.nextShiftAt = 0;
        this.generation++;
        this.state = { ...initialTrainingState, mode: policy ? train ? 'training' : 'evaluation' : 'manual',
            source: this.selector.source, steps: policy?.steps ?? this.state.steps };
        this.state.status = this.driverStatus();
        this.publish(this.state);
        this.timer = setInterval(() => { void this.tick(policy); }, this.readManual ? this.config.runtime.manualControlIntervalMs : this.config.runtime.controlIntervalMs);
        void this.tick(policy);
    }

    async stop(status = 'Stopped. Controller released.'): Promise<void> {
        const generation = ++this.generation;
        this.modelRevision++;
        clearInterval(this.timer);
        this.modelTiming?.finish(false);
        this.discardPrevious();
        this.lease = undefined;
        this.memory.reset();
        this.planTraining.reset();
        this.policy = undefined;
        this.state = { ...this.state, mode: 'idle', status, action: NEUTRAL, source: null, plan: null };
        this.publish(this.state);
        try { await this.output.release(); }
        catch (error) {
            if (generation !== this.generation) return;
            this.state = { ...this.state, status: `${status} ${error instanceof Error ? error.message : 'Controller disconnected.'}` };
            this.publish(this.state);
        }
    }

    private driverStatus() {
        if (this.state.source === 'manual') return this.state.mode === 'training'
            ? 'Human control • learning from demonstrations' : 'Human control • learning off';
        return this.state.mode === 'training' ? 'AI control • online actor–critic' : 'AI control • fixed policy';
    }

    private discardPrevious() {
        this.policy?.releaseDecision(this.previous?.decision);
        this.previous = undefined;
    }

    private changeSource(source: ControlSource) {
        if (source === this.state.source) return;
        this.modelRevision++;
        this.modelTiming?.finish(false);
        this.selector.source = source;
        this.lease = this.output.acquire(source);
        // No reward, held paddle or demonstration may cross an ownership boundary.
        this.discardPrevious();
        this.memory.reset();
        this.planTraining.reset();
        this.controlChangedAt = this.now();
        this.shiftReleaseAt = 0;
        this.state = { ...this.state, source, action: NEUTRAL, plan: null };
        this.state.status = this.driverStatus();
        this.publish(this.state);
    }

    private observationError(input: ObservationResult): string | undefined {
        if (!input.observation) return input.error || 'Waiting for live inputs.';
        if (!this.policy?.matches(input.observation)) return 'Game, car, map or observation configuration changed. Waiting for matching model inputs.';
        if (!input.observation.features.every(Number.isFinite)) return 'Invalid policy observation.';
        if (input.observation.session !== this.boundary?.session) return 'Live session changed. Waiting for the original live session.';
        if (input.observation.reference !== this.boundary?.reference) return 'Live Map reference changed. Waiting for the original reference.';
        if (this.previous) {
            const elapsedSeconds = (input.observation.capturedAt - this.previous.observation.capturedAt) / 1000;
            const progress = Math.abs(input.observation.normalizedPosition - this.previous.observation.normalizedPosition);
            if (elapsedSeconds > 0 && elapsedSeconds <= this.config.runtime.maxTransitionSeconds && Math.min(progress, 1 - progress) > this.config.runtime.maxPositionJump) {
                return 'Lap position jumped. Waiting for fresh telemetry.';
            }
        }
    }

    private async tick(policy?: DrivingPolicy) {
        if (this.controlBusy || this.state.mode === 'idle') return;
        this.controlBusy = true;
        const generation = this.generation;
        let lease = this.lease!;
        try {
            // Release paddles on the control clock, even if no new image has arrived.
            if ((this.state.action.shiftUp || this.state.action.shiftDown) && this.now() >= this.shiftReleaseAt) {
                const released = { ...this.state.action, shiftUp: false, shiftDown: false };
                if (!await lease.apply(released) || generation !== this.generation) return;
                this.state = { ...this.state, action: released };
                this.publish(this.state);
            }
            const manual = this.readManual ? await this.readManual() : null;
            if (generation !== this.generation || lease !== this.lease) return;
            const input: ObservationResult = policy ? this.read() : { error: 'AI not started.' };
            const inputError = policy ? this.observationError(input) : undefined;
            const selected = this.selector.select(manual, Boolean(policy && !inputError));
            this.changeSource(selected.source);
            lease = this.lease!;
            const observation = !inputError && input.observation ? { ...input.observation } : undefined;
            if (!observation) {
                this.modelRevision++;
                this.modelTiming?.finish(false);
                this.discardPrevious();
                this.memory.reset();
                this.planTraining.reset();
                this.controlChangedAt = this.now();
                this.state = { ...this.state, reward: null, plan: null };
            }
            if (inputError) {
                this.state = { ...this.state, status: selected.source === 'manual'
                    ? `Human control • learning paused: ${inputError}`
                    : `AI control • waiting for valid inputs; retrying: ${inputError}` };
                if (selected.source === 'ai') {
                    // Keep polling with the same policy, but discard the interrupted transition.
                    if (!await lease.apply(NEUTRAL) || generation !== this.generation) return;
                    this.controlChangedAt = this.now();
                    this.shiftReleaseAt = 0;
                    this.state = { ...this.state, action: NEUTRAL };
                    this.publish(this.state);
                    return;
                }
            }
            const previous = this.previous;
            const fresh = observation && observation.capturedAt > this.lastFrame && observation.capturedAt > this.controlChangedAt
                && observation.measuredAt > this.controlChangedAt
                && (!previous || (observation.capturedAt > previous.appliedAt && observation.measuredAt > previous.appliedAt))
                && !this.state.action.shiftUp && !this.state.action.shiftDown;
            if (selected.source === 'manual') {
                const canShift = this.now() >= this.nextShiftAt;
                const action = { ...selected.action!,
                    shiftUp: Boolean(this.state.action.shiftUp || (canShift && selected.action!.shiftUp)),
                    shiftDown: Boolean(this.state.action.shiftDown || (canShift && selected.action!.shiftDown)) };
                if (!await lease.apply(action) || generation !== this.generation) return;
                const appliedAt = this.now();
                if (canShift && (action.shiftUp || action.shiftDown)) {
                    this.shiftReleaseAt = appliedAt + this.config.actions.shiftPulseMs;
                    this.nextShiftAt = appliedAt + this.config.actions.shiftCooldownMs;
                }
                this.state = { ...this.state, action, status: inputError ? this.state.status : this.driverStatus() };
                this.publish(this.state);
                if (fresh && observation && policy && !this.modelLease) {
                    void this.updateModel(policy, observation, lease, { action, shifts: { up: canShift, down: canShift }, appliedAt });
                } else if (previous?.source === 'manual' && canShift && (action.shiftUp || action.shiftDown)) {
                    // Retain a brief paddle press between images as the demonstration for
                    // this frame. Use the whole acknowledged action, not a synthetic mix.
                    this.previous = { ...previous, action, shifts: { up: true, down: true }, appliedAt };
                }
                return;
            }
            // An old owner's model work may still be finishing. Neutralize a new
            // AI lease immediately, but never queue stale heartbeats behind its decision.
            if (this.modelLease) {
                if (this.modelLease !== lease) await lease.apply(this.state.action);
                return;
            }
            // Maintain the current action without learning twice from the same image or from a capture predating the action.
            if (!fresh) {
                if (!await lease.apply(this.state.action)) return;
                if (generation === this.generation) this.publish(this.state);
                return;
            }
            if (!observation || !policy) return;
            void this.updateModel(policy, observation, lease);
        } catch (error) {
            if (generation === this.generation && lease === this.lease) await this.stop(error instanceof Error ? error.message : 'Controller failed.');
        } finally { this.controlBusy = false; }
    }

    /** Only one model operation runs at a time; controller polling never awaits it. */
    private async updateModel(policy: DrivingPolicy, observation: DrivingObservation, lease: ControlLease,
        manual?: Pick<TrainingSample, 'action' | 'shifts' | 'appliedAt'>) {
        this.modelLease = lease;
        const generation = this.generation, revision = this.modelRevision;
        const active = () => generation === this.generation && lease === this.lease && revision === this.modelRevision;
        const timing = this.modelTiming = new DrivingModelTimer(observation.capturedAt, manual ? 'manual' : 'ai', this.state.mode === 'training');
        const previous = this.previous;
        try {
            timing.start('memory');
            const policyInput = this.memory.encode(observation);
            timing.end();
            // Pending learning owns the old sample until it releases its GPU buffers.
            this.previous = undefined;
            if (manual) {
                // Save the acknowledged action before learning, so later paddle
                // presses can still be retained while the model is busy.
                this.previous = { observation, input: policyInput, ...manual, source: 'manual' };
                this.lastFrame = observation.capturedAt;
            }
            if (previous) await this.learnPrevious(policy, previous, observation, policyInput, active, timing);
            if (!active()) return;
            if (manual) {
                timing.finish(true);
                this.publish(this.state);
                return;
            }
            const canShift = this.now() >= this.nextShiftAt;
            timing.start('inference');
            const decision = await policy.decideAsync(policyInput, this.state.mode === 'training', {
                up: canShift, down: canShift,
            });
            if (!active()) { policy.releaseDecision(decision); return; }
            const action = decision.plan[0].action;
            try {
                timing.start('control');
                if (!await lease.apply(action) || !active()) { policy.releaseDecision(decision); return; }
            } catch (error) { policy.releaseDecision(decision); throw error; }
            timing.end();
            const appliedAt = this.now();
            if (action.shiftUp || action.shiftDown) {
                this.shiftReleaseAt = appliedAt + this.config.actions.shiftPulseMs;
                this.nextShiftAt = appliedAt + this.config.actions.shiftCooldownMs;
            }
            this.previous = { observation, input: policyInput, decision, action, source: 'ai',
                shifts: { up: canShift, down: canShift }, appliedAt };
            this.lastFrame = observation.capturedAt;
            this.state = { ...this.state, action, plan: decision.plan.map(({ timeSeconds, action: planned }) =>
                ({ timeSeconds, action: { ...planned } })), status: this.driverStatus() };
            timing.finish(true);
            this.publish(this.state);
        } catch (error) {
            if (active()) await this.stop(error instanceof Error ? error.message : 'Training failed.');
        } finally { timing.finish(false); this.modelLease = undefined; }
    }

    private async learnPrevious(policy: DrivingPolicy, previous: TrainingSample, observation: DrivingObservation, input: DrivingPolicyInput, active: () => boolean,
        timing: DrivingModelTimer) {
        // This operation owns the sample while GPU work is in flight. Stop/handoff
        // can clear the loop without destroying buffers still needed by backprop.
        try {
            const elapsedSeconds = (observation.capturedAt - previous.observation.capturedAt) / 1000;
            // Gaps between usable frames are recoverable after input validation.
            // The caller starts the next transition from this fresh observation.
            if (elapsedSeconds > this.config.runtime.maxTransitionSeconds) {
                this.planTraining.reset();
                this.state = { ...this.state, reward: null };
                return;
            }
            const reward = drivingReward(previous.observation, observation, this.config);
            if (this.state.mode === 'training') {
                timing.start('learning');
                await policy.trainBatchAsync(async () => {
                    if (previous.source === 'manual') await policy.learnDemonstrationAsync(previous.input, previous.action, previous.shifts);
                    else await policy.learnAsync(previous.decision!, reward.total, input, elapsedSeconds);
                    if (active()) await this.planTraining.recordAsync(policy, previous.input, previous.action, previous.shifts, previous.appliedAt, active);
                });
                timing.end();
            }
            if (active()) this.state = { ...this.state, steps: policy.steps, reward, episodeReward: this.state.episodeReward + reward.total };
        } finally { policy.releaseDecision(previous.decision); }
    }
}
