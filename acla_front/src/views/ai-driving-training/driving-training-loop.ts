import { DrivingAction, DrivingObservation, DrivingReward, drivingReward,
    NEUTRAL, ObservationResult } from './driving-observation';
import { DrivingPolicy, PlannedAction, PlanStepCredit, PolicyDecision, ShiftAvailability } from './driving-policy';
import { ControlSource, DrivingControlSelector, ManualControls } from './driving-control';
import { ControlLease, DrivingControlOutput } from './driving-control-output';
import { DrivingMemory, DrivingPolicyInput, drivingMemorySlotInputs } from './driving-memory';
import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingModelTimer } from './driving-model-timing';
import type { DrivingImpact } from './driving-impact';
import type { DrivingRolloutInterval } from './driving-rollout';

import { DEFAULT_DRIVING_CONFIG, drivingMemorySlots } from './driving-model-config';

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
    action: DrivingAction;
    shifts: ShiftAvailability;
    source: ControlSource;
    appliedAt: number;
}

interface ExecutedPlan {
    decision: PolicyDecision;
    startedAt: number;
    endedAt?: number;
    trained?: boolean;
    executions: (Omit<PlanStepCredit, 'weight'> & { appliedAt: number; action: DrivingAction; nextWeight: number })[];
}

export class DrivingTrainingLoop {
    state = { ...initialTrainingState };
    private get config() { return this.policy?.configuration ?? DEFAULT_DRIVING_CONFIG; }
    private previous?: TrainingSample;
    private activePlan?: ExecutedPlan;
    private plans: ExecutedPlan[] = [];
    private sequence: DrivingRolloutInterval[] = [];
    private learningPlans = new Set<ExecutedPlan>();
    private installingPlan = false;
    private planWrite?: Promise<boolean>;
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
    private lease?: ControlLease;
    private policy?: DrivingPolicy;
    private memory = new DrivingMemory();
    private planTraining = new DrivingPlanTraining();
    private modelTiming?: DrivingModelTimer;

    constructor(private output: DrivingControlOutput, private read: () => ObservationResult,
        private publish: (state: TrainingState) => void, private now = Date.now,
        private readManual?: () => Promise<ManualControls | null>,
        private readImpact?: (previous: DrivingObservation, current: DrivingObservation) => DrivingImpact | undefined) {}

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
            if (!policy.matches(input.observation)) throw new Error('Live input layout or encoding differs from the model configuration.');
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
        this.generation++;
        this.state = { ...initialTrainingState, mode: policy ? train ? 'training' : 'evaluation' : 'manual',
            source: this.selector.source, steps: policy?.steps ?? this.state.steps };
        this.state.status = this.driverStatus();
        this.publish(this.state);
        const interval = this.readManual ? this.config.runtime.manualControlIntervalMs : this.config.runtime.controlIntervalMs;
        // Sample the plan between its 100 ms knots, including while GPU work is pending.
        this.timer = setInterval(() => { void this.tick(policy); }, policy ? Math.min(interval, 20) : interval);
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
        this.discardPlans();
        this.previous = undefined;
    }

    private discardPlans() {
        this.sequence = [];
        for (const plan of this.plans) {
            if (!this.learningPlans.has(plan)) this.policy?.releaseDecision(plan.decision);
        }
        this.plans = [];
        this.activePlan = undefined;
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
        if (!this.policy?.matches(input.observation)) return 'Observation configuration changed. Waiting for matching model inputs.';
        if (!input.observation.features.every(Number.isFinite)) return 'Invalid policy observation.';
        if (input.observation.session !== this.boundary?.session) return 'Live session changed. Stop and restart to use the same model in the current session.';
        if ((['game', 'car', 'map'] as const).some((key) => input.observation!.context[key] !== this.boundary?.context[key])) {
            return 'Game, car or map changed during this episode. Stop and restart to continue with the same model.';
        }
        if (input.observation.reference !== this.boundary?.reference) return 'Live Map reference changed. Waiting for the original reference.';
        if (this.previous) {
            const elapsedSeconds = (input.observation.capturedAt - this.previous.observation.capturedAt) / 1000;
            const progress = Math.abs(input.observation.normalizedPosition - this.previous.observation.normalizedPosition);
            if (elapsedSeconds > 0 && elapsedSeconds <= this.config.runtime.maxTransitionSeconds && Math.min(progress, 1 - progress) > this.config.runtime.maxPositionJump) {
                return 'Lap position jumped. Waiting for fresh telemetry.';
            }
        }
    }

    private async releasePaddles(lease: ControlLease) {
        const action = { ...this.state.action, shiftUp: false, shiftDown: false };
        if (!await lease.apply(action) || lease !== this.lease) return false;
        this.state = { ...this.state, action };
        this.publish(this.state);
        return true;
    }

    private async tick(policy?: DrivingPolicy) {
        if (this.controlBusy || this.installingPlan || this.state.mode === 'idle') return;
        this.controlBusy = true;
        const generation = this.generation;
        let lease = this.lease!;
        try {
            // Release paddles on the control clock, even if no new image has arrived.
            if ((this.state.action.shiftUp || this.state.action.shiftDown) && this.now() >= this.shiftReleaseAt) {
                this.planWrite = this.releasePaddles(lease);
                if (!await this.planWrite) return;
            }
            const manual = this.readManual ? await this.readManual() : null;
            if (generation !== this.generation || lease !== this.lease || this.installingPlan) return;
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
                && (selected.source === 'ai' || (!this.state.action.shiftUp && !this.state.action.shiftDown));
            if (selected.source === 'manual') {
                const newShift = selected.action!.shiftUp || selected.action!.shiftDown;
                // A fresh pull interrupts the old pulse so rapid pulls remain distinct presses.
                if (newShift && (this.state.action.shiftUp || this.state.action.shiftDown)
                    && !await this.releasePaddles(lease)) return;
                const action = { ...selected.action!,
                    shiftUp: Boolean(this.state.action.shiftUp || selected.action!.shiftUp),
                    shiftDown: Boolean(this.state.action.shiftDown || selected.action!.shiftDown) };
                if (!await lease.apply(action) || generation !== this.generation) return;
                const appliedAt = this.now();
                if (newShift) {
                    this.shiftReleaseAt = appliedAt + this.config.actions.shiftPulseMs;
                }
                this.state = { ...this.state, action, status: inputError ? this.state.status : this.driverStatus() };
                this.publish(this.state);
                if (fresh && observation && policy && !this.modelLease) {
                    void this.updateModel(policy, observation, lease, { action, shifts: { up: true, down: true }, appliedAt });
                } else if (previous?.source === 'manual' && newShift) {
                    // Retain a brief paddle press between images as the demonstration for
                    // this frame. Use the whole acknowledged action, not a synthetic mix.
                    this.previous = { ...previous, action, shifts: { up: true, down: true }, appliedAt };
                }
                return;
            }
            // Advance the current plan on the control clock, including during GPU work.
            if (this.activePlan) {
                this.planWrite = this.applyPlan(this.activePlan, lease);
                if (!await this.planWrite) return;
            }
            // An old owner's model work may still be finishing.
            if (this.modelLease) {
                if (this.modelLease !== lease) await lease.apply(this.state.action);
                return;
            }
            // Maintain the current action without learning twice from the same image or from a capture predating the action.
            if (!fresh) {
                if (!this.activePlan && !await lease.apply(this.state.action)) return;
                if (generation === this.generation) this.publish(this.state);
                return;
            }
            if (!observation || !policy) return;
            void this.updateModel(policy, observation, lease);
        } catch (error) {
            if (generation === this.generation && lease === this.lease) await this.stop(error instanceof Error ? error.message : 'Controller failed.');
        } finally { this.controlBusy = false; this.planWrite = undefined; }
    }

    private async applyPlan(plan: ExecutedPlan, lease: ControlLease) {
        const elapsed = Math.max(0, (this.now() - plan.startedAt) / 1000);
        let step = 0;
        while (step + 1 < plan.decision.plan.length && plan.decision.plan[step + 1].timeSeconds <= elapsed) step++;
        const previous = plan.executions[plan.executions.length - 1];
        const sameStep = step === previous?.step;
        const selected = plan.decision.plan[step];
        const next = plan.decision.plan[step + 1];
        const nextWeight = next ? (elapsed - selected.timeSeconds) / (next.timeSeconds - selected.timeSeconds) : 0;
        const pedal = selected.action.throttle - selected.action.brake;
        const blendedPedal = next ? pedal + (next.action.throttle - next.action.brake - pedal) * nextWeight : pedal;
        // Keep the original categorical choice/mask for credit, but never pulse it twice.
        const shifts = sameStep ? previous.shifts : selected.shifts;
        const shift = sameStep ? previous.shift : selected.shift === 1 && shifts.up ? 1 : selected.shift === 2 && shifts.down ? 2 : 0;
        if (!sameStep && shift && (this.state.action.shiftUp || this.state.action.shiftDown)) {
            if (!await this.releasePaddles(lease) || plan !== this.activePlan) return false;
        }
        const action = { ...selected.action,
            throttle: Math.max(0, blendedPedal), brake: Math.max(0, -blendedPedal),
            steering: next ? selected.action.steering + (next.action.steering - selected.action.steering) * nextWeight : selected.action.steering,
            shiftUp: Boolean(this.state.action.shiftUp || (!sameStep && shift === 1)),
            shiftDown: Boolean(this.state.action.shiftDown || (!sameStep && shift === 2)) };
        if (!await lease.apply(action) || plan !== this.activePlan) return false;
        const appliedAt = this.now();
        if (!plan.executions.length) plan.startedAt = appliedAt;
        plan.executions.push({ step, appliedAt, shift, shifts, action: { ...action }, nextWeight });
        if (!sameStep && shift) {
            this.shiftReleaseAt = appliedAt + this.config.actions.shiftPulseMs;
        }
        this.state = { ...this.state, action };
        this.publish(this.state);
        return true;
    }

    /** Attribute each observed interval to acknowledged controls, including an old
     * plan's tail while its replacement was being computed. */
    private planCredits(from: number, until: number) {
        return this.plans.map((plan) => {
            const intervals = plan.executions.map((execution, i) => {
                const end = plan.executions[i + 1]?.appliedAt ?? plan.endedAt ?? until;
                const duration = Math.max(0, Math.min(end, until) - Math.max(from, execution.appliedAt));
                return { execution, duration };
            }).filter(({ duration }) => duration > 0);
            const duration = intervals.reduce((sum, interval) => sum + interval.duration, 0);
            const credits: PlanStepCredit[] = intervals.flatMap(({ execution, duration: held }) => {
                const { step, shift, shifts, action, appliedAt, nextWeight } = execution;
                const weight = held / duration;
                // Both analog endpoints contributed; only the due slot chose a gear.
                return [{ step, shift, shifts, action, appliedAt, weight: weight * (1 - nextWeight), shiftWeight: weight },
                    ...(nextWeight > 0 ? [{ step: step + 1, shift: 0 as const, shifts: { up: false, down: false },
                        action, appliedAt, weight: weight * nextWeight, shiftWeight: 0 }] : [])];
            });
            return { plan, duration, credits };
        }).filter(({ duration }) => duration > 0);
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
            timing.historyLength = 0;
            const slotInputs = drivingMemorySlotInputs(this.config);
            for (let bin = 0; bin < drivingMemorySlots(this.config); bin++) {
                if (policyInput.features[bin * slotInputs]) timing.historyLength++;
            }
            timing.end();
            // Reward uses this observation once; plans retain their own GPU buffers.
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
            timing.start('inference');
            const decision = await policy.decideAsync(policyInput, this.state.mode === 'training', {
                up: true, down: true,
            });
            if (!active()) { policy.releaseDecision(decision); return; }
            this.installingPlan = true;
            timing.start('control');
            // Finish an acknowledged plan step before applying the new plan.
            try { await this.planWrite; }
            catch (error) { policy.releaseDecision(decision); throw error; }
            if (!active()) { policy.releaseDecision(decision); return; }
            const outgoing = this.activePlan;
            const plan: ExecutedPlan = { decision, startedAt: this.now(), executions: [] };
            this.activePlan = plan;
            try {
                if (!await this.applyPlan(plan, lease) || !active()) { policy.releaseDecision(decision); return; }
            } catch (error) { policy.releaseDecision(decision); throw error; }
            timing.end();
            const appliedAt = plan.startedAt, action = this.state.action;
            if (outgoing) outgoing.endedAt = appliedAt;
            this.plans.push(plan);
            this.previous = { observation, input: policyInput, action, source: 'ai',
                shifts: { up: true, down: true }, appliedAt };
            this.lastFrame = observation.capturedAt;
            this.state = { ...this.state, action, plan: decision.plan.map(({ timeSeconds, action: planned }) =>
                ({ timeSeconds, action: { ...planned } })), status: this.driverStatus() };
            timing.finish(true);
            this.publish(this.state);
        } catch (error) {
            if (active()) await this.stop(error instanceof Error ? error.message : 'Training failed.');
        } finally { timing.finish(false); this.modelLease = undefined; this.installingPlan = false; }
    }

    private async learnPrevious(policy: DrivingPolicy, previous: TrainingSample, observation: DrivingObservation, input: DrivingPolicyInput, active: () => boolean,
        timing: DrivingModelTimer) {
        const samples = this.planCredits(previous.observation.measuredAt, observation.measuredAt);
        const duration = samples.reduce((sum, sample) => sum + sample.duration, 0);
        samples.forEach(({ plan }) => this.learningPlans.add(plan));
        try {
            const elapsedSeconds = (observation.capturedAt - previous.observation.capturedAt) / 1000;
            // Gaps between usable frames are recoverable after input validation.
            // The caller starts the next transition from this fresh observation.
            if (elapsedSeconds > this.config.runtime.maxTransitionSeconds) {
                this.planTraining.reset();
                this.discardPlans();
                this.state = { ...this.state, reward: null };
                return;
            }
            const reward = drivingReward(previous.observation, observation, this.config, this.readImpact?.(previous.observation, observation));
            if (this.state.mode === 'training') {
                if (previous.source === 'manual') {
                    timing.start('learning');
                    await policy.trainBatchAsync(async () => {
                        await policy.learnDemonstrationAsync(previous.input, previous.action, previous.shifts);
                        if (active()) await this.planTraining.recordAsync(policy, previous.input, previous.action, previous.shifts,
                            previous.appliedAt, active, () => { timing.replayCount++; });
                    }, () => timing.start('learningWait'));
                    timing.end();
                } else if (samples.some(({ plan }) => !plan.trained)) {
                    this.sequence.push({ startedAt: previous.observation.capturedAt, endedAt: observation.capturedAt,
                        reward: reward.total, samples: samples.filter(({ plan }) => !plan.trained).map(({ plan, credits, duration: appliedDuration }) => ({
                            decision: plan.decision, credits, rewardShare: appliedDuration / duration,
                        })) });
                    if (observation.capturedAt - this.sequence[0].startedAt >= this.config.learning.sequenceSeconds * 1000) {
                        const sequence = this.sequence;
                        this.sequence = [];
                        timing.start('learning');
                        await policy.trainBatchAsync(() => policy.learnSequenceAsync(sequence, input, active), () => timing.start('learningWait'));
                        timing.end();
                        // An old plan may keep driving while the batch finishes,
                        // but must never supply gradients to the next policy.
                        samples.forEach(({ plan }) => { plan.trained = true; });
                    }
                }
            }
            if (active()) this.state = { ...this.state, steps: policy.steps, reward, episodeReward: this.state.episodeReward + reward.total };
        } finally {
            for (const { plan } of samples) {
                this.learningPlans.delete(plan);
                if (!this.plans.includes(plan)) policy.releaseDecision(plan.decision);
            }
            this.plans = this.plans.filter((plan) => {
                if (plan.trained) policy.releaseDecision(plan.decision);
                if (plan.endedAt === undefined || plan.endedAt > observation.measuredAt) return true;
                policy.releaseDecision(plan.decision);
                return false;
            });
        }
    }
}
