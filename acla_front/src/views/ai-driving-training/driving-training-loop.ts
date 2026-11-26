import { DrivingAction, drivingControllerAction, DrivingObservation, DrivingReward, DrivingTelemetrySample, drivingReward,
    NEUTRAL, ObservationResult } from './driving-observation';
import { ControlChanges, DrivingControlChanges } from './driving-control-changes';
import { DrivingPolicy, PlannedAction, PlanStepCredit, PolicyDecision } from './driving-policy';
import { ControlSource, DrivingControlSelector, ManualControls } from './driving-control';
import { ControlLease, DrivingControlOutput } from './driving-control-output';
import { DrivingMemory, DrivingPolicyInput, drivingMemorySlotInputs } from './driving-memory';
import { DrivingConsequenceBatch, DrivingConsequenceBuffer } from './driving-consequence-buffer';
import { DrivingModelTimer } from './driving-model-timing';
import { DRIVING_PLAN_REFRESH_SECONDS, drivingPlanControlsDiffer } from './driving-plan-agreement';
import type { ControllerAction } from './virtual-controller';
import type { DrivingImpact } from './driving-impact';
import { DrivingBlockedMotionMonitor } from './driving-blocked-motion';

import { DEFAULT_DRIVING_CONFIG, drivingMemorySlots } from './driving-model-config';

export interface TrainingState {
    mode: 'idle' | 'manual' | 'training' | 'evaluation' | 'finishing';
    status: string;
    action: ControllerAction;
    plan: PlannedAction[] | null;
    /** Null while stopped or while exploration learning finishes with the controls neutral. */
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
}

interface ExecutedPlan {
    decision: PolicyDecision;
    startedAt: number;
    endedAt?: number;
    executions: (Omit<PlanStepCredit, 'weight'> & { appliedAt: number; action: DrivingAction; nextWeight: number;
        shiftWeight: number; controlChanges: ControlChanges })[];
}

export class DrivingTrainingLoop {
    state = { ...initialTrainingState };
    private get config() { return this.policy?.configuration ?? DEFAULT_DRIVING_CONFIG; }
    private previous?: TrainingSample;
    private activePlan?: ExecutedPlan;
    private plans: ExecutedPlan[] = [];
    private consequences = new DrivingConsequenceBuffer(DEFAULT_DRIVING_CONFIG.learning.sequenceSeconds);
    private learningDecisions = new Set<PolicyDecision>();
    private installingPlan = false;
    private planWrite?: Promise<boolean>;
    private boundary?: DrivingObservation;
    private selector = new DrivingControlSelector();
    private controlChangedAt = -Infinity;
    private lastFrame = -Infinity;
    private lastInferenceFrame = -Infinity;
    private generation = 0;
    private controlBusy = false;
    private modelLease?: ControlLease;
    private modelWork?: Promise<void>;
    private handoffWork?: Promise<void>;
    private finishing?: Promise<void>;
    private modelRevision = 0;
    private learningRevision = 0;
    private starting = false;
    private timer?: ReturnType<typeof setInterval>;
    private shiftReleaseAt = 0;
    private lease?: ControlLease;
    private policy?: DrivingPolicy;
    private memory = new DrivingMemory();
    private modelTiming?: DrivingModelTimer;
    private measuredSpeedKmh = 0;
    private measuredGear = 0;
    private blockedMotion = new DrivingBlockedMotionMonitor();
    private controlChanges = new DrivingControlChanges();

    constructor(private output: DrivingControlOutput, private read: () => ObservationResult,
        private publish: (state: TrainingState) => void, private now = Date.now,
        private readManual?: () => Promise<ManualControls | null>,
        private readImpact?: (previous: DrivingObservation, current: DrivingObservation) => DrivingImpact | undefined) {}

    getModelTimingSnapshot() { return this.modelTiming?.snapshot(); }

    private positionJumped(previous: number | null, current: number | null) {
        if (previous === null || current === null) return false;
        const progress = Math.abs(current - previous);
        return Math.min(progress, 1 - progress) > this.config.runtime.maxPositionJump;
    }

    /** The telemetry subscription runs independently of controller and GPU work. */
    recordTelemetry(sample: DrivingTelemetrySample) {
        if (this.state.source !== 'ai' || sample.session !== this.boundary?.session
            || sample.measuredAt <= this.controlChangedAt) return;
        this.blockedMotion.observe(sample);
    }

    setControlSource(source: ControlSource) {
        if (!this.state.source) return;
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
            let input = this.read();
            // Loading/copying weights and GPU preparation can block renderer event delivery
            // longer than the freshness limit. Yield until new inputs arrive, without
            // accepting stale observations or taking control while startup is pending.
            const deadline = performance.now() + 5000;
            while (!input.observation && input.waitingForFreshInputs && performance.now() < deadline) {
                await new Promise<void>((resolve) => setTimeout(resolve, 50));
                if (generation !== this.generation) return;
                input = this.read();
            }
            if (!input.observation) throw new Error(input.error);
            if (!policy.matches(input.observation)) throw new Error('Live input layout or encoding differs from the model configuration.');
            this.begin(policy, train, input.observation);
        } finally { this.starting = false; }
    }

    startManual() { this.begin(); }

    private begin(policy?: DrivingPolicy, train = false, boundary?: DrivingObservation) {
        const humanDriving = this.state.mode === 'manual';
        if ((!humanDriving && (this.state.mode !== 'idle' || this.controlBusy || this.modelLease || this.handoffWork)) || (humanDriving && !policy)) {
            throw new Error('Stop the current session before starting another.');
        }
        clearInterval(this.timer);
        this.discardPrevious();
        this.policy = policy;
        policy?.resetExploration();
        this.memory = new DrivingMemory(this.config);
        this.consequences = new DrivingConsequenceBuffer(this.config.learning.sequenceSeconds);
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

    /** The Stop button releases the wheel while the captured training segment finishes. */
    finishTraining(): Promise<void> {
        if (this.finishing) return this.finishing;
        if (this.state.mode !== 'training' || !this.policy) return this.stop();
        const generation = this.generation;
        const policy = this.policy;
        clearInterval(this.timer);
        this.lease = undefined;
        this.activePlan = undefined;
        this.modelRevision++;
        this.modelTiming?.finish(false);
        this.state = { ...this.state, mode: 'finishing', status: 'Finishing training…', action: NEUTRAL, source: null, plan: null };
        this.finishing = this.output.release().then(async () => {
            if (generation !== this.generation) return;
            await this.finishBufferedTraining(policy);
            if (generation === this.generation) this.endSession('Training finished. Controller released.', policy.steps);
        }).catch(async (error) => {
            if (generation === this.generation) await this.stop(error instanceof Error ? error.message : 'Controller failed.');
        });
        this.publish(this.state);
        return this.finishing;
    }

    private endSession(status: string, steps = this.state.steps) {
        this.generation++;
        this.finishing = undefined;
        this.modelRevision++;
        this.learningRevision++;
        clearInterval(this.timer);
        this.modelTiming?.finish(false);
        this.discardPrevious();
        this.lease = undefined;
        this.memory.reset();
        this.state = { ...this.state, mode: 'idle', status, steps, action: NEUTRAL, source: null, plan: null };
        this.policy = undefined;
        this.publish(this.state);
    }

    async stop(status = 'Stopped. Controller released.'): Promise<void> {
        this.endSession(status);
        const generation = this.generation;
        try { await this.output.release(); }
        catch (error) {
            if (generation !== this.generation) return;
            this.state = { ...this.state, status: `${status} ${error instanceof Error ? error.message : 'Controller disconnected.'}` };
            this.publish(this.state);
        }
    }

    private driverStatus() {
        if (this.state.source === 'manual') return 'Human control • learning off';
        return this.state.mode === 'training' ? 'AI control • soft actor–critic' : 'AI control • fixed policy';
    }

    private discardPrevious() {
        this.policy?.resetExploration();
        this.discardPlans();
        this.blockedMotion.reset();
        this.previous = undefined;
        this.lastInferenceFrame = -Infinity;
    }

    private discardPlans() {
        this.controlChanges = new DrivingControlChanges();
        this.consequences.reset();
        for (const plan of this.plans) {
            if (!this.learningDecisions.has(plan.decision)) this.policy?.releaseDecision(plan.decision);
        }
        this.plans = [];
        this.activePlan = undefined;
    }

    private changeSource(source: ControlSource) {
        if (source === this.state.source) return;
        const learning = source === 'manual' && this.policy && this.state.mode === 'training'
            ? this.finishBufferedTraining(this.policy) : undefined;
        this.modelRevision++;
        this.modelTiming?.finish(false);
        this.selector.source = source;
        this.discardPrevious();
        this.memory.reset();
        this.shiftReleaseAt = 0;
        if (learning) {
            this.lease = undefined;
            this.state = { ...this.state, source: null, action: NEUTRAL, plan: null, reward: null,
                status: 'Finishing exploration learning before human takeover…' };
            void this.completeHumanTakeover(learning);
            this.publish(this.state);
            return;
        }
        this.grantControl(source);
    }

    private async completeHumanTakeover(learning: Promise<void>) {
        const generation = this.generation;
        try {
            await Promise.all([this.output.release(), learning]);
            if (generation === this.generation && this.state.mode === 'training') this.grantControl('manual');
        } catch (error) {
            if (generation === this.generation) await this.stop(error instanceof Error ? error.message : 'Controller failed.');
        }
    }

    private grantControl(source: ControlSource) {
        const returningToAI = this.state.source === 'manual' && source === 'ai';
        const lease = this.lease = this.output.acquire(source);
        this.controlChangedAt = this.now();
        if (returningToAI) {
            // A queued manual write can outlast the ownership change. Wait for the
            // neutral output before accepting telemetry or camera samples for AI.
            this.controlChangedAt = Infinity;
            void lease.apply(NEUTRAL).then((applied) => {
                if (applied && lease === this.lease) this.controlChangedAt = this.now();
            }).catch((error) => {
                if (lease === this.lease) return this.stop(error instanceof Error ? error.message : 'Controller failed.');
            });
        }
        this.state = { ...this.state, source, action: NEUTRAL, plan: null, reward: null };
        this.state.status = this.driverStatus();
        this.publish(this.state);
    }

    private finishBufferedTraining(policy: DrivingPolicy) {
        const pending = this.handoffWork ?? (this.modelLease && this.modelWork);
        const batches = this.consequences.finish();
        if (!pending && !batches.length) return;
        const generation = this.generation, revision = this.learningRevision;
        const active = () => generation === this.generation && revision === this.learningRevision;
        let work!: Promise<void>;
        work = (async () => {
            try {
                await pending;
                if (!active()) return;
                for (const batch of batches) {
                    if (!active()) return;
                    await this.trainConsequences(policy, batch, active);
                }
                if (active()) {
                    this.state = { ...this.state, steps: policy.steps };
                    this.publish(this.state);
                }
            } catch (error) {
                if (active()) await this.stop(error instanceof Error ? error.message : 'Training failed.');
            } finally {
                if (this.handoffWork === work) this.handoffWork = undefined;
            }
        })();
        this.handoffWork = work;
        return work;
    }

    private observationError(input: ObservationResult): string | undefined {
        if (input.telemetry && input.telemetry.session !== this.boundary?.session) {
            return 'Live session changed. Stop and restart to use the same model in the current session.';
        }
        if (!input.observation) return input.error || 'Waiting for live inputs.';
        if (!this.policy?.matches(input.observation)) return 'Observation configuration changed. Waiting for matching model inputs.';
        if (!input.observation.features.every(Number.isFinite)) return 'Invalid policy observation.';
        if (input.observation.session !== this.boundary?.session) return 'Live session changed. Stop and restart to use the same model in the current session.';
        if ((['game', 'car', 'map'] as const).some((key) => input.observation!.context[key] !== this.boundary?.context[key])) {
            return 'Game, car or map changed during this episode. Stop and restart to continue with the same model.';
        }
        if (input.observation.reference !== this.boundary?.reference) return 'Live Map reference changed. Waiting for the original reference.';
        if (this.previous) {
            const elapsedSeconds = (input.observation.measuredAt - this.previous.observation.measuredAt) / 1000;
            if (elapsedSeconds > 0 && elapsedSeconds <= this.config.runtime.maxTransitionSeconds
                && this.positionJumped(this.previous.observation.normalizedPosition, input.observation.normalizedPosition)) {
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
        let lease = this.lease;
        try {
            // Release paddles on the control clock, even if no new image has arrived.
            if (lease && (this.state.action.shiftUp || this.state.action.shiftDown) && this.now() >= this.shiftReleaseAt) {
                this.planWrite = this.releasePaddles(lease);
                if (!await this.planWrite) return;
            }
            const manual = this.readManual ? await this.readManual() : null;
            if (generation !== this.generation || lease !== this.lease || this.installingPlan) return;
            if (!lease) {
                // Keep checking the controller and consuming button edges during takeover.
                this.selector.select(manual, false);
                return;
            }
            const input: ObservationResult = policy ? this.read() : { error: 'AI not started.' };
            const inputError = policy ? this.observationError(input) : undefined;
            const selected = this.selector.select(manual, Boolean(policy && !inputError));
            this.changeSource(selected.source);
            lease = this.lease;
            if (!lease) return;
            if (selected.source === 'manual') {
                const newShift = selected.action!.shiftUp || selected.action!.shiftDown;
                // A fresh pull interrupts the old pulse so rapid pulls remain distinct presses.
                if (newShift && (this.state.action.shiftUp || this.state.action.shiftDown)
                    && !await this.releasePaddles(lease)) return;
                const action = { ...selected.action!,
                    shiftUp: Boolean(this.state.action.shiftUp || selected.action!.shiftUp),
                    shiftDown: Boolean(this.state.action.shiftDown || selected.action!.shiftDown) };
                if (!await lease.apply(action) || generation !== this.generation || lease !== this.lease) return;
                if (newShift) this.shiftReleaseAt = this.now() + this.config.actions.shiftPulseMs;
                const status = this.driverStatus();
                this.state = { ...this.state, action, status: inputError ? `${status} • ${inputError}` : status };
                this.publish(this.state);
                return;
            }
            if (this.controlChangedAt === Infinity) return;
            if (inputError || !input.observation) {
                // Keep polling with the same policy, but discard the interrupted transition.
                this.modelRevision++;
                this.learningRevision++;
                this.modelTiming?.finish(false);
                this.discardPrevious();
                this.memory.reset();
                if (!await lease.apply(NEUTRAL) || generation !== this.generation || lease !== this.lease) return;
                this.controlChangedAt = this.now();
                this.shiftReleaseAt = 0;
                this.state = { ...this.state, action: NEUTRAL, reward: null, plan: null,
                    status: `AI control • waiting for valid inputs; retrying: ${inputError ?? input.error}` };
                this.publish(this.state);
                return;
            }
            const observation = { ...input.observation };
            const telemetry = input.telemetry && input.telemetry.measuredAt > this.controlChangedAt ? input.telemetry : undefined;
            this.measuredSpeedKmh = telemetry?.speed ?? observation.speed;
            this.measuredGear = telemetry?.gear ?? observation.gear;
            const previous = this.previous;
            const fresh = observation.capturedAt > this.lastFrame && observation.capturedAt > this.controlChangedAt
                && observation.measuredAt > this.controlChangedAt
                && (!previous || (observation.capturedAt > previous.observation.measuredAt
                    && observation.measuredAt > previous.observation.measuredAt));
            // Consequences and history advance on every fresh observation, even
            // while the model is waiting for training or inference to finish.
            if (fresh && policy) this.collectConsequences(policy, observation);
            // Advance the current plan on the control clock, including during GPU work.
            if (this.activePlan) {
                this.planWrite = this.applyPlan(this.activePlan, lease);
                if (!await this.planWrite) return;
            }
            // An old owner's model work may still be finishing.
            if (this.modelLease || this.handoffWork) {
                if (this.modelLease !== lease) await lease.apply(this.state.action);
                return;
            }
            // Ready batches remain consumable even without another camera frame.
            if (!this.previous || (this.previous.observation.capturedAt <= this.lastInferenceFrame
                && !(this.state.mode === 'training' && this.consequences.hasReady))) {
                if (!this.activePlan && !await lease.apply(this.state.action)) return;
                if (generation === this.generation) this.publish(this.state);
                return;
            }
            if (!policy) return;
            this.modelWork = this.updateModel(policy, this.previous, lease);
        } catch (error) {
            if (generation === this.generation && lease === this.lease) await this.stop(error instanceof Error ? error.message : 'Controller failed.');
        } finally { this.controlBusy = false; this.planWrite = undefined; }
    }

    private scoreReward(previous: DrivingObservation, current: DrivingObservation,
        controlChanges?: ControlChanges) {
        return drivingReward(previous, current, this.config, this.readImpact?.(previous, current), controlChanges,
            this.blockedMotion.penaltyBetween(previous.measuredAt, current.measuredAt));
    }

    private shiftsAtGear(gear: number) {
        const limit = this.config.learning.maxExplorationGear;
        return { up: this.state.mode !== 'training' || limit === 0 || gear < limit, down: true };
    }

    private throttleExplorationLimited() {
        return this.state.mode === 'training' && this.measuredSpeedKmh >= this.config.learning.throttleExplorationSpeedLimitKmh;
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
        let pedal = next ? selected.action.pedal + (next.action.pedal - selected.action.pedal) * nextWeight : selected.action.pedal;
        if (pedal > 0 && this.throttleExplorationLimited()) {
            const preferred = Math.tanh(selected.means[0]);
            const nextPreferred = next ? Math.tanh(next.means[0]) : preferred;
            pedal = Math.min(pedal, Math.max(0, preferred + (nextPreferred - preferred) * nextWeight));
        }
        const steering = next ? selected.action.steering + (next.action.steering - selected.action.steering) * nextWeight : selected.action.steering;
        // Recheck planned upshifts against live gear and keep the executed mask for credit.
        const shifts = sameStep ? previous.shifts : {
            up: selected.shifts.up && this.shiftsAtGear(this.measuredGear).up,
            down: selected.shifts.down,
        };
        const shift = sameStep ? 0 : selected.shift === 1 && shifts.up ? 1 : selected.shift === 2 && shifts.down ? 2 : 0;
        if (shift && (this.state.action.shiftUp || this.state.action.shiftDown)) {
            if (!await this.releasePaddles(lease) || plan !== this.activePlan) return false;
        }
        const action = { pedal, steering, shiftUp: shift === 1, shiftDown: shift === 2 };
        const outgoing = this.plans[this.plans.length - 1];
        const previousCommand = previous ?? outgoing?.executions[outgoing.executions.length - 1];
        // Keep the vJoy pulse alive while recording only the new press for learning.
        const controls = { ...drivingControllerAction(action),
            shiftUp: Boolean(this.state.action.shiftUp || action.shiftUp),
            shiftDown: Boolean(this.state.action.shiftDown || action.shiftDown) };
        if (!await lease.apply(controls) || plan !== this.activePlan) return false;
        const appliedAt = this.now();
        const controlChanges = previousCommand ? this.controlChanges.record(drivingControllerAction(previousCommand.action),
            controls, appliedAt - previousCommand.appliedAt) : NEUTRAL;
        if (previousCommand?.appliedAt === appliedAt) {
            // Only the final command acknowledged in a tick receives duration credit.
            controlChanges.steering += previousCommand.controlChanges.steering;
            controlChanges.throttle += previousCommand.controlChanges.throttle;
        }
        if (!plan.executions.length) plan.startedAt = appliedAt;
        plan.executions.push({ step, appliedAt, shift, shifts, action, nextWeight, controlChanges, shiftWeight: sameStep ? 0 : 1 });
        if (shift) this.shiftReleaseAt = appliedAt + this.config.actions.shiftPulseMs;
        this.state = { ...this.state, action: controls };
        this.publish(this.state);
        return true;
    }

    /** Attribute each observed interval to acknowledged controls, including an old
     * plan's tail while its replacement was being computed. */
    private planCredits(from: number, until: number) {
        return this.plans.flatMap((plan) => {
            const intervals = plan.executions.map((execution, i) => {
                const end = plan.executions[i + 1]?.appliedAt ?? plan.endedAt ?? until;
                const startedAt = Math.max(from, execution.appliedAt);
                const duration = Math.max(0, Math.min(end, until) - startedAt);
                return { execution, startedAt, duration };
            }).filter(({ duration }) => duration > 0);
            return intervals.map(({ execution, startedAt, duration }) => {
                const { step, shifts, appliedAt, nextWeight, shiftWeight } = execution;
                const changed = appliedAt >= from;
                // A command can span several telemetry intervals. Credit its gear choice only at acknowledgement.
                const shift = changed ? execution.shift : 0;
                const action = shift === execution.shift ? execution.action : { ...execution.action, shiftUp: false, shiftDown: false };
                // Each acknowledged command has its own consequence clock.
                const credits: PlanStepCredit[] = [{ step, shift, shifts, action, appliedAt, weight: 1 - nextWeight,
                    shiftWeight: changed ? shiftWeight : 0 },
                    ...(nextWeight > 0 ? [{ step: step + 1, shift: 0 as const, shifts: { up: false, down: false },
                        action, appliedAt, weight: nextWeight, shiftWeight: 0 }] : [])];
                return { plan, startedAt, duration, credits, controlChanges: changed ? execution.controlChanges : NEUTRAL };
            });
        });
    }

    /** Only one model operation runs at a time; controller polling never awaits it. */
    private async updateModel(policy: DrivingPolicy, sample: TrainingSample, lease: ControlLease) {
        this.modelLease = lease;
        const generation = this.generation, revision = this.modelRevision;
        const active = () => generation === this.generation && lease === this.lease && revision === this.modelRevision;
        const learningRevision = this.learningRevision;
        const learningActive = () => generation === this.generation && learningRevision === this.learningRevision;
        const timing = this.modelTiming = new DrivingModelTimer(sample.observation.capturedAt, 'ai', this.state.mode === 'training');
        try {
            timing.historyLength = 0;
            const slotInputs = drivingMemorySlotInputs(this.config);
            for (let bin = 0; bin < drivingMemorySlots(this.config); bin++) {
                if (sample.input.features[bin * slotInputs]) timing.historyLength++;
            }
            if (!active()) return;
            // Refresh control before learning from earlier executed actions.
            sample = this.previous ?? sample;
            this.lastInferenceFrame = sample.observation.capturedAt;
            timing.start('inference');
            const decision = await policy.decideAsync(sample.input, this.state.mode === 'training', this.shiftsAtGear(this.measuredGear));
            if (!active()) { policy.releaseDecision(decision); return; }
            this.installingPlan = true;
            timing.start('control');
            try {
                // Finish an acknowledged plan step before checking its remaining tail.
                await this.planWrite;
                if (!active()) { policy.releaseDecision(decision); return; }
                const outgoing = this.activePlan;
                // Plan zero is its acknowledged execution start. The candidate
                // would start now; the outgoing plan keeps its original clock.
                const elapsed = outgoing ? Math.max(0, (this.now() - outgoing.startedAt) / 1000) : 0;
                // The old per-step tolerance is kept in checkpoints for compatibility,
                // but refresh thresholds are a runtime rule shared by all models.
                const keep = outgoing && elapsed < DRIVING_PLAN_REFRESH_SECONDS
                    && !drivingPlanControlsDiffer(outgoing.decision.plan, elapsed, decision.plan,
                        outgoing.executions[outgoing.executions.length - 1]?.step);
                if (keep) {
                    // A confirming inference did not execute: keep credit and
                    // sampled exploration attached to the original decision.
                    policy.releaseDecision(decision);
                } else {
                    const plan: ExecutedPlan = { decision, startedAt: this.now(), executions: [] };
                    this.activePlan = plan;
                    if (!await this.applyPlan(plan, lease) || !active()) { policy.releaseDecision(decision); return; }
                    if (outgoing) outgoing.endedAt = plan.startedAt;
                    this.plans.push(plan);
                    this.state = { ...this.state, plan: decision.plan.map(({ timeSeconds, action }) =>
                        ({ timeSeconds, action: { ...action } })) };
                }
            } catch (error) { policy.releaseDecision(decision); throw error; }
            finally { this.installingPlan = false; }
            timing.end();
            this.state = { ...this.state, status: this.driverStatus() };
            this.publish(this.state);
            // The controller continues advancing the selected plan while this
            // bounded update consumes a previous action's rolling consequences.
            const batch = this.state.mode === 'training' ? this.consequences.take() : undefined;
            if (batch) {
                timing.start('learning');
                await this.trainConsequences(policy, batch, learningActive, () => timing.start('learningWait'));
                timing.end();
            }
            if (active()) timing.finish(true);
        } catch (error) {
            if (active() || (learningActive() && (this.handoffWork || this.state.mode === 'finishing'))) {
                await this.stop(error instanceof Error ? error.message : 'Training failed.');
            }
        } finally { timing.finish(false); this.modelLease = undefined; this.installingPlan = false; }
    }

    private async trainConsequences(policy: DrivingPolicy, batch: DrivingConsequenceBatch, active: () => boolean, onSubmitted?: () => void) {
        const decisions = new Set(batch.sequence.flatMap(({ samples }) => samples.map(({ decision }) => decision)));
        decisions.forEach((decision) => this.learningDecisions.add(decision));
        try {
            await policy.trainBatchAsync(() => policy.learnSequenceAsync(batch.sequence, batch.input, active), onSubmitted);
            if (active()) {
                this.state = { ...this.state, steps: policy.steps };
                this.publish(this.state);
            }
        } finally {
            decisions.forEach((decision) => {
                this.learningDecisions.delete(decision);
                policy.releaseDecision(decision);
            });
        }
    }

    private collectConsequences(policy: DrivingPolicy, observation: DrivingObservation) {
        const previous = this.previous;
        const input = this.memory.encode(observation);
        this.previous = { observation, input };
        this.lastFrame = observation.capturedAt;
        if (!previous) return;
        const elapsed = observation.measuredAt - previous.observation.measuredAt;
        if (elapsed > this.config.runtime.maxTransitionSeconds * 1000
            || observation.capturedAt - previous.observation.capturedAt > this.config.runtime.maxTransitionSeconds * 1000) {
            this.modelRevision++;
            this.learningRevision++;
            this.modelTiming?.finish(false);
            this.discardPlans();
            policy.resetExploration();
            this.state = { ...this.state, reward: null, plan: null };
            return;
        }
        const samples = this.planCredits(previous.observation.measuredAt, observation.measuredAt);
        const duration = samples.reduce((sum, sample) => sum + sample.duration, 0);
        const controlChanges = samples.reduce((sum, { controlChanges }) => ({
            steering: sum.steering + controlChanges.steering, throttle: sum.throttle + controlChanges.throttle,
            brake: sum.brake + controlChanges.brake,
        }), { steering: 0, throttle: 0, brake: 0 });
        const reward = this.scoreReward(previous.observation, observation, controlChanges);
        if (samples.length) this.consequences.add({
            startedAt: previous.observation.measuredAt, endedAt: observation.measuredAt, reward: reward.total,
            input: previous.input, next: input, nextShifts: this.shiftsAtGear(observation.gear), terminal: false,
            samples: samples.map(({ plan, startedAt, credits, duration: held }) =>
                ({ decision: plan.decision, startedAt, credits, rewardShare: held / duration })),
        }, input);
        this.state = { ...this.state, reward, episodeReward: this.state.episodeReward + reward.total };
        this.publish(this.state);
        this.plans = this.plans.filter((plan) => {
            if (plan.endedAt !== undefined && plan.endedAt <= observation.measuredAt) {
                if (!this.learningDecisions.has(plan.decision)) policy.releaseDecision(plan.decision);
                return false;
            }
            // Retain the last acknowledged command at the boundary for its held
            // duration, plus newer commands if the observation arrived late.
            const next = plan.executions.findIndex(({ appliedAt }) => appliedAt > observation.measuredAt);
            const keep = next < 0 ? plan.executions.length - 1 : Math.max(0, next - 1);
            plan.executions = plan.executions.slice(keep);
            return true;
        });
    }
}
