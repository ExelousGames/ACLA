import { DrivingAction, DrivingObservation, DrivingReward, drivingReward, MAX_TRAINING_TRANSITION_SECONDS,
    NEUTRAL, ObservationResult } from './driving-observation';
import { DrivingPolicy, PlannedAction, PolicyDecision, ShiftAvailability } from './driving-policy';
import { ControlSource, DrivingControlSelector, ManualControls } from './driving-control';
import { ControlLease, DrivingControlOutput } from './driving-control-output';
import { DrivingMemory } from './driving-memory';
import { DrivingPlanTraining } from './driving-plan-training';

const SHIFT_PULSE_MS = 100;
const SHIFT_COOLDOWN_MS = 500;

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

export class DrivingTrainingLoop {
    state = { ...initialTrainingState };
    private previous?: { observation: DrivingObservation; decision?: PolicyDecision; action: DrivingAction;
        shifts: ShiftAvailability; source: ControlSource; appliedAt: number };
    private boundary?: DrivingObservation;
    private selector = new DrivingControlSelector();
    private controlChangedAt = -Infinity;
    private lastFrame = -Infinity;
    private generation = 0;
    private busy = false;
    private timer?: ReturnType<typeof setInterval>;
    private shiftReleaseAt = 0;
    private nextShiftAt = 0;
    private lease?: ControlLease;
    private policy?: DrivingPolicy;
    private memory = new DrivingMemory();
    private planTraining = new DrivingPlanTraining();

    constructor(private output: DrivingControlOutput, private read: () => ObservationResult,
        private publish: (state: TrainingState) => void, private now = Date.now,
        private readManual?: () => Promise<ManualControls | null>) {}

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

    start(policy: DrivingPolicy, train: boolean) {
        const input = this.read();
        if (!input.observation) throw new Error(input.error);
        if (!policy.matches(input.observation)) throw new Error('Model inputs differ from this game, car, map or segmentation model. Create a new model.');
        this.begin(policy, train, input.observation);
    }

    startManual() { this.begin(); }

    private begin(policy?: DrivingPolicy, train = false, boundary?: DrivingObservation) {
        const humanDriving = this.state.mode === 'manual';
        if ((!humanDriving && (this.state.mode !== 'idle' || this.busy)) || (humanDriving && !policy)) {
            throw new Error('Stop the current session before starting another.');
        }
        clearInterval(this.timer);
        this.policy = policy;
        this.memory.reset();
        this.planTraining.reset();
        this.previous = undefined;
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
        this.timer = setInterval(() => { void this.tick(policy); }, this.readManual ? 50 : 100);
        void this.tick(policy);
    }

    async stop(status = 'Stopped. Controller released.'): Promise<void> {
        const generation = ++this.generation;
        clearInterval(this.timer);
        this.previous = undefined;
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

    private changeSource(source: ControlSource) {
        if (source === this.state.source) return;
        this.selector.source = source;
        this.lease = this.output.acquire(source);
        // No reward, held paddle or demonstration may cross an ownership boundary.
        this.previous = undefined;
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
        if (!this.policy?.matches(input.observation)) return 'Game, car, map or segmentation model changed. Waiting for matching model inputs.';
        if (!input.observation.features.every(Number.isFinite)) return 'Invalid policy observation.';
        if (input.observation.session !== this.boundary?.session) return 'Live session changed. Waiting for the original live session.';
        if (input.observation.reference !== this.boundary?.reference) return 'Live Map reference changed. Waiting for the original reference.';
        if (this.previous) {
            const elapsedSeconds = (input.observation.capturedAt - this.previous.observation.capturedAt) / 1000;
            const progress = Math.abs(input.observation.normalizedPosition - this.previous.observation.normalizedPosition);
            if (elapsedSeconds > 0 && elapsedSeconds <= MAX_TRAINING_TRANSITION_SECONDS && Math.min(progress, 1 - progress) > 0.1) {
                return 'Lap position jumped. Waiting for fresh telemetry.';
            }
        }
    }

    private async tick(policy?: DrivingPolicy) {
        if (this.busy || this.state.mode === 'idle') return;
        this.busy = true;
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
                this.previous = undefined;
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
                    this.shiftReleaseAt = appliedAt + SHIFT_PULSE_MS;
                    this.nextShiftAt = appliedAt + SHIFT_COOLDOWN_MS;
                }
                if (fresh && observation && policy) {
                    observation.features = this.memory.encode(observation);
                    if (previous) this.learnPrevious(policy, observation);
                    this.previous = { observation, action, source: 'manual', shifts: { up: canShift, down: canShift }, appliedAt };
                    this.lastFrame = observation.capturedAt;
                } else if (previous?.source === 'manual' && canShift && (action.shiftUp || action.shiftDown)) {
                    // Retain a brief paddle press between images as the demonstration for
                    // this frame. Use the whole acknowledged action, not a synthetic mix.
                    this.previous = { ...previous, action, shifts: { up: true, down: true }, appliedAt };
                }
                this.state = { ...this.state, action, status: inputError ? this.state.status : this.driverStatus() };
                this.publish(this.state);
                return;
            }
            // Refresh the hardware watchdog, but never learn twice from the same image or from a capture predating the action.
            if (!fresh) {
                if (!await lease.apply(this.state.action)) return;
                if (generation === this.generation) this.publish(this.state);
                return;
            }
            if (!observation || !policy) return;
            observation.features = this.memory.encode(observation);
            if (previous) this.learnPrevious(policy, observation);
            const canShift = this.now() >= this.nextShiftAt;
            const decision = policy.decide(observation.features, this.state.mode === 'training', {
                up: canShift, down: canShift,
            });
            const action = decision.plan[0].action;
            if (!await lease.apply(action) || generation !== this.generation) return;
            const appliedAt = this.now();
            if (action.shiftUp || action.shiftDown) {
                this.shiftReleaseAt = appliedAt + SHIFT_PULSE_MS;
                this.nextShiftAt = appliedAt + SHIFT_COOLDOWN_MS;
            }
            this.previous = { observation, decision, action, source: 'ai',
                shifts: { up: canShift, down: canShift }, appliedAt };
            this.lastFrame = observation.capturedAt;
            this.state = { ...this.state, action, plan: decision.plan.map(({ timeSeconds, action: planned }) =>
                ({ timeSeconds, action: { ...planned } })), status: this.driverStatus() };
            this.publish(this.state);
        } catch (error) {
            if (generation === this.generation) await this.stop(error instanceof Error ? error.message : 'Training failed.');
        } finally { this.busy = false; }
    }

    private learnPrevious(policy: DrivingPolicy, observation: DrivingObservation) {
        const previous = this.previous!;
        const elapsedSeconds = (observation.capturedAt - previous.observation.capturedAt) / 1000;
        // Gaps between usable frames are recoverable after input validation.
        // The caller starts the next transition from this fresh observation.
        if (elapsedSeconds > MAX_TRAINING_TRANSITION_SECONDS) {
            this.planTraining.reset();
            this.state = { ...this.state, reward: null };
            return;
        }
        const reward = drivingReward(previous.observation, observation, previous.action);
        if (this.state.mode === 'training') {
            if (previous.source === 'manual') policy.learnDemonstration(previous.observation.features, previous.action, previous.shifts);
            else policy.learn(previous.decision!, reward.total, observation.features, elapsedSeconds);
            this.planTraining.record(policy, previous.observation.features, previous.action, previous.shifts, previous.appliedAt);
        }
        this.state = { ...this.state, steps: policy.steps, reward, episodeReward: this.state.episodeReward + reward.total };
    }
}
