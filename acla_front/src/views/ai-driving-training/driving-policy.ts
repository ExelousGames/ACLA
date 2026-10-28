import { clamp, DrivingAction, DrivingObservation } from './driving-observation';
import { DRIVING_MEMORY_INPUTS } from './driving-memory';

const HIDDEN = 32;
const SIGMA = 0.25;
export const DRIVING_PLAN_TIMES = [0, 0.5, 1, 1.5, 2, 2.5, 3] as const;
const ACTION_OUTPUTS = 6; // Throttle, brake, steering, hold, up, down per time step.
const CRITIC_OUTPUT = DRIVING_PLAN_TIMES.length * ACTION_OUTPUTS;
/** Categorical order: hold gear, shift up, shift down. */
export type ShiftChoice = 0 | 1 | 2;
export interface ShiftAvailability { up: boolean; down: boolean }
export interface PlannedAction {
    timeSeconds: number;
    action: DrivingAction;
}
interface ActionDecision extends PlannedAction {
    means: number[];
    latent: number[];
    shift: ShiftChoice;
    shiftProbabilities: number[];
}
export interface PolicyDecision {
    features: number[];
    hidden: number[];
    value: number;
    plan: ActionDecision[];
}
export interface PlannedActionTarget {
    step: number;
    action: DrivingAction;
    shifts: ShiftAvailability;
}
export interface PolicyCheckpoint {
    /** Version 10 uses ACC's swapped Y/Z motion inputs, including the three-second history. */
    version: 10;
    algorithm: 'online-actor-critic';
    context: DrivingObservation['context'];
    inputs: number;
    steps: number;
    hiddenWeights: number[][];
    outputWeights: number[][];
}

/** Gaussian axes, categorical shifting and a TD(0) critic share a trainable tanh hidden layer. */
export class DrivingPolicy {
    private hiddenWeights: number[][];
    private outputWeights: number[][];
    readonly inputs: number;
    steps = 0;

    constructor(readonly context: DrivingObservation['context'], readonly observationInputs: number,
        private random: () => number = Math.random) {
        const inputs = this.inputs = observationInputs + DRIVING_MEMORY_INPUTS;
        this.hiddenWeights = Array.from({ length: HIDDEN }, () => Array.from({ length: inputs + 1 },
            () => (random() * 2 - 1) / Math.sqrt(inputs)));
        this.outputWeights = Array.from({ length: CRITIC_OUTPUT + 1 }, (_, i) => Array.from({ length: HIDDEN + 1 },
            (_, j) => j === HIDDEN ? (i === CRITIC_OUTPUT ? 0 : [-2, -3, 0, 1, 0, 0][i % ACTION_OUTPUTS])
                : (random() * 2 - 1) * 0.01));
    }

    matches(observation: DrivingObservation): boolean {
        return this.inputMismatches(observation).length === 0;
    }

    inputMismatches(observation: DrivingObservation): string[] {
        const differences: string[] = [];
        // Compare values, not JSON object key order. Class order defines feature channels.
        for (const [key, label] of [['game', 'Game'], ['car', 'Car'], ['map', 'Track']] as const) {
            if (this.context[key] !== observation.context[key]) {
                differences.push(`${label}: saved “${this.context[key]}”; live “${observation.context[key]}”.`);
            }
        }
        if (this.context.classes.length !== observation.context.classes.length
            || this.context.classes.some((name, i) => name !== observation.context.classes[i])) {
            differences.push(`Segmentation labels (in channel order): saved ${JSON.stringify(this.context.classes)}; live ${JSON.stringify(observation.context.classes)}.`);
        }
        if (this.observationInputs !== observation.features.length) {
            differences.push(`Input count: saved ${this.observationInputs}; live ${observation.features.length}.`);
        }
        return differences;
    }

    decide(features: number[], explore: boolean, shifts: ShiftAvailability = { up: true, down: true }): PolicyDecision {
        if (features.length !== this.inputs || !features.every(Number.isFinite)) throw new Error('Invalid policy observation.');
        const dot = (weights: number[], values: number[]) => values.reduce((sum, value, i) => sum + weights[i] * value, weights[values.length]);
        const hidden = this.hiddenWeights.map((weights) => Math.tanh(dot(weights, features)));
        const outputs = this.outputWeights.map((weights) => dot(weights, hidden));
        const plan = DRIVING_PLAN_TIMES.map((timeSeconds, step) => this.actionDecision(
            outputs.slice(step * ACTION_OUTPUTS, (step + 1) * ACTION_OUTPUTS), timeSeconds,
            explore && step === 0, step === 0 ? shifts : { up: true, down: true }));
        return { features, hidden, value: outputs[CRITIC_OUTPUT], plan };
    }

    private actionDecision(outputs: number[], timeSeconds: number, explore: boolean, shifts: ShiftAvailability): ActionDecision {
        const means = outputs.slice(0, 3);
        const latent = means.map((mean) => mean + (explore ? SIGMA * Math.sqrt(-2 * Math.log(Math.max(1e-9, this.random())))
            * Math.cos(2 * Math.PI * this.random()) : 0));
        // Mask before sampling and retain these probabilities for the policy gradient.
        // A cooldown-forced hold therefore contributes no shifting gradient.
        const logits = outputs.slice(3).map((logit, i) => [true, shifts.up, shifts.down][i] ? logit : -Infinity);
        const maximum = Math.max(...logits);
        const weights = logits.map((logit) => Math.exp(logit - maximum));
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        const shiftProbabilities = weights.map((weight) => weight / total);
        let shift: ShiftChoice = 0;
        if (explore) {
            const draw = this.random();
            let cumulative = 0;
            for (let i = 0; i < shiftProbabilities.length; i++) {
                cumulative += shiftProbabilities[i];
                if (shiftProbabilities[i] > 0) shift = i as ShiftChoice;
                if (draw < cumulative) break;
            }
        } else shift = shiftProbabilities.indexOf(Math.max(...shiftProbabilities)) as ShiftChoice;
        const action = { throttle: 1 / (1 + Math.exp(-latent[0])), brake: 1 / (1 + Math.exp(-latent[1])),
            steering: Math.tanh(latent[2]), shiftUp: shift === 1, shiftDown: shift === 2 };
        return { timeSeconds, means, latent, shift, shiftProbabilities, action };
    }

    learn(decision: PolicyDecision, reward: number, next: number[], elapsedSeconds: number, terminal = false): number {
        if (!Number.isFinite(reward) || !Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) throw new Error('Invalid training transition.');
        const bootstrap = terminal ? 0 : this.decide(next, false).value;
        const advantage = clamp(reward + Math.pow(0.99, elapsedSeconds / 0.2) * bootstrap - decision.value, -5, 5);
        // The action transform is fixed and invertible, so its Jacobian is independent of the policy mean.
        // Only the first action was executed; later actions must not receive its reward gradient.
        const current = decision.plan[0];
        const gradients = new Array(CRITIC_OUTPUT + 1).fill(0);
        current.latent.forEach((value, i) => {
            gradients[i] = 0.0003 * advantage * clamp((value - current.means[i]) / SIGMA ** 2, -10, 10);
        });
        current.shiftProbabilities.forEach((probability, i) => {
            gradients[3 + i] = 0.0003 * advantage * ((i === current.shift ? 1 : 0) - probability);
        });
        gradients[CRITIC_OUTPUT] = 0.001 * advantage;
        this.update(decision, gradients);
        return advantage;
    }

    /** Behavior cloning uses the applied human action as a target, never as another input.
     * Human actions are off-policy: do not apply the on-policy reward gradient to them. */
    learnDemonstration(features: number[], action: DrivingAction, shifts: ShiftAvailability): void {
        this.learnTargets(features, [{ step: 0, action, shifts }], true);
    }

    /** Delayed, acknowledged actions supervise the remaining slots of the same plan.
     * This distills subsequent AI behavior or human demonstrations without treating
     * unexecuted plans as on-policy samples. Steps count completed transitions only. */
    learnPlannedActions(features: number[], targets: PlannedActionTarget[]): void {
        this.learnTargets(features, targets, false);
    }

    private learnTargets(features: number[], targets: PlannedActionTarget[], countStep: boolean): void {
        if (!targets.length) return;
        const decision = this.decide(features, false);
        const gradients = new Array(CRITIC_OUTPUT + 1).fill(0);
        for (const { step, action, shifts } of targets) {
            if (!Number.isInteger(step) || step < 0 || step >= DRIVING_PLAN_TIMES.length) throw new Error('Invalid action plan step.');
            const current = decision.plan[step];
            // Apply the availability from the target's execution time, not the original frame.
            const masked = current.shiftProbabilities.map((p, i) => [true, shifts.up, shifts.down][i] ? p : 0);
            const total = masked.reduce((sum, p) => sum + p, 0);
            const targetGradients = this.demonstrationGradients(current.means, masked.map((p) => p / total), action);
            targetGradients.forEach((gradient, i) => { gradients[step * ACTION_OUTPUTS + i] = gradient; });
        }
        this.update(decision, gradients, countStep);
    }

    private demonstrationGradients(means: number[], shiftProbabilities: number[], action: DrivingAction): number[] {
        if (![action.throttle, action.brake, action.steering].every(Number.isFinite)
            || action.throttle < 0 || action.throttle > 1 || action.brake < 0 || action.brake > 1
            || Math.abs(action.steering) > 1 || (action.shiftUp && action.shiftDown)) {
            throw new Error('Invalid demonstration action.');
        }
        const logit = (value: number) => { const p = clamp(value, 0.001, 0.999); return Math.log(p / (1 - p)); };
        const targets = [logit(action.throttle), logit(action.brake), Math.atanh(clamp(action.steering, -0.999, 0.999))];
        const gradients = targets.map((target, i) => 0.001 * clamp(target - means[i], -5, 5));
        const shift = action.shiftUp ? 1 : action.shiftDown ? 2 : 0;
        gradients.push(...shiftProbabilities.map((probability, i) =>
            0.001 * ((i === shift ? 1 : 0) - probability)));
        return gradients;
    }

    private update(decision: PolicyDecision, gradients: number[], countStep = true): void {
        const hiddenGradient = decision.hidden.map((value, j) => (1 - value ** 2)
            * gradients.reduce((sum, gradient, i) => sum + gradient * this.outputWeights[i][j], 0));
        this.outputWeights.forEach((row, i) => row.forEach((value, j) => {
            row[j] = clamp(value + gradients[i] * (j === HIDDEN ? 1 : decision.hidden[j]), -5, 5);
        }));
        this.hiddenWeights.forEach((row, i) => row.forEach((value, j) => {
            row[j] = clamp(value + hiddenGradient[i] * (j === this.inputs ? 1 : decision.features[j]), -5, 5);
        }));
        if (countStep) this.steps++;
    }

    checkpoint(): PolicyCheckpoint {
        return { version: 10, algorithm: 'online-actor-critic', context: this.context, inputs: this.inputs, steps: this.steps,
            hiddenWeights: this.hiddenWeights.map((row) => row.slice()), outputWeights: this.outputWeights.map((row) => row.slice()) };
    }

    static restore(value: unknown): DrivingPolicy {
        const data = value as PolicyCheckpoint;
        const matrix = (rows: unknown, height: number, width: number) => Array.isArray(rows) && rows.length === height
            && rows.every((row) => Array.isArray(row) && row.length === width && row.every((v) => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 5));
        if (!data || data.version !== 10 || data.algorithm !== 'online-actor-critic'
            || !Number.isInteger(data.inputs) || data.inputs < 1 || data.inputs > 1200 + DRIVING_MEMORY_INPUTS
            || !Number.isSafeInteger(data.steps) || data.steps < 0
            || !data.context || !['game', 'car', 'map'].every((key) => typeof (data.context as any)[key] === 'string' && (data.context as any)[key].length > 0)
            || !Array.isArray(data.context.classes) || !data.context.classes.length || data.context.classes.length > 64
            || data.context.classes.some((name) => typeof name !== 'string' || !name)
            || new Set(data.context.classes).size !== data.context.classes.length
            || data.inputs !== 68 + data.context.classes.length * 16 + DRIVING_MEMORY_INPUTS
            || !matrix(data.hiddenWeights, HIDDEN, data.inputs + 1) || !matrix(data.outputWeights, CRITIC_OUTPUT + 1, HIDDEN + 1)) {
            throw new Error('Invalid or incompatible self-driving model checkpoint.');
        }
        const policy = new DrivingPolicy(data.context, data.inputs - DRIVING_MEMORY_INPUTS);
        policy.hiddenWeights = data.hiddenWeights.map((row) => row.slice());
        policy.outputWeights = data.outputWeights.map((row) => row.slice());
        policy.steps = data.steps;
        return policy;
    }
}
