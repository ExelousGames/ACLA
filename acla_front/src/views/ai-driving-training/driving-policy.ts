import { clamp, DrivingAction, DrivingObservation } from './driving-observation';

const HIDDEN = 32;
const SIGMA = 0.25;
/** Categorical order: hold gear, shift up, shift down. */
export type ShiftChoice = 0 | 1 | 2;
export interface ShiftAvailability { up: boolean; down: boolean }
export interface PolicyDecision {
    features: number[];
    hidden: number[];
    means: number[];
    latent: number[];
    shift: ShiftChoice;
    shiftProbabilities: number[];
    value: number;
    action: DrivingAction;
}
export interface PolicyCheckpoint {
    version: 3;
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
    steps = 0;

    constructor(readonly context: DrivingObservation['context'], readonly inputs: number,
        private random: () => number = Math.random) {
        this.hiddenWeights = Array.from({ length: HIDDEN }, () => Array.from({ length: inputs + 1 },
            () => (random() * 2 - 1) / Math.sqrt(inputs)));
        this.outputWeights = Array.from({ length: 7 }, (_, i) => Array.from({ length: HIDDEN + 1 },
            (_, j) => j === HIDDEN ? [-2, -3, 0, 0, 1, 0, 0][i] : (random() * 2 - 1) * 0.01));
    }

    matches(observation: DrivingObservation): boolean {
        return observation.features.length === this.inputs && JSON.stringify(this.context) === JSON.stringify(observation.context);
    }

    decide(features: number[], explore: boolean, shifts: ShiftAvailability = { up: true, down: true }): PolicyDecision {
        if (features.length !== this.inputs || !features.every(Number.isFinite)) throw new Error('Invalid policy observation.');
        const dot = (weights: number[], values: number[]) => values.reduce((sum, value, i) => sum + weights[i] * value, weights[values.length]);
        const hidden = this.hiddenWeights.map((weights) => Math.tanh(dot(weights, features)));
        const outputs = this.outputWeights.map((weights) => dot(weights, hidden));
        const means = outputs.slice(0, 3);
        const latent = means.map((mean) => mean + (explore ? SIGMA * Math.sqrt(-2 * Math.log(Math.max(1e-9, this.random())))
            * Math.cos(2 * Math.PI * this.random()) : 0));
        // Mask before sampling and retain these probabilities for the policy gradient.
        // A cooldown-forced hold therefore contributes no shifting gradient.
        const logits = outputs.slice(4).map((logit, i) => [true, shifts.up, shifts.down][i] ? logit : -Infinity);
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
        return { features, hidden, means, latent, shift, shiftProbabilities, value: outputs[3], action };
    }

    learn(decision: PolicyDecision, reward: number, next: number[], elapsedSeconds: number, terminal = false): number {
        if (!Number.isFinite(reward) || !Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) throw new Error('Invalid training transition.');
        const bootstrap = terminal ? 0 : this.decide(next, false).value;
        const advantage = clamp(reward + Math.pow(0.99, elapsedSeconds / 0.2) * bootstrap - decision.value, -5, 5);
        // The action transform is fixed and invertible, so its Jacobian is independent of the policy mean.
        const gradients = decision.latent.map((value, i) => 0.0003 * advantage * clamp((value - decision.means[i]) / SIGMA ** 2, -10, 10));
        gradients.push(0.001 * advantage);
        gradients.push(...decision.shiftProbabilities.map((probability, i) =>
            0.0003 * advantage * ((i === decision.shift ? 1 : 0) - probability)));
        const hiddenGradient = decision.hidden.map((value, j) => (1 - value ** 2)
            * gradients.reduce((sum, gradient, i) => sum + gradient * this.outputWeights[i][j], 0));
        this.outputWeights.forEach((row, i) => row.forEach((value, j) => {
            row[j] = clamp(value + gradients[i] * (j === HIDDEN ? 1 : decision.hidden[j]), -5, 5);
        }));
        this.hiddenWeights.forEach((row, i) => row.forEach((value, j) => {
            row[j] = clamp(value + hiddenGradient[i] * (j === this.inputs ? 1 : decision.features[j]), -5, 5);
        }));
        this.steps++;
        return advantage;
    }

    checkpoint(): PolicyCheckpoint {
        return { version: 3, algorithm: 'online-actor-critic', context: this.context, inputs: this.inputs, steps: this.steps,
            hiddenWeights: this.hiddenWeights.map((row) => row.slice()), outputWeights: this.outputWeights.map((row) => row.slice()) };
    }

    static restore(value: unknown): DrivingPolicy {
        const data = value as PolicyCheckpoint;
        const matrix = (rows: unknown, height: number, width: number) => Array.isArray(rows) && rows.length === height
            && rows.every((row) => Array.isArray(row) && row.length === width && row.every((v) => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 5));
        if (!data || data.version !== 3 || data.algorithm !== 'online-actor-critic'
            || !Number.isInteger(data.inputs) || data.inputs < 1 || data.inputs > 1200
            || !Number.isSafeInteger(data.steps) || data.steps < 0
            || !data.context || !['game', 'car', 'map'].every((key) => typeof (data.context as any)[key] === 'string' && (data.context as any)[key].length > 0)
            || !Array.isArray(data.context.classes) || !data.context.classes.length || data.context.classes.length > 64
            || data.context.classes.some((name) => typeof name !== 'string' || !name)
            || new Set(data.context.classes).size !== data.context.classes.length
            || data.inputs !== 71 + data.context.classes.length * 16
            || !matrix(data.hiddenWeights, HIDDEN, data.inputs + 1) || !matrix(data.outputWeights, 7, HIDDEN + 1)) {
            throw new Error('Invalid or incompatible self-driving model checkpoint.');
        }
        const policy = new DrivingPolicy(data.context, data.inputs);
        policy.hiddenWeights = data.hiddenWeights.map((row) => row.slice());
        policy.outputWeights = data.outputWeights.map((row) => row.slice());
        policy.steps = data.steps;
        return policy;
    }
}
