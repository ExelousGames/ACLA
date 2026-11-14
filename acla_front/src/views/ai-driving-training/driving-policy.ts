import { DrivingAction, DrivingObservation } from './driving-observation';
import { DrivingPolicyInput } from './driving-memory';
import { cloneConfig, DEFAULT_DRIVING_CONFIG, DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingFeatureEncoding, drivingObservationInputs, drivingPolicyInputs, upgradeDrivingConfig, validateDrivingConfig } from './driving-model-config';
import { DrivingGpuModel, GpuPolicyPass } from './driving-gpu-model';
import { DrivingRolloutInterval, drivingSequenceTargets } from './driving-rollout';

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
    shifts: ShiftAvailability;
}
export interface PolicyDecision {
    gpu: GpuPolicyPass;
    input: DrivingPolicyInput;
    value: number;
    plan: ActionDecision[];
}
export interface PlannedActionTarget {
    step: number;
    action: DrivingAction;
    shifts: ShiftAvailability;
}
export interface PlanStepCredit {
    step: number;
    weight: number;
    /** Interpolated analog endpoints can contribute without executing a gear choice. */
    shiftWeight?: number;
    shift: ShiftChoice;
    shifts: ShiftAvailability;
    action?: DrivingAction;
    appliedAt?: number;
}
export interface PolicyModelState {
    /** Creation context retained for checkpoint compatibility; not a restriction on live inputs. */
    context: DrivingObservation['context'];
    inputs: number;
    steps: number;
    cameraWeights: number[][][];
    cameraProjectionWeights: number[][];
    lstmWeights: number[][];
    outputWeights: number[][];
}

export interface DrivingTrainingDocument {
    format: 'kestrel-self-driving-model';
    algorithm: 'online-actor-critic';
    configuration: DrivingModelConfig;
    model: PolicyModelState | null;
}
export interface PolicyCheckpoint extends DrivingTrainingDocument { model: PolicyModelState }

export function emptyTrainingDocument(configuration = DEFAULT_DRIVING_CONFIG): DrivingTrainingDocument {
    return { format: 'kestrel-self-driving-model', algorithm: 'online-actor-critic', configuration: cloneConfig(configuration), model: null };
}

export function readTrainingDocument(value: unknown): DrivingTrainingDocument {
    const data = value as DrivingTrainingDocument;
    if (!data || data.format !== 'kestrel-self-driving-model' || data.algorithm !== 'online-actor-critic'
        || Object.keys(data).some((key) => !['format', 'algorithm', 'configuration', 'model'].includes(key))
        || !Object.prototype.hasOwnProperty.call(data, 'model')) throw new Error('Invalid or incompatible self-driving model checkpoint.');
    const configuration = upgradeDrivingConfig(data.configuration);
    validateDrivingConfig(configuration);
    if (data.model !== null) return DrivingPolicy.restore(data).checkpoint();
    return emptyTrainingDocument(configuration);
}

/** Gaussian axes, categorical shifting and a multi-step critic share a CNN and temporal LSTM. */
export class DrivingPolicy {
    private initialWeights?: Pick<PolicyModelState, 'cameraWeights' | 'cameraProjectionWeights' | 'lstmWeights' | 'outputWeights'>;
    private gpu?: DrivingGpuModel;
    private gpuInitialization?: Promise<void>;
    private gpuBatch?: Promise<void>;
    private disposed = false;
    readonly inputs: number;
    steps = 0;
    readonly configuration: DrivingModelConfig;
    private get actionOutputs() { return this.configuration.actions.outputs.length; }
    private get criticOutput() { return this.planTimes.length * this.actionOutputs; }
    get planTimes() { return this.configuration.actions.planTimesSeconds; }

    constructor(readonly context: DrivingObservation['context'], readonly observationInputs: number,
        private random: () => number = Math.random, configuration = DEFAULT_DRIVING_CONFIG) {
        validateDrivingConfig(configuration);
        this.configuration = cloneConfig(configuration);
        this.context = { game: context.game, car: context.car, map: context.map };
        const inputs = this.inputs = drivingPolicyInputs(this.configuration);
        if (observationInputs !== drivingObservationInputs(this.configuration)
            || !Number.isSafeInteger(inputs) || inputs < 1) {
            throw new Error('The self-driving model input layout is too large or invalid.');
        }
        const units = configuration.network.hiddenUnits, limit = configuration.learning.weightLimit;
        // CPU arrays only seed or restore the GPU model; they never run inference or learning.
        const cameraWeights = drivingCameraLayers(configuration).map((layer) => Array.from({ length: layer.channels }, () =>
            Array.from({ length: layer.columns }, (_, i) => i === layer.columns - 1 ? 0
                : Math.max(-limit, Math.min(limit, (random() * 2 - 1) * Math.sqrt(6 / (layer.columns - 1)))))));
        const pooled = drivingCameraPooledInputs(configuration), projected = drivingCameraInputs(configuration);
        const cameraProjectionWeights = Array.from({ length: projected }, () =>
            Array.from({ length: pooled + 1 }, (_, i) => i === pooled ? 0
                : Math.max(-limit, Math.min(limit, (random() * 2 - 1) * Math.sqrt(6 / (pooled + projected))))));
        const lstmWeights = Array.from({ length: 4 * units }, (_, row) =>
            Array.from({ length: inputs + units + 1 }, (_, column) => {
                const value = column === inputs + units ? Number(row >= units && row < 2 * units)
                    : (random() * 2 - 1) / Math.sqrt(inputs + units);
                return Math.max(-limit, Math.min(limit, value));
            }));
        const outputWeights = Array.from({ length: this.criticOutput + 1 }, (_, i) => Array.from({ length: this.configuration.network.hiddenUnits + 1 },
            (_, j) => j === this.configuration.network.hiddenUnits ? (i === this.criticOutput ? 0 : this.configuration.network.initialActionBias[i % this.actionOutputs])
                : (random() * 2 - 1) * this.configuration.network.initialOutputWeightScale));
        this.initialWeights = { cameraWeights, cameraProjectionWeights, lstmWeights, outputWeights };
    }

    matches(observation: DrivingObservation): boolean {
        return this.inputMismatches(observation).length === 0;
    }

    inputMismatches(observation: DrivingObservation): string[] {
        const differences: string[] = [];
        // Game, car and map are observation features, not model compatibility constraints.
        if (this.observationInputs !== observation.features.length) {
            differences.push(`Input count: saved ${this.observationInputs}; live ${observation.features.length}.`);
        }
        if (observation.featureEncoding !== drivingFeatureEncoding(this.configuration)) {
            differences.push('Input encoding differs from the model JSON. Read live inputs using this model’s configuration.');
        }
        return differences;
    }

    /** Inference and learning always require the hardware GPU. */
    initializeGpu(): Promise<void> {
        if (this.disposed) return Promise.reject(new Error('The driving model has been released.'));
        if (!this.gpuInitialization) this.gpuInitialization = DrivingGpuModel.create(this.configuration, this.checkpoint().model).then((gpu) => {
            if (this.disposed) { gpu.dispose(); throw new Error('The driving model has been released.'); }
            this.gpu = gpu;
            this.initialWeights = undefined;
        }).catch((error) => { this.gpuInitialization = undefined; throw error; });
        return this.gpuInitialization;
    }

    async decideAsync(input: DrivingPolicyInput, explore: boolean, shifts: ShiftAvailability = { up: true, down: true }): Promise<PolicyDecision> {
        const { pass, outputs } = await this.requireGpu().forward(input);
        const plan = this.planTimes.map((timeSeconds, step) => this.actionDecision(
            outputs.slice(step * this.actionOutputs, (step + 1) * this.actionOutputs), timeSeconds,
            explore, step === 0 ? shifts : { up: true, down: true }));
        return { input, gpu: pass, value: outputs[this.criticOutput], plan };
    }

    private requireGpu(): DrivingGpuModel {
        if (this.disposed) throw new Error('The driving model has been released.');
        if (!this.gpu) throw new Error('Initialize the training GPU before prediction or learning.');
        return this.gpu;
    }

    releaseDecision(decision?: PolicyDecision) { if (decision?.gpu) this.gpu?.release(decision.gpu); }

    private actionDecision(outputs: number[], timeSeconds: number, explore: boolean, shifts: ShiftAvailability): ActionDecision {
        const means = outputs.slice(0, 2);
        const latent = means.map((mean) => mean + (explore ? this.configuration.learning.explorationSigma * Math.sqrt(-2 * Math.log(Math.max(1e-9, this.random())))
            * Math.cos(2 * Math.PI * this.random()) : 0));
        // Mask before sampling. GPU learning reuses the execution-time mask
        // with the retained GPU logits to calculate the policy gradient.
        const shiftLogits = outputs.slice(2);
        const shiftProbabilities = this.shiftProbabilities(shiftLogits, shifts);
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
        const pedal = Math.tanh(latent[0]);
        const action = { throttle: Math.max(0, pedal), brake: Math.max(0, -pedal),
            steering: Math.tanh(latent[1]), shiftUp: shift === 1, shiftDown: shift === 2 };
        return { timeSeconds, means, latent, shift, shiftProbabilities, shifts: { ...shifts }, action };
    }

    private shiftProbabilities(values: number[], shifts: ShiftAvailability): number[] {
        const logits = values.map((logit, i) => [true, shifts.up, shifts.down][i] ? logit : -Infinity);
        const maximum = Math.max(...logits);
        const weights = logits.map((logit) => Math.exp(logit - maximum));
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        return weights.map((weight) => weight / total);
    }

    /** Submit learning without reading back the bootstrap value or advantage. */
    async learnAsync(decision: PolicyDecision, reward: number, next: DrivingPolicyInput, elapsedSeconds: number,
        credits: PlanStepCredit[], terminal = false): Promise<void> {
        const gpu = this.requireGpu();
        if (!Number.isFinite(reward) || !Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) throw new Error('Invalid training transition.');
        if (!decision.gpu) throw new Error('Missing GPU training sample.');
        const pass = decision.gpu.released ? gpu.replay(decision.input) : decision.gpu;
        try {
            gpu.learnTransition(pass, next, { reward, elapsedSeconds, terminal,
                actions: credits.map((credit) => ({ ...credit, latent: decision.plan[credit.step].latent })) });
        } finally { if (pass !== decision.gpu) gpu.release(pass); }
        this.steps++;
    }

    /** One pass over a rollout, with all gradients and bootstrap values evaluated
     * on its collection policy. Publish weights only after the whole sequence. */
    async learnSequenceAsync(sequence: DrivingRolloutInterval[], next: DrivingPolicyInput, active = () => true): Promise<void> {
        const gpu = this.requireGpu(), steps = this.steps;
        const targets = drivingSequenceTargets(sequence, this.configuration.learning);
        if (!targets.length || !active()) return;
        gpu.beginSequence(next);
        let commit = false;
        try {
            for (const target of targets) {
                if (!active()) return;
                await this.learnAsync(target.decision, target.reward, next, target.elapsedSeconds, target.credits);
            }
            commit = active();
        } finally {
            gpu.endSequence(commit);
            if (!commit) this.steps = steps;
        }
    }

    async learnDemonstrationAsync(input: DrivingPolicyInput, action: DrivingAction, shifts: ShiftAvailability): Promise<void> {
        await this.learnTargetsAsync(input, [{ step: 0, action, shifts }], true);
    }

    async learnPlannedActionsAsync(input: DrivingPolicyInput, targets: PlannedActionTarget[]): Promise<void> {
        await this.learnTargetsAsync(input, targets, false);
    }

    private async learnTargetsAsync(input: DrivingPolicyInput, targets: PlannedActionTarget[], countStep: boolean) {
        const gpu = this.requireGpu();
        if (!targets.length) return;
        this.validateTargets(targets);
        gpu.learnTargets(input, targets);
        if (countStep) this.steps++;
    }

    private validateTargets(targets: PlannedActionTarget[]) {
        for (const { step, action } of targets) {
            if (!Number.isInteger(step) || step < 0 || step >= this.planTimes.length) throw new Error('Invalid action plan step.');
            if (![action.throttle, action.brake, action.steering].every(Number.isFinite)
                || action.throttle < 0 || action.throttle > 1 || action.brake < 0 || action.brake > 1
                || Math.abs(action.steering) > 1 || (action.shiftUp && action.shiftDown)) {
                throw new Error('Invalid demonstration action.');
            }
        }
    }

    /** Keep a checkpoint from observing only part of a training tick. */
    async trainBatchAsync(train: () => Promise<void>, onSubmitted?: () => void): Promise<void> {
        const gpu = this.requireGpu();
        let finish!: () => void;
        this.gpuBatch = new Promise<void>((resolve) => { finish = resolve; });
        try {
            await train();
            onSubmitted?.();
            await gpu.finishTraining();
        }
        finally { this.gpuBatch = undefined; finish(); }
    }

    /** Serialize seed/restored weights before upload; trained weights require checkpointAsync. */
    checkpoint(): PolicyCheckpoint {
        if (!this.initialWeights) throw new Error('GPU checkpoints must be awaited.');
        return { ...emptyTrainingDocument(this.configuration), model: {
            context: { ...this.context }, inputs: this.inputs, steps: this.steps,
            cameraWeights: this.initialWeights.cameraWeights.map((rows) => rows.map((row) => row.slice())),
            cameraProjectionWeights: this.initialWeights.cameraProjectionWeights.map((row) => row.slice()),
            lstmWeights: this.initialWeights.lstmWeights.map((row) => row.slice()), outputWeights: this.initialWeights.outputWeights.map((row) => row.slice()) } };
    }

    async checkpointAsync(): Promise<PolicyCheckpoint> {
        if (!this.gpu) return this.checkpoint();
        await this.gpuBatch;
        const steps = this.steps, document = emptyTrainingDocument(this.configuration);
        const weights = await this.gpu.weights();
        return { ...document, model: { context: { ...this.context }, inputs: this.inputs, steps, ...weights } };
    }

    dispose() { this.disposed = true; this.gpu?.dispose(); }

    static restore(value: unknown): DrivingPolicy {
        const document = value as PolicyCheckpoint;
        const invalid = () => { throw new Error('Invalid or incompatible self-driving model checkpoint.'); };
        if (!document || document.format !== 'kestrel-self-driving-model' || document.algorithm !== 'online-actor-critic'
            || Object.keys(document).some((key) => !['format', 'algorithm', 'configuration', 'model'].includes(key))) return invalid();
        const config = upgradeDrivingConfig(document.configuration), data = document.model;
        validateDrivingConfig(config);
        const matrix = (rows: unknown, height: number, width: number) => Array.isArray(rows) && rows.length === height
            && rows.every((row) => Array.isArray(row) && row.length === width && row.every((v) => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= config.learning.weightLimit));
        if (!data || Object.keys(data).some((key) => !['context', 'inputs', 'steps', 'cameraWeights', 'cameraProjectionWeights', 'lstmWeights', 'outputWeights'].includes(key))
            || !Number.isInteger(data.inputs) || data.inputs < 1
            || !Number.isSafeInteger(data.steps) || data.steps < 0
            || !data.context || !['game', 'car', 'map'].every((key) => typeof (data.context as any)[key] === 'string' && (data.context as any)[key].length > 0)
            || Object.keys(data.context).some((key) => !['game', 'car', 'map'].includes(key))
            || data.inputs !== drivingPolicyInputs(config)
            || !Array.isArray(data.cameraWeights) || data.cameraWeights.length !== config.network.cameraLayers.length
            || drivingCameraLayers(config).some((layer, i) => !matrix(data.cameraWeights[i], layer.channels, layer.columns))
            || !matrix(data.cameraProjectionWeights, drivingCameraInputs(config), drivingCameraPooledInputs(config) + 1)
            || !matrix(data.lstmWeights, 4 * config.network.hiddenUnits, data.inputs + config.network.hiddenUnits + 1)
            || !matrix(data.outputWeights, config.actions.planTimesSeconds.length * config.actions.outputs.length + 1, config.network.hiddenUnits + 1)) return invalid();
        const policy = new DrivingPolicy(data.context, drivingObservationInputs(config), Math.random, config);
        policy.initialWeights = {
            cameraWeights: data.cameraWeights.map((rows) => rows.map((row) => row.slice())),
            cameraProjectionWeights: data.cameraProjectionWeights.map((row) => row.slice()),
            lstmWeights: data.lstmWeights.map((row) => row.slice()),
            outputWeights: data.outputWeights.map((row) => row.slice()),
        };
        policy.steps = data.steps;
        return policy;
    }
}
