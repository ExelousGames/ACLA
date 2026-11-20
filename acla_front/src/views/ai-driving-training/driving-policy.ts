import { DrivingAction, DrivingObservation } from './driving-observation';
import { DrivingPolicyInput } from './driving-memory';
import { cloneConfig, DEFAULT_DRIVING_CONFIG, DRIVING_ANALOG_AXES, DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingExplorationSigmas, drivingFeatureEncoding, drivingObservationInputs, drivingPolicyInputs, readDrivingConfig, validateDrivingConfig } from './driving-model-config';
import { DrivingExploration } from './driving-exploration';
import { DrivingGpuModel, GpuPolicyPass } from './driving-gpu-model';
import { DrivingRolloutInterval, drivingSequenceTargets } from './driving-rollout';
import type { PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
import type { DrivingCameraInspection } from './driving-camera-inspection';

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
    /** Sampling variances for exploration diagnostics. */
    variances?: number[];
    shift: ShiftChoice;
    shiftProbabilities: number[];
    shifts: ShiftAvailability;
}
interface ExplorationSample { axes: number[]; variances: number[]; shift: number }
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
    /** LayerNorm rows are [gain, bias]; absent for the original unnormalized computation. */
    cameraNormWeights?: number[][];
    lstmNormWeights?: number[][];
    lstmWeights: number[][];
    outputWeights: number[][];
    sac: DrivingSacWeights;
}

export type DrivingEncoderWeights = Pick<PolicyModelState, 'cameraWeights' | 'cameraProjectionWeights'
    | 'cameraNormWeights' | 'lstmNormWeights' | 'lstmWeights'>;
export interface DrivingSacWeights {
    /** Learned log spread at reference sigma 0.05; configured exploration scales it. */
    logStds: number[];
    criticWeights: number[][];
    targetCriticWeights: number[][];
    targetEncoder: DrivingEncoderWeights;
}

interface ReplayTransition {
    input: DrivingPolicyInput;
    originInput: DrivingPolicyInput;
    next: DrivingPolicyInput;
    reward: number;
    elapsedSeconds: number;
    terminal: boolean;
    nextShifts: ShiftAvailability;
    actions: (PlanStepCredit & { action: DrivingAction })[];
}

export interface DrivingTrainingDocument {
    format: 'kestrel-self-driving-model';
    algorithm: 'soft-actor-critic';
    configuration: DrivingModelConfig;
    model: PolicyModelState | null;
}
export interface PolicyCheckpoint extends DrivingTrainingDocument { model: PolicyModelState }
export type PolicyWeights = Pick<PolicyModelState, 'cameraWeights' | 'cameraProjectionWeights' | 'cameraNormWeights'
    | 'lstmNormWeights' | 'lstmWeights' | 'outputWeights' | 'sac'>;

export function emptyTrainingDocument(configuration = DEFAULT_DRIVING_CONFIG): DrivingTrainingDocument {
    return { format: 'kestrel-self-driving-model', algorithm: 'soft-actor-critic', configuration: cloneConfig(configuration), model: null };
}

export function readTrainingDocument(value: unknown): DrivingTrainingDocument {
    const data = value as DrivingTrainingDocument;
    if (!data || data.format !== 'kestrel-self-driving-model' || data.algorithm !== 'soft-actor-critic'
        || Object.keys(data).some((key) => !['format', 'algorithm', 'configuration', 'model'].includes(key))
        || !Object.prototype.hasOwnProperty.call(data, 'model')) throw new Error('Invalid or incompatible self-driving model checkpoint.');
    const configuration = readDrivingConfig(data.configuration);
    if (data.model !== null) return DrivingPolicy.restore({ ...data, configuration }).checkpoint();
    return emptyTrainingDocument(configuration);
}

/** Hybrid SAC with the existing camera/history inputs and signed pedal action plan. */
export class DrivingPolicy {
    private initialWeights?: PolicyWeights;
    private gpu?: DrivingGpuModel;
    private gpuInitialization?: Promise<void>;
    private gpuBatch?: Promise<void>;
    private disposed = false;
    private exploration: DrivingExploration;
    private explorationRevision = 0;
    private replayBuffer: ReplayTransition[] = [];
    readonly inputs: number;
    steps = 0;
    readonly configuration: DrivingModelConfig;
    private get actionOutputs() { return this.configuration.actions.outputs.length; }
    private get criticOutput() { return this.planTimes.length * this.actionOutputs; }
    get planTimes() { return this.configuration.actions.planTimesSeconds; }

    constructor(readonly context: DrivingObservation['context'], readonly observationInputs: number,
        private random: () => number = Math.random, configuration = DEFAULT_DRIVING_CONFIG, restoredWeights?: PolicyWeights) {
        validateDrivingConfig(configuration);
        this.configuration = cloneConfig(configuration);
        this.exploration = new DrivingExploration(this.configuration, () => this.random());
        this.context = { game: context.game, car: context.car, map: context.map };
        const inputs = this.inputs = drivingPolicyInputs(this.configuration);
        if (observationInputs !== drivingObservationInputs(this.configuration)
            || !Number.isSafeInteger(inputs) || inputs < 1) {
            throw new Error('The self-driving model input layout is too large or invalid.');
        }
        // A validated SAC checkpoint already includes online and target weights.
        // Avoid allocating and randomizing a second full model before restoring it.
        if (restoredWeights) { this.initialWeights = restoredWeights; return; }
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
                const value = column === inputs + units ? Number(!configuration.network.lstmLayerNorm && row >= units && row < 2 * units)
                    : (random() * 2 - 1) / Math.sqrt(inputs + units);
                return Math.max(-limit, Math.min(limit, value));
            }));
        // Preserve the public scalar output as zero; SAC values live in private Q networks.
        const outputWeights = Array.from({ length: this.criticOutput + 1 }, (_, i) => Array.from({ length: this.configuration.network.hiddenUnits + 1 },
            (_, j) => i === this.criticOutput ? 0 : j === this.configuration.network.hiddenUnits ? this.configuration.network.initialActionBias[i % this.actionOutputs]
                : (random() * 2 - 1) * this.configuration.network.initialOutputWeightScale));
        const encoder: DrivingEncoderWeights = { cameraWeights, cameraProjectionWeights, lstmWeights };
        if (configuration.network.cameraLayerNorm) {
            encoder.cameraNormWeights = [Array(projected).fill(1), Array(projected).fill(0)];
        }
        if (configuration.network.lstmLayerNorm) {
            encoder.lstmNormWeights = [Array(4 * units).fill(1),
                Array.from({ length: 4 * units }, (_, row) => Number(row >= units && row < 2 * units))];
        }
        const columns = units + this.actionOutputs + 1;
        const criticWeights = Array.from({ length: 2 }, () => Array.from({ length: units * columns + units + 1 }, (_, i) =>
            i < units * columns && i % columns === columns - 1 || i === units * columns + units ? 0
                : Math.max(-limit, Math.min(limit, (random() * 2 - 1) / Math.sqrt(columns - 1)))));
        this.initialWeights = { ...encoder, outputWeights, sac: {
            logStds: new Array(this.planTimes.length * DRIVING_ANALOG_AXES).fill(Math.log(0.05)),
            criticWeights, targetCriticWeights: criticWeights.map((weights) => weights.slice()),
            targetEncoder: JSON.parse(JSON.stringify(encoder)),
        } };
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
        if (!explore) this.resetExploration();
        const revision = this.explorationRevision;
        const { pass, outputs, logStds } = await this.requireGpu().forward(input);
        const samples = explore && revision === this.explorationRevision ? this.exploration.plan(input, Date.now()) : undefined;
        const plan = this.planTimes.map((timeSeconds, step) => this.actionDecision(
            outputs.slice(step * this.actionOutputs, (step + 1) * this.actionOutputs), timeSeconds,
            samples?.[step], logStds.slice(step * DRIVING_ANALOG_AXES, (step + 1) * DRIVING_ANALOG_AXES),
            step === 0 ? shifts : { up: true, down: true }));
        return { input, gpu: pass, value: outputs[this.criticOutput], plan };
    }

    resetExploration() {
        this.explorationRevision++;
        this.exploration.reset();
    }

    async inspectCameraAsync(image: PolicyCameraImage): Promise<DrivingCameraInspection> {
        await this.initializeGpu();
        await this.gpuBatch;
        return this.requireGpu().inspectCamera(image);
    }

    private requireGpu(): DrivingGpuModel {
        if (this.disposed) throw new Error('The driving model has been released.');
        if (!this.gpu) throw new Error('Initialize the training GPU before prediction or learning.');
        return this.gpu;
    }

    releaseDecision(decision?: PolicyDecision) { if (decision?.gpu) this.gpu?.release(decision.gpu); }

    private actionDecision(outputs: number[], timeSeconds: number, exploration: ExplorationSample | undefined,
        logStds: number[], shifts: ShiftAvailability): ActionDecision {
        const means = outputs.slice(0, DRIVING_ANALOG_AXES);
        const initialSigmas = drivingExplorationSigmas(this.configuration);
        const sigmas = logStds.map((value, axis) => Math.exp(value) * initialSigmas[axis] / 0.05);
        const latent = means.map((mean, axis) => mean + (exploration ? sigmas[axis] * exploration.axes[axis] : 0));
        // The execution mask is also retained with the off-policy transition.
        const shiftLogits = outputs.slice(DRIVING_ANALOG_AXES);
        const shiftProbabilities = this.shiftProbabilities(shiftLogits, shifts);
        const shift = this.chooseShift(shiftProbabilities, exploration?.shift);
        const action = { pedal: Math.tanh(latent[0]), steering: Math.tanh(latent[1]), shiftUp: shift === 1, shiftDown: shift === 2 };
        return { timeSeconds, means, latent, variances: sigmas.map((sigma, axis) => sigma * sigma
            * (exploration ? exploration.variances[axis] / initialSigmas[axis] ** 2 : 1)),
            shift, shiftProbabilities, shifts: { ...shifts }, action };
    }

    private chooseShift(probabilities: number[], draw?: number): ShiftChoice {
        if (draw === undefined) return probabilities.indexOf(Math.max(...probabilities)) as ShiftChoice;
        let cumulative = 0, shift: ShiftChoice = 0;
        for (const gear of [0, 1, 2] as const) {
            cumulative += probabilities[gear];
            if (probabilities[gear] > 0) shift = gear;
            if (draw < cumulative) break;
        }
        return shift;
    }

    private shiftProbabilities(values: number[], shifts: ShiftAvailability): number[] {
        const logits = values.map((logit, i) => [true, shifts.up, shifts.down][i] ? logit : -Infinity);
        const maximum = Math.max(...logits);
        // Concentrate exploration around the learned gear choice without changing its argmax.
        const weights = logits.map((logit) => Math.exp((logit - maximum) / this.configuration.learning.shiftTemperature));
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        return weights.map((weight) => weight / total);
    }

    /** Re-encode replay data using current weights. Behavior actions train Q only;
     * the GPU samples the current actor independently for its SAC objective. */
    async learnAsync(decision: Pick<PolicyDecision, 'input' | 'plan'>, reward: number, next: DrivingPolicyInput, elapsedSeconds: number,
        credits: PlanStepCredit[], terminal = false, nextShifts: ShiftAvailability = { up: true, down: true }, originInput = decision.input): Promise<void> {
        const gpu = this.requireGpu();
        const transition = this.transition(decision, reward, next, elapsedSeconds, credits, terminal, nextShifts, originInput);
        if (!transition.actions.length) return;
        const pass = gpu.replay(transition.input);
        try {
            gpu.learnTransition(pass, next, transition);
        } finally { gpu.release(pass); }
        this.steps++;
    }

    private transition(decision: Pick<PolicyDecision, 'input' | 'plan'>, reward: number, next: DrivingPolicyInput, elapsedSeconds: number,
        credits: PlanStepCredit[], terminal: boolean, nextShifts: ShiftAvailability, originInput = decision.input): ReplayTransition {
        if (!Number.isFinite(reward) || !Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) throw new Error('Invalid training transition.');
        const actions = credits.map((credit) => {
            if (!Number.isFinite(credit.weight) || credit.weight < 0 || !Number.isFinite(credit.shiftWeight ?? credit.weight)
                || (credit.shiftWeight ?? credit.weight) < 0) throw new Error('Invalid training transition credit.');
            const source = credit.action ?? decision.plan[credit.step]?.action;
            if (!source) throw new Error('Missing executed training action.');
            return { ...credit, shifts: { ...credit.shifts }, action: { ...source } };
        });
        this.validateTargets(actions);
        return { input: decision.input, originInput, next, reward, elapsedSeconds, actions, terminal, nextShifts: { ...nextShifts } };
    }

    /** Bounded off-policy replay stores multi-step returns with CPU observations
     * and executed controls, without retaining behavior activations. */
    async learnSequenceAsync(sequence: DrivingRolloutInterval[], next: DrivingPolicyInput, active = () => true): Promise<void> {
        const gpu = this.requireGpu(), steps = this.steps;
        const targets = drivingSequenceTargets(sequence, this.configuration.learning, next);
        if (!targets.length || !active()) return;
        const incoming = targets.flatMap((target) => {
            const endpoint = target.next ?? (target.terminal ? target.input : undefined);
            if (!endpoint) return [];
            return this.transition({ input: target.input, plan: target.decision.plan },
                target.reward, endpoint, target.elapsedSeconds, target.credits, target.terminal,
                target.nextShifts ?? { up: true, down: true }, target.decision.input);
        }).filter((target) => target.actions.length);
        const replay = [...this.replayBuffer, ...incoming].slice(-this.configuration.learning.replayCapacity);
        const indices = replay.map((_, index) => index);
        const batch: ReplayTransition[] = [];
        while (indices.length && batch.length < this.configuration.learning.replayBatchSize) {
            const index = Math.min(indices.length - 1, Math.max(0, Math.floor(this.random() * indices.length)));
            batch.push(replay[indices.splice(index, 1)[0]]);
        }
        if (!batch.length) return;
        gpu.beginSequence(next);
        let commit = false;
        try {
            for (const target of batch) {
                if (!active()) return;
                await this.learnAsync({ input: target.input, plan: [] }, target.reward, target.next, target.elapsedSeconds,
                    target.actions, target.terminal, target.nextShifts, target.originInput);
            }
            commit = active();
        } finally {
            gpu.endSequence(commit);
            if (!commit) this.steps = steps;
            else this.replayBuffer = replay;
        }
    }

    async learnDemonstrationAsync(input: DrivingPolicyInput, action: DrivingAction, shifts: ShiftAvailability): Promise<void> {
        await this.learnTargetsAsync(input, [{ step: 0, action, shifts }], true);
    }

    /** Train one observation against its available demonstrated future in one update. */
    async learnDemonstratedPlanAsync(input: DrivingPolicyInput, targets: PlannedActionTarget[]): Promise<void> {
        await this.learnTargetsAsync(input, targets, true);
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
            if (![action.pedal, action.steering].every(Number.isFinite) || Math.abs(action.pedal) > 1
                || Math.abs(action.steering) > 1 || (action.shiftUp && action.shiftDown)) {
                throw new Error('Invalid demonstration action.');
            }
        }
    }

    /** Keep a checkpoint from observing only part of a training batch.
     * Each demonstrated origin trains its available targets together in one globally
     * clipped update. Autonomous rollouts accumulate via learnSequenceAsync. */
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

    /** Explicit measurement only; optimizer diagnostics are not stored in checkpoints. */
    async trainingDiagnosticsAsync() {
        await this.gpuBatch;
        return this.gpu ? this.requireGpu().trainingDiagnostics() : undefined;
    }

    /** Serialize seed/restored weights before upload; trained weights require checkpointAsync. */
    checkpoint(): PolicyCheckpoint {
        if (!this.initialWeights) throw new Error('GPU checkpoints must be awaited.');
        return { ...emptyTrainingDocument(this.configuration), model: {
            context: { ...this.context }, inputs: this.inputs, steps: this.steps,
            cameraWeights: this.initialWeights.cameraWeights.map((rows) => rows.map((row) => row.slice())),
            cameraProjectionWeights: this.initialWeights.cameraProjectionWeights.map((row) => row.slice()),
            ...(this.initialWeights.cameraNormWeights && { cameraNormWeights: this.initialWeights.cameraNormWeights.map((row) => row.slice()) }),
            ...(this.initialWeights.lstmNormWeights && { lstmNormWeights: this.initialWeights.lstmNormWeights.map((row) => row.slice()) }),
            lstmWeights: this.initialWeights.lstmWeights.map((row) => row.slice()), outputWeights: this.initialWeights.outputWeights.map((row) => row.slice()),
            sac: JSON.parse(JSON.stringify(this.initialWeights.sac)) } };
    }

    async checkpointAsync(): Promise<PolicyCheckpoint> {
        if (!this.gpu) return this.checkpoint();
        await this.gpuBatch;
        const steps = this.steps, document = emptyTrainingDocument(this.configuration);
        const weights = await this.gpu.weights();
        return { ...document, model: { context: { ...this.context }, inputs: this.inputs, steps, ...weights } };
    }

    dispose() { this.disposed = true; this.resetExploration(); this.replayBuffer = []; this.gpu?.dispose(); }

    static restore(value: unknown): DrivingPolicy {
        const document = value as PolicyCheckpoint;
        const invalid = () => { throw new Error('Invalid or incompatible self-driving model checkpoint.'); };
        if (!document || document.format !== 'kestrel-self-driving-model' || document.algorithm !== 'soft-actor-critic'
            || Object.keys(document).some((key) => !['format', 'algorithm', 'configuration', 'model'].includes(key))) return invalid();
        const config = readDrivingConfig(document.configuration), data = document.model;
        const matrix = (rows: unknown, height: number, width: number, limit = config.learning.weightLimit) => Array.isArray(rows) && rows.length === height
            && rows.every((row) => Array.isArray(row) && row.length === width && row.every((v) => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= limit));
        const validEncoder = (encoder: DrivingEncoderWeights) => encoder
            && !Object.keys(encoder).some((key) => !['cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'lstmNormWeights', 'lstmWeights'].includes(key))
            && Array.isArray(encoder.cameraWeights) && encoder.cameraWeights.length === config.network.cameraLayers.length
            && drivingCameraLayers(config).every((layer, i) => matrix(encoder.cameraWeights[i], layer.channels, layer.columns))
            && matrix(encoder.cameraProjectionWeights, drivingCameraInputs(config), drivingCameraPooledInputs(config) + 1)
            && matrix(encoder.lstmWeights, 4 * config.network.hiddenUnits, drivingPolicyInputs(config) + config.network.hiddenUnits + 1)
            && (config.network.cameraLayerNorm ? matrix(encoder.cameraNormWeights, 2, drivingCameraInputs(config), Math.max(1, config.learning.weightLimit)) : encoder.cameraNormWeights === undefined)
            && (config.network.lstmLayerNorm ? matrix(encoder.lstmNormWeights, 2, 4 * config.network.hiddenUnits, Math.max(1, config.learning.weightLimit)) : encoder.lstmNormWeights === undefined);
        const criticSize = config.network.hiddenUnits * (config.network.hiddenUnits + config.actions.outputs.length + 1) + config.network.hiddenUnits + 1;
        if (!data || Object.keys(data).some((key) => !['context', 'inputs', 'steps', 'cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'lstmNormWeights', 'lstmWeights', 'outputWeights', 'sac'].includes(key))
            || !Number.isInteger(data.inputs) || data.inputs < 1
            || !Number.isSafeInteger(data.steps) || data.steps < 0
            || !data.context || !['game', 'car', 'map'].every((key) => typeof (data.context as any)[key] === 'string' && (data.context as any)[key].length > 0)
            || Object.keys(data.context).some((key) => !['game', 'car', 'map'].includes(key))
            || data.inputs !== drivingPolicyInputs(config)
            || !Array.isArray(data.cameraWeights) || data.cameraWeights.length !== config.network.cameraLayers.length
            || drivingCameraLayers(config).some((layer, i) => !matrix(data.cameraWeights[i], layer.channels, layer.columns))
            || !matrix(data.cameraProjectionWeights, drivingCameraInputs(config), drivingCameraPooledInputs(config) + 1)
            || (config.network.cameraLayerNorm ? !matrix(data.cameraNormWeights, 2, drivingCameraInputs(config), Math.max(1, config.learning.weightLimit))
                : data.cameraNormWeights !== undefined)
            || (config.network.lstmLayerNorm ? !matrix(data.lstmNormWeights, 2, 4 * config.network.hiddenUnits, Math.max(1, config.learning.weightLimit))
                : data.lstmNormWeights !== undefined)
            || !matrix(data.lstmWeights, 4 * config.network.hiddenUnits, data.inputs + config.network.hiddenUnits + 1)
            || !matrix(data.outputWeights, config.actions.planTimesSeconds.length * config.actions.outputs.length + 1, config.network.hiddenUnits + 1)
            || !data.sac || Object.keys(data.sac).some((key) => !['logStds', 'criticWeights', 'targetCriticWeights', 'targetEncoder'].includes(key))
            || !Array.isArray(data.sac.logStds) || data.sac.logStds.length !== config.actions.planTimesSeconds.length * DRIVING_ANALOG_AXES
            || !data.sac.logStds.every((v) => typeof v === 'number' && Number.isFinite(v) && v >= -7 && v <= 1)
            || !matrix(data.sac.criticWeights, 2, criticSize) || !matrix(data.sac.targetCriticWeights, 2, criticSize)
            || !validEncoder(data.sac.targetEncoder)) return invalid();
        const policy = new DrivingPolicy(data.context, drivingObservationInputs(config), Math.random, config, {
            cameraWeights: data.cameraWeights.map((rows) => rows.map((row) => row.slice())),
            cameraProjectionWeights: data.cameraProjectionWeights.map((row) => row.slice()),
            ...(data.cameraNormWeights && { cameraNormWeights: data.cameraNormWeights.map((row) => row.slice()) }),
            ...(data.lstmNormWeights && { lstmNormWeights: data.lstmNormWeights.map((row) => row.slice()) }),
            lstmWeights: data.lstmWeights.map((row) => row.slice()),
            outputWeights: data.outputWeights.map((row) => row.slice()),
            sac: JSON.parse(JSON.stringify(data.sac)),
        });
        policy.steps = data.steps;
        return policy;
    }
}
