import type { PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
import { RECORDED_BPTT_STEPS, type DrivingPolicyInput } from './driving-memory';
import { DRIVING_ANALOG_AXES, DRIVING_DEMONSTRATION_BACKLOG, DRIVING_GRU_GATES, DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingMemorySlots, drivingPolicyInputs } from './driving-model-config';
import { drivingGruDeltaWidth, drivingGruShaders, drivingGruStateWidth } from './driving-gru-shaders';
import { drivingCameraShaders } from './driving-camera-shaders';
import { drivingDemonstrationTargetStride, drivingLearningShaders } from './driving-learning-shaders';
import { drivingSacShaders, sacPolicySampleFloats } from './driving-sac-shaders';
import { drivingNormalizationShaders } from './driving-normalization-shaders';
import { drivingUpdateShaders } from './driving-update-shaders';
import type { DrivingImitationObjective, PlannedActionTarget, PolicyModelState, PolicyWeights, ShiftAvailability } from './driving-policy';
import { assembleDrivingInput } from './driving-policy-input';
import type { DrivingCameraInspection } from './driving-camera-inspection';
import type { DrivingAction } from './driving-observation';

export interface GpuPolicyPass {
    features: GPUBuffer;
    states: GPUBuffer;
    count: GPUBuffer;
    outputs: GPUBuffer;
    gateCache?: GPUBuffer;
    cameras: { offset: GPUBuffer; activations: GPUBuffer[] }[];
    steps: number;
    released: boolean;
}

type GpuWorkspace = Map<string, GPUBuffer>;
type RecurrentPoint = Pick<NonNullable<DrivingPolicyInput['recurrent']>, 'episode' | 'capturedAt'>;

/** GPU-resident CNN/GRU inference and backpropagation through each replayed sequence. */
export class DrivingGpuModel {
    private pipelines: Record<string, GPUComputePipeline> = {};
    private buffers = new Set<GPUBuffer>();
    private bufferIds = new WeakMap<GPUBuffer, number>();
    private nextBufferId = 0;
    private bindGroups = new Map<string, GPUBindGroup>();
    private idlePasses: GpuWorkspace[] = [];
    private passWorkspaces = new WeakMap<GpuPolicyPass, GpuWorkspace>();
    private training: GpuWorkspace = new Map();
    private sequence?: { samples: number; recorded: boolean };
    private readonly recordedAdamMoments = new WeakMap<GPUBuffer, [GPUBuffer, GPUBuffer]>();
    private recordedAdamCorrection?: GPUBuffer;
    private recurrentStates?: GPUBuffer;
    private readonly recurrentSnapshots: (RecurrentPoint | undefined)[];
    private recurrentCursor = 0;
    private readonly pendingPredictions = new WeakMap<object, Promise<void>>();
    private random: () => number = Math.random;
    private readbacks = new Map<number, GPUBuffer>();
    private failure?: string;
    private disposed = false;
    private gruWeights: GPUBuffer;
    private outputWeights: GPUBuffer;
    private cameraWeights: GPUBuffer[];
    private cameraProjectionWeights: GPUBuffer;
    private cameraNormWeights?: GPUBuffer;
    private gruNormWeights?: GPUBuffer;
    private logStds: GPUBuffer;
    private criticWeights: GPUBuffer;
    private targetCriticWeights: GPUBuffer;
    private targetEncoder: { cameraWeights: GPUBuffer[]; cameraProjectionWeights: GPUBuffer;
        gruWeights: GPUBuffer; cameraNormWeights?: GPUBuffer; gruNormWeights?: GPUBuffer };
    private layers;
    private times: GPUBuffer[];
    private readonly inputs: number;
    private readonly hidden: number;
    private readonly outputs: number;
    private readonly cameraInputs: number;
    private readonly pooledInputs: number;

    private constructor(private device: GPUDevice, private config: DrivingModelConfig, state: PolicyModelState) {
        this.inputs = drivingPolicyInputs(config);
        this.hidden = config.network.hiddenUnits;
        this.outputs = config.actions.planTimesSeconds.length * config.actions.outputs.length + 1;
        this.cameraInputs = drivingCameraInputs(config);
        this.pooledInputs = drivingCameraPooledInputs(config);
        // Include the demonstration backlog, replay transitions and the history bins.
        this.recurrentSnapshots = new Array(DRIVING_DEMONSTRATION_BACKLOG + config.learning.replayCapacity + drivingMemorySlots(config));
        this.layers = drivingCameraLayers(config);
        this.cameraWeights = state.cameraWeights.map((rows) => this.upload(rows.flat()));
        this.cameraProjectionWeights = this.upload(state.cameraProjectionWeights.flat());
        this.gruWeights = this.upload(state.gruWeights.flat());
        this.outputWeights = this.upload(state.outputWeights.flat());
        if (config.network.cameraLayerNorm) this.cameraNormWeights = this.upload(state.cameraNormWeights!.flat());
        if (config.network.gruLayerNorm) this.gruNormWeights = this.upload(state.gruNormWeights!.flat());
        this.logStds = this.upload(state.sac.logStds);
        this.criticWeights = this.upload(state.sac.criticWeights.flat());
        this.targetCriticWeights = this.upload(state.sac.targetCriticWeights.flat());
        const target = state.sac.targetEncoder;
        this.targetEncoder = { cameraWeights: target.cameraWeights.map((rows) => this.upload(rows.flat())),
            cameraProjectionWeights: this.upload(target.cameraProjectionWeights.flat()),
            gruWeights: this.upload(target.gruWeights.flat()),
            ...(target.cameraNormWeights && { cameraNormWeights: this.upload(target.cameraNormWeights.flat()) }),
            ...(target.gruNormWeights && { gruNormWeights: this.upload(target.gruNormWeights.flat()) }) };
        this.times = Array.from({ length: Math.max(drivingMemorySlots(config) + 1, RECORDED_BPTT_STEPS) },
            (_, t) => this.upload(new Uint32Array([t])));
        device.lost.then((info) => { if (!this.disposed) this.failure = `Training GPU lost: ${info.message || info.reason}. Restart training.`; });
        device.addEventListener('uncapturederror', (event) => { this.failure = `Training GPU error: ${event.error.message}`; });
    }

    static async create(config: DrivingModelConfig, state: PolicyModelState): Promise<DrivingGpuModel> {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter || adapter.info?.isFallbackAdapter) {
            throw new Error('AI driving requires a hardware WebGPU device for camera encoding and training. Enable GPU acceleration and restart the desktop app.');
        }
        const device = await adapter.requestDevice();
        let model: DrivingGpuModel | undefined;
        try {
            model = new DrivingGpuModel(device, config, state);
            await model.compile();
            model.assertReady();
            return model;
        } catch (error) { if (model) model.dispose(); else device.destroy(); throw error; }
    }

    private assertReady() {
        if (this.disposed) throw new Error('The training GPU has been released.');
        if (this.failure) throw new Error(this.failure);
    }

    private buffer(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
        const buffer = this.device.createBuffer({ size, usage });
        this.buffers.add(buffer);
        this.bufferIds.set(buffer, this.nextBufferId++);
        return buffer;
    }
    private destroy(buffer: GPUBuffer) {
        if (!this.buffers.delete(buffer)) return;
        buffer.destroy();
        // Eviction is rare; drop bindings so none retain retired resources.
        this.bindGroups.clear();
    }
    private scratch(workspace: GpuWorkspace, name: string, size: number) {
        let buffer = workspace.get(name);
        if (!buffer) { buffer = this.buffer(size); workspace.set(name, buffer); }
        return buffer;
    }
    private recycle(workspace: GpuWorkspace) {
        if (this.disposed) return;
        // SAC replays the state, next policy and target encoder concurrently.
        // Active decisions keep exclusive workspaces regardless of this limit.
        if (!this.failure && this.idlePasses.length < 3) this.idlePasses.push(workspace);
        else workspace.forEach((buffer) => this.destroy(buffer));
    }
    private upload(values: number[] | Uint32Array) {
        const data = values instanceof Uint32Array ? values : new Float32Array(values);
        const buffer = this.buffer(data.byteLength);
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }
    private dispatch(encoder: GPUCommandEncoder, name: string, buffers: GPUBuffer[], x: number, y = 1) {
        const pipeline = this.pipelines[name], pass = encoder.beginComputePass();
        const key = `${name}:${buffers.map((buffer) => this.bufferIds.get(buffer)).join(',')}`;
        let group = this.bindGroups.get(key);
        if (!group) {
            group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
            this.bindGroups.set(key, group);
        }
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }

    private async compile() {
        const shaders = { ...drivingGruShaders(this.config), ...drivingCameraShaders(this.config),
            ...drivingLearningShaders(this.config), ...drivingSacShaders(this.config), ...drivingUpdateShaders(this.config),
            ...(this.cameraNormWeights ? drivingNormalizationShaders(this.config) : {}) };
        if (!this.gruNormWeights) {
            ['gateSums', 'normalizedForward', 'candidateNormBackward', 'gateNormBackward'].forEach((name) => { delete shaders[name]; });
        } else delete shaders.forward;
        await Promise.all(Object.entries(shaders).map(async ([name, shader]) => {
            try {
                this.pipelines[name] = await this.device.createComputePipelineAsync({ label: `Driving ${name}`, layout: 'auto',
                    compute: { module: this.device.createShaderModule({ code: shader }), entryPoint: 'main' } });
            } catch (error) { throw new Error(`Driving ${name}: ${(error as Error).message || String(error)}`); }
        }));
    }

    private encode(image: PolicyCameraImage, commands: GPUCommandEncoder, workspace: GpuWorkspace, index: number, target = false) {
        const weights = target ? this.targetEncoder : { cameraWeights: this.cameraWeights,
            cameraProjectionWeights: this.cameraProjectionWeights, cameraNormWeights: this.cameraNormWeights };
        const buffer = (name: string, size: number) => this.scratch(workspace, `camera${index}/${name}`, size);
        const pixels = buffer('pixels', image.rgb.byteLength), activations = [pixels];
        this.device.queue.writeBuffer(pixels, 0, image.rgb);
        this.layers.forEach((layer, l) => {
            const count = layer.width * layer.height * layer.channels, output = buffer(`layer${l}`, count * 4);
            this.dispatch(commands, `cameraForward${l}`, [activations[l], weights.cameraWeights[l], output], Math.ceil(count / 64));
            activations.push(output);
        });
        const pooled = buffer('pooled', this.pooledInputs * 4), projected = buffer('projected', this.cameraInputs * 4);
        this.dispatch(commands, 'cameraPool', [activations[activations.length - 1], pooled], Math.ceil(this.pooledInputs / 64));
        this.dispatch(commands, 'cameraProject', [pooled, weights.cameraProjectionWeights, projected], this.cameraInputs);
        activations.push(pooled, projected);
        if (weights.cameraNormWeights) {
            const cache = buffer('normalizationCache', (this.cameraInputs + 1) * 4);
            const normalized = buffer('normalized', this.cameraInputs * 4);
            this.dispatch(commands, 'cameraNormalize', [projected, weights.cameraNormWeights, normalized, cache], 1);
            activations.push(cache, normalized);
        }
        return activations;
    }

    /** Read the same CNN activations used by the policy without updating any model weights. */
    async inspectCamera(image: PolicyCameraImage): Promise<DrivingCameraInspection> {
        this.assertReady();
        const { cameraWidth, cameraHeight } = this.config.observation;
        if (!image || image.width !== cameraWidth || image.height !== cameraHeight
            || !(image.rgb instanceof Uint8Array) || image.rgb.length !== cameraWidth * cameraHeight * 3) {
            throw new Error(`Camera inspection requires ${cameraWidth} × ${cameraHeight} RGB pixels.`);
        }
        const workspace = this.idlePasses.pop() ?? new Map<string, GPUBuffer>();
        let reusable = false;
        try {
            const commands = this.device.createCommandEncoder();
            const activations = this.encode(image, commands, workspace, 0);
            this.device.queue.submit([commands.finish()]);
            const values = await this.download(activations.slice(1, this.layers.length + 1));
            if (!values.every(Number.isFinite)) throw new Error('The training GPU produced invalid camera activations.');
            let offset = 0;
            const layers = this.layers.map(({ width, height, channels }, index) => {
                const count = width * height * channels;
                const layer = { index, width, height, channels, values: values.subarray(offset, offset + count) };
                offset += count;
                return layer;
            });
            reusable = true;
            return { layers };
        } finally {
            // Concurrent inspections and retained policy decisions own separate workspaces.
            if (reusable) this.recycle(workspace);
            else workspace.forEach((buffer) => this.destroy(buffer));
        }
    }

    forward(input: DrivingPolicyInput): Promise<{ pass: GpuPolicyPass; outputs: number[]; logStds: number[] }> {
        const episode = input.recurrent?.episode;
        if (!episode) return this.predict(input);
        const prior = this.pendingPredictions.get(episode);
        const prediction = prior ? prior.then(() => this.predict(input)) : this.predict(input);
        this.pendingPredictions.set(episode, prediction.then(() => undefined, () => undefined));
        return prediction;
    }

    private async predict(input: DrivingPolicyInput) {
        const pass = this.forwardPass(input, false, 'sequential');
        try {
            const values = await this.download([pass.outputs, this.logStds]);
            if (!values.every(Number.isFinite)) throw new Error('The training GPU produced invalid model outputs.');
            this.rememberRecurrentState(input, pass);
            return { pass, outputs: Array.from(values.subarray(0, this.outputs)), logStds: Array.from(values.subarray(this.outputs)) };
        } catch (error) { this.release(pass); throw error; }
    }

    private recurrentStateBefore(point: RecurrentPoint, before: number, includeCutoff = false) {
        const capacity = this.recurrentSnapshots.length;
        for (let i = 1; i <= capacity; i++) {
            const slot = (this.recurrentCursor + capacity - i) % capacity;
            const saved = this.recurrentSnapshots[slot];
            if (saved?.episode === point.episode
                && (saved.capturedAt < before || (includeCutoff && saved.capturedAt === before))) {
                return { slot, capturedAt: saved.capturedAt };
            }
        }
        return undefined;
    }

    private rememberRecurrentState(input: DrivingPolicyInput, pass: GpuPolicyPass) {
        const point = input.recurrent;
        if (!point) return;
        const latest = this.recurrentStateBefore(point, Infinity);
        if (latest && latest.capturedAt >= point.capturedAt) return;
        const width = this.hidden * 4, slot = this.recurrentCursor;
        const states = this.recurrentStates ??= this.buffer(this.recurrentSnapshots.length * width);
        const commands = this.device.createCommandEncoder();
        commands.copyBufferToBuffer(pass.states, pass.steps * drivingGruStateWidth * width, states, slot * width, width);
        this.device.queue.submit([commands.finish()]);
        this.recurrentSnapshots[slot] = { episode: point.episode, capturedAt: point.capturedAt };
        this.recurrentCursor = (slot + 1) % this.recurrentSnapshots.length;
    }

    /** Queue a prediction without mapping it back to the CPU. Sequence replays
     * use frozen weights; other replays see the preceding training update. */
    private forwardPass(input: DrivingPolicyInput, target = false, mode: 'replay' | 'sequential' | 'recorded' = 'replay'): GpuPolicyPass {
        this.assertReady();
        const weights = target ? this.targetEncoder : { gruWeights: this.gruWeights, gruNormWeights: this.gruNormWeights };
        const point = input.recurrent;
        const recordedHistory = mode === 'recorded' ? input.recordedHistory : undefined;
        const sequential = mode === 'sequential';
        let seed;
        if (point) {
            // Replay includes all retained inputs in backpropagation. Sequential
            // prediction can reuse a state already saved for this exact capture.
            let cutoff = point.capturedAt;
            if (!sequential) {
                cutoff = recordedHistory ? recordedHistory[0]?.capturedAt ?? cutoff
                    : Math.min(cutoff, ...point.historyTimes.map((time) => time ?? Infinity));
            }
            seed = this.recurrentStateBefore(point, cutoff, sequential);
        }
        const assembled = assembleDrivingInput(input, this.config, seed?.capturedAt, recordedHistory);
        const workspace = this.idlePasses.pop() ?? new Map<string, GPUBuffer>();
        try {
            const commands = this.device.createCommandEncoder(), capacity = this.times.length;
            const pass: GpuPolicyPass = { features: this.scratch(workspace, 'features', capacity * this.inputs * 4),
                states: this.scratch(workspace, 'states', (capacity + 1) * drivingGruStateWidth * this.hidden * 4),
                count: this.scratch(workspace, 'count', 4), steps: assembled.steps, released: false,
                outputs: this.scratch(workspace, 'outputs', this.outputs * 4), cameras: [] };
            if (this.gruNormWeights) pass.gateCache = this.scratch(workspace, 'gateCache', capacity * DRIVING_GRU_GATES * (this.hidden + 1) * 4);
            if (pass.steps) this.device.queue.writeBuffer(pass.features, 0, new Float32Array(assembled.features));
            this.device.queue.writeBuffer(pass.count, 0, new Uint32Array([assembled.steps]));
            const width = this.hidden * 4;
            if (seed) commands.copyBufferToBuffer(this.recurrentStates!, seed.slot * width, pass.states, 0, width);
            else commands.clearBuffer(pass.states, 0, width);
            // Share repeated captures only inside this pass. A later forward must
            // use the latest CNN weights, and pending decisions own their activations.
            const encoded = new Map<PolicyCameraImage, GPUBuffer[]>();
            for (let index = 0; index < assembled.cameras.length; index++) {
                const { offset, image } = assembled.cameras[index];
                const activations = encoded.get(image) ?? this.encode(image, commands, workspace, encoded.size, target);
                encoded.set(image, activations);
                const location = this.scratch(workspace, `offset${index}`, 4);
                this.device.queue.writeBuffer(location, 0, new Uint32Array([offset]));
                pass.cameras.push({ offset: location, activations });
                commands.copyBufferToBuffer(activations[activations.length - 1], 0, pass.features, offset * 4, this.cameraInputs * 4);
            }
            for (let t = 0; t < pass.steps; t++) {
                if (weights.gruNormWeights) {
                    this.dispatch(commands, 'gateSums', [weights.gruWeights, pass.features, pass.states, this.times[t], pass.gateCache!], this.hidden);
                    this.dispatch(commands, 'normalizedForward', [pass.gateCache!, weights.gruNormWeights, pass.states, this.times[t]], 1);
                } else this.dispatch(commands, 'forward', [weights.gruWeights, pass.features, pass.states, this.times[t]], this.hidden);
            }
            this.dispatch(commands, 'output', [this.outputWeights, pass.states, pass.count, pass.outputs], Math.ceil(this.outputs / 64));
            this.device.queue.submit([commands.finish()]);
            this.passWorkspaces.set(pass, workspace);
            return pass;
        } catch (error) { workspace.forEach((buffer) => this.destroy(buffer)); throw error; }
    }

    /** Replay stored inputs without retaining a rollout's CNN activations. */
    replay(input: DrivingPolicyInput) { return this.forwardPass(input); }

    beginSequence(_next?: DrivingPolicyInput) {
        this.beginGradientBatch(false);
    }

    beginRecordedSequence() {
        this.beginGradientBatch(true);
    }

    private beginGradientBatch(recorded: boolean) {
        this.assertReady();
        if (this.sequence) throw new Error('A training sequence is already being updated.');
        const commands = this.device.createCommandEncoder();
        this.parameters().forEach(({ gradient }) => commands.clearBuffer(gradient));
        this.device.queue.submit([commands.finish()]);
        this.sequence = { samples: 0, recorded };
    }

    endSequence(commit: boolean) {
        const sequence = this.sequence;
        if (!sequence) return;
        this.sequence = undefined;
        if (commit && sequence.samples) {
            this.assertReady();
            const commands = this.device.createCommandEncoder();
            this.applyGradients(commands, sequence.samples, sequence.recorded);
            this.device.queue.submit([commands.finish()]);
        }
    }

    private parameters() {
        const entries: [string, GPUBuffer, boolean][] = [
            ...this.cameraWeights.map((weights, i): [string, GPUBuffer, boolean] => [`cameraGradient${i}`, weights, false]),
            ['projectionGradient', this.cameraProjectionWeights, false],
            ['gruGradient', this.gruWeights, false], ['headGradient', this.outputWeights, false],
            ['criticGradient', this.criticWeights, false], ['logStdGradient', this.logStds, false],
        ];
        if (this.cameraNormWeights) entries.push(['cameraNormGradient', this.cameraNormWeights, true]);
        if (this.gruNormWeights) entries.push(['gruNormGradient', this.gruNormWeights, true]);
        return entries.map(([name, weights, normalized]) => ({ weights, normalized,
            gradient: this.scratch(this.training, name, weights.size) }));
    }

    private applyGradients(commands: GPUCommandEncoder, samples: number, recorded = false) {
        const parameters = this.parameters();
        // Optimize as if the smaller telemetry inputs were scaled, while retaining the
        // checkpoint's original parameter coordinates for inference and other training.
        if (recorded) {
            const gradient = this.scratch(this.training, 'gruGradient', this.gruWeights.size);
            this.dispatch(commands, 'recordedAdamConditionGru', [gradient], Math.ceil(gradient.size / (4 * 64)));
        }
        const count = this.scratch(this.training, 'gradientSamples', 4);
        const stats = this.scratch(this.training, 'gradientStats', 6 * 4);
        const partialCount = parameters.reduce((sum, { gradient }) => sum + Math.ceil(gradient.size / (4 * 256)), 0);
        const partials = this.scratch(this.training, 'gradientPartials', partialCount * 4);
        this.device.queue.writeBuffer(count, 0, new Uint32Array([samples]));
        let offset = 0;
        const ranges = parameters.map(({ gradient }, i) => {
            const position = this.scratch(this.training, `gradientOffset${i}`, 4);
            this.device.queue.writeBuffer(position, 0, new Uint32Array([offset]));
            const groups = Math.ceil(gradient.size / (4 * 256));
            this.dispatch(commands, 'gradientPartialNorm', [gradient, partials, position], groups);
            offset += groups;
            return { position, groups };
        });
        this.dispatch(commands, 'gradientGlobalNorm', [partials, count, stats], 1);
        if (recorded) {
            const correction = this.recordedAdamCorrection ??= this.upload([1, 1, 0, 0]);
            this.dispatch(commands, 'recordedAdamAdvance', [stats, correction], 1);
            parameters.forEach(({ weights, gradient, normalized }, index) => {
                if (weights === this.criticWeights || weights === this.logStds) return;
                let moments = this.recordedAdamMoments.get(weights);
                if (!moments) {
                    moments = [this.buffer(weights.size), this.buffer(weights.size)];
                    this.recordedAdamMoments.set(weights, moments);
                }
                const { position, groups } = ranges[index];
                const shader = weights === this.gruWeights ? 'recordedAdamApplyGru' : normalized ? 'recordedAdamApplyNorm' : 'recordedAdamApply';
                this.dispatch(commands, shader,
                    [weights, gradient, stats, ...moments, correction, partials, position], groups);
            });
            this.dispatch(commands, 'recordedAdamUpdateNorm', [partials, stats], 1);
        } else {
            parameters.forEach(({ weights, gradient, normalized }) => this.dispatch(commands,
                weights === this.logStds ? 'sacApplyLogStd' : normalized ? 'applyNormGradient' : 'applyGradient',
                [weights, gradient, stats], Math.ceil(weights.size / (4 * 64))));
        }
        const targets: [GPUBuffer, GPUBuffer][] = [
            ...this.cameraWeights.map((weights, index): [GPUBuffer, GPUBuffer] => [weights, this.targetEncoder.cameraWeights[index]]),
            [this.cameraProjectionWeights, this.targetEncoder.cameraProjectionWeights],
            [this.gruWeights, this.targetEncoder.gruWeights],
        ];
        if (!recorded) targets.push([this.criticWeights, this.targetCriticWeights]);
        if (this.cameraNormWeights) targets.push([this.cameraNormWeights, this.targetEncoder.cameraNormWeights!]);
        if (this.gruNormWeights) targets.push([this.gruNormWeights, this.targetEncoder.gruNormWeights!]);
        targets.forEach(([weights, target]) => this.dispatch(commands, 'sacTargetUpdate', [weights, target, stats], Math.ceil(weights.size / (4 * 64))));
    }

    /** Optional diagnostic readback; normal training keeps all gradients on the GPU.
     * updateNorm describes the SGD proposal or the applied Adam step. */
    async trainingDiagnostics() {
        const stats = this.training.get('gradientStats');
        if (!stats) return undefined;
        const [gradientNorm, clipScale, updateNorm, learningRate, samples, valid] = Array.from(await this.download([stats]));
        return { gradientNorm, clipScale, updateNorm, learningRate, samples, finite: valid === 1 };
    }

    /** Keep the recurrent history of a demonstration origin without updating weights. */
    observeDemonstration(input: DrivingPolicyInput) {
        const pass = this.forwardPass(input, false, 'sequential');
        try { this.rememberRecurrentState(input, pass); }
        finally { this.release(pass); }
    }

    learnTargets(input: DrivingPolicyInput, targets: PlannedActionTarget[], rememberState = false,
        objective: DrivingImitationObjective = 'standard') {
        const recorded = objective === 'recorded';
        if (this.sequence?.recorded && !recorded) throw new Error('Recorded training batches require recorded action targets.');
        const stride = drivingDemonstrationTargetStride;
        const hasImmediateTarget = recorded && targets.some(({ step }) => step === 0);
        const delayedSlots = hasImmediateTarget ? new Set(targets.map(({ step }) => step)).size - 1 : 0;
        const values = new Float32Array(this.config.actions.planTimesSeconds.length * stride);
        for (const { step, action, shifts } of targets) {
            const offset = step * stride;
            // A full recorded plan reserves the stronger signal for the action used now.
            const weight = step > 0 && delayedSlots > 0 ? 0.5 / delayedSlots : 1;
            values.set([weight, action.pedal, action.steering, action.shiftUp ? 1 : action.shiftDown ? 2 : 0,
                Number(shifts.up), Number(shifts.down)], offset);
        }
        const pass = this.forwardPass(input, false, recorded ? 'recorded' : 'replay');
        try {
            const target = this.scratch(this.training, recorded ? 'recordedDemonstrationTarget' : 'demonstrationTarget', values.byteLength);
            this.device.queue.writeBuffer(target, 0, values);
            this.update(pass, recorded ? 'recordedDemonstrationGradient' : 'demonstrationGradient', [pass.outputs, target]);
            if (rememberState) this.rememberRecurrentState(input, pass);
        } finally { this.release(pass); }
    }

    learnTransition(pass: GpuPolicyPass, next: DrivingPolicyInput, target: {
        reward: number; elapsedSeconds: number; terminal: boolean;
        actions: { step: number; weight: number; shiftWeight?: number; action: DrivingAction; shift: number; shifts: ShiftAvailability }[];
        nextShifts: ShiftAvailability;
        originInput?: DrivingPolicyInput;
    }) {
        this.assertReady();
        if (this.sequence?.recorded) throw new Error('Recorded training batches require recorded action targets.');
        if (pass.released) throw new Error('The GPU training sample was already released.');
        const actions = target.actions.filter(({ weight, shiftWeight = weight }) => weight > 0 || shiftWeight > 0);
        if (!actions.length) return;
        const localSequence = !this.sequence;
        if (localSequence) this.beginSequence();
        let commit = false;
        let nextPolicy: GpuPolicyPass | undefined, nextTarget: GpuPolicyPass | undefined;
        try {
            if (!target.terminal) {
                nextPolicy = this.forwardPass(next);
                nextTarget = this.forwardPass(next, true);
            }
            const analogWeight = actions.reduce((sum, { weight }) => sum + weight, 0);
            const weightSum = analogWeight || actions.reduce((sum, { weight, shiftWeight = weight }) => sum + shiftWeight, 0);
            const commandShifts = actions.find(({ weight, shiftWeight = weight }) => shiftWeight > 0)?.shifts;
            const normal = () => Math.sqrt(-2 * Math.log(Math.max(1e-12, this.random()))) * Math.cos(2 * Math.PI * this.random());
            const transitions = actions.map(({ step, weight, shiftWeight = weight, action, shifts }, index) => {
                // The immediate actor learns at the observed execution state.
                const samplingShifts = commandShifts ?? shifts;
                const data = new Float32Array([target.reward, target.elapsedSeconds, Number(target.terminal), 0,
                    weight, shiftWeight, (analogWeight ? weight : shiftWeight) / weightSum,
                    Number(samplingShifts.up), Number(samplingShifts.down), Number(target.nextShifts.up), Number(target.nextShifts.down),
                    action.pedal, action.steering, action.shiftUp ? 1 : action.shiftDown ? 2 : 0,
                    ...Array.from({ length: 2 * DRIVING_ANALOG_AXES }, normal)]);
                const values = this.scratch(this.training, `sacTransition${index}`, data.byteLength);
                this.device.queue.writeBuffer(values, 0, data);
                return { values, step, shifts };
            });
            const features = this.scratch(this.training, 'sacFeatures', this.hidden * 2 * 4);
            this.update(pass, undefined, [], (commands, outputGradient, criticHidden) => {
                const stateStride = drivingGruStateWidth * this.hidden * 4;
                commands.copyBufferToBuffer(pass.states, pass.steps * stateStride, features, 0, this.hidden * 4);
                const endpoint = nextTarget ?? pass;
                commands.copyBufferToBuffer(endpoint.states, endpoint.steps * stateStride, features, this.hidden * 4, this.hidden * 4);
                for (const { values } of transitions) {
                    this.sacGradients(commands, pass, features, values, outputGradient, criticHidden, nextPolicy);
                }
            });
            if (nextPolicy) { this.release(nextPolicy); nextPolicy = undefined; }
            if (nextTarget) { this.release(nextTarget); nextTarget = undefined; }
            const future = transitions.filter(({ step }) => step > 0);
            if (future.length) {
                // Sample each credited head from its original plan input. Score its actions
                // with the critics at the state reached when that step controlled the car.
                const origin = target.originInput ? this.forwardPass(target.originInput) : pass;
                try {
                    for (const { values, step, shifts } of future) {
                        this.device.queue.writeBuffer(values, 3 * 4, new Float32Array([step]));
                        this.device.queue.writeBuffer(values, 7 * 4, new Float32Array([Number(shifts.up), Number(shifts.down)]));
                    }
                    this.update(origin, undefined, [], (commands, outputGradient) => {
                        for (const { values } of future) this.sacGradients(commands, origin, features, values, outputGradient);
                    }, false);
                } finally { if (origin !== pass) this.release(origin); }
            }
            commit = true;
        } finally {
            if (nextPolicy) this.release(nextPolicy);
            if (nextTarget) this.release(nextTarget);
            if (localSequence) this.endSequence(commit);
        }
    }

    private sacGradients(commands: GPUCommandEncoder, actor: GpuPolicyPass, features: GPUBuffer, transition: GPUBuffer,
        outputGradient: GPUBuffer, criticHidden?: GPUBuffer, nextPolicy?: GpuPolicyPass) {
        const buffer = (name: string, size: number) => this.scratch(this.training, name, size);
        const samples = buffer('sacSamples', 2 * sacPolicySampleFloats * 4), activations = buffer('sacActivations', 14 * this.hidden * 4);
        const values = buffer('sacValues', 14 * 4), errors = buffer('sacErrors', 2 * 4);
        this.dispatch(commands, 'sacSample', [actor.outputs, nextPolicy?.outputs ?? actor.outputs, this.logStds, transition, samples], criticHidden ? 2 : 1);
        this.dispatch(commands, 'sacCriticForward', [this.criticWeights, this.targetCriticWeights, features, transition, samples, activations, values], criticHidden ? 14 : 8);
        this.dispatch(commands, 'sacLoss', [transition, samples, values, errors, outputGradient,
            buffer('logStdGradient', this.logStds.size), this.criticWeights, activations], 1);
        if (criticHidden) {
            this.dispatch(commands, 'sacCriticWeightGradient', [this.criticWeights, features, transition, samples, activations, errors,
                buffer('criticGradient', this.criticWeights.size)], Math.ceil(this.criticWeights.size / (4 * 64)));
            this.dispatch(commands, 'sacCriticHiddenGradient', [this.criticWeights, activations, errors, criticHidden], Math.ceil(this.hidden / 64));
        }
    }

    private update(pass: GpuPolicyPass, gradientShader: string | undefined, targets: GPUBuffer[],
        sac?: (commands: GPUCommandEncoder, outputGradient: GPUBuffer, criticHidden: GPUBuffer) => void, countSample = true) {
        this.assertReady();
        if (pass.released) throw new Error('The GPU training sample was already released.');
        // This method encodes and submits synchronously. Later writes/clears on
        // the same queue execute after this update, even while it is in flight.
        const buffer = (name: string, size: number) => this.scratch(this.training, name, size);
        const commands = this.device.createCommandEncoder(), outputGradient = buffer('outputGradient', this.outputs * 4);
        const upstream = buffer('upstream', this.hidden * 4);
        const deltas = buffer('deltas', this.times.length * drivingGruDeltaWidth * this.hidden * 4);
        const normWeights = this.gruNormWeights;
        const normGradient = normWeights && buffer('gruNormGradient', normWeights.size);
        const cameraGradients = this.layers.map((layer, l) => buffer(`cameraGradient${l}`, layer.channels * layer.columns * 4));
        const cameraDeltas = this.layers.map((layer, l) => buffer(`cameraDelta${l}`, layer.width * layer.height * layer.channels * 4));
        const projectionGradient = buffer('projectionGradient', this.cameraInputs * (this.pooledInputs + 1) * 4);
        const projectionDelta = buffer('projectionDelta', this.cameraInputs * 4), pooledDelta = buffer('pooledDelta', this.pooledInputs * 4);
        if (!this.sequence) this.parameters().forEach(({ gradient }) => commands.clearBuffer(gradient));
        const criticHidden = buffer('criticHidden', this.hidden * 4);
        if (sac) {
            commands.clearBuffer(outputGradient);
            commands.clearBuffer(criticHidden);
            sac(commands, outputGradient, criticHidden);
        } else this.dispatch(commands, gradientShader!, [...targets, outputGradient], Math.ceil(this.outputs / 64));
        this.dispatch(commands, 'hiddenGradient', [this.outputWeights, outputGradient, upstream], Math.ceil(this.hidden / 64));
        if (sac) this.dispatch(commands, 'sacAddHidden', [criticHidden, upstream], Math.ceil(this.hidden / 64));
        const hiddenGroups = Math.ceil(this.hidden / 64);
        for (let t = pass.steps - 1; t >= 0; t--) {
            const time = this.times[t];
            const normBuffers = normWeights && normGradient && [pass.gateCache!, normWeights, deltas, time, normGradient];
            this.dispatch(commands, 'gateGradient', [pass.states, upstream, deltas, time], hiddenGroups);
            if (normBuffers) this.dispatch(commands, 'candidateNormBackward', normBuffers, 1);
            this.dispatch(commands, 'resetGradient', [pass.states, deltas, time], hiddenGroups);
            if (normBuffers) this.dispatch(commands, 'gateNormBackward', normBuffers, 2);
            if (t > 0) this.dispatch(commands, 'recurrentGradient', [this.gruWeights, deltas, upstream, time], hiddenGroups);
        }
        for (const camera of pass.cameras) {
            this.dispatch(commands, 'cameraGradient', [this.gruWeights, deltas, camera.offset, projectionDelta],
                Math.ceil(this.cameraInputs / 64));
            if (this.cameraNormWeights) this.dispatch(commands, 'cameraNormalizeBackward',
                [camera.activations[this.layers.length + 3], this.cameraNormWeights, projectionDelta,
                    buffer('cameraNormGradient', this.cameraNormWeights.size)], 1);
            this.dispatch(commands, 'cameraProjectionBackward', [this.cameraProjectionWeights, projectionDelta, pooledDelta],
                Math.ceil(this.pooledInputs / 64));
            this.dispatch(commands, 'cameraProjectionWeightGradient', [camera.activations[this.layers.length + 1], projectionDelta, projectionGradient],
                Math.ceil(this.cameraInputs * (this.pooledInputs + 1) / 64));
            const last = this.layers[this.layers.length - 1];
            this.dispatch(commands, 'cameraPoolBackward', [pooledDelta, cameraDeltas[cameraDeltas.length - 1]],
                Math.ceil(last.width * last.height * last.channels / 64));
            for (let l = this.layers.length - 1; l >= 0; l--) {
                const layer = this.layers[l];
                this.dispatch(commands, `cameraDelta${l}`, [camera.activations[l + 1], cameraDeltas[l]],
                    Math.ceil(layer.width * layer.height * layer.channels / 64));
                if (l) this.dispatch(commands, `cameraBackward${l}`, [this.cameraWeights[l], cameraDeltas[l], cameraDeltas[l - 1]],
                    Math.ceil(layer.inputWidth * layer.inputHeight * layer.inputChannels / 64));
                this.dispatch(commands, `cameraWeightGradient${l}`, [camera.activations[l], cameraDeltas[l], cameraGradients[l]],
                    layer.channels * layer.columns);
            }
        }
        // History sums follow BPTT's chain rule. Replay samples accumulate into
        // a mean gradient, clipped only when the entire batch is committed.
        this.dispatch(commands, 'gruWeightGradient', [buffer('gruGradient', this.gruWeights.size), pass.features, pass.states, deltas, pass.count],
            Math.ceil(this.gruWeights.size / (4 * 64)));
        this.dispatch(commands, 'outputWeightGradient', [buffer('headGradient', this.outputWeights.size), pass.states, outputGradient, pass.count], Math.ceil(this.outputs * (this.hidden + 1) / 64));
        if (this.sequence) { if (countSample) this.sequence.samples++; }
        else this.applyGradients(commands, 1);
        this.device.queue.submit([commands.finish()]);
    }

    release(pass: GpuPolicyPass) {
        if (pass.released) return;
        pass.released = true;
        const workspace = this.passWorkspaces.get(pass);
        this.passWorkspaces.delete(pass);
        // A fresh pass object is issued for every lease, so a delayed duplicate
        // release cannot return a workspace that now belongs to another decision.
        if (workspace) this.recycle(workspace);
    }

    /** Bound queued work even during human driving, which has no control readback.
     * One completion wait per batch transfers no predictions or gradients. */
    async finishTraining() {
        this.assertReady();
        await this.device.queue.onSubmittedWorkDone();
        this.assertReady();
    }

    private async download(buffers: GPUBuffer[]) {
        this.assertReady();
        const size = buffers.reduce((total, buffer) => total + buffer.size, 0);
        const staging = this.readbacks.get(size) ?? this.buffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        this.readbacks.delete(size);
        let reusable = false;
        try {
            const commands = this.device.createCommandEncoder(); let offset = 0;
            for (const buffer of buffers) { commands.copyBufferToBuffer(buffer, 0, staging, offset, buffer.size); offset += buffer.size; }
            this.device.queue.submit([commands.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            this.assertReady();
            const values = new Float32Array(staging.getMappedRange()).slice();
            staging.unmap();
            reusable = true;
            return values;
        } finally {
            // Pending/mapped readbacks are exclusive until copied and unmapped.
            // Keep one idle buffer per size; retire failed or excess readbacks.
            if (reusable && !this.disposed && !this.failure && !this.readbacks.has(size)) this.readbacks.set(size, staging);
            else this.destroy(staging);
        }
    }

    async weights(): Promise<PolicyWeights> {
        const target = this.targetEncoder;
        const values = await this.download([...this.cameraWeights, this.cameraProjectionWeights, this.gruWeights, this.outputWeights,
            ...(this.cameraNormWeights ? [this.cameraNormWeights] : []), ...(this.gruNormWeights ? [this.gruNormWeights] : []),
            this.logStds, this.criticWeights, this.targetCriticWeights,
            ...target.cameraWeights, target.cameraProjectionWeights, target.gruWeights,
            ...(target.cameraNormWeights ? [target.cameraNormWeights] : []), ...(target.gruNormWeights ? [target.gruNormWeights] : [])]);
        if (!values.every(Number.isFinite)) throw new Error('Invalid GPU model weights; the checkpoint was not saved.');
        let offset = 0;
        const matrix = (rows: number, columns: number, limit = this.config.learning.weightLimit) => Array.from({ length: rows }, () => {
            const row = Array.from(values.subarray(offset, offset + columns), (value) => Math.max(-limit, Math.min(limit, value)));
            offset += columns; return row;
        });
        const encoder = { cameraWeights: this.layers.map((layer) => matrix(layer.channels, layer.columns)),
            cameraProjectionWeights: matrix(this.cameraInputs, this.pooledInputs + 1),
            gruWeights: matrix(DRIVING_GRU_GATES * this.hidden, this.inputs + this.hidden + 1),
            outputWeights: matrix(this.outputs, this.hidden + 1),
            ...(this.cameraNormWeights ? { cameraNormWeights: matrix(2, this.cameraInputs, Math.max(1, this.config.learning.weightLimit)) } : {}),
            ...(this.gruNormWeights ? { gruNormWeights: matrix(2, DRIVING_GRU_GATES * this.hidden, Math.max(1, this.config.learning.weightLimit)) } : {}) };
        const logStds = Array.from(values.subarray(offset, offset + this.logStds.size / 4)); offset += logStds.length;
        const criticSize = this.criticWeights.size / 8;
        return { ...encoder, sac: { logStds, criticWeights: matrix(2, criticSize), targetCriticWeights: matrix(2, criticSize),
            targetEncoder: { cameraWeights: this.layers.map((layer) => matrix(layer.channels, layer.columns)),
                cameraProjectionWeights: matrix(this.cameraInputs, this.pooledInputs + 1),
                gruWeights: matrix(DRIVING_GRU_GATES * this.hidden, this.inputs + this.hidden + 1),
                ...(target.cameraNormWeights ? { cameraNormWeights: matrix(2, this.cameraInputs, Math.max(1, this.config.learning.weightLimit)) } : {}),
                ...(target.gruNormWeights ? { gruNormWeights: matrix(2, DRIVING_GRU_GATES * this.hidden, Math.max(1, this.config.learning.weightLimit)) } : {}) } } };
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.buffers.forEach((buffer) => buffer.destroy()); this.buffers.clear();
        this.bindGroups.clear(); this.idlePasses = []; this.training.clear(); this.readbacks.clear();
        this.passWorkspaces = new WeakMap();
        this.recurrentSnapshots.fill(undefined);
        this.recurrentStates = undefined;
        this.sequence = undefined;
        this.device.destroy();
    }
}
