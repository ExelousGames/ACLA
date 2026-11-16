import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';
import type { DrivingPolicyInput } from './driving-memory';
import { DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingMemorySlots, drivingPolicyInputs } from './driving-model-config';
import { drivingLstmShaders } from './driving-lstm-shaders';
import { drivingCameraShaders } from './driving-camera-shaders';
import { drivingLearningShaders } from './driving-learning-shaders';
import { drivingNormalizationShaders } from './driving-normalization-shaders';
import { drivingUpdateShaders } from './driving-update-shaders';
import type { PlannedActionTarget, PolicyModelState, PolicyWeights, ShiftAvailability } from './driving-policy';
import { assembleDrivingInput } from './driving-policy-input';
import type { DrivingCameraInspection } from './driving-camera-inspection';

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

/** GPU-resident CNN/LSTM inference and backpropagation through each replayed sequence. */
export class DrivingGpuModel {
    private pipelines: Record<string, GPUComputePipeline> = {};
    private buffers = new Set<GPUBuffer>();
    private bufferIds = new WeakMap<GPUBuffer, number>();
    private nextBufferId = 0;
    private bindGroups = new Map<string, GPUBindGroup>();
    private idlePasses: GpuWorkspace[] = [];
    private passWorkspaces = new WeakMap<GpuPolicyPass, GpuWorkspace>();
    private training: GpuWorkspace = new Map();
    private sequence?: { samples: number; bootstrap: GpuPolicyPass };
    private readbacks = new Map<number, GPUBuffer>();
    private failure?: string;
    private disposed = false;
    private lstmWeights: GPUBuffer;
    private outputWeights: GPUBuffer;
    private cameraWeights: GPUBuffer[];
    private cameraProjectionWeights: GPUBuffer;
    private cameraNormWeights?: GPUBuffer;
    private lstmNormWeights?: GPUBuffer;
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
        this.layers = drivingCameraLayers(config);
        this.cameraWeights = state.cameraWeights.map((rows) => this.upload(rows.flat()));
        this.cameraProjectionWeights = this.upload(state.cameraProjectionWeights.flat());
        this.lstmWeights = this.upload(state.lstmWeights.flat());
        this.outputWeights = this.upload(state.outputWeights.flat());
        if (config.network.cameraLayerNorm) this.cameraNormWeights = this.upload(state.cameraNormWeights!.flat());
        if (config.network.lstmLayerNorm) this.lstmNormWeights = this.upload(state.lstmNormWeights!.flat());
        this.times = Array.from({ length: drivingMemorySlots(config) + 1 },
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
        // Two idle passes cover a retained decision plus bootstrap/replay work.
        // Active decisions keep exclusive workspaces regardless of this limit.
        if (!this.failure && this.idlePasses.length < 2) this.idlePasses.push(workspace);
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
        const shaders = { ...drivingLstmShaders(this.config), ...drivingCameraShaders(this.config),
            ...drivingLearningShaders(this.config), ...drivingUpdateShaders(this.config),
            ...(this.cameraNormWeights ? drivingNormalizationShaders(this.config) : {}) };
        if (!this.lstmNormWeights) {
            ['gateSums', 'normalizedForward', 'gateNormBackward'].forEach((name) => { delete shaders[name]; });
        } else delete shaders.forward;
        await Promise.all(Object.entries(shaders).map(async ([name, shader]) => {
            try {
                this.pipelines[name] = await this.device.createComputePipelineAsync({ label: `Driving ${name}`, layout: 'auto',
                    compute: { module: this.device.createShaderModule({ code: shader }), entryPoint: 'main' } });
            } catch (error) { throw new Error(`Driving ${name}: ${(error as Error).message || String(error)}`); }
        }));
    }

    private encode(image: PolicyCameraImage, commands: GPUCommandEncoder, workspace: GpuWorkspace, index: number) {
        const buffer = (name: string, size: number) => this.scratch(workspace, `camera${index}/${name}`, size);
        const pixels = buffer('pixels', image.rgb.byteLength), activations = [pixels];
        this.device.queue.writeBuffer(pixels, 0, image.rgb);
        this.layers.forEach((layer, l) => {
            const count = layer.width * layer.height * layer.channels, output = buffer(`layer${l}`, count * 4);
            this.dispatch(commands, `cameraForward${l}`, [activations[l], this.cameraWeights[l], output], Math.ceil(count / 64));
            activations.push(output);
        });
        const pooled = buffer('pooled', this.pooledInputs * 4), projected = buffer('projected', this.cameraInputs * 4);
        this.dispatch(commands, 'cameraPool', [activations[activations.length - 1], pooled], Math.ceil(this.pooledInputs / 64));
        this.dispatch(commands, 'cameraProject', [pooled, this.cameraProjectionWeights, projected], this.cameraInputs);
        activations.push(pooled, projected);
        if (this.cameraNormWeights) {
            const cache = buffer('normalizationCache', (this.cameraInputs + 1) * 4);
            const normalized = buffer('normalized', this.cameraInputs * 4);
            this.dispatch(commands, 'cameraNormalize', [projected, this.cameraNormWeights, normalized, cache], 1);
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

    async forward(input: DrivingPolicyInput): Promise<{ pass: GpuPolicyPass; outputs: number[] }> {
        const pass = this.forwardPass(input);
        try {
            const values = await this.download([pass.outputs]);
            if (!values.every(Number.isFinite)) throw new Error('The training GPU produced invalid model outputs.');
            return { pass, outputs: Array.from(values) };
        } catch (error) { this.release(pass); throw error; }
    }

    /** Queue a prediction without mapping it back to the CPU. Sequence replays
     * use frozen weights; other replays see the preceding training update. */
    private forwardPass(input: DrivingPolicyInput): GpuPolicyPass {
        this.assertReady();
        const assembled = assembleDrivingInput(input, this.config);
        const workspace = this.idlePasses.pop() ?? new Map<string, GPUBuffer>();
        try {
            const commands = this.device.createCommandEncoder(), capacity = this.times.length;
            const pass: GpuPolicyPass = { features: this.scratch(workspace, 'features', capacity * this.inputs * 4),
                states: this.scratch(workspace, 'states', (capacity + 1) * 6 * this.hidden * 4),
                count: this.scratch(workspace, 'count', 4), steps: assembled.steps, released: false,
                outputs: this.scratch(workspace, 'outputs', this.outputs * 4), cameras: [] };
            if (this.lstmNormWeights) pass.gateCache = this.scratch(workspace, 'gateCache', capacity * (4 * this.hidden + 4) * 4);
            this.device.queue.writeBuffer(pass.features, 0, new Float32Array(assembled.features));
            this.device.queue.writeBuffer(pass.count, 0, new Uint32Array([assembled.steps]));
            // All subsequent states are overwritten; only the initial hidden/cell
            // state is read before being written on a reused workspace.
            commands.clearBuffer(pass.states, 0, 2 * this.hidden * 4);
            // Share repeated captures only inside this pass. A later forward must
            // use the latest CNN weights, and pending decisions own their activations.
            const encoded = new Map<PolicyCameraImage, GPUBuffer[]>();
            for (let index = 0; index < assembled.cameras.length; index++) {
                const { offset, image } = assembled.cameras[index];
                const activations = encoded.get(image) ?? this.encode(image, commands, workspace, encoded.size);
                encoded.set(image, activations);
                const location = this.scratch(workspace, `offset${index}`, 4);
                this.device.queue.writeBuffer(location, 0, new Uint32Array([offset]));
                pass.cameras.push({ offset: location, activations });
                commands.copyBufferToBuffer(activations[activations.length - 1], 0, pass.features, offset * 4, this.cameraInputs * 4);
            }
            for (let t = 0; t < pass.steps; t++) {
                if (this.lstmNormWeights) {
                    this.dispatch(commands, 'gateSums', [this.lstmWeights, pass.features, pass.states, this.times[t], pass.gateCache!], this.hidden);
                    this.dispatch(commands, 'normalizedForward', [pass.gateCache!, this.lstmNormWeights, pass.states, this.times[t]], 1);
                } else this.dispatch(commands, 'forward', [this.lstmWeights, pass.features, pass.states, this.times[t]], this.hidden);
            }
            this.dispatch(commands, 'output', [this.outputWeights, pass.states, pass.count, pass.outputs], Math.ceil(this.outputs / 64));
            this.device.queue.submit([commands.finish()]);
            this.passWorkspaces.set(pass, workspace);
            return pass;
        } catch (error) { workspace.forEach((buffer) => this.destroy(buffer)); throw error; }
    }

    /** Replay stored inputs without retaining a rollout's CNN activations. */
    replay(input: DrivingPolicyInput) { return this.forwardPass(input); }

    beginSequence(next: DrivingPolicyInput) {
        this.assertReady();
        if (this.sequence) throw new Error('A training sequence is already being updated.');
        const commands = this.device.createCommandEncoder();
        this.parameters().forEach(({ gradient }) => commands.clearBuffer(gradient));
        this.device.queue.submit([commands.finish()]);
        this.sequence = { samples: 0, bootstrap: this.forwardPass(next) };
    }

    endSequence(commit: boolean) {
        const sequence = this.sequence;
        if (!sequence) return;
        this.sequence = undefined;
        try {
            if (commit && sequence.samples) {
                this.assertReady();
                const commands = this.device.createCommandEncoder();
                this.applyGradients(commands, sequence.samples);
                this.device.queue.submit([commands.finish()]);
            }
        } finally { this.release(sequence.bootstrap); }
    }

    private parameters() {
        const entries: [string, GPUBuffer, boolean][] = [
            ...this.cameraWeights.map((weights, i): [string, GPUBuffer, boolean] => [`cameraGradient${i}`, weights, false]),
            ['projectionGradient', this.cameraProjectionWeights, false],
            ['lstmGradient', this.lstmWeights, false], ['headGradient', this.outputWeights, false],
        ];
        if (this.cameraNormWeights) entries.push(['cameraNormGradient', this.cameraNormWeights, true]);
        if (this.lstmNormWeights) entries.push(['lstmNormGradient', this.lstmNormWeights, true]);
        return entries.map(([name, weights, normalized]) => ({ weights, normalized,
            gradient: this.scratch(this.training, name, weights.size) }));
    }

    private applyGradients(commands: GPUCommandEncoder, samples: number) {
        const parameters = this.parameters();
        const count = this.scratch(this.training, 'gradientSamples', 4);
        const stats = this.scratch(this.training, 'gradientStats', 6 * 4);
        const partialCount = parameters.reduce((sum, { gradient }) => sum + Math.ceil(gradient.size / (4 * 256)), 0);
        const partials = this.scratch(this.training, 'gradientPartials', partialCount * 4);
        this.device.queue.writeBuffer(count, 0, new Uint32Array([samples]));
        let offset = 0;
        parameters.forEach(({ gradient }, i) => {
            const position = this.scratch(this.training, `gradientOffset${i}`, 4);
            this.device.queue.writeBuffer(position, 0, new Uint32Array([offset]));
            const groups = Math.ceil(gradient.size / (4 * 256));
            this.dispatch(commands, 'gradientPartialNorm', [gradient, partials, position], groups);
            offset += groups;
        });
        this.dispatch(commands, 'gradientGlobalNorm', [partials, count, stats], 1);
        parameters.forEach(({ weights, gradient, normalized }) => this.dispatch(commands,
            normalized ? 'applyNormGradient' : 'applyGradient', [weights, gradient, stats], Math.ceil(weights.size / (4 * 64))));
    }

    /** Optional diagnostic readback; normal training keeps all gradients on the GPU.
     * updateNorm is the proposed step before the final weight guard, an upper bound on the actual step. */
    async trainingDiagnostics() {
        const stats = this.training.get('gradientStats');
        if (!stats) return undefined;
        const [gradientNorm, clipScale, updateNorm, learningRate, samples, valid] = Array.from(await this.download([stats]));
        return { gradientNorm, clipScale, updateNorm, learningRate, samples, finite: valid === 1 };
    }

    learnTargets(input: DrivingPolicyInput, targets: PlannedActionTarget[]) {
        const values = new Array(this.config.actions.planTimesSeconds.length * 7).fill(0);
        for (const { step, action, shifts } of targets) {
            values.splice(step * 7, 7, 1, action.throttle, action.brake, action.steering,
                action.shiftUp ? 1 : action.shiftDown ? 2 : 0, Number(shifts.up), Number(shifts.down));
        }
        const pass = this.forwardPass(input);
        try {
            const target = this.scratch(this.training, 'demonstrationTarget', values.length * 4);
            this.device.queue.writeBuffer(target, 0, new Float32Array(values));
            this.update(pass, 'demonstrationGradient', [pass.outputs, target]);
        } finally { this.release(pass); }
    }

    learnTransition(pass: GpuPolicyPass, next: DrivingPolicyInput, target: {
        reward: number; elapsedSeconds: number; terminal: boolean;
        actions: { step: number; weight: number; shiftWeight?: number; latent: number[]; shift: number; shifts: ShiftAvailability }[];
    }) {
        this.assertReady();
        if (pass.released) throw new Error('The GPU training sample was already released.');
        const bootstrap = target.terminal ? undefined : this.sequence?.bootstrap ?? this.forwardPass(next);
        try {
            // Header followed by per-slot axis weight, latent axes, shift, mask and shift weight.
            const data = new Float32Array(3 + this.config.actions.planTimesSeconds.length * 7);
            data.set([target.reward, target.elapsedSeconds, Number(target.terminal)]);
            for (const { step, weight, shiftWeight = weight, latent, shift, shifts } of target.actions) {
                const base = 3 + step * 7;
                // Multiple acknowledged interpolation samples may credit the same slot.
                data[base] += weight;
                data.set(latent, base + 1);
                if (shiftWeight > 0) data.set([shift, Number(shifts.up), Number(shifts.down)], base + 3);
                data[base + 6] += shiftWeight;
            }
            const values = this.scratch(this.training, 'transitionTarget', data.byteLength);
            this.device.queue.writeBuffer(values, 0, data);
            this.update(pass, 'transitionGradient', [pass.outputs, bootstrap?.outputs ?? pass.outputs, values]);
        } finally { if (bootstrap && bootstrap !== this.sequence?.bootstrap) this.release(bootstrap); }
    }

    private update(pass: GpuPolicyPass, gradientShader: string, targets: GPUBuffer[]) {
        this.assertReady();
        if (pass.released) throw new Error('The GPU training sample was already released.');
        // This method encodes and submits synchronously. Later writes/clears on
        // the same queue execute after this update, even while it is in flight.
        const buffer = (name: string, size: number) => this.scratch(this.training, name, size);
        const commands = this.device.createCommandEncoder(), outputGradient = buffer('outputGradient', this.outputs * 4);
        const upstream = buffer('upstream', 2 * this.hidden * 4), deltas = buffer('deltas', this.times.length * 4 * this.hidden * 4);
        const cameraGradients = this.layers.map((layer, l) => buffer(`cameraGradient${l}`, layer.channels * layer.columns * 4));
        const cameraDeltas = this.layers.map((layer, l) => buffer(`cameraDelta${l}`, layer.width * layer.height * layer.channels * 4));
        const projectionGradient = buffer('projectionGradient', this.cameraInputs * (this.pooledInputs + 1) * 4);
        const projectionDelta = buffer('projectionDelta', this.cameraInputs * 4), pooledDelta = buffer('pooledDelta', this.pooledInputs * 4);
        // Hidden gradients and deltas are overwritten. Cell gradients and weight
        // accumulators must start at zero on every update, including reuse.
        commands.clearBuffer(upstream, this.hidden * 4, this.hidden * 4);
        if (!this.sequence) this.parameters().forEach(({ gradient }) => commands.clearBuffer(gradient));
        this.dispatch(commands, gradientShader, [...targets, outputGradient], Math.ceil(this.outputs / 64));
        this.dispatch(commands, 'hiddenGradient', [this.outputWeights, outputGradient, upstream], Math.ceil(this.hidden / 64));
        for (let t = pass.steps - 1; t >= 0; t--) {
            this.dispatch(commands, 'gateGradient', [pass.states, upstream, deltas, this.times[t]], Math.ceil(this.hidden / 64));
            if (this.lstmNormWeights) this.dispatch(commands, 'gateNormBackward',
                [pass.gateCache!, this.lstmNormWeights, deltas, this.times[t], buffer('lstmNormGradient', this.lstmNormWeights.size)], 4);
            if (t > 0) this.dispatch(commands, 'recurrentGradient', [this.lstmWeights, deltas, upstream, this.times[t]], Math.ceil(this.hidden / 64));
        }
        for (const camera of pass.cameras) {
            this.dispatch(commands, 'cameraGradient', [this.lstmWeights, deltas, camera.offset, projectionDelta],
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
        // History sums follow BPTT's chain rule. Frozen sequence samples accumulate
        // into one summed gradient, clipped only when the entire sequence is committed.
        this.dispatch(commands, 'lstmWeightGradient', [buffer('lstmGradient', this.lstmWeights.size), pass.features, pass.states, deltas, pass.count],
            Math.ceil(4 * this.hidden * (this.inputs + this.hidden + 1) / 64));
        this.dispatch(commands, 'outputWeightGradient', [buffer('headGradient', this.outputWeights.size), pass.states, outputGradient, pass.count], Math.ceil(this.outputs * (this.hidden + 1) / 64));
        if (this.sequence) this.sequence.samples++;
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
        const values = await this.download([...this.cameraWeights, this.cameraProjectionWeights, this.lstmWeights, this.outputWeights,
            ...(this.cameraNormWeights ? [this.cameraNormWeights] : []), ...(this.lstmNormWeights ? [this.lstmNormWeights] : [])]);
        if (!values.every(Number.isFinite)) throw new Error('Invalid GPU model weights; the checkpoint was not saved.');
        let offset = 0;
        const matrix = (rows: number, columns: number, limit = this.config.learning.weightLimit) => Array.from({ length: rows }, () => {
            const row = Array.from(values.subarray(offset, offset + columns), (value) => Math.max(-limit, Math.min(limit, value)));
            offset += columns; return row;
        });
        return { cameraWeights: this.layers.map((layer) => matrix(layer.channels, layer.columns)),
            cameraProjectionWeights: matrix(this.cameraInputs, this.pooledInputs + 1),
            lstmWeights: matrix(4 * this.hidden, this.inputs + this.hidden + 1),
            outputWeights: matrix(this.outputs, this.hidden + 1),
            ...(this.cameraNormWeights ? { cameraNormWeights: matrix(2, this.cameraInputs, Math.max(1, this.config.learning.weightLimit)) } : {}),
            ...(this.lstmNormWeights ? { lstmNormWeights: matrix(2, 4 * this.hidden, Math.max(1, this.config.learning.weightLimit)) } : {}) };
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.buffers.forEach((buffer) => buffer.destroy()); this.buffers.clear();
        this.bindGroups.clear(); this.idlePasses = []; this.training.clear(); this.readbacks.clear();
        this.passWorkspaces = new WeakMap();
        this.sequence = undefined;
        this.device.destroy();
    }
}
