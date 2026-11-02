import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';
import type { DrivingPolicyInput } from './driving-memory';
import { DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingMemorySlots, drivingPolicyInputs } from './driving-model-config';
import { drivingLstmShaders } from './driving-lstm-shaders';
import { drivingCameraShaders } from './driving-camera-shaders';
import { drivingLearningShaders } from './driving-learning-shaders';
import type { PlannedActionTarget, PolicyModelState, ShiftAvailability } from './driving-policy';
import { assembleDrivingInput } from './driving-policy-input';

export interface GpuPolicyPass {
    features: GPUBuffer;
    states: GPUBuffer;
    count: GPUBuffer;
    outputs: GPUBuffer;
    cameras: { offset: GPUBuffer; activations: GPUBuffer[] }[];
    cameraBuffers: GPUBuffer[];
    steps: number;
    released: boolean;
}

/** GPU-resident CNN/LSTM inference and backpropagation through each replayed sequence. */
export class DrivingGpuModel {
    private pipelines: Record<string, GPUComputePipeline> = {};
    private buffers = new Set<GPUBuffer>();
    private failure?: string;
    private disposed = false;
    private lstmWeights: GPUBuffer;
    private outputWeights: GPUBuffer;
    private cameraWeights: GPUBuffer[];
    private cameraProjectionWeights: GPUBuffer;
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
        return buffer;
    }
    private destroy(buffer: GPUBuffer) { buffer.destroy(); this.buffers.delete(buffer); }
    private upload(values: number[] | Uint32Array) {
        const data = values instanceof Uint32Array ? values : new Float32Array(values);
        const buffer = this.buffer(data.byteLength);
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }
    private dispatch(encoder: GPUCommandEncoder, name: string, buffers: GPUBuffer[], x: number, y = 1) {
        const pipeline = this.pipelines[name], pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }

    private async compile() {
        await Promise.all(Object.entries({ ...drivingLstmShaders(this.config), ...drivingCameraShaders(this.config),
            ...drivingLearningShaders(this.config) }).map(async ([name, shader]) => {
            this.pipelines[name] = await this.device.createComputePipelineAsync({ label: `Driving ${name}`, layout: 'auto',
                compute: { module: this.device.createShaderModule({ code: shader }), entryPoint: 'main' } });
        }));
    }

    private encode(image: PolicyCameraImage, commands: GPUCommandEncoder, owned: GPUBuffer[]) {
        const pixels = this.buffer(image.rgb.byteLength), activations = [pixels];
        owned.push(pixels);
        this.device.queue.writeBuffer(pixels, 0, image.rgb);
        this.layers.forEach((layer, l) => {
            const count = layer.width * layer.height * layer.channels, output = this.buffer(count * 4);
            owned.push(output);
            this.dispatch(commands, `cameraForward${l}`, [activations[l], this.cameraWeights[l], output], Math.ceil(count / 64));
            activations.push(output);
        });
        const pooled = this.buffer(this.pooledInputs * 4), projected = this.buffer(this.cameraInputs * 4);
        owned.push(pooled, projected);
        this.dispatch(commands, 'cameraPool', [activations[activations.length - 1], pooled], Math.ceil(this.pooledInputs / 64));
        this.dispatch(commands, 'cameraProject', [pooled, this.cameraProjectionWeights, projected], this.cameraInputs);
        activations.push(pooled, projected);
        return activations;
    }

    async forward(input: DrivingPolicyInput): Promise<{ pass: GpuPolicyPass; outputs: number[] }> {
        const pass = this.forwardPass(input);
        try {
            const values = await this.download([pass.outputs]);
            if (!values.every(Number.isFinite)) throw new Error('The training GPU produced invalid model outputs.');
            return { pass, outputs: Array.from(values) };
        } catch (error) { this.release(pass); throw error; }
    }

    /** Queue a prediction without mapping it back to the CPU. Queue ordering keeps
     * each replay on the weights produced by the preceding training update. */
    private forwardPass(input: DrivingPolicyInput): GpuPolicyPass {
        this.assertReady();
        const assembled = assembleDrivingInput(input, this.config);
        const commands = this.device.createCommandEncoder();
        const pass: GpuPolicyPass = { features: this.upload(assembled.features),
            states: this.buffer((assembled.steps + 1) * 6 * this.hidden * 4),
            count: this.upload(new Uint32Array([assembled.steps])), steps: assembled.steps, released: false,
            outputs: this.buffer(this.outputs * 4),
            cameras: [], cameraBuffers: [] };
        try {
            // Share repeated captures only inside this pass. A later forward must
            // use the latest CNN weights, and pending decisions own their activations.
            const encoded = new Map<PolicyCameraImage, GPUBuffer[]>();
            for (const { offset, image } of assembled.cameras) {
                const activations = encoded.get(image) ?? this.encode(image, commands, pass.cameraBuffers);
                encoded.set(image, activations);
                const location = this.upload(new Uint32Array([offset]));
                pass.cameraBuffers.push(location);
                pass.cameras.push({ offset: location, activations });
                commands.copyBufferToBuffer(activations[activations.length - 1], 0, pass.features, offset * 4, this.cameraInputs * 4);
            }
            for (let t = 0; t < pass.steps; t++) {
                this.dispatch(commands, 'forward', [this.lstmWeights, pass.features, pass.states, this.times[t]], this.hidden);
            }
            this.dispatch(commands, 'output', [this.outputWeights, pass.states, pass.count, pass.outputs], Math.ceil(this.outputs / 64));
            this.device.queue.submit([commands.finish()]);
            return pass;
        } catch (error) { this.release(pass); throw error; }
    }

    learnTargets(input: DrivingPolicyInput, targets: PlannedActionTarget[]) {
        const values = new Array(this.config.actions.planTimesSeconds.length * 7).fill(0);
        for (const { step, action, shifts } of targets) {
            values.splice(step * 7, 7, 1, action.throttle, action.brake, action.steering,
                action.shiftUp ? 1 : action.shiftDown ? 2 : 0, Number(shifts.up), Number(shifts.down));
        }
        const pass = this.forwardPass(input), target = this.upload(values);
        try { this.update(pass, 'demonstrationGradient', [pass.outputs, target]); }
        finally { this.release(pass); this.destroy(target); }
    }

    learnTransition(pass: GpuPolicyPass, next: DrivingPolicyInput, target: {
        reward: number; elapsedSeconds: number; terminal: boolean;
        latent: number[]; shift: number; shifts: ShiftAvailability;
    }) {
        this.assertReady();
        if (pass.released) throw new Error('The GPU training sample was already released.');
        const bootstrap = target.terminal ? undefined : this.forwardPass(next);
        const values = this.upload([target.reward, target.elapsedSeconds, Number(target.terminal), ...target.latent,
            target.shift, Number(target.shifts.up), Number(target.shifts.down)]);
        try { this.update(pass, 'transitionGradient', [pass.outputs, bootstrap?.outputs ?? pass.outputs, values]); }
        finally { if (bootstrap) this.release(bootstrap); this.destroy(values); }
    }

    private update(pass: GpuPolicyPass, gradientShader: string, targets: GPUBuffer[]) {
        this.assertReady();
        if (pass.released) throw new Error('The GPU training sample was already released.');
        const commands = this.device.createCommandEncoder(), outputGradient = this.buffer(this.outputs * 4);
        const upstream = this.buffer(2 * this.hidden * 4), deltas = this.buffer(pass.steps * 4 * this.hidden * 4);
        const cameraGradients = this.layers.map((layer) => this.buffer(layer.channels * layer.columns * 4));
        const cameraDeltas = this.layers.map((layer) => this.buffer(layer.width * layer.height * layer.channels * 4));
        const projectionGradient = this.buffer(this.cameraInputs * (this.pooledInputs + 1) * 4);
        const projectionDelta = this.buffer(this.cameraInputs * 4), pooledDelta = this.buffer(this.pooledInputs * 4);
        try {
            this.dispatch(commands, gradientShader, [...targets, outputGradient], Math.ceil(this.outputs / 64));
            this.dispatch(commands, 'hiddenGradient', [this.outputWeights, outputGradient, upstream], Math.ceil(this.hidden / 64));
            for (let t = pass.steps - 1; t >= 0; t--) {
                this.dispatch(commands, 'gateGradient', [pass.states, upstream, deltas, this.times[t]], Math.ceil(this.hidden / 64));
                if (t > 0) this.dispatch(commands, 'recurrentGradient', [this.lstmWeights, deltas, upstream, this.times[t]], Math.ceil(this.hidden / 64));
            }
            for (const camera of pass.cameras) {
                this.dispatch(commands, 'cameraGradient', [this.lstmWeights, deltas, camera.offset, projectionDelta],
                    Math.ceil(this.cameraInputs / 64));
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
            this.layers.forEach((layer, l) => this.dispatch(commands, `cameraUpdate${l}`, [this.cameraWeights[l], cameraGradients[l]],
                Math.ceil(layer.channels * layer.columns / 64)));
            this.dispatch(commands, 'cameraProjectionUpdate', [this.cameraProjectionWeights, projectionGradient],
                Math.ceil(this.cameraInputs * (this.pooledInputs + 1) / 64));
            this.dispatch(commands, 'updateLstm', [this.lstmWeights, pass.features, pass.states, deltas, pass.count],
                Math.ceil(4 * this.hidden * (this.inputs + this.hidden + 1) / 64));
            this.dispatch(commands, 'updateOutput', [this.outputWeights, pass.states, outputGradient, pass.count], Math.ceil(this.outputs * (this.hidden + 1) / 64));
            this.device.queue.submit([commands.finish()]);
        } finally { [outputGradient, upstream, deltas, projectionGradient, projectionDelta, pooledDelta,
            ...cameraGradients, ...cameraDeltas].forEach((buffer) => this.destroy(buffer)); }
    }

    release(pass: GpuPolicyPass) {
        if (pass.released) return;
        pass.released = true;
        [pass.features, pass.states, pass.count, pass.outputs, ...pass.cameraBuffers].forEach((buffer) => this.destroy(buffer));
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
        const staging = this.buffer(buffers.reduce((size, buffer) => size + buffer.size, 0), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        try {
            const commands = this.device.createCommandEncoder(); let offset = 0;
            for (const buffer of buffers) { commands.copyBufferToBuffer(buffer, 0, staging, offset, buffer.size); offset += buffer.size; }
            this.device.queue.submit([commands.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            this.assertReady();
            return new Float32Array(staging.getMappedRange()).slice();
        } finally { this.destroy(staging); }
    }

    async weights(): Promise<Pick<PolicyModelState, 'cameraWeights' | 'cameraProjectionWeights' | 'lstmWeights' | 'outputWeights'>> {
        const values = await this.download([...this.cameraWeights, this.cameraProjectionWeights, this.lstmWeights, this.outputWeights]);
        if (!values.every(Number.isFinite)) throw new Error('Invalid GPU model weights; the checkpoint was not saved.');
        let offset = 0;
        const matrix = (rows: number, columns: number) => Array.from({ length: rows }, () => {
            const limit = this.config.learning.weightLimit;
            const row = Array.from(values.subarray(offset, offset + columns), (value) => Math.max(-limit, Math.min(limit, value)));
            offset += columns; return row;
        });
        return { cameraWeights: this.layers.map((layer) => matrix(layer.channels, layer.columns)),
            cameraProjectionWeights: matrix(this.cameraInputs, this.pooledInputs + 1),
            lstmWeights: matrix(4 * this.hidden, this.inputs + this.hidden + 1),
            outputWeights: matrix(this.outputs, this.hidden + 1) };
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.buffers.forEach((buffer) => buffer.destroy()); this.buffers.clear();
        this.device.destroy();
    }
}
