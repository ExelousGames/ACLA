import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { loadBackendPositionModel } from './backend-vision-model';
import { GpuInferenceError } from './track-vision-model';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import { visionAssetUrl } from './vision-assets';
import { POSITION_MODEL_NAMES, PositionModelVersion } from './vision-config';
import { VisionGpuInput } from './vision-gpu-input';
import type { PositionReferences, TrackPositionResult } from './track-vision-types';

// Python/Pillow training uses ties-to-even rounding for the letterboxed image.
const roundEven = (value: number) => value % 1 === 0.5 ? 2 * Math.round(value / 2) : Math.round(value);

export interface PositionContext {
    normalized_position: number;
    game?: string;
    map_name?: string | string[];
    /** World heading, pitch, roll in radians, captured with this frame. */
    orientation?: [number, number, number];
}

interface PositionFrame {
    references: Float32Array;
    context?: Float32Array;
    orientation?: Float32Array;
}

export class TrackPositionModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private canvas = document.createElement('canvas');
    private history: PositionFrame[] = [];
    private input: VisionGpuInput;
    private imageBuffers: GPUBuffer[] = [];
    private imageTensors: Tensor[] = [];
    private imageIndex = 0;

    get requiresOrientation() { return this.session.inputNames.includes('orientation'); }

    private constructor(private runtime: typeof import('onnxruntime-web'), private session: InferenceSession,
        readonly name: string, readonly inputSize: number, private categories: Record<string, string[]> = {},
        readonly sequenceLength = 1, private device: GPUDevice) {
        this.canvas.width = this.canvas.height = inputSize;
        this.input = new VisionGpuInput(device, runtime, inputSize);
        try {
            if (sequenceLength > 1) {
                // Alternate complete windows so the retained frames need only one GPU copy.
                for (let i = 0; i < 2; i++) {
                    const buffer = device.createBuffer({ size: Math.ceil(sequenceLength * 3 * inputSize ** 2 * 4 / 16) * 16,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
                    this.imageBuffers.push(buffer);
                    this.imageTensors.push(runtime.Tensor.fromGpuBuffer(buffer,
                        { dataType: 'float32', dims: [1, sequenceLength, 3, inputSize, inputSize] }));
                }
            }
        } catch (error) { this.disposeInputs(); throw error; }
    }

    static async load(version: PositionModelVersion = 'v1'): Promise<TrackPositionModel> {
        const loaded = await loadBackendPositionModel(version);
        if (!loaded) throw new Error(`No ONNX ${POSITION_MODEL_NAMES[version]} model has been uploaded for position.`);
        if (!('gpu' in navigator) || !navigator.gpu) throw new GpuInferenceError('GPU acceleration is unavailable for Track Vision.');
        return runWithVisionGpuQueue(async () => {
            const runtime = await import('onnxruntime-web/webgpu');
            runtime.env.wasm.wasmPaths = visionAssetUrl('vision-runtime/');
            runtime.env.wasm.numThreads = 1;
            runtime.env.wasm.proxy = false;
            runtime.env.webgpu.powerPreference = 'high-performance';
            let session: InferenceSession | undefined;
            let model: TrackPositionModel | undefined;
            try {
                session = await runtime.InferenceSession.create(loaded.bytes, { executionProviders: ['webgpu'] });
                model = new TrackPositionModel(runtime, session, loaded.metadata.name, loaded.inputSize,
                    loaded.categories, loaded.sequenceLength, await runtime.env.webgpu.device);
                // Run the actual graph instead of inferring compatibility from a version or input count.
                // Warm-up never becomes part of the captured history.
                await model.run(model.imageTensors[0] ?? model.input.tensor, [{ references: new Float32Array(3),
                    context: model.encodeContext({ normalized_position: 0 }),
                    orientation: model.requiresOrientation ? new Float32Array(3) : undefined }]);
                return model;
            } catch (error) {
                model?.disposeInputs();
                await session?.release().catch(() => undefined);
                throw new GpuInferenceError(`Track Vision GPU initialization failed. ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }

    private encodeContext(context?: PositionContext) {
        if (this.session.inputNames.includes('context')) {
            if (!context || !Number.isFinite(context.normalized_position)
                || context.normalized_position < 0 || context.normalized_position > 1) {
                throw new Error('Position inference requires normalized_position between 0 and 1.');
            }
            const values = [context.normalized_position];
            for (const field of ['game', 'map_name'] as const) {
                const vocabulary = this.categories[field] ?? [];
                const supplied = context[field];
                const candidates = Array.isArray(supplied) ? supplied : [supplied];
                // A map can have both a display name and a simulator key. Use one known label.
                const value = candidates.find((candidate) => candidate !== undefined && vocabulary.includes(candidate));
                values.push(...vocabulary.map((label) => Number(label === value)));
            }
            return new Float32Array(values);
        }
    }

    private prepareImages(firstFrame: boolean) {
        // Keep canvas high-quality resizing and ties-to-even letterboxing, without pixel readback.
        const image = this.input.prepare(this.canvas, { rect: [0, 0, 1, 1] }, false);
        if (this.sequenceLength === 1) return image;
        const next = 1 - this.imageIndex;
        const target = this.imageBuffers[next];
        const frameBytes = 3 * this.inputSize ** 2 * 4;
        const encoder = this.device.createCommandEncoder();
        if (firstFrame) {
            for (let i = 0; i < this.sequenceLength - 1; i++) {
                encoder.copyBufferToBuffer(image.gpuBuffer, 0, target, i * frameBytes, frameBytes);
            }
        } else {
            encoder.copyBufferToBuffer(this.imageBuffers[this.imageIndex], frameBytes, target, 0,
                (this.sequenceLength - 1) * frameBytes);
        }
        encoder.copyBufferToBuffer(image.gpuBuffer, 0, target, (this.sequenceLength - 1) * frameBytes, frameBytes);
        this.device.queue.submit([encoder.finish()]);
        this.imageIndex = next;
        return this.imageTensors[next];
    }

    private disposeInputs() {
        this.input.tensor.dispose();
        this.input.dispose();
        this.imageTensors.forEach((tensor) => tensor.dispose());
        this.imageBuffers.forEach((buffer) => buffer.destroy());
    }

    private async run(images: Tensor, frames: PositionFrame[]) {
        // Left padding matches training; each graph call resets its recurrent state.
        const window = [...Array<PositionFrame>(this.sequenceLength - frames.length).fill(frames[0]), ...frames];
        const stack = (select: (frame: PositionFrame) => Float32Array) => {
            const first = select(window[0]);
            if (this.sequenceLength === 1) return first;
            const data = new Float32Array(this.sequenceLength * first.length);
            window.forEach((frame, index) => data.set(select(frame), index * first.length));
            return data;
        };
        const batch = this.sequenceLength > 1 ? [1, this.sequenceLength] : [1];
        const feeds: Record<string, import('onnxruntime-web').Tensor> = {};
        let outputs: InferenceSession.ReturnType | undefined;
        try {
            feeds.references = new this.runtime.Tensor('float32', stack((frame) => frame.references), [...batch, 1, 3]);
            if (window[0].context) {
                feeds.context = new this.runtime.Tensor('float32', stack((frame) => frame.context!), [...batch, window[0].context.length]);
            }
            if (this.requiresOrientation) {
                feeds.orientation = new this.runtime.Tensor('float32', stack((frame) => frame.orientation!), [...batch, 3]);
            }
            // Images borrow persistent GPU storage; only the small telemetry tensors are temporary.
            outputs = await this.session.run({ images, ...feeds });
            const result = outputs.deltas;
            if (result?.dims.join(',') !== '1,1,3' || !(result.data instanceof Float32Array)
                || result.data.length !== 3 || !result.data.every(Number.isFinite)) throw new Error('Invalid track-position output.');
            // The model already predicts local axes; do not rotate by telemetry again.
            return { coordinateFrame: 'car_local' as const, referenceDelta: Array.from(result.data) as [number, number, number] };
        } finally {
            Object.values(feeds).forEach((input) => input.dispose());
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
        }
    }

    async detect(frame: HTMLCanvasElement, references: PositionReferences, positionContext?: PositionContext): Promise<TrackPositionResult> {
        if (this.disposed) throw new Error('The track-position model has been released.');
        if (this.busy) throw new Error('Track Vision inference is already running.');
        const points = new Float32Array([references.reference.x, references.reference.y, references.reference.z]);
        if (!points.every(Number.isFinite) || !frame.width || !frame.height) {
            this.resetHistory();
            throw new Error('Position inference requires a frame and finite reference points.');
        }
        const orientation = this.requiresOrientation && positionContext?.orientation
            ? new Float32Array(positionContext.orientation) : undefined;
        if (this.requiresOrientation && (!orientation || orientation.length !== 3 || !orientation.every(Number.isFinite))) {
            this.resetHistory();
            throw new Error('Position inference requires finite orientation: world heading, pitch, roll in radians.');
        }
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const context = this.canvas.getContext('2d');
            if (!context) throw new Error('A canvas is required for position inference.');
            const ratio = Math.min(this.inputSize / frame.width, this.inputSize / frame.height);
            const width = Math.max(1, roundEven(frame.width * ratio)), height = Math.max(1, roundEven(frame.height * ratio));
            context.fillStyle = 'rgb(114, 114, 114)';
            context.fillRect(0, 0, this.inputSize, this.inputSize);
            context.imageSmoothingEnabled = true;
            context.imageSmoothingQuality = 'high';
            context.drawImage(frame, Math.floor((this.inputSize - width) / 2), Math.floor((this.inputSize - height) / 2), width, height);
            const firstFrame = this.history.length === 0;
            this.history = [...this.history, { references: points, context: this.encodeContext(positionContext), orientation }]
                .slice(-this.sequenceLength);
            // Snapshot before queueing so resetHistory during an in-flight call stays effective.
            const frames = this.history;
            const position = await runWithVisionGpuQueue(() => this.run(this.prepareImages(firstFrame), frames));
            return { ...position, inferenceMs: performance.now() - started };
        })();
        this.pending = operation;
        try { return await operation; }
        catch (error) { this.resetHistory(); throw error; }
        finally { this.busy = false; }
    }

    resetHistory() {
        this.history = [];
    }

    async dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.resetHistory();
        await this.pending.catch(() => undefined);
        await runWithVisionGpuQueue(async () => {
            this.disposeInputs();
            await this.session.release();
        });
    }
}
