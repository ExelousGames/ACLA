import type { InferenceSession } from 'onnxruntime-web';
import { letterbox, rgbaToChw, FloatTensor } from './yolo-segmentation';
import { decodeDepth, decodeSegments, decodeSemantic } from './vision-decoding';
import { DETECTION_TASKS, DetectionTask, VisionResult } from './track-vision-types';
import { readVisionModel, visionAssetUrl } from './vision-assets';
import { runWithVisionGpuQueue } from './vision-gpu-queue';

export const VISION_INPUT_SIZE = 640;

export class GpuInferenceError extends Error {}

export class TrackVisionModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private inputCanvas = document.createElement('canvas');

    private constructor(
        private runtime: typeof import('onnxruntime-web'),
        private session: InferenceSession,
        readonly task: DetectionTask,
        readonly executionProvider: 'webgpu' | 'wasm',
        readonly fallbackReason?: string,
        readonly classNames: Record<number, string> = {},
    ) {
        this.inputCanvas.width = VISION_INPUT_SIZE;
        this.inputCanvas.height = VISION_INPUT_SIZE;
    }

    static async loadBuiltin(task: DetectionTask, allowCpuFallback = false): Promise<TrackVisionModel> {
        const definition = DETECTION_TASKS.find(({ id }) => id === task)!;
        let bytes: ArrayBuffer;
        let classNames: Record<number, string>;
        try {
            const [model, metadata] = await Promise.all([
                readVisionModel(visionAssetUrl(`vision-models/${definition.file}`)),
                readVisionModel(visionAssetUrl(`vision-models/${definition.file.replace('.onnx', '.json')}`)),
            ]);
            bytes = model;
            classNames = JSON.parse(new TextDecoder().decode(metadata)).names;
        }
        catch { throw new Error(`${definition.label} weights unavailable. Run npm run setup:vision-models, then restart the app.`); }
        let fallbackReason = 'GPU acceleration is unavailable in the desktop app.';
        if ('gpu' in navigator && navigator.gpu) {
            try { return await TrackVisionModel.loadWithProvider(bytes, task, 'webgpu', undefined, classNames); }
            catch (error) {
                if (!allowCpuFallback) throw new GpuInferenceError(`GPU inference failed. ${error instanceof Error ? error.message : String(error)}`);
                fallbackReason = 'GPU could not run this model; using CPU.';
            }
        }
        if (!allowCpuFallback) throw new GpuInferenceError(fallbackReason);
        return TrackVisionModel.loadWithProvider(bytes, task, 'wasm', fallbackReason, classNames);
    }

    private static async loadWithProvider(bytes: ArrayBuffer, task: DetectionTask, provider: 'webgpu' | 'wasm', fallbackReason?: string, classNames?: Record<number, string>) {
        return runWithVisionGpuQueue(provider, async () => {
            const runtime = provider === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm');
            runtime.env.wasm.wasmPaths = visionAssetUrl('vision-runtime/');
            runtime.env.wasm.numThreads = 1;
            runtime.env.wasm.proxy = provider === 'wasm';
            if (provider === 'webgpu') runtime.env.webgpu.powerPreference = 'high-performance';
            const session = await runtime.InferenceSession.create(bytes, { executionProviders: [provider] });
            const model = new TrackVisionModel(runtime, session, task, provider, fallbackReason, classNames);
            try {
                if (session.inputNames.length !== 1 || session.outputNames.length !== (task === 'segment' ? 2 : 1)) {
                    throw new Error(`Unexpected ${task} model inputs or outputs. Re-export the bundled weights.`);
                }
                await model.run(new Float32Array(3 * VISION_INPUT_SIZE ** 2), 0.5);
                return model;
            } catch (error) {
                // Initialization already owns the queue; clean up without re-entering it.
                await session.release().catch(() => undefined);
                throw error;
            }
        });
    }

    private async run(input: Float32Array, threshold: number) {
        const tensor = new this.runtime.Tensor('float32', input, [1, 3, VISION_INPUT_SIZE, VISION_INPUT_SIZE]);
        let outputs: InferenceSession.ReturnType | undefined;
        try {
            outputs = await this.session.run({ [this.session.inputNames[0]]: tensor });
            if (this.task === 'semantic') {
                const value = Object.values(outputs)[0];
                if (value.type !== 'float32' && value.type !== 'int64' && value.type !== 'int32' && value.type !== 'uint8') {
                    throw new Error('Unsupported semantic tensor type.');
                }
                return decodeSemantic({ dims: value.dims, data: value.data as Float32Array | BigInt64Array | Int32Array | Uint8Array });
            }
            const values: FloatTensor[] = Object.values(outputs).map((value) => {
                if (!(value.data instanceof Float32Array)) throw new Error('Track Vision requires float32 model outputs.');
                return { dims: value.dims, data: value.data };
            });
            if (this.task === 'depth') return decodeDepth(values[0]);
            const predictions = values.find(({ dims }) => dims.length === 3);
            const prototypes = values.find(({ dims }) => dims.length === 4);
            if (!predictions || !prototypes) throw new Error('Segment requires prediction and mask-prototype outputs.');
            return decodeSegments(predictions, prototypes, VISION_INPUT_SIZE, threshold);
        } finally {
            tensor.dispose();
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
        }
    }

    async detect(frame: HTMLCanvasElement, threshold: number): Promise<VisionResult> {
        if (this.disposed) throw new Error('The vision model has been released.');
        if (this.busy) throw new Error('Vision inference is already running.');
        if (!frame.width || !frame.height) throw new Error('No captured frame is available.');
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const context = this.inputCanvas.getContext('2d', { willReadFrequently: true });
            if (!context) throw new Error('A canvas is required for vision detection.');
            const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, VISION_INPUT_SIZE);
            context.fillStyle = 'rgb(114, 114, 114)';
            context.fillRect(0, 0, VISION_INPUT_SIZE, VISION_INPUT_SIZE);
            context.drawImage(frame, padX, padY, resizedWidth, resizedHeight);
            const input = rgbaToChw(context.getImageData(0, 0, VISION_INPUT_SIZE, VISION_INPUT_SIZE).data);
            const result = await runWithVisionGpuQueue(this.executionProvider, () => this.run(input, threshold));
            return { ...result, inferenceMs: performance.now() - started, classNames: this.classNames };
        })();
        this.pending = operation;
        try { return await operation; } finally { this.busy = false; }
    }

    async dispose() {
        if (this.disposed) return;
        this.disposed = true;
        await this.pending.catch(() => undefined);
        await runWithVisionGpuQueue(this.executionProvider, () => this.session.release());
    }
}
