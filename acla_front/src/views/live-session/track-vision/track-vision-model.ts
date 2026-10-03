import type { InferenceSession } from 'onnxruntime-web';
import { letterbox, rgbaToChw, FloatTensor } from './yolo-segmentation';
import { decodeDepth, decodeSegments } from './vision-decoding';
import { DETECTION_TASKS, DetectionTask, VisionResult } from './track-vision-types';
import { readVisionModel, visionAssetUrl } from './vision-assets';
import { loadBackendVisionModel } from './backend-vision-model';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import { MaskRegion, resizeMask } from './world-mask';
import { DEPTH_INPUT_SIZE, VISION_INPUT_SIZE } from './vision-config';

export class GpuInferenceError extends Error {}

type ModelMetadata = { task: DetectionTask; name: string; classNames: string[]; inputSize: number };

export class TrackVisionModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private inputCanvas = document.createElement('canvas');
    readonly executionProvider = 'webgpu';

    private constructor(
        private runtime: typeof import('onnxruntime-web'),
        private session: InferenceSession,
        readonly task: DetectionTask,
        readonly name: string,
        readonly classNames: string[],
        readonly inputSize: number,
    ) {
        this.inputCanvas.width = this.inputSize;
        this.inputCanvas.height = this.inputSize;
    }

    static async loadBackend(inputSize = VISION_INPUT_SIZE): Promise<TrackVisionModel> {
        const { bytes, metadata } = await loadBackendVisionModel(inputSize);
        return TrackVisionModel.load(bytes, { task: 'segment', name: metadata.name, classNames: metadata.classNames, inputSize });
    }

    static async loadBuiltin(task: 'depth', inputSize = DEPTH_INPUT_SIZE): Promise<TrackVisionModel> {
        const definition = DETECTION_TASKS.find((definition): definition is Extract<typeof DETECTION_TASKS[number], { id: 'depth' }> => definition.id === task)!;
        const file = inputSize === DEPTH_INPUT_SIZE ? definition.file : definition.file.replace('.onnx', `-${inputSize}.onnx`);
        const bytes = await readVisionModel(visionAssetUrl(`vision-models/${file}`));
        return TrackVisionModel.load(bytes, { task, name: 'Depth-Anything-V2-Small', classNames: [], inputSize });
    }

    private static async load(bytes: ArrayBuffer, metadata: ModelMetadata): Promise<TrackVisionModel> {
        if (!('gpu' in navigator) || !navigator.gpu) {
            throw new GpuInferenceError('GPU acceleration is unavailable in the desktop app.');
        }
        try { return await TrackVisionModel.loadOnGpu(bytes, metadata); }
        catch (error) {
            throw new GpuInferenceError(`GPU inference failed. ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private static async loadOnGpu(bytes: ArrayBuffer, metadata: ModelMetadata) {
        return runWithVisionGpuQueue(async () => {
            const runtime = await import('onnxruntime-web/webgpu');
            runtime.env.wasm.wasmPaths = visionAssetUrl('vision-runtime/');
            runtime.env.wasm.numThreads = 1;
            runtime.env.wasm.proxy = false;
            runtime.env.webgpu.powerPreference = 'high-performance';
            const session = await runtime.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
            const model = new TrackVisionModel(runtime, session, metadata.task, metadata.name, metadata.classNames, metadata.inputSize);
            try {
                if (session.inputNames.length !== 1 || session.outputNames.length !== (metadata.task === 'segment' ? 2 : 1)) {
                    throw new Error(metadata.task === 'depth'
                        ? 'Unexpected depth model inputs or outputs. Re-export the bundled weights.'
                        : 'Unexpected segmentation model inputs or outputs. Upload compatible segmentation weights.');
                }
                await model.run(new Float32Array(3 * model.inputSize ** 2), 0.5);
                return model;
            } catch (error) {
                // Initialization already owns the queue; clean up without re-entering it.
                await session.release().catch(() => undefined);
                throw error;
            }
        });
    }

    private async run(input: Float32Array, threshold: number) {
        const tensor = new this.runtime.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]);
        let outputs: InferenceSession.ReturnType | undefined;
        try {
            outputs = await this.session.run({ [this.session.inputNames[0]]: tensor });
            const values: FloatTensor[] = Object.values(outputs).map((value) => {
                if (!(value.data instanceof Float32Array)) throw new Error('Track Vision requires float32 model outputs.');
                return { dims: value.dims, data: value.data };
            });
            if (this.task === 'depth') return decodeDepth(values[0]);
            const predictions = values.find(({ dims }) => dims.length === 3);
            const prototypes = values.find(({ dims }) => dims.length === 4);
            if (!predictions || !prototypes) throw new Error('Segment requires prediction and mask-prototype outputs.');
            return decodeSegments(predictions, prototypes, this.inputSize, threshold, this.classNames.length);
        } finally {
            tensor.dispose();
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
        }
    }

    async detect(frame: HTMLCanvasElement, threshold: number, region?: MaskRegion): Promise<VisionResult> {
        if (this.disposed) throw new Error('The vision model has been released.');
        if (this.busy) throw new Error('Vision inference is already running.');
        if (!frame.width || !frame.height) throw new Error('No captured frame is available.');
        if (this.task === 'depth' && !region) throw new Error('Depth requires the current frame\'s segmentation mask.');
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const context = this.inputCanvas.getContext('2d', { willReadFrequently: true });
            if (!context) throw new Error('A canvas is required for vision detection.');
            const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, VISION_INPUT_SIZE);
            // Scale the same letterbox geometry so depth stays aligned with segmentation.
            const scale = this.inputSize / VISION_INPUT_SIZE;
            context.fillStyle = 'rgb(114, 114, 114)';
            context.fillRect(0, 0, this.inputSize, this.inputSize);
            context.drawImage(frame, padX * scale, padY * scale, resizedWidth * scale, resizedHeight * scale);
            const input = rgbaToChw(context.getImageData(0, 0, this.inputSize, this.inputSize).data);
            if (this.task === 'depth' && region) {
                const mask = resizeMask(region, this.inputSize, this.inputSize);
                for (let i = 0; i < mask.length; i++) {
                    if (mask[i]) continue;
                    // Neutralize excluded pixels without cropping or changing the camera's image coordinates.
                    input[i] = input[i + mask.length] = input[i + mask.length * 2] = 114 / 255;
                }
                const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
                for (let channel = 0; channel < 3; channel++) {
                    const offset = channel * mask.length;
                    for (let i = 0; i < mask.length; i++) input[offset + i] = (input[offset + i] - mean[channel]) / std[channel];
                }
            }
            const result = await runWithVisionGpuQueue(() => this.run(input, threshold));
            if (result.task === 'depth' && region) {
                const mask = resizeMask(region, result.width, result.height);
                for (let i = 0; i < mask.length; i++) {
                    const x = (i % result.width + 0.5) / result.width * VISION_INPUT_SIZE;
                    const y = (Math.floor(i / result.width) + 0.5) / result.height * VISION_INPUT_SIZE;
                    if (!mask[i] || x < padX || x >= padX + resizedWidth || y < padY || y >= padY + resizedHeight) result.values[i] = 0;
                }
            }
            return { ...result, inferenceMs: performance.now() - started, classNames: this.classNames };
        })();
        this.pending = operation;
        try { return await operation; } finally { this.busy = false; }
    }

    async dispose() {
        if (this.disposed) return;
        this.disposed = true;
        await this.pending.catch(() => undefined);
        await runWithVisionGpuQueue(() => this.session.release());
    }
}
