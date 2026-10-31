import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { loadBackendSegmentationModel } from './backend-vision-model';
import { DepthResult, DETECTION_TASKS, SegmentResult, VisionResult } from './track-vision-types';
import { readVisionModel, visionAssetUrl } from './vision-assets';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import { MaskRegion } from './world-mask';
import { DEPTH_INPUT_SIZE } from './vision-config';
import { GpuImageGeometry, imageGeometry, VisionGpuInput } from './vision-gpu-input';
import { GpuDepth, gpuDepthShape } from './vision-gpu-depth';
import { GpuSegments, gpuSegmentShape } from './vision-gpu-segments';
import type { VisionModelTiming } from './vision-timing';

export class GpuInferenceError extends Error {}

type ModelMetadata = { task: 'segment' | 'depth'; name: string; classNames: string[]; inputSize: number };

export class TrackVisionModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private input: VisionGpuInput;
    private depths = new Set<GpuDepth>();
    private segments = new Set<GpuSegments>();
    readonly executionProvider = 'webgpu';

    private constructor(
        runtime: typeof import('onnxruntime-web'),
        private session: InferenceSession,
        readonly task: ModelMetadata['task'],
        readonly name: string,
        readonly classNames: string[],
        readonly inputSize: number,
        private device: GPUDevice,
    ) {
        this.input = new VisionGpuInput(device, runtime, inputSize);
    }

    static async loadBuiltin(task: 'depth', inputSize = DEPTH_INPUT_SIZE): Promise<TrackVisionModel> {
        const definition = DETECTION_TASKS.find((definition): definition is Extract<typeof DETECTION_TASKS[number], { id: 'depth' }> => definition.id === task)!;
        const file = inputSize === DEPTH_INPUT_SIZE ? definition.file : definition.file.replace('.onnx', `-${inputSize}.onnx`);
        const bytes = await readVisionModel(visionAssetUrl(`vision-models/${file}`));
        return TrackVisionModel.load(bytes, { task, name: 'Depth-Anything-V2-Small', classNames: [], inputSize });
    }

    static async loadBackend(): Promise<TrackVisionModel> {
        const loaded = await loadBackendSegmentationModel();
        if (!loaded) throw new Error('No ONNX track-segments model has been uploaded for segmentation.');
        return TrackVisionModel.load(loaded.bytes, { task: 'segment', name: loaded.metadata.name,
            classNames: loaded.metadata.classNames, inputSize: loaded.inputSize });
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
            const session = await runtime.InferenceSession.create(bytes, { executionProviders: ['webgpu'],
                preferredOutputLocation: 'gpu-buffer' });
            let model: TrackVisionModel | undefined;
            try {
                model = new TrackVisionModel(runtime, session, metadata.task, metadata.name, metadata.classNames, metadata.inputSize,
                    await runtime.env.webgpu.device);
                if (session.inputNames.length !== 1 || session.outputNames.length !== (metadata.task === 'segment' ? 2 : 1)) {
                    throw new Error(`Unexpected ${metadata.task} model inputs or outputs.`);
                }
                // WebGPU buffers are zero-initialized. Warm up without uploading an image.
                await model.run(model.input.tensor);
                return model;
            } catch (error) {
                // Initialization already owns the queue; clean up without re-entering it.
                model?.input.dispose();
                await session.release().catch(() => undefined);
                throw error;
            }
        });
    }

    private async run(tensor: Tensor, threshold = 0.5, geometry?: GpuImageGeometry, timing?: VisionModelTiming) {
        let outputs: InferenceSession.ReturnType | undefined;
        let outputStarted: number | undefined;
        try {
            const runStarted = performance.now();
            outputs = await this.session.run({ [this.session.inputNames[0]]: tensor });
            outputStarted = performance.now();
            if (timing) timing.runMs = outputStarted - runStarted;
            if (this.task === 'depth') {
                const name = this.session.outputNames[0], output = outputs[name];
                const result: DepthResult = { task: 'depth', scale: 'relative', ...gpuDepthShape(output) };
                if (geometry) {
                    const depth: GpuDepth = new GpuDepth(this.device, output, geometry, () => this.depths.delete(depth));
                    this.depths.add(depth);
                    result.gpu = depth;
                    // The captured frame now owns this tensor, including any deferred readback.
                    outputs = undefined;
                }
                return result;
            }
            const [predictions, prototypes] = this.session.outputNames.map((name) => outputs![name]);
            const { width, height } = gpuSegmentShape(predictions, prototypes, this.classNames.length);
            const result: SegmentResult = { task: 'segment', width, height, instances: [] };
            const segment: GpuSegments = new GpuSegments(this.device, predictions, prototypes, this.inputSize,
                threshold, this.classNames.length, () => this.segments.delete(segment));
            if (geometry) {
                this.segments.add(segment);
                result.gpu = segment;
            } else await segment.dispose(); // Compile and submit the mask passes during warm-up too.
            // All uses of the raw outputs have been submitted on the same GPU queue.
            // ORT can recycle them after those commands; the frame owns the decoded buffers.
            return result;
        } finally {
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
            if (timing && outputStarted !== undefined) timing.outputMs = performance.now() - outputStarted;
        }
    }

    /** Omitting the region keeps the full frame; depth always discards letterbox padding. */
    async detect(frame: HTMLCanvasElement, threshold: number, region?: MaskRegion): Promise<VisionResult> {
        if (this.disposed) throw new Error('The vision model has been released.');
        if (this.busy) throw new Error('Vision inference is already running.');
        if (!frame.width || !frame.height) throw new Error('No captured frame is available.');
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const geometry = imageGeometry(frame, this.inputSize, this.task === 'depth', region);
            const queuedAt = performance.now();
            const timing: VisionModelTiming = { inputMs: queuedAt - started, queueMs: 0, runMs: 0, outputMs: 0 };
            const result = await runWithVisionGpuQueue(() => {
                const inputStarted = performance.now();
                timing.queueMs = inputStarted - queuedAt;
                const tensor = this.input.prepare(frame, geometry, this.task === 'depth');
                timing.inputMs += performance.now() - inputStarted;
                return this.run(tensor, threshold, geometry, timing);
            });
            return { ...result, inferenceMs: performance.now() - started, inferenceTiming: timing, classNames: this.classNames };
        })();
        this.pending = operation;
        try { return await operation; } finally { this.busy = false; }
    }

    async dispose() {
        if (this.disposed) return;
        this.disposed = true;
        await this.pending.catch(() => undefined);
        await Promise.all(Array.from(this.depths, (depth) => depth.dispose()));
        await Promise.all(Array.from(this.segments, (segment) => segment.dispose()));
        await runWithVisionGpuQueue(async () => {
            this.input.dispose();
            await this.session.release();
        });
    }
}
