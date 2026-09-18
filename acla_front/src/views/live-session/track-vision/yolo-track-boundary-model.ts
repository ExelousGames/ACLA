import type { InferenceSession } from 'onnxruntime-web';
import { decodeTrackMask, letterbox, rgbaToChw, traceTrackBoundaries, TrackBoundaryDetection } from './yolo-segmentation';
import { decodeYolopMask, rgbaToYolopInput } from './yolop-segmentation';
import { readVisionModel, visionAssetUrl } from './vision-assets';
import { runWithVisionGpuQueue } from './vision-gpu-queue';

const BUILTIN_INPUT_SIZE = 320;

/** Reusable local inference engine; capture and UI are owned by the panel. */
export class YoloTrackBoundaryModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private inputCanvas = document.createElement('canvas');

    private constructor(
        private runtime: typeof import('onnxruntime-web'),
        private session: InferenceSession,
        private format: 'yolo-seg' | 'yolop',
        private inputSize: number,
        readonly executionProvider: 'webgpu' | 'wasm',
        readonly fallbackReason?: string,
    ) {
        this.inputCanvas.width = inputSize;
        this.inputCanvas.height = inputSize;
    }

    static async loadBuiltin(): Promise<YoloTrackBoundaryModel> {
        const bytes = await readVisionModel(visionAssetUrl('vision-models/yolop-320-320.onnx'));
        return YoloTrackBoundaryModel.load(bytes, 'yolop');
    }

    static async load(bytes: ArrayBuffer, format: 'yolo-seg' | 'yolop' = 'yolo-seg'): Promise<YoloTrackBoundaryModel> {
        let fallbackReason = 'GPU acceleration is unavailable in the desktop app.';
        if (typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
            try {
                return await YoloTrackBoundaryModel.loadWithProvider(bytes, format, 'webgpu');
            } catch (error) {
                console.warn('Track vision GPU initialization failed; falling back to CPU.', error);
                fallbackReason = 'GPU could not run this model; using CPU.';
            }
        }
        try {
            return await YoloTrackBoundaryModel.loadWithProvider(bytes, format, 'wasm', fallbackReason);
        } catch (error) {
            const guidance = format === 'yolop' ? 'Run npm run setup:vision to restore the bundled weights.'
                : 'Export a float32, 640px YOLOv8/YOLO11 segmentation ONNX model without NMS.';
            throw new Error(`Unable to load model. ${guidance} ${error instanceof Error ? error.message : ''}`);
        }
    }

    private static async loadWithProvider(
        bytes: ArrayBuffer, format: 'yolo-seg' | 'yolop', executionProvider: 'webgpu' | 'wasm', fallbackReason?: string,
    ): Promise<YoloTrackBoundaryModel> {
        return runWithVisionGpuQueue(executionProvider, async () => {
            // Separate runtime bundles keep WebGPU initialization and CPU proxy state independent.
            const runtime = executionProvider === 'webgpu' ? await import('onnxruntime-web/webgpu')
                : await import('onnxruntime-web/wasm');
            runtime.env.wasm.wasmPaths = visionAssetUrl('vision-runtime/');
            runtime.env.wasm.numThreads = 1;
            // ONNX's proxy worker cannot run WebGPU. Keep CPU fallback off the React thread.
            runtime.env.wasm.proxy = executionProvider === 'wasm';
            if (executionProvider === 'webgpu') runtime.env.webgpu.powerPreference = 'high-performance';
            const session = await runtime.InferenceSession.create(bytes, { executionProviders: [executionProvider] });
            const size = format === 'yolop' ? BUILTIN_INPUT_SIZE : 640;
            const model = new YoloTrackBoundaryModel(runtime, session, format, size, executionProvider, fallbackReason);
            try {
                if (format === 'yolop' && (session.inputNames.length !== 1 || !session.outputNames.includes('drive_area_seg'))) {
                    throw new Error('The built-in YOLOP model is missing its drivable-area output.');
                }
                if (format === 'yolo-seg' && (session.inputNames.length !== 1 || session.outputNames.length !== 2)) {
                    throw new Error('Choose a YOLOv8 or YOLO11 segmentation model with one image input and two raw outputs.');
                }
                // Validate the input size and output contract before allowing capture inference.
                await model.run(new Float32Array(3 * size * size), 0, 0.5);
                return model;
            } catch (error) {
                // Initialization already owns the queue; clean up without re-entering it.
                await session.release().catch(() => undefined);
                throw error;
            }
        });
    }

    private async run(input: Float32Array, classId: number, threshold: number) {
        const tensor = new this.runtime.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]);
        let outputs: InferenceSession.ReturnType | undefined;
        try {
            outputs = await this.session.run({ [this.session.inputNames[0]]: tensor },
                this.format === 'yolop' ? ['drive_area_seg'] : this.session.outputNames);
            if (this.format === 'yolop') {
                const road = outputs.drive_area_seg;
                if (!road || !(road.data instanceof Float32Array)) throw new Error('YOLOP must return float32 road probabilities.');
                return decodeYolopMask({ dims: road.dims, data: road.data }, threshold);
            }
            const values = Object.values(outputs);
            const predictions = values.find((value) => value.dims.length === 3);
            const prototypes = values.find((value) => value.dims.length === 4);
            if (!predictions || !prototypes || !(predictions.data instanceof Float32Array) || !(prototypes.data instanceof Float32Array)) {
                throw new Error('The model must return float32 detection and mask-prototype tensors.');
            }
            return decodeTrackMask(
                { dims: predictions.dims, data: predictions.data },
                { dims: prototypes.dims, data: prototypes.data }, this.inputSize, classId, threshold,
            );
        } finally {
            tensor.dispose();
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
        }
    }

    async detect(frame: HTMLCanvasElement, classId = 0, threshold = 0.5): Promise<TrackBoundaryDetection> {
        if (this.disposed) throw new Error('The track model has been released.');
        if (this.busy) throw new Error('Track inference is already running.');
        if (!frame.width || !frame.height) throw new Error('No captured frame is available.');
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const capturedAt = Date.now();
            const context = this.inputCanvas.getContext('2d', { willReadFrequently: true });
            if (!context) throw new Error('A canvas is required for track detection.');
            const size = this.inputSize;
            const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, size);
            context.fillStyle = 'rgb(114, 114, 114)';
            context.fillRect(0, 0, size, size);
            context.drawImage(frame, padX, padY, resizedWidth, resizedHeight);
            const rgba = context.getImageData(0, 0, size, size).data;
            const input = this.format === 'yolop' ? rgbaToYolopInput(rgba) : rgbaToChw(rgba);
            const { mask, width, height, confidence } = await runWithVisionGpuQueue(this.executionProvider, () => this.run(input, classId, threshold));
            const segments = traceTrackBoundaries(mask, width, height, frame.width, frame.height, size);
            return {
                capturedAt, width: frame.width, height: frame.height,
                inferenceMs: performance.now() - started, confidence: segments.length ? confidence : 0, segments,
            };
        })();
        this.pending = operation;
        try { return await operation; } finally { this.busy = false; }
    }

    async dispose(): Promise<void> {
        if (this.disposed) return;
        this.disposed = true;
        await this.pending.catch(() => undefined);
        await runWithVisionGpuQueue(this.executionProvider, () => this.session.release());
    }
}
