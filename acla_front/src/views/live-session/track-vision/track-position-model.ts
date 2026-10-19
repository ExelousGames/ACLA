import type { InferenceSession } from 'onnxruntime-web';
import { loadBackendPositionModel } from './backend-vision-model';
import { GpuInferenceError } from './track-vision-model';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import { visionAssetUrl } from './vision-assets';
import { rgbaToChw } from './yolo-segmentation';
import type { PositionReferences, TrackPositionResult } from './track-vision-types';

// Python/Pillow training uses ties-to-even rounding for the letterboxed image.
const roundEven = (value: number) => value % 1 === 0.5 ? 2 * Math.round(value / 2) : Math.round(value);

export interface PositionContext {
    normalized_position: number;
    game?: string;
    map_name?: string | string[];
}

export class TrackPositionModel {
    private pending: Promise<unknown> = Promise.resolve();
    private disposed = false;
    private busy = false;
    private canvas = document.createElement('canvas');

    private constructor(private runtime: typeof import('onnxruntime-web'), private session: InferenceSession,
        readonly name: string, readonly inputSize: number, private categories: Record<string, string[]> = {}) {
        this.canvas.width = this.canvas.height = inputSize;
    }

    static async load(): Promise<TrackPositionModel> {
        const loaded = await loadBackendPositionModel();
        if (!loaded) throw new Error('No ONNX track-position model has been uploaded for position.');
        if (!('gpu' in navigator) || !navigator.gpu) throw new GpuInferenceError('GPU acceleration is unavailable for Track Vision.');
        return runWithVisionGpuQueue(async () => {
            const runtime = await import('onnxruntime-web/webgpu');
            runtime.env.wasm.wasmPaths = visionAssetUrl('vision-runtime/');
            runtime.env.wasm.numThreads = 1;
            runtime.env.wasm.proxy = false;
            runtime.env.webgpu.powerPreference = 'high-performance';
            let session: InferenceSession | undefined;
            try {
                session = await runtime.InferenceSession.create(loaded.bytes, { executionProviders: ['webgpu'] });
                const model = new TrackPositionModel(runtime, session, loaded.metadata.name, loaded.inputSize, loaded.categories);
                // Run the actual graph instead of inferring compatibility from a version or input count.
                await model.run(new Float32Array(3 * loaded.inputSize ** 2), new Float32Array(6), { normalized_position: 0 });
                return model;
            } catch (error) {
                await session?.release().catch(() => undefined);
                throw new GpuInferenceError(`Track Vision GPU initialization failed. ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }

    private async run(image: Float32Array, references: Float32Array, context?: PositionContext) {
        const feeds: Record<string, import('onnxruntime-web').Tensor> = {};
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
            feeds.context = new this.runtime.Tensor('float32', new Float32Array(values), [1, values.length]);
        }
        const images = new this.runtime.Tensor('float32', image, [1, 3, this.inputSize, this.inputSize]);
        const points = new this.runtime.Tensor('float32', references, [1, 2, 3]);
        feeds.images = images;
        feeds.references = points;
        let outputs: InferenceSession.ReturnType | undefined;
        try {
            outputs = await this.session.run(feeds);
            const result = outputs.deltas;
            if (result?.dims.join(',') !== '1,2,3' || !(result.data instanceof Float32Array)
                || result.data.length !== 6 || !result.data.every(Number.isFinite)) throw new Error('Invalid track-position output.');
            return { referenceDelta: Array.from(result.data.slice(0, 3)) as [number, number, number],
                centerlineDelta: Array.from(result.data.slice(3, 6)) as [number, number, number] };
        } finally {
            Object.values(feeds).forEach((input) => input.dispose());
            if (outputs) Object.values(outputs).forEach((output) => output.dispose());
        }
    }

    async detect(frame: HTMLCanvasElement, references: PositionReferences, positionContext?: PositionContext): Promise<TrackPositionResult> {
        if (this.disposed) throw new Error('The track-position model has been released.');
        if (this.busy) throw new Error('Track Vision inference is already running.');
        const points = new Float32Array([references.reference.x, references.reference.y, references.reference.z,
            references.centerline.x, references.centerline.y, references.centerline.z]);
        if (!points.every(Number.isFinite) || !frame.width || !frame.height) throw new Error('Position inference requires a frame and finite reference points.');
        this.busy = true;
        const operation = (async () => {
            const started = performance.now();
            const context = this.canvas.getContext('2d', { willReadFrequently: true });
            if (!context) throw new Error('A canvas is required for position inference.');
            const ratio = Math.min(this.inputSize / frame.width, this.inputSize / frame.height);
            const width = Math.max(1, roundEven(frame.width * ratio)), height = Math.max(1, roundEven(frame.height * ratio));
            context.fillStyle = 'rgb(114, 114, 114)';
            context.fillRect(0, 0, this.inputSize, this.inputSize);
            context.imageSmoothingEnabled = true;
            context.imageSmoothingQuality = 'high';
            context.drawImage(frame, Math.floor((this.inputSize - width) / 2), Math.floor((this.inputSize - height) / 2), width, height);
            const image = rgbaToChw(context.getImageData(0, 0, this.inputSize, this.inputSize).data);
            const position = await runWithVisionGpuQueue(() => this.run(image, points, positionContext));
            return { ...position, inferenceMs: performance.now() - started };
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
