import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { readVisionModel } from './vision-assets';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import * as worldMask from './world-mask';
import { loadBackendSegmentationModel } from './backend-vision-model';
import { mockGpuDevice } from './vision-gpu-test-device';
import { readDepthValues } from './vision-gpu-depth';
import { runWithVisionGpuQueue } from './vision-gpu-queue';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: Object.assign(jest.fn(), { fromGpuBuffer: jest.fn() }),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./vision-assets', () => ({ ...jest.requireActual('./vision-assets'), readVisionModel: jest.fn() }));
jest.mock('./backend-vision-model', () => ({ loadBackendSegmentationModel: jest.fn() }));

const tensor = (dims: number[], data: Float32Array | Uint8Array) => ({ dims, data, type: data instanceof Float32Array ? 'float32' : 'uint8',
    location: 'gpu-buffer', gpuBuffer: {}, getData: jest.fn().mockResolvedValue(data), dispose: jest.fn() });
const session = () => ({
    inputNames: ['images'], outputNames: ['output0'],
    run: jest.fn().mockResolvedValue({
        output0: tensor([1, 2, 2], new Float32Array(4)),
    }),
    release: jest.fn().mockResolvedValue(undefined),
});

it('loads segmentation ONNX separately and runs only its image input at its published resolution', async () => {
    const bytes = new ArrayBuffer(16);
    (loadBackendSegmentationModel as jest.Mock).mockResolvedValue({ bytes, inputSize: 512,
        metadata: { name: 'track-segments', classNames: ['track', 'curb'] } });
    gpu.outputNames = ['output0', 'output1'];
    const predictions = tensor([1, 7, 1], new Float32Array([256, 256, 256, 256, 0.9, 0, 1]));
    const prototypes = tensor([1, 1, 4, 4], new Float32Array(16).fill(1));
    gpu.run.mockResolvedValue({ output1: prototypes, output0: predictions });
    const drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage, getImageData: () => ({ data: new Uint8ClampedArray(512 * 512 * 4).fill(255) }),
    } as any);
    const model = await TrackVisionModel.loadBackend();
    const frame = document.createElement('canvas');
    frame.width = 1280; frame.height = 720;
    let now = 100;
    jest.spyOn(performance, 'now').mockImplementation(() => now);
    device.calls.queue.copyExternalImageToTexture.mockImplementation(() => { now += 3; });
    gpu.run.mockImplementation(async () => { now += 20; return { output1: prototypes, output0: predictions }; });
    Object.defineProperty(predictions, 'data', { get() { throw new Error('CPU prediction access'); } });
    Object.defineProperty(prototypes, 'data', { get() { throw new Error('CPU prototype access'); } });
    const priorWork = runWithVisionGpuQueue(async () => { now += 11; });
    const result = await model.detect(frame, 0.5);
    await priorWork;
    expect(result.inferenceTiming).toEqual({ queueMs: 11, inputMs: 3, runMs: 20, outputMs: 0 });
    expect(result.inferenceMs).toBe(34);
    expect(gpuRuntime.InferenceSession.create).toHaveBeenCalledWith(bytes, { executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer' });
    expect(readVisionModel).not.toHaveBeenCalled();
    expect(gpu.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, 512, 512] }) });
    expect(drawImage).not.toHaveBeenCalled();
    expect(device.calls.queue.copyExternalImageToTexture).toHaveBeenCalledWith({ source: frame }, expect.anything(), [1280, 720]);
    expect(gpu.run.mock.calls[1][0].images.location).toBe('gpu-buffer');
    expect(device.calls.queue.writeBuffer).toHaveBeenCalledWith(expect.anything(), 0,
        new Float32Array([512, 512, 0, 0, 0, 112 / 512, 1, 288 / 512, 0, 0, 0, 0]));
    expect(result).toMatchObject({ task: 'segment', classNames: ['track', 'curb'], width: 4, height: 4,
        instances: [], gpu: expect.anything() });
    expect(predictions.getData).not.toHaveBeenCalled();
    expect(prototypes.getData).not.toHaveBeenCalled();
    expect(device.calls.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    expect(device.calls.createBuffer.mock.results.every(({ value }) => value.mapAsync.mock.calls.length === 0)).toBe(true);
    expect(await model.detect(frame, 0.95)).toMatchObject({ instances: [] });
    expect(predictions.dispose).toHaveBeenCalledTimes(3);
    expect(prototypes.dispose).toHaveBeenCalledTimes(3);
    await model.dispose();
    expect(gpu.release).toHaveBeenCalledTimes(1);
});

it('reports missing segmentation independently of position', async () => {
    (loadBackendSegmentationModel as jest.Mock).mockResolvedValue(null);
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('No ONNX track-segments model');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('releases a segmentation session when its output does not match its labels', async () => {
    (loadBackendSegmentationModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(16), inputSize: 512,
        metadata: { name: 'track-segments', classNames: ['track', 'curb'] } });
    gpu.outputNames = ['output0', 'output1'];
    gpu.run.mockResolvedValue({ output0: tensor([1, 8, 1], new Float32Array(8)), output1: tensor([1, 1, 4, 4], new Float32Array(16)) });
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('backend labels');
    expect(gpu.release).toHaveBeenCalledTimes(1);
});
let gpu: ReturnType<typeof session>;
let device: ReturnType<typeof mockGpuDevice>;
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');

beforeEach(() => {
    device = mockGpuDevice();
    (gpuRuntime.env.webgpu as any).device = device.device;
    (gpuRuntime.Tensor.fromGpuBuffer as jest.Mock).mockImplementation((gpuBuffer, options) => ({
        dims: options.dims, type: options.dataType, location: 'gpu-buffer', gpuBuffer,
    }));
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    gpu = session();
    (gpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(gpu);
    (gpuRuntime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    (readVisionModel as jest.Mock).mockResolvedValue(new ArrayBuffer(8));
});

afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('rejects GPU initialization failure without creating a CPU session', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No GPU adapter'));
    await expect(TrackVisionModel.loadBuiltin('depth')).rejects.toThrow('No GPU adapter');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('releases failed GPU warm-up without falling back to CPU', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    await expect(TrackVisionModel.loadBuiltin('depth')).rejects.toBeInstanceOf(GpuInferenceError);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('requires WebGPU and never initializes CPU inference when it is unavailable', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    await expect(TrackVisionModel.loadBuiltin('depth')).rejects.toBeInstanceOf(GpuInferenceError);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('preserves unlabelled depth input and output while excluding the interior', async () => {
    const output = tensor([1, 2, 2], new Float32Array([1, 5, 15, 50]));
    const depthSession = gpu;
    gpu.outputNames = ['output0'];
    depthSession.run.mockResolvedValue({ output0: output });
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage: jest.fn(), getImageData: () => ({ data: new Uint8ClampedArray(518 * 518 * 4).fill(255) }),
    } as any);
    const model = await TrackVisionModel.loadBuiltin('depth');
    expect(model.executionProvider).toBe('webgpu');
    expect(model.name).toBe('Depth-Anything-V2-Small');
    expect(readVisionModel).toHaveBeenCalledWith('http://localhost/vision-models/depth-anything-v2-small.onnx');
    expect(depthSession.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 518, 518] }) });
    const frame = document.createElement('canvas');
    frame.width = frame.height = 640;
    const region = worldMask.createWorldMask({ task: 'segment', width: 2, height: 2, classNames: ['track', 'car interior', 'fence'], instances: [
        { classId: 0, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([1, 1, 0, 0]) },
        { classId: 1, confidence: 0.6, box: [0, 0, 1, 1], mask: new Uint8Array([0, 1, 0, 0]) },
        { classId: 2, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([0, 0, 0, 1]) },
    ] })!;
    const result = await model.detect(frame, 0.5, region);
    expect(depthSession.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, 518, 518] }) });
    expect(gpuRuntime.InferenceSession.create).toHaveBeenCalledWith(expect.any(ArrayBuffer), {
        executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer',
    });
    expect(output.getData).not.toHaveBeenCalled();
    expect(device.calls.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    expect(result).not.toHaveProperty('values');
    if (result.task !== 'depth') throw new Error('depth');
    await Promise.all([readDepthValues(result), readDepthValues(result)]);
    expect(output.getData).toHaveBeenCalledTimes(1);
    expect(output.data).toEqual(new Float32Array([1, 5, 15, 50]));
    output.data.fill(0);
    expect(result).toMatchObject({ task: 'depth', scale: 'relative', width: 2, height: 2, values: new Float32Array([1 / 2, 0, 1 / 16, 1 / 51]) });
    expect(output.dispose).toHaveBeenCalledTimes(1); // Warm-up only; the live frame owns its tensor.
    await model.dispose();
    expect(output.dispose).toHaveBeenCalledTimes(2);
    expect(depthSession.release).toHaveBeenCalledTimes(1);
});

it.each([252, 392, 518].flatMap((inputSize) => [[1280, 640, inputSize], [640, 1280, inputSize], [640, 640, inputSize]]))('matches all-ones masking without resizing masks and removes padding for a %i x %i capture at %i pixels', async (width, height, inputSize) => {
    const resizeMask = jest.spyOn(worldMask, 'resizeMask');
    gpu.outputNames = ['output0'];
    gpu.run.mockResolvedValue({ output0: tensor([1, 1, 4, 4], new Float32Array(16).fill(10)) });
    const rgba = Uint8ClampedArray.from({ length: inputSize * inputSize * 4 }, (_, i) => i % 256);
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage: jest.fn(), getImageData: () => ({ data: rgba }),
    } as any);
    const model = await TrackVisionModel.loadBuiltin('depth', inputSize);
    expect(readVisionModel).toHaveBeenCalledWith(`http://localhost/vision-models/depth-anything-v2-small${inputSize === 518 ? '' : `-${inputSize}`}.onnx`);
    const frame = document.createElement('canvas');
    frame.width = width; frame.height = height;
    const result = await model.detect(frame, 0.5);
    expect(resizeMask).not.toHaveBeenCalled();
    expect(gpu.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, inputSize, inputSize] }) });
    if (result.task !== 'depth') throw new Error('depth');
    await readDepthValues(result);
    const expected = width === height ? Array(16).fill(10)
        : width > height ? [0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0]
            : [0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0];
    expect(result.values).toEqual(Float32Array.from(expected, (value) => value ? 1 / 11 : 0));
    const masked = await model.detect(frame, 0.5, { width: 2, height: 2, mask: new Uint8Array(4).fill(1) });
    expect(resizeMask).not.toHaveBeenCalled();
    expect(gpu.run.mock.calls[2][0].images).toBe(gpu.run.mock.calls[1][0].images);
    if (masked.task !== 'depth') throw new Error('depth');
    await readDepthValues(masked);
    expect(masked).toMatchObject({ task: 'depth', width: result.width, height: result.height, values: result.values });
    await model.dispose();
});
