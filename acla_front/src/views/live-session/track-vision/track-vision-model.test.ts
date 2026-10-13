import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { loadBackendVisionModel } from './backend-vision-model';
import { readVisionModel } from './vision-assets';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { createWorldMask } from './world-mask';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./backend-vision-model', () => ({ loadBackendVisionModel: jest.fn() }));
jest.mock('./vision-assets', () => ({ ...jest.requireActual('./vision-assets'), readVisionModel: jest.fn() }));

const tensor = (dims: number[], data: Float32Array | Uint8Array) => ({ dims, data, type: data instanceof Float32Array ? 'float32' : 'uint8', dispose: jest.fn() });
const session = () => ({
    inputNames: ['images'], outputNames: ['output0', 'output1'],
    run: jest.fn().mockResolvedValue({
        output0: tensor([1, 7, 12096], new Float32Array(7 * 12096)),
        output1: tensor([1, 1, 2, 2], new Float32Array(4)),
    }),
    release: jest.fn().mockResolvedValue(undefined),
});
let gpu: ReturnType<typeof session>;
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');

beforeEach(() => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    gpu = session();
    (gpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(gpu);
    (gpuRuntime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    (loadBackendVisionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), metadata: { name: 'track-v2', classNames: ['track', 'curb'] } });
    (readVisionModel as jest.Mock).mockResolvedValue(new ArrayBuffer(8));
});

afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('loads metadata and warms up the backend segmentation export before reporting GPU readiness', async () => {
    const model = await TrackVisionModel.loadBackend();
    expect(loadBackendVisionModel).toHaveBeenCalledTimes(1);
    expect(model.classNames).toEqual(['track', 'curb']);
    expect(model.executionProvider).toBe('webgpu');
    expect(gpu.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 768, 768] }) });
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
    expect(gpu.release).toHaveBeenCalledTimes(1);
});

it.each([384, 640, 768])('runs segmentation at %i pixels with normalized boxes and aligned letterboxing', async (inputSize) => {
    gpu.run.mockResolvedValue({
        output0: tensor([1, 7, 1], new Float32Array([inputSize / 2, inputSize / 2, inputSize / 2, inputSize / 2, 0.9, 0, 1])),
        output1: tensor([1, 1, 4, 4], new Float32Array(16).fill(1)),
    });
    const drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage, getImageData: (_x: number, _y: number, width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    } as any);
    const model = await TrackVisionModel.loadBackend(inputSize);
    expect(loadBackendVisionModel).toHaveBeenCalledWith(inputSize);
    const frame = document.createElement('canvas');
    frame.width = 1280; frame.height = 640;
    const result = await model.detect(frame, 0.5);
    expect(model.inputSize).toBe(inputSize);
    expect(gpu.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, inputSize, inputSize], data: expect.any(Float32Array) }) });
    expect(gpu.run.mock.calls[1][0].images.data).toHaveLength(3 * inputSize ** 2);
    expect(drawImage).toHaveBeenCalledWith(frame, 0, inputSize / 4, inputSize, inputSize / 2);
    expect(result).toMatchObject({ task: 'segment', instances: [{ box: [0.25, 0.25, 0.75, 0.75], mask: new Uint8Array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]) }] });
    await model.dispose();
});

it('rejects GPU initialization failure without creating a CPU session', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No GPU adapter'));
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('No GPU adapter');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('releases failed GPU warm-up without falling back to CPU', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    await expect(TrackVisionModel.loadBackend()).rejects.toBeInstanceOf(GpuInferenceError);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('requires WebGPU and never initializes CPU inference when it is unavailable', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    await expect(TrackVisionModel.loadBackend()).rejects.toBeInstanceOf(GpuInferenceError);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('loads both raw segment outputs and releases tensors', async () => {
    gpu.outputNames = ['output0', 'output1'];
    const predictions = tensor([1, 7, 12096], new Float32Array(7 * 12096));
    const prototypes = tensor([1, 1, 2, 2], new Float32Array(4));
    gpu.run.mockResolvedValue({ output0: predictions, output1: prototypes });
    const model = await TrackVisionModel.loadBackend();
    expect(model.name).toBe('track-v2');
    expect(predictions.dispose).toHaveBeenCalledTimes(1);
    expect(prototypes.dispose).toHaveBeenCalledTimes(1);
    await model.dispose();
});

it('reports backend loading failures without initializing a bundled model', async () => {
    (loadBackendVisionModel as jest.Mock).mockRejectedValue(new Error('Not found'));
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('Not found');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('rejects model outputs that do not match the backend label count', async () => {
    gpu.run.mockResolvedValue({
        output0: tensor([1, 8, 12096], new Float32Array(8 * 12096)),
        output1: tensor([1, 1, 2, 2], new Float32Array(4)),
    });
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('backend labels');
    expect(gpu.release).toHaveBeenCalledTimes(1);
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
    expect(loadBackendVisionModel).not.toHaveBeenCalled();
    const frame = document.createElement('canvas');
    frame.width = frame.height = 640;
    await expect(model.detect(frame, 0.5)).rejects.toThrow('segmentation mask');
    const region = createWorldMask({ task: 'segment', width: 2, height: 2, classNames: ['track', 'car interior', 'fence'], instances: [
        { classId: 0, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([1, 1, 0, 0]) },
        { classId: 1, confidence: 0.6, box: [0, 0, 1, 1], mask: new Uint8Array([0, 1, 0, 0]) },
        { classId: 2, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([0, 0, 0, 1]) },
    ] })!;
    const result = await model.detect(frame, 0.5, region);
    expect(depthSession.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, 518, 518] }) });
    const input = depthSession.run.mock.calls[1][0].images.data;
    expect(input).toHaveLength(3 * 518 * 518);
    for (const channel of [0, 1, 2]) {
        const offset = channel * 518 * 518;
        const mean = [0.485, 0.456, 0.406][channel], std = [0.229, 0.224, 0.225][channel];
        expect(input[offset + 129 * 518 + 129]).toBeCloseTo((1 - mean) / std);
        expect(input[offset + 388 * 518 + 388]).toBeCloseTo((1 - mean) / std);
        expect(input[offset + 129 * 518 + 388]).toBeCloseTo((114 / 255 - mean) / std);
        expect(input[offset + 388 * 518 + 129]).toBeCloseTo((1 - mean) / std);
    }
    expect(output.data).toEqual(new Float32Array([1, 5, 15, 50]));
    output.data.fill(0);
    expect(result).toMatchObject({ task: 'depth', scale: 'relative', width: 2, height: 2, values: new Float32Array([1 / 2, 0, 1 / 16, 1 / 51]) });
    expect(output.dispose).toHaveBeenCalledTimes(2);
    await model.dispose();
    expect(depthSession.release).toHaveBeenCalledTimes(1);
});

it.each([252, 392, 518].flatMap((inputSize) => [[1280, 640, inputSize], [640, 1280, inputSize]]))('excludes letterbox padding from retained depth for a %i x %i capture at %i pixels', async (width, height, inputSize) => {
    gpu.outputNames = ['output0'];
    gpu.run.mockResolvedValue({ output0: tensor([1, 1, 4, 4], new Float32Array(16).fill(10)) });
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage: jest.fn(), getImageData: () => ({ data: new Uint8ClampedArray(inputSize * inputSize * 4) }),
    } as any);
    const model = await TrackVisionModel.loadBuiltin('depth', inputSize);
    expect(readVisionModel).toHaveBeenCalledWith(`http://localhost/vision-models/depth-anything-v2-small${inputSize === 518 ? '' : `-${inputSize}`}.onnx`);
    const frame = document.createElement('canvas');
    frame.width = width; frame.height = height;
    const result = await model.detect(frame, 0.5, { width: 2, height: 2, mask: new Uint8Array(4).fill(1) });
    expect(gpu.run).toHaveBeenLastCalledWith({ images: expect.objectContaining({ dims: [1, 3, inputSize, inputSize] }) });
    if (result.task !== 'depth') throw new Error('depth');
    const expected = width > height
        ? [0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0]
        : [0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0];
    expect(result.values).toEqual(Float32Array.from(expected, (value) => value ? 1 / 11 : 0));
    await model.dispose();
});
