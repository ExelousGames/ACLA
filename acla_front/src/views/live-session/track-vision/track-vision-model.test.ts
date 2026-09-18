import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { readVisionModel } from './vision-assets';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./vision-assets', () => ({ ...jest.requireActual('./vision-assets'), readVisionModel: jest.fn() }));

const tensor = (dims: number[], data: Float32Array | Uint8Array) => ({ dims, data, type: data instanceof Float32Array ? 'float32' : 'uint8', dispose: jest.fn() });
const session = () => ({
    inputNames: ['images'], outputNames: ['output0'],
    run: jest.fn().mockResolvedValue({ output0: tensor([1, 2, 2], new Uint8Array([0, 1, 2, 3])) }),
    release: jest.fn().mockResolvedValue(undefined),
});
let gpu: ReturnType<typeof session>;
let cpu: ReturnType<typeof session>;
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');

beforeEach(() => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    gpu = session();
    cpu = session();
    (gpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(gpu);
    (cpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(cpu);
    for (const runtime of [gpuRuntime, cpuRuntime]) {
        (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    }
    (readVisionModel as jest.Mock).mockImplementation(async (url: string) => url.endsWith('.json')
        ? new TextEncoder().encode('{"names":{"0":"road"}}').buffer : new ArrayBuffer(8));
});

afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('loads metadata and warms up the integer semantic export before reporting GPU readiness', async () => {
    const model = await TrackVisionModel.loadBuiltin('semantic');
    expect(readVisionModel).toHaveBeenCalledWith('http://localhost/vision-models/yolo26n-sem.onnx');
    expect(model.classNames[0]).toBe('road');
    expect(model.executionProvider).toBe('webgpu');
    expect(gpu.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 640, 640] }) });
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
    expect(gpu.release).toHaveBeenCalledTimes(1);
});

it('rejects GPU initialization failure without creating a CPU session by default', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No GPU adapter'));
    await expect(TrackVisionModel.loadBuiltin('semantic')).rejects.toThrow('No GPU adapter');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('releases failed GPU warm-up without falling back to CPU by default', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    await expect(TrackVisionModel.loadBuiltin('depth')).rejects.toBeInstanceOf(GpuInferenceError);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('requires explicit CPU fallback when WebGPU is unavailable', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    await expect(TrackVisionModel.loadBuiltin('semantic')).rejects.toBeInstanceOf(GpuInferenceError);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    const model = await TrackVisionModel.loadBuiltin('semantic', true);
    expect(model.executionProvider).toBe('wasm');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
});

it('still prefers GPU when CPU fallback is allowed', async () => {
    const model = await TrackVisionModel.loadBuiltin('semantic', true);
    expect(model.executionProvider).toBe('webgpu');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
});

it('releases failed GPU warm-up and validates depth through the CPU worker when fallback is allowed', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    cpu.run.mockResolvedValue({ output0: tensor([1, 1, 2, 2], new Float32Array([1, 2, 3, 4])) });
    const model = await TrackVisionModel.loadBuiltin('depth', true);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(model.executionProvider).toBe('wasm');
    expect(cpuRuntime.env.wasm.proxy).toBe(true);
    expect(model.fallbackReason).toMatch(/GPU could not run/);
    await model.dispose();
});

it('loads both raw segment outputs and releases tensors', async () => {
    gpu.outputNames = ['output0', 'output1'];
    const predictions = tensor([1, 6, 8400], new Float32Array(6 * 8400));
    const prototypes = tensor([1, 1, 2, 2], new Float32Array(4));
    gpu.run.mockResolvedValue({ output0: predictions, output1: prototypes });
    const model = await TrackVisionModel.loadBuiltin('segment');
    expect(readVisionModel).toHaveBeenCalledWith('http://localhost/vision-models/yolo11n-seg.onnx');
    expect(predictions.dispose).toHaveBeenCalledTimes(1);
    expect(prototypes.dispose).toHaveBeenCalledTimes(1);
    await model.dispose();
});

it('reports missing weights with the task name and setup command', async () => {
    (readVisionModel as jest.Mock).mockRejectedValue(new Error('Not found'));
    await expect(TrackVisionModel.loadBuiltin('depth')).rejects.toThrow('Depth weights unavailable. Run npm run setup:vision-models');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});
