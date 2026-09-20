import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { loadBackendVisionModel } from './backend-vision-model';
import { readVisionModel } from './vision-assets';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';

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
        output0: tensor([1, 7, 8400], new Float32Array(7 * 8400)),
        output1: tensor([1, 1, 2, 2], new Float32Array(4)),
    }),
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
    expect(gpu.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 640, 640] }) });
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
    expect(gpu.release).toHaveBeenCalledTimes(1);
});

it('rejects GPU initialization failure without creating a CPU session by default', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No GPU adapter'));
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('No GPU adapter');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('releases failed GPU warm-up without falling back to CPU by default', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    await expect(TrackVisionModel.loadBackend()).rejects.toBeInstanceOf(GpuInferenceError);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('requires explicit CPU fallback when WebGPU is unavailable', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    await expect(TrackVisionModel.loadBackend()).rejects.toBeInstanceOf(GpuInferenceError);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    const model = await TrackVisionModel.loadBackend(true);
    expect(model.executionProvider).toBe('wasm');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
});

it('still prefers GPU when CPU fallback is allowed', async () => {
    const model = await TrackVisionModel.loadBackend(true);
    expect(model.executionProvider).toBe('webgpu');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    await model.dispose();
});

it('releases failed GPU warm-up and validates segmentation through the CPU worker when fallback is allowed', async () => {
    gpu.run.mockRejectedValue(new Error('GPU device lost'));
    const model = await TrackVisionModel.loadBackend(true);
    expect(gpu.release).toHaveBeenCalledTimes(1);
    expect(model.executionProvider).toBe('wasm');
    expect(cpuRuntime.env.wasm.proxy).toBe(true);
    expect(model.fallbackReason).toMatch(/GPU could not run/);
    await model.dispose();
});

it('loads both raw segment outputs and releases tensors', async () => {
    gpu.outputNames = ['output0', 'output1'];
    const predictions = tensor([1, 7, 8400], new Float32Array(7 * 8400));
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
        output0: tensor([1, 8, 8400], new Float32Array(8 * 8400)),
        output1: tensor([1, 1, 2, 2], new Float32Array(4)),
    });
    await expect(TrackVisionModel.loadBackend()).rejects.toThrow('backend labels');
    expect(gpu.release).toHaveBeenCalledTimes(1);
});

it.each([false, true])('runs bundled depth independently with CPU fallback %s and retains copied distances', async (fallback) => {
    const output = tensor([1, 1, 2, 2], new Float32Array([1, 5, 15, 50]));
    const depthSession = fallback ? cpu : gpu;
    if (fallback) gpu.run.mockRejectedValue(new Error('GPU device lost'));
    gpu.outputNames = ['output0'];
    cpu.outputNames = ['output0'];
    depthSession.run.mockResolvedValue({ output0: output });
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage: jest.fn(), getImageData: () => ({ data: new Uint8ClampedArray(640 * 640 * 4) }),
    } as any);
    const model = await TrackVisionModel.loadBuiltin('depth', fallback);
    expect(model.executionProvider).toBe(fallback ? 'wasm' : 'webgpu');
    expect(readVisionModel).toHaveBeenCalledWith('http://localhost/vision-models/yolo26n-depth.onnx');
    expect(loadBackendVisionModel).not.toHaveBeenCalled();
    const frame = document.createElement('canvas');
    const result = await model.detect(frame, 0.5);
    output.data.fill(0);
    expect(result).toMatchObject({ task: 'depth', width: 2, height: 2, values: new Float32Array([1, 5, 15, 50]) });
    expect(output.dispose).toHaveBeenCalledTimes(2);
    await model.dispose();
    expect(depthSession.release).toHaveBeenCalledTimes(1);
});
