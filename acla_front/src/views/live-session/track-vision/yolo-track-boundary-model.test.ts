import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { readVisionModel } from './vision-assets';
import { YoloTrackBoundaryModel } from './yolo-track-boundary-model';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./vision-assets', () => ({ ...jest.requireActual('./vision-assets'), readVisionModel: jest.fn() }));

const bytes = new ArrayBuffer(8);
const tensor = (dims: number[], data: Float32Array) => ({ dims, data, dispose: jest.fn() });
const session = () => ({
    inputNames: ['images'], outputNames: ['drive_area_seg'],
    run: jest.fn().mockResolvedValue({ drive_area_seg: tensor([1, 2, 2, 2], new Float32Array(8)) }),
    release: jest.fn().mockResolvedValue(undefined),
});
let gpuSession: ReturnType<typeof session>;
let cpuSession: ReturnType<typeof session>;
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');

beforeEach(() => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    gpuSession = session();
    cpuSession = session();
    (gpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(gpuSession);
    (cpuRuntime.InferenceSession.create as jest.Mock).mockResolvedValue(cpuSession);
    for (const runtime of [gpuRuntime, cpuRuntime]) {
        (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    }
    (readVisionModel as jest.Mock).mockResolvedValue(bytes);
    jest.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('loads and validates the built-in model on GPU before reporting GPU acceleration', async () => {
    const model = await YoloTrackBoundaryModel.loadBuiltin();
    expect(readVisionModel).toHaveBeenCalledWith('http://localhost/vision-models/yolop-320-320.onnx');
    expect(gpuRuntime.InferenceSession.create).toHaveBeenCalledWith(bytes, { executionProviders: ['webgpu'] });
    expect(gpuRuntime.env.wasm).toMatchObject({ proxy: false, numThreads: 1, wasmPaths: 'http://localhost/vision-runtime/' });
    expect(gpuRuntime.env.webgpu.powerPreference).toBe('high-performance');
    expect(gpuSession.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 320, 320] }) }, ['drive_area_seg']);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    expect(model.executionProvider).toBe('webgpu');
    expect(model.fallbackReason).toBeUndefined();
    await model.dispose();
    expect(gpuSession.release).toHaveBeenCalledTimes(1);
});

it('also accelerates custom segmentation models with the 640px input contract', async () => {
    gpuSession.outputNames = ['predictions', 'prototypes'];
    const predictions = tensor([1, 6, 8400], new Float32Array(6 * 8400));
    const prototypes = tensor([1, 1, 4, 4], new Float32Array(16));
    gpuSession.run.mockResolvedValue({ predictions, prototypes });
    const model = await YoloTrackBoundaryModel.load(bytes);
    expect(model.executionProvider).toBe('webgpu');
    expect(gpuSession.run).toHaveBeenCalledWith({ images: expect.objectContaining({ dims: [1, 3, 640, 640] }) }, gpuSession.outputNames);
    expect(predictions.dispose).toHaveBeenCalledTimes(1);
    expect(prototypes.dispose).toHaveBeenCalledTimes(1);
    await model.dispose();
});

it('uses the CPU worker when WebGPU is unavailable', async () => {
    Reflect.deleteProperty(navigator, 'gpu');
    const model = await YoloTrackBoundaryModel.load(bytes, 'yolop');
    expect(gpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
    expect(cpuRuntime.InferenceSession.create).toHaveBeenCalledWith(bytes, { executionProviders: ['wasm'] });
    expect(cpuRuntime.env.wasm).toMatchObject({ proxy: true, numThreads: 1, wasmPaths: 'http://localhost/vision-runtime/' });
    expect(cpuSession.run).toHaveBeenCalledTimes(1);
    expect(model.executionProvider).toBe('wasm');
    expect(model.fallbackReason).toMatch(/GPU acceleration is unavailable/);
    await model.dispose();
});

it('retries on CPU when GPU session creation fails', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No GPU adapter found.'));
    const model = await YoloTrackBoundaryModel.load(bytes, 'yolop');
    expect(cpuRuntime.InferenceSession.create).toHaveBeenCalledWith(bytes, { executionProviders: ['wasm'] });
    expect(model.executionProvider).toBe('wasm');
    expect(model.fallbackReason).toMatch(/GPU could not run this model/);
    expect(cpuRuntime.env.wasm.proxy).toBe(true);
    expect(gpuRuntime.env.wasm.proxy).toBe(false);
    await model.dispose();
});

it('releases a GPU session that fails warm-up before retrying on CPU', async () => {
    gpuSession.run.mockRejectedValue(new Error('Shader compilation failed.'));
    const model = await YoloTrackBoundaryModel.load(bytes, 'yolop');
    expect(gpuSession.release).toHaveBeenCalledTimes(1);
    expect(gpuSession.release.mock.invocationCallOrder[0]).toBeLessThan((cpuRuntime.InferenceSession.create as jest.Mock).mock.invocationCallOrder[0]);
    expect(model.executionProvider).toBe('wasm');
    await model.dispose();
});

it('still falls back if releasing a failed GPU session also fails', async () => {
    gpuSession.run.mockRejectedValue(new Error('GPU device lost.'));
    gpuSession.release.mockRejectedValue(new Error('GPU device lost.'));
    const model = await YoloTrackBoundaryModel.load(bytes, 'yolop');
    expect(model.executionProvider).toBe('wasm');
    await model.dispose();
});

it('reports actionable errors when both providers fail', async () => {
    (gpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('No adapter.'));
    (cpuRuntime.InferenceSession.create as jest.Mock).mockRejectedValue(new Error('Invalid ONNX data.'));
    await expect(YoloTrackBoundaryModel.load(bytes, 'yolop')).rejects.toThrow('setup:vision');
    await expect(YoloTrackBoundaryModel.load(bytes)).rejects.toThrow('Export a float32, 640px');
});

it('rejects invalid model outputs on both providers and releases both sessions', async () => {
    gpuSession.outputNames = ['unsupported'];
    cpuSession.outputNames = ['unsupported'];
    await expect(YoloTrackBoundaryModel.load(bytes, 'yolop')).rejects.toThrow('missing its drivable-area output');
    expect(gpuSession.release).toHaveBeenCalledTimes(1);
    expect(cpuSession.release).toHaveBeenCalledTimes(1);
});
