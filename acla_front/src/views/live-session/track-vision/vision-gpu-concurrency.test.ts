import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { readVisionModel } from './vision-assets';
import { TrackVisionModel } from './track-vision-model';
import { DetectionTask } from './track-vision-types';
import { YoloTrackBoundaryModel } from './yolo-track-boundary-model';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./vision-assets', () => ({ ...jest.requireActual('./vision-assets'), readVisionModel: jest.fn() }));

type Task = DetectionTask | 'boundary';
const tasks: Task[] = ['semantic', 'depth', 'segment', 'boundary'];
const bytesFor = (task: Task) => new Uint8Array([tasks.indexOf(task)]).buffer;
const nextTurn = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const tensor = (dims: number[], data: Float32Array | Uint8Array) => ({
    dims, data, type: data instanceof Float32Array ? 'float32' : 'uint8', dispose: jest.fn(),
});
const outputsFor = (task: Task) => {
    if (task === 'semantic') return { output0: tensor([1, 2, 2], new Uint8Array([0, 1, 2, 3])) };
    if (task === 'depth') return { output0: tensor([1, 1, 2, 2], new Float32Array([1, 2, 3, 4])) };
    if (task === 'boundary') return { drive_area_seg: tensor([1, 2, 2, 2], new Float32Array(8)) };
    return {
        predictions: tensor([1, 6, 8400], new Float32Array(6 * 8400)),
        prototypes: tensor([1, 1, 2, 2], new Float32Array(4)),
    };
};
let active: number;
let maxActive: number;
let events: string[];
const gpuOperation = <T,>(name: string, work: () => T | Promise<T>) => async () => {
    events.push(`${name}:start`);
    maxActive = Math.max(maxActive, ++active);
    try {
        // Keep each runtime call in flight long enough to expose cross-model overlap.
        await nextTurn();
        return await work();
    } finally {
        active--;
        events.push(`${name}:end`);
    }
};
const sessionFor = (task: Task) => ({
    inputNames: ['images'], outputNames: Object.keys(outputsFor(task)),
    run: jest.fn(gpuOperation(`${task}:run`, () => outputsFor(task))),
    release: jest.fn(gpuOperation(`${task}:release`, () => undefined)),
});
let sessions: Record<Task, ReturnType<typeof sessionFor>>;
let frame: HTMLCanvasElement;
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');

beforeEach(() => {
    active = 0;
    maxActive = 0;
    events = [];
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    sessions = Object.fromEntries(tasks.map((task) => [task, sessionFor(task)])) as typeof sessions;
    (gpuRuntime.InferenceSession.create as jest.Mock).mockImplementation((bytes: ArrayBuffer) => {
        const task = tasks[new Uint8Array(bytes)[0]];
        return gpuOperation(`${task}:create`, () => sessions[task])();
    });
    for (const runtime of [gpuRuntime, cpuRuntime]) {
        (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    }
    (readVisionModel as jest.Mock).mockImplementation(async (url: string) => {
        if (url.endsWith('.json')) return new TextEncoder().encode('{"names":{"0":"road"}}').buffer;
        return bytesFor(url.includes('-depth') ? 'depth' : url.includes('-seg') ? 'segment' : 'semantic');
    });
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        fillRect: jest.fn(), drawImage: jest.fn(),
        getImageData: jest.fn((_x, _y, width, height) => ({ data: new Uint8ClampedArray(width * height * 4) })),
    } as any);
    frame = document.createElement('canvas');
    frame.width = 1280;
    frame.height = 720;
});

afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('serializes initialization and warm-up across all enabled detector tasks', async () => {
    const models = await Promise.all((['semantic', 'depth', 'segment'] as const).map((task) => TrackVisionModel.loadBuiltin(task)));
    expect(models.map((model) => model.executionProvider)).toEqual(['webgpu', 'webgpu', 'webgpu']);
    expect(events).toEqual(['semantic', 'depth', 'segment'].flatMap((task) => [
        `${task}:create:start`, `${task}:create:end`, `${task}:run:start`, `${task}:run:end`,
    ]));
    await Promise.all(models.map((model) => model.dispose()));
    expect(maxActive).toBe(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('queues live inference, another detector load, and disposal on the same GPU', async () => {
    const semantic = await TrackVisionModel.loadBuiltin('semantic');
    const depth = await TrackVisionModel.loadBuiltin('depth');
    events = [];
    const semanticResult = semantic.detect(frame, 0.5);
    const segmentLoad = TrackVisionModel.loadBuiltin('segment');
    const depthResult = depth.detect(frame, 0.5);
    const depthDisposal = depth.dispose();
    const [first, segment, second] = await Promise.all([semanticResult, segmentLoad, depthResult, depthDisposal]);
    expect(first.task).toBe('semantic');
    expect(second.task).toBe('depth');
    expect(segment.executionProvider).toBe('webgpu');
    expect(events.indexOf('depth:release:start')).toBeGreaterThan(events.indexOf('depth:run:end'));
    await Promise.all([semantic.dispose(), depth.dispose(), segment.dispose()]);
    expect(sessions.depth.release).toHaveBeenCalledTimes(1);
    expect(maxActive).toBe(1);
    await expect(depth.detect(frame, 0.5)).rejects.toThrow('released');
});

it('shares GPU access with the track-boundary detector too', async () => {
    const [semantic, boundary] = await Promise.all([
        TrackVisionModel.loadBuiltin('semantic'), YoloTrackBoundaryModel.load(bytesFor('boundary'), 'yolop'),
    ]);
    await Promise.all([semantic.detect(frame, 0.5), boundary.detect(frame)]);
    await Promise.all([semantic.dispose(), boundary.dispose()]);
    expect(maxActive).toBe(1);
    expect(boundary.executionProvider).toBe('webgpu');
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('finishes failed warm-up cleanup before allowing another detector to initialize', async () => {
    sessions.semantic.run.mockImplementationOnce(gpuOperation('semantic:run', () => { throw new Error('Warm-up failed'); }));
    sessions.semantic.release.mockImplementationOnce(gpuOperation('semantic:release', () => { throw new Error('Release failed'); }));
    const [failed, working] = await Promise.allSettled([
        TrackVisionModel.loadBuiltin('semantic'), TrackVisionModel.loadBuiltin('depth'),
    ]);
    expect(failed).toMatchObject({ status: 'rejected', reason: new Error('GPU inference failed. Warm-up failed') });
    expect(working.status).toBe('fulfilled');
    if (working.status === 'fulfilled') await working.value.dispose();
    expect(events.indexOf('depth:create:start')).toBeGreaterThan(events.indexOf('semantic:release:end'));
    expect(maxActive).toBe(1);
});

it('continues processing other detectors after inference and disposal fail', async () => {
    const semantic = await TrackVisionModel.loadBuiltin('semantic');
    const depth = await TrackVisionModel.loadBuiltin('depth');
    sessions.semantic.run.mockImplementationOnce(gpuOperation('semantic:run', () => { throw new Error('Inference failed'); }));
    sessions.semantic.release.mockImplementationOnce(gpuOperation('semantic:release', () => { throw new Error('Release failed'); }));
    const results = await Promise.allSettled([semantic.detect(frame, 0.5), depth.detect(frame, 0.5), semantic.dispose()]);
    expect(results.map(({ status }) => status)).toEqual(['rejected', 'fulfilled', 'rejected']);
    await expect(depth.detect(frame, 0.5)).resolves.toMatchObject({ task: 'depth' });
    await depth.dispose();
    expect(maxActive).toBe(1);
});
