import * as gpuRuntime from 'onnxruntime-web/webgpu';
import * as cpuRuntime from 'onnxruntime-web/wasm';
import { loadBackendVisionModel } from './backend-vision-model';
import { TrackVisionModel } from './track-vision-model';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('onnxruntime-web/wasm', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./backend-vision-model', () => ({ loadBackendVisionModel: jest.fn() }));

type Task = 'first' | 'second' | 'third';
const tasks: Task[] = ['first', 'second', 'third'];
const bytesFor = (task: Task) => new Uint8Array([tasks.indexOf(task)]).buffer;
const nextTurn = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const tensor = (dims: number[], data: Float32Array | Uint8Array) => ({
    dims, data, type: data instanceof Float32Array ? 'float32' : 'uint8', dispose: jest.fn(),
});
const outputsFor = (_task: Task) => ({
    predictions: tensor([1, 6, 8400], new Float32Array(6 * 8400)),
    prototypes: tensor([1, 1, 2, 2], new Float32Array(4)),
});
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
    let loadIndex = 0;
    (loadBackendVisionModel as jest.Mock).mockImplementation(async () => {
        const task = tasks[loadIndex++];
        return { bytes: bytesFor(task), metadata: { name: task, classNames: ['track'] } };
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
    const models = await Promise.all((['first', 'second', 'third'] as const).map(() => TrackVisionModel.loadBackend()));
    expect(models.map((model) => model.executionProvider)).toEqual(['webgpu', 'webgpu', 'webgpu']);
    expect(events).toEqual(['first', 'second', 'third'].flatMap((task) => [
        `${task}:create:start`, `${task}:create:end`, `${task}:run:start`, `${task}:run:end`,
    ]));
    await Promise.all(models.map((model) => model.dispose()));
    expect(maxActive).toBe(1);
    expect(cpuRuntime.InferenceSession.create).not.toHaveBeenCalled();
});

it('queues live inference, another detector load, and disposal on the same GPU', async () => {
    const first = await TrackVisionModel.loadBackend();
    const second = await TrackVisionModel.loadBackend();
    events = [];
    const firstResult = first.detect(frame, 0.5);
    const thirdLoad = TrackVisionModel.loadBackend();
    const secondResult = second.detect(frame, 0.5);
    const secondDisposal = second.dispose();
    const [firstDetection, third, secondDetection] = await Promise.all([firstResult, thirdLoad, secondResult, secondDisposal]);
    expect(firstDetection.task).toBe('segment');
    expect(secondDetection.task).toBe('segment');
    expect(third.executionProvider).toBe('webgpu');
    expect(events.indexOf('second:release:start')).toBeGreaterThan(events.indexOf('second:run:end'));
    await Promise.all([first.dispose(), second.dispose(), third.dispose()]);
    expect(sessions.second.release).toHaveBeenCalledTimes(1);
    expect(maxActive).toBe(1);
    await expect(second.detect(frame, 0.5)).rejects.toThrow('released');
});

it('finishes failed warm-up cleanup before allowing another detector to initialize', async () => {
    sessions.first.run.mockImplementationOnce(gpuOperation('first:run', () => { throw new Error('Warm-up failed'); }));
    sessions.first.release.mockImplementationOnce(gpuOperation('first:release', () => { throw new Error('Release failed'); }));
    const [failed, working] = await Promise.allSettled([
        TrackVisionModel.loadBackend(), TrackVisionModel.loadBackend(),
    ]);
    expect(failed).toMatchObject({ status: 'rejected', reason: new Error('GPU inference failed. Warm-up failed') });
    expect(working.status).toBe('fulfilled');
    if (working.status === 'fulfilled') await working.value.dispose();
    expect(events.indexOf('second:create:start')).toBeGreaterThan(events.indexOf('first:release:end'));
    expect(maxActive).toBe(1);
});

it('continues processing other detectors after inference and disposal fail', async () => {
    const first = await TrackVisionModel.loadBackend();
    const second = await TrackVisionModel.loadBackend();
    sessions.first.run.mockImplementationOnce(gpuOperation('first:run', () => { throw new Error('Inference failed'); }));
    sessions.first.release.mockImplementationOnce(gpuOperation('first:release', () => { throw new Error('Release failed'); }));
    const results = await Promise.allSettled([first.detect(frame, 0.5), second.detect(frame, 0.5), first.dispose()]);
    expect(results.map(({ status }) => status)).toEqual(['rejected', 'fulfilled', 'rejected']);
    await expect(second.detect(frame, 0.5)).resolves.toMatchObject({ task: 'segment' });
    await second.dispose();
    expect(maxActive).toBe(1);
});
