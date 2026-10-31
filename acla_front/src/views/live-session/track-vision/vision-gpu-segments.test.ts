import type { Tensor } from 'onnxruntime-web';
import { GpuSegments, readSegmentInstances } from './vision-gpu-segments';
import { mockGpuDevice } from './vision-gpu-test-device';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import type { SegmentResult } from './track-vision-types';

afterEach(() => jest.restoreAllMocks());

function fixture() {
    const { device, calls } = mockGpuDevice();
    const data = new ArrayBuffer(976 + 30 * 8);
    const ints = new Uint32Array(data), floats = new Float32Array(data);
    ints[0] = 1;
    floats.set([0, 0, 1, 1, 0.9], 4);
    ints[9] = 1; ints[11] = 1;
    new Uint8Array(data, 976).set([1, 0, 1, 0, 1, 0]);
    calls.createBuffer.mockImplementation(({ size }) => ({ size, destroy: jest.fn(),
        mapAsync: jest.fn().mockResolvedValue(undefined), getMappedRange: () => data }));
    const tensor = (dims: number[]) => ({ dims, type: 'float32', location: 'gpu-buffer', gpuBuffer: {} } as unknown as Tensor);
    const onRelease = jest.fn();
    const gpu = new GpuSegments(device, tensor([1, 7, 1]), tensor([1, 1, 2, 3]), 512, 0.5, 2, onRelease);
    const result: SegmentResult = { task: 'segment', width: 3, height: 2, instances: [], gpu };
    return { gpu, result, calls, data, onRelease };
}

it('copies final byte masks once, preserves the GPU result and caches concurrent reads', async () => {
    const { gpu, result, calls, data } = fixture();
    const instances = await Promise.all([readSegmentInstances(result), readSegmentInstances(result)]);
    expect(instances[0]).toBe(instances[1]);
    expect(result.instances).toEqual([{ classId: 1, confidence: expect.closeTo(0.9), box: [0, 0, 1, 1], mask: new Uint8Array([1, 0, 1, 0, 1, 0]) }]);
    new Uint8Array(data).fill(0);
    expect(result.instances[0].mask).toEqual(new Uint8Array([1, 0, 1, 0, 1, 0]));
    const buffers = calls.createBuffer.mock.results.map(({ value }) => value);
    expect(buffers.reduce((sum, buffer) => sum + buffer.mapAsync.mock.calls.length, 0)).toBe(1);
    expect(calls.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    await gpu.dispose();
    expect(buffers.every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});

it('allows another model to submit during mapping and waits for mapping before disposal', async () => {
    const { gpu, calls, onRelease } = fixture();
    let finish!: () => void;
    const mapping = new Promise<void>((resolve) => { finish = resolve; });
    const createBuffer = calls.createBuffer.getMockImplementation()!;
    calls.createBuffer.mockImplementation((args) => ({ ...createBuffer(args), mapAsync: jest.fn(() => mapping) }));
    const read = gpu.read(), release = gpu.dispose();
    const nextModel = jest.fn(async () => undefined);
    await runWithVisionGpuQueue(nextModel);
    expect(nextModel).toHaveBeenCalledTimes(1);
    expect(onRelease).toHaveBeenCalledTimes(1);
    const selected = calls.createBuffer.mock.results[0].value;
    expect(selected.destroy).not.toHaveBeenCalled();
    finish();
    await Promise.all([read, release, gpu.dispose()]);
    expect(selected.destroy).toHaveBeenCalledTimes(1);
});

it('measures copy submission, mapping and CPU assembly once for concurrent and cached reads', async () => {
    const { gpu, result, calls } = fixture();
    let now = 100;
    jest.spyOn(performance, 'now').mockImplementation(() => now);
    calls.queue.submit.mockImplementation(() => { now += 4; });
    const createBuffer = calls.createBuffer.getMockImplementation()!;
    calls.createBuffer.mockImplementation((args) => {
        const buffer = createBuffer(args);
        return { ...buffer,
            mapAsync: jest.fn(async () => { now += 120; }),
            getMappedRange: () => { now += 7; return buffer.getMappedRange(); } };
    });
    await Promise.all([readSegmentInstances(result), readSegmentInstances(result)]);
    expect(result.readbackTiming).toEqual({ submissionMs: 4, mapMs: 120, assemblyMs: 7, totalMs: 131 });
    expect(result.readbackMs).toBe(131);
    const timing = result.readbackTiming;
    now += 1000;
    await readSegmentInstances(result);
    expect(result.readbackTiming).toBe(timing);
    expect(result.readbackMs).toBe(131);
    expect(calls.queue.submit).toHaveBeenCalledTimes(2); // GPU postprocessing and one readback.
    expect(calls.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    await gpu.dispose();
});

it('propagates invalid backend labels and cleans up failed readbacks', async () => {
    const { gpu, result, data, calls } = fixture();
    new Uint32Array(data)[1] = 1;
    await expect(readSegmentInstances(result)).rejects.toThrow('class ID');
    expect(result.readbackMs).toBeUndefined();
    expect(result.readbackTiming).toBeUndefined();
    expect(gpu.readbackTiming).toBeUndefined();
    await gpu.dispose();
    expect(calls.createBuffer.mock.results.every(({ value }) => value.destroy.mock.calls.length === 1)).toBe(true);
    const discarded = fixture();
    await discarded.gpu.dispose();
    await expect(discarded.gpu.read()).rejects.toThrow('released');
});
