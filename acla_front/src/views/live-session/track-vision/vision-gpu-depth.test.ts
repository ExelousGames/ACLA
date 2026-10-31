import type { Tensor } from 'onnxruntime-web';
import { GpuDepth, readDepthValues } from './vision-gpu-depth';
import { mockGpuDevice } from './vision-gpu-test-device';
import { createDepthMap, drawDepthMap } from './depth-map';
import type { DepthResult } from './track-vision-types';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import * as decoding from './vision-decoding';

const makeDepth = () => {
    const gpu = mockGpuDevice();
    const raw = new Float32Array([0, 3, -1, NaN]);
    const output = { dims: [1, 2, 2], type: 'float32', location: 'gpu-buffer', gpuBuffer: {},
        getData: jest.fn().mockResolvedValue(raw), dispose: jest.fn(),
        get data() { throw new Error('GPU data must not be accessed synchronously'); } };
    const depth = new GpuDepth(gpu.device, output as unknown as Tensor, { rect: [0, 0, 1, 1] });
    const result: DepthResult = { task: 'depth', scale: 'relative', width: 2, height: 2, gpu: depth };
    return { ...gpu, output, depth, result };
};
afterEach(() => jest.restoreAllMocks());

it('renders the depth map and its color range on the GPU without accessing or downloading the output', async () => {
    const { depth, output, result, calls, pass } = makeDepth();
    const canvasContext = { configure: jest.fn(), unconfigure: jest.fn(), getCurrentTexture: () => ({ createView: () => ({}) }) };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext as any);
    const context = { save: jest.fn(), restore: jest.fn(), fillRect: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    const map = createDepthMap({ capturedAt: 1, width: 4, height: 4,
        detections: { depth: { ...result, classNames: [], inferenceMs: 1 } } })!;
    drawDepthMap(context, map);
    expect(output.getData).not.toHaveBeenCalled();
    expect(output.dispose).not.toHaveBeenCalled();
    expect(pass.draw).toHaveBeenCalledWith(3);
    expect(context.drawImage).toHaveBeenCalled();
    await depth.readRange();
    expect(output.getData).not.toHaveBeenCalled();
    expect(calls.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    expect(calls.createBuffer).toHaveBeenLastCalledWith({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    await depth.dispose();
    expect(canvasContext.unconfigure).toHaveBeenCalledTimes(1);
});

it('shares one lazy readback, copies decoded values, and retains the GPU tensor for further rendering', async () => {
    const { result, output, depth } = makeDepth();
    const [first, second] = await Promise.all([readDepthValues(result), readDepthValues(result)]);
    expect(first).toBe(second);
    expect(result.values).toEqual(new Float32Array([1, 0.25, 0, 0]));
    expect(output.getData).toHaveBeenCalledTimes(1);
    expect(output.dispose).not.toHaveBeenCalled();
    output.getData.mock.results[0].value.then((raw: Float32Array) => raw.fill(0));
    expect(await readDepthValues(result)).toEqual(new Float32Array([1, 0.25, 0, 0]));
    await depth.dispose();
    expect(output.dispose).toHaveBeenCalledTimes(1);
});

it('separates app queue, GPU completion, download and conversion delays and retains them for cached reads', async () => {
    const { result, output, depth, calls } = makeDepth();
    let now = 100;
    jest.spyOn(performance, 'now').mockImplementation(() => now);
    calls.queue.onSubmittedWorkDone.mockImplementation(async () => { now += 100; });
    output.getData.mockImplementation(async () => { now += 35; return new Float32Array(4); });
    const decode = decoding.decodeDepth;
    jest.spyOn(decoding, 'decodeDepth').mockImplementation((value) => { now += 7; return decode(value); });
    const priorWork = runWithVisionGpuQueue(async () => { now += 15; });
    await Promise.all([readDepthValues(result), readDepthValues(result)]);
    await priorWork;
    expect(result.readbackTiming).toEqual({ queueMs: 15, gpuWaitMs: 100, downloadMs: 35, conversionMs: 7, totalMs: 157 });
    expect(result.readbackMs).toBe(157);
    const timing = result.readbackTiming;
    now += 1000;
    await readDepthValues(result);
    expect(result.readbackTiming).toBe(timing);
    expect(result.readbackMs).toBe(157);
    expect(calls.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
    expect(output.getData).toHaveBeenCalledTimes(1);
    await depth.dispose();
});

it('waits for a pending readback before releasing its tensor and disposes exactly once', async () => {
    const { depth, output } = makeDepth();
    let resolve!: (value: Float32Array) => void;
    let downloadStarted!: () => void;
    const downloading = new Promise<void>((done) => { downloadStarted = done; });
    output.getData.mockImplementation(() => new Promise((done) => { resolve = done; downloadStarted(); }));
    const read = depth.read();
    const release = depth.dispose();
    await downloading;
    expect(output.dispose).not.toHaveBeenCalled();
    resolve(new Float32Array(4));
    await Promise.all([read, release, depth.dispose()]);
    expect(output.dispose).toHaveBeenCalledTimes(1);
});

it('releases the tensor if the GPU completion wait fails without reporting a completed readback', async () => {
    const { depth, result, output, calls } = makeDepth();
    calls.queue.onSubmittedWorkDone.mockRejectedValue(new Error('GPU completion failed'));
    await expect(readDepthValues(result)).rejects.toThrow('GPU completion failed');
    expect(output.getData).not.toHaveBeenCalled();
    expect(result.readbackTiming).toBeUndefined();
    expect(result.readbackMs).toBeUndefined();
    await depth.dispose();
    expect(output.dispose).toHaveBeenCalledTimes(1);
});

it('releases after failed downloads and rejects new downloads on a discarded frame', async () => {
    const first = makeDepth();
    first.output.getData.mockRejectedValue(new Error('device lost'));
    const reading = first.depth.read();
    const release = first.depth.dispose();
    await expect(reading).rejects.toThrow('device lost');
    await release;
    expect(first.output.dispose).toHaveBeenCalledTimes(1);
    const second = makeDepth();
    await second.depth.dispose();
    await expect(second.depth.read()).rejects.toThrow('released');
    expect(second.output.getData).not.toHaveBeenCalled();
});
