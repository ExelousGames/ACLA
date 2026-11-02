import * as runtime from 'onnxruntime-web/webgpu';
import { loadBackendPositionModel } from './backend-vision-model';
import { TrackPositionModel } from './track-position-model';
import { POSITION_MODEL_NAMES } from './vision-config';
import { mockGpuDevice } from './vision-gpu-test-device';
import { runWithVisionGpuQueue } from './vision-gpu-queue';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: Object.assign(jest.fn(), { fromGpuBuffer: jest.fn() }),
}), { virtual: true });
jest.mock('./backend-vision-model', () => ({ loadBackendPositionModel: jest.fn() }));
const tensor = (dims: number[], data: Float32Array) => ({ dims, data, dispose: jest.fn() });
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');
let session: { inputNames: string[]; outputNames: string[]; run: jest.Mock; release: jest.Mock };
let drawImage: jest.Mock;
let device: ReturnType<typeof mockGpuDevice>;
let pixel: number;
let imageWindows: Float32Array[];

const snapshotImages = () => {
    const feeds = session.run.mock.calls[session.run.mock.calls.length - 1][0];
    imageWindows[session.run.mock.calls.length - 1] = feeds.images.gpuBuffer.values.slice();
};

beforeEach(() => {
    pixel = 255;
    imageWindows = [];
    device = mockGpuDevice();
    (runtime.env.webgpu as any).device = device.device;
    // Model tests simulate GPU storage/copies; the Chromium check verifies the actual shader pixels.
    device.calls.createBuffer.mockImplementation(({ size }) => ({ size, values: new Float32Array(size / 4),
        destroy: jest.fn(), mapAsync: jest.fn(), getMappedRange: jest.fn() }));
    device.pass.dispatchWorkgroups.mockImplementation(() => {
        const entries = device.calls.createBindGroup.mock.calls.slice(-1)[0][0].entries;
        entries.find((entry: any) => entry.binding === 4).resource.buffer.values.fill(pixel / 255);
    });
    device.calls.createCommandEncoder.mockImplementation(() => ({ beginComputePass: () => device.pass,
        beginRenderPass: () => device.pass, finish: jest.fn(() => ({})),
        copyBufferToBuffer: jest.fn((source, sourceOffset, target, targetOffset, bytes) => {
            target.values.set(source.values.subarray(sourceOffset / 4, (sourceOffset + bytes) / 4), targetOffset / 4);
        }) }));
    (runtime.Tensor.fromGpuBuffer as jest.Mock).mockImplementation((gpuBuffer, options) => ({
        dims: options.dims, gpuBuffer, location: 'gpu-buffer', dispose: jest.fn(),
    }));
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    session = { inputNames: ['images', 'references'], outputNames: ['deltas'],
        run: jest.fn().mockResolvedValue({ deltas: tensor([1, 1, 3], new Float32Array([-1, 2, -3])) }),
        release: jest.fn().mockResolvedValue(undefined) };
    (runtime.InferenceSession.create as jest.Mock).mockResolvedValue(session);
    (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 512, metadata: { name: 'track-position', classNames: ['reference_delta_x', 'reference_delta_y', 'reference_delta_z'] } });
    drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ fillRect: jest.fn(), drawImage,
        getImageData: () => { throw new Error('Position preprocessing must not read CPU pixels'); } } as any);
});
afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('runs the image and ordered raw reference XYZ on GPU and preserves all three signed car-local reference offsets', async () => {
    const model = (await TrackPositionModel.load())!;
    const frame = document.createElement('canvas');
    frame.width = 1280; frame.height = 721;
    const result = await model.detect(frame, { reference: { x: 10, y: -20, z: 30 } });
    const downloaded = await (loadBackendPositionModel as jest.Mock).mock.results[0].value;
    expect((runtime.InferenceSession.create as jest.Mock).mock.calls[0][0]).toBe(downloaded.bytes);
    expect(runtime.InferenceSession.create).toHaveBeenCalledWith(downloaded.bytes, { executionProviders: ['webgpu'] });
    expect(session.run).toHaveBeenLastCalledWith({
        images: expect.objectContaining({ dims: [1, 3, 512, 512] }),
        references: expect.objectContaining({ dims: [1, 1, 3], data: new Float32Array([10, -20, 30]) }),
    });
    expect(drawImage).toHaveBeenCalledWith(frame, 0, 112, 512, 288);
    expect(session.run.mock.calls[1][0].images.location).toBe('gpu-buffer');
    expect(session.run.mock.calls[1][0].images).toBe(session.run.mock.calls[0][0].images);
    expect(runtime.Tensor).not.toHaveBeenCalledWith('float32', expect.anything(), [1, 3, 512, 512]);
    expect(HTMLCanvasElement.prototype.getContext).toHaveBeenCalledWith('2d');
    expect(device.calls.queue.copyExternalImageToTexture).toHaveBeenCalledWith(
        { source: expect.any(HTMLCanvasElement) }, expect.anything(), [512, 512]);
    expect(result).toEqual({ coordinateFrame: 'car_local', referenceDelta: [-1, 2, -3], inferenceMs: expect.any(Number) });
    expect(session.run).toHaveBeenCalledTimes(2); // One warm-up and one position inference.
    expect(session.run.mock.calls[1][0].references.dispose).toHaveBeenCalledTimes(1);
    expect(session.run.mock.calls[1][0].images.dispose).not.toHaveBeenCalled();
    await model.dispose();
    await model.dispose();
    expect(session.release).toHaveBeenCalledTimes(1);
    expect(session.run.mock.calls[1][0].images.dispose).toHaveBeenCalledTimes(1);
    expect(device.calls.createBuffer.mock.results.every(({ value }) => value.destroy.mock.calls.length === 1)).toBe(true);
});

it.each(['v1', 'v2'] as const)('reports missing %s weights without initializing another session', async (version) => {
    (loadBackendPositionModel as jest.Mock).mockResolvedValue(null);
    await expect(TrackPositionModel.load(version)).rejects.toThrow(`No ONNX ${POSITION_MODEL_NAMES[version]} model`);
    expect(loadBackendPositionModel).toHaveBeenCalledWith(version);
    expect(runtime.InferenceSession.create).not.toHaveBeenCalled();
});

it('feeds progress, game and map encodings in training order when the graph requests context', async () => {
    session.inputNames = ['images', 'references', 'context'];
    (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 512,
        metadata: { name: 'future-position-model', classNames: ['reference_delta_x', 'reference_delta_y', 'reference_delta_z'] },
        categories: { game: ['iracing', 'acc'], map_name: ['Autodromo Nazionale Monza', 'nurburgring_24h'] } });
    const model = (await TrackPositionModel.load())!;
    const references = { reference: { x: 1, y: 2, z: 3 } };
    const frame = document.createElement('canvas');
    await model.detect(frame, references, { normalized_position: 0.25, game: 'acc',
        map_name: ['Nürburgring', 'nurburgring_24h'] });
    const context = session.run.mock.calls[1][0].context;
    expect(context).toMatchObject({ dims: [1, 5], data: new Float32Array([0.25, 0, 1, 0, 1]) });
    expect(context.dispose).toHaveBeenCalledTimes(1);
    await model.detect(frame, references, { normalized_position: 0.75, game: 'new-game', map_name: 'new-track' });
    expect(session.run.mock.calls[2][0].context.data).toEqual(new Float32Array([0.75, 0, 0, 0, 0]));
    await expect(model.detect(frame, references)).rejects.toThrow('normalized_position');
    expect(session.run).toHaveBeenCalledTimes(3);
    await model.dispose();
});

it.each([
    [[1, 1, 3], [NaN, 0, 0]],
    [[1, 2, 3], [0, 0, 0, 0, 0, 0]],
])('rejects nonfinite or old centerline output and releases the failed session (%j)', async (dims, values) => {
    session.run.mockResolvedValue({ deltas: tensor(dims, new Float32Array(values)) });
    await expect(TrackPositionModel.load()).rejects.toThrow('Invalid track-position output');
    expect(session.release).toHaveBeenCalledTimes(1);
    expect(session.run.mock.calls[0][0].references.dispose).toHaveBeenCalledTimes(1);
    expect(device.calls.createBuffer.mock.results.every(({ value }) => value.destroy.mock.calls.length === 1)).toBe(true);
});

it('waits for an in-flight prediction before releasing its GPU session', async () => {
    const model = (await TrackPositionModel.load())!;
    let finish!: (value: unknown) => void;
    session.run.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const frame = document.createElement('canvas');
    const pending = model.detect(frame, { reference: { x: 1, y: 2, z: 3 } });
    await Promise.resolve();
    const disposed = model.dispose();
    expect(session.release).not.toHaveBeenCalled();
    finish({ deltas: tensor([1, 1, 3], new Float32Array(3)) });
    await pending;
    await disposed;
    expect(session.release).toHaveBeenCalledTimes(1);
});

it('feeds single-frame world orientation in radians and rejects missing or nonfinite angles', async () => {
    session.inputNames = ['images', 'references', 'orientation'];
    const model = await TrackPositionModel.load();
    expect(model.requiresOrientation).toBe(true);
    const frame = document.createElement('canvas');
    const references = { reference: { x: 1, y: 2, z: 3 } };
    await model.detect(frame, references, { normalized_position: 0.25, orientation: [1.2, -0.1, 0.3] });
    expect(session.run.mock.calls[1][0].orientation).toMatchObject({ dims: [1, 3], data: new Float32Array([1.2, -0.1, 0.3]) });
    for (const orientation of [undefined, [0, NaN, 0], [0, Infinity, 0], [0, 0]]) {
        await expect(model.detect(frame, references, { normalized_position: 0.25, orientation: orientation as any }))
            .rejects.toThrow('finite orientation');
    }
    expect(session.run).toHaveBeenCalledTimes(2);
    await model.dispose();
});

describe.each(['v1', 'v2'] as const)('temporal position inference (%s)', (version) => {
    beforeEach(() => {
        pixel = 0;
        session.inputNames = ['images', 'references', 'context', 'orientation'];
        (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 32,
            sequenceLength: 3, metadata: { name: POSITION_MODEL_NAMES[version] }, categories: { game: ['acc'], map_name: ['monza'] } });
    });

    const detect = async (model: TrackPositionModel, step: number) => {
        pixel = step * 32;
        const result = await model.detect(document.createElement('canvas'), { reference: { x: step, y: -step, z: step * 10 } },
            { normalized_position: step / 10, game: 'acc', map_name: 'monza', orientation: [step / 2, -step / 10, step / 20] });
        snapshotImages();
        return result;
    };
    const expectWindow = (call: number, steps: number[]) => {
        const { images, references, context, orientation } = session.run.mock.calls[call][0];
        expect(images.dims).toEqual([1, 3, 3, 32, 32]);
        expect(images.location).toBe('gpu-buffer');
        expect(orientation).toMatchObject({ dims: [1, 3, 3],
            data: new Float32Array(steps.flatMap((step) => [step / 2, -step / 10, step / 20])) });
        expect(references).toMatchObject({ dims: [1, 3, 1, 3],
            data: new Float32Array(steps.flatMap((step) => [step, -step, step * 10])) });
        expect(context).toMatchObject({ dims: [1, 3, 3],
            data: new Float32Array(steps.flatMap((step) => [step / 10, 1, 1])) });
        steps.forEach((step, index) => {
            expect(imageWindows[call][index * 3 * 32 * 32]).toBeCloseTo(step * 32 / 255);
            expect(imageWindows[call][(index + 1) * 3 * 32 * 32 - 1]).toBeCloseTo(step * 32 / 255);
        });
    };

    it('pads, aligns and rolls the frame window without retaining warm-up samples', async () => {
        const model = await TrackPositionModel.load(version);
        expect(loadBackendPositionModel).toHaveBeenCalledWith(version);
        expect(model.name).toBe(POSITION_MODEL_NAMES[version]);
        expect(session.run.mock.calls[0][0].images.dims).toEqual([1, 3, 3, 32, 32]);
        for (let step = 1; step <= 4; step++) await detect(model, step);
        expectWindow(1, [1, 1, 1]);
        expectWindow(2, [1, 1, 2]);
        expectWindow(3, [1, 2, 3]);
        expectWindow(4, [2, 3, 4]);
        expect(runtime.Tensor.fromGpuBuffer).toHaveBeenCalledTimes(3); // One frame and two reusable windows.
        expect(session.run.mock.calls[1][0].images).toBe(session.run.mock.calls[3][0].images);
        expect(session.run.mock.calls[2][0].images).toBe(session.run.mock.calls[4][0].images);
        expect(device.calls.queue.writeBuffer.mock.calls.every(([, , data]) => data.byteLength <= 48)).toBe(true);
        for (const [{ images, ...feeds }] of session.run.mock.calls) {
            expect(images.dispose).not.toHaveBeenCalled();
            for (const input of Object.values(feeds) as ReturnType<typeof tensor>[]) expect(input.dispose).toHaveBeenCalledTimes(1);
        }
        await model.dispose();
        expect(device.calls.createBuffer.mock.results.every(({ value }) => value.destroy.mock.calls.length === 1)).toBe(true);
    });

    it('does not restore history when a prediction finishes after a reset', async () => {
        const model = await TrackPositionModel.load(version);
        await detect(model, 1);
        let finish!: (value: unknown) => void;
        session.run.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
        const pending = detect(model, 2);
        await Promise.resolve();
        model.resetHistory();
        finish({ deltas: tensor([1, 1, 3], new Float32Array(3)) });
        await pending;
        await detect(model, 3);
        expectWindow(2, [1, 1, 2]);
        expectWindow(3, [3, 3, 3]);
        await model.dispose();
    });

    it('keeps resets effective while a frame is waiting for another GPU operation', async () => {
        const model = await TrackPositionModel.load(version);
        await detect(model, 1);
        let unblock!: () => void;
        const blocked = runWithVisionGpuQueue(() => new Promise<void>((resolve) => { unblock = resolve; }));
        await Promise.resolve();
        const pending = detect(model, 2);
        model.resetHistory();
        unblock();
        await blocked;
        await pending;
        expectWindow(2, [1, 1, 2]);
        await detect(model, 3);
        expectWindow(3, [3, 3, 3]);
        await model.dispose();
    });

    it('releases both history buffers when warm-up fails', async () => {
        session.run.mockRejectedValueOnce(new Error('Warm-up failed'));
        await expect(TrackPositionModel.load(version)).rejects.toThrow('Warm-up failed');
        expect(session.release).toHaveBeenCalledTimes(1);
        expect(runtime.Tensor.fromGpuBuffer).toHaveBeenCalledTimes(3);
        expect(device.calls.createBuffer.mock.results.every(({ value }) => value.destroy.mock.calls.length === 1)).toBe(true);
    });

    it('clears interrupted history after invalid inputs or failed inference', async () => {
        const model = await TrackPositionModel.load(version);
        await detect(model, 1);
        await expect(model.detect(document.createElement('canvas'), { reference: { x: NaN, y: 0, z: 0 } }))
            .rejects.toThrow('finite reference points');
        await detect(model, 2);
        expectWindow(2, [2, 2, 2]);
        await expect(model.detect(document.createElement('canvas'), { reference: { x: 0, y: 0, z: 0 } },
            { normalized_position: 0.25 })).rejects.toThrow('finite orientation');
        session.run.mockRejectedValueOnce(new Error('GPU failure'));
        await expect(detect(model, 3)).rejects.toThrow('GPU failure');
        expect(session.run.mock.calls[3][0].images.dispose).not.toHaveBeenCalled();
        await detect(model, 4);
        expectWindow(4, [4, 4, 4]);
        await model.dispose();
    });
});
