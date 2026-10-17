import * as runtime from 'onnxruntime-web/webgpu';
import { loadBackendPositionModel } from './backend-vision-model';
import { TrackPositionModel } from './track-position-model';

jest.mock('onnxruntime-web/webgpu', () => ({
    env: { wasm: {}, webgpu: {} }, InferenceSession: { create: jest.fn() }, Tensor: jest.fn(),
}), { virtual: true });
jest.mock('./backend-vision-model', () => ({ loadBackendPositionModel: jest.fn() }));
const tensor = (dims: number[], data: Float32Array) => ({ dims, data, dispose: jest.fn() });
const originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');
let session: { inputNames: string[]; outputNames: string[]; run: jest.Mock; release: jest.Mock };
let drawImage: jest.Mock;

beforeEach(() => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {} });
    session = { inputNames: ['images', 'references'], outputNames: ['deltas'],
        run: jest.fn().mockResolvedValue({ deltas: tensor([1, 2, 3], new Float32Array([-1, 2, -3, 4, -5, 6])) }),
        release: jest.fn().mockResolvedValue(undefined) };
    (runtime.InferenceSession.create as jest.Mock).mockResolvedValue(session);
    (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 512, metadata: { name: 'position-v1' } });
    drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ fillRect: jest.fn(), drawImage,
        getImageData: () => ({ data: new Uint8ClampedArray(512 * 512 * 4).fill(255) }) } as any);
});
afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('runs the image and ordered raw reference XYZ on GPU and preserves all six signed outputs', async () => {
    const model = (await TrackPositionModel.load())!;
    const frame = document.createElement('canvas');
    frame.width = 1280; frame.height = 721;
    const result = await model.predict(frame, { reference: { x: 10, y: -20, z: 30 }, centerline: { x: -40, y: 50, z: -60 } });
    expect(runtime.InferenceSession.create).toHaveBeenCalledWith(expect.any(ArrayBuffer), { executionProviders: ['webgpu'] });
    expect(session.run).toHaveBeenLastCalledWith({
        images: expect.objectContaining({ dims: [1, 3, 512, 512] }),
        references: expect.objectContaining({ dims: [1, 2, 3], data: new Float32Array([10, -20, 30, -40, 50, -60]) }),
    });
    expect(drawImage).toHaveBeenCalledWith(frame, 0, 112, 512, 288);
    expect(session.run.mock.calls[1][0].images.data[0]).toBe(1);
    expect(result).toEqual({ referenceDelta: [-1, 2, -3], centerlineDelta: [4, -5, 6], inferenceMs: expect.any(Number) });
    expect(session.run.mock.calls[1][0].references.dispose).toHaveBeenCalledTimes(1);
    await model.dispose();
    await model.dispose();
    expect(session.release).toHaveBeenCalledTimes(1);
});

it('returns no model when no add-on is published', async () => {
    (loadBackendPositionModel as jest.Mock).mockResolvedValue(null);
    await expect(TrackPositionModel.load()).resolves.toBeNull();
    expect(runtime.InferenceSession.create).not.toHaveBeenCalled();
});

it('feeds progress, game and map encodings in training order when the graph requests context', async () => {
    session.inputNames = ['images', 'references', 'context'];
    (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 512,
        metadata: { name: 'future-position-model' },
        categories: { game: ['iracing', 'acc'], map_name: ['Autodromo Nazionale Monza', 'nurburgring_24h'] } });
    const model = (await TrackPositionModel.load())!;
    const references = { reference: { x: 1, y: 2, z: 3 }, centerline: { x: 4, y: 5, z: 6 } };
    const frame = document.createElement('canvas');
    await model.predict(frame, references, { normalized_position: 0.25, game: 'acc',
        map_name: ['Nürburgring', 'nurburgring_24h'] });
    const context = session.run.mock.calls[1][0].context;
    expect(context).toMatchObject({ dims: [1, 5], data: new Float32Array([0.25, 0, 1, 0, 1]) });
    expect(context.dispose).toHaveBeenCalledTimes(1);
    await model.predict(frame, references, { normalized_position: 0.75, game: 'new-game', map_name: 'new-track' });
    expect(session.run.mock.calls[2][0].context.data).toEqual(new Float32Array([0.75, 0, 0, 0, 0]));
    await expect(model.predict(frame, references)).rejects.toThrow('normalized_position');
    expect(session.run).toHaveBeenCalledTimes(3);
    await model.dispose();
});

it('rejects malformed or nonfinite output and releases the failed session', async () => {
    session.run.mockResolvedValue({ deltas: tensor([1, 2, 3], new Float32Array([NaN, 0, 0, 0, 0, 0])) });
    await expect(TrackPositionModel.load()).rejects.toThrow('Invalid position add-on output');
    expect(session.release).toHaveBeenCalledTimes(1);
    expect(session.run.mock.calls[0][0].references.dispose).toHaveBeenCalledTimes(1);
});

it('waits for an in-flight prediction before releasing its GPU session', async () => {
    const model = (await TrackPositionModel.load())!;
    let finish!: (value: unknown) => void;
    session.run.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const frame = document.createElement('canvas');
    const pending = model.predict(frame, { reference: { x: 1, y: 2, z: 3 }, centerline: { x: 4, y: 5, z: 6 } });
    await Promise.resolve();
    const disposed = model.dispose();
    expect(session.release).not.toHaveBeenCalled();
    finish({ deltas: tensor([1, 2, 3], new Float32Array(6)) });
    await pending;
    await disposed;
    expect(session.release).toHaveBeenCalledTimes(1);
});
