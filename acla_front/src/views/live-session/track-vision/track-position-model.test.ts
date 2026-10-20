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
        run: jest.fn().mockResolvedValue({ deltas: tensor([1, 1, 3], new Float32Array([-1, 2, -3])) }),
        release: jest.fn().mockResolvedValue(undefined) };
    (runtime.InferenceSession.create as jest.Mock).mockResolvedValue(session);
    (runtime.Tensor as unknown as jest.Mock).mockImplementation((_type, data, dims) => tensor(dims, data));
    (loadBackendPositionModel as jest.Mock).mockResolvedValue({ bytes: new ArrayBuffer(8), inputSize: 512, metadata: { name: 'track-position', classNames: ['reference_delta_x', 'reference_delta_y', 'reference_delta_z'] } });
    drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ fillRect: jest.fn(), drawImage,
        getImageData: () => ({ data: new Uint8ClampedArray(512 * 512 * 4).fill(255) }) } as any);
});
afterEach(() => {
    jest.restoreAllMocks();
    if (originalGpu) Object.defineProperty(navigator, 'gpu', originalGpu);
    else Reflect.deleteProperty(navigator, 'gpu');
});

it('runs the image and ordered raw reference XYZ on GPU and preserves all three signed reference-lap outputs', async () => {
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
    expect(session.run.mock.calls[1][0].images.data[0]).toBe(1);
    expect(result).toEqual({ referenceDelta: [-1, 2, -3], inferenceMs: expect.any(Number) });
    expect(session.run).toHaveBeenCalledTimes(2); // One warm-up and one position inference.
    expect(session.run.mock.calls[1][0].references.dispose).toHaveBeenCalledTimes(1);
    await model.dispose();
    await model.dispose();
    expect(session.release).toHaveBeenCalledTimes(1);
});

it('reports a missing position model without initializing another session', async () => {
    (loadBackendPositionModel as jest.Mock).mockResolvedValue(null);
    await expect(TrackPositionModel.load()).rejects.toThrow('No ONNX track-position model');
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
