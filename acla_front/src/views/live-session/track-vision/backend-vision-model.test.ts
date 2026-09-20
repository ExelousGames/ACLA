import apiService from 'services/api.service';
import { BackendVisionModel, loadBackendVisionModel } from './backend-vision-model';

jest.mock('services/api.service', () => ({ __esModule: true, default: { get: jest.fn(), getBinary: jest.fn() } }));
const metadata: BackendVisionModel = {
    id: 'a'.repeat(24), name: 'track-v2', task: 'segment', classNames: ['curb', 'track'],
    sha256: 'b'.repeat(64), sizeBytes: 4, downloadPath: `/ai-model/ultralytics/${'a'.repeat(24)}/file`,
};
const prepare = jest.fn();

beforeEach(() => {
    window.trackVisionModels = { prepare };
    (apiService.get as jest.Mock).mockResolvedValue({ data: metadata });
    (apiService.getBinary as jest.Mock).mockResolvedValue(new ArrayBuffer(4));
});
afterEach(() => { delete window.trackVisionModels; });

it('checks the backend version and reuses local weights without downloading', async () => {
    prepare.mockResolvedValue(new Uint8Array([1, 2]));
    const result = await loadBackendVisionModel();
    expect(apiService.get).toHaveBeenCalledWith('/ai-model/ultralytics/track-vision');
    expect(prepare).toHaveBeenCalledWith(metadata);
    expect(apiService.getBinary).not.toHaveBeenCalled();
    expect(result.metadata.classNames).toEqual(['curb', 'track']);
    expect(new Uint8Array(result.bytes)).toEqual(new Uint8Array([1, 2]));
});

it('downloads missing weights from the backend and saves them before inference', async () => {
    prepare.mockResolvedValueOnce(null).mockResolvedValueOnce(new Uint8Array([7]));
    await loadBackendVisionModel();
    expect(apiService.getBinary).toHaveBeenCalledWith(metadata.downloadPath, { timeoutMs: 300000 });
    expect(prepare).toHaveBeenLastCalledWith(metadata, new ArrayBuffer(4));
});

it('coalesces simultaneous panel loads into one download and export', async () => {
    prepare.mockResolvedValueOnce(null).mockResolvedValueOnce(new Uint8Array([7]));
    const [first, second] = await Promise.all([loadBackendVisionModel(), loadBackendVisionModel()]);
    expect(first).toBe(second);
    expect(apiService.getBinary).toHaveBeenCalledTimes(1);
    expect(prepare).toHaveBeenCalledTimes(2);
});

it('rejects external download paths without fetching any weights', async () => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: { ...metadata, downloadPath: 'https://example.com/model.pt' } });
    await expect(loadBackendVisionModel()).rejects.toThrow('invalid Track Vision model metadata');
    expect(prepare).not.toHaveBeenCalled();
    expect(apiService.getBinary).not.toHaveBeenCalled();
});

it('reports backend errors and permits retry without any bundled model fallback', async () => {
    (apiService.get as jest.Mock).mockRejectedValueOnce({ status: 404, data: { message: 'No segmentation model uploaded' } });
    await expect(loadBackendVisionModel()).rejects.toThrow('No segmentation model uploaded');
    expect(prepare).not.toHaveBeenCalled();
    prepare.mockResolvedValue(new Uint8Array([1]));
    await expect(loadBackendVisionModel()).resolves.toMatchObject({ metadata });
});

it('preserves conversion errors and never downloads another model as a fallback', async () => {
    prepare.mockRejectedValue(new Error('Backend labels do not match the model'));
    await expect(loadBackendVisionModel()).rejects.toThrow('Backend labels do not match');
    expect(apiService.getBinary).not.toHaveBeenCalled();
});
