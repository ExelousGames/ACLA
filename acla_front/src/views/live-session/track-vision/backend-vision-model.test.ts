import apiService from 'services/api.service';
import { BackendVisionModel, loadBackendPositionModel } from './backend-vision-model';

jest.mock('services/api.service', () => ({ __esModule: true, default: { get: jest.fn(), getBinary: jest.fn() } }));
const metadata: BackendVisionModel = {
    id: 'a'.repeat(24), name: 'track-segments', task: 'segment', classNames: ['curb', 'track'],
    sha256: 'b'.repeat(64), sizeBytes: 4, downloadPath: `/ai-model/ultralytics/${'a'.repeat(24)}/file`,
};
const catalogueModel = { ...metadata, _id: metadata.id, createdAt: '2026-10-15T12:00:00Z' };
const positionMetadata: BackendVisionModel = { ...metadata, name: 'track-position', task: 'position' };
const positionCatalogueModel = { ...catalogueModel, ...positionMetadata,
    metadata: { format: 'acla-yolo-position-v1', imgsz: 512 } };
const prepare = jest.fn();

beforeEach(() => {
    window.trackVisionModels = { prepare };
    (apiService.get as jest.Mock).mockResolvedValue({ data: [positionCatalogueModel] });
    (apiService.getBinary as jest.Mock).mockResolvedValue(new ArrayBuffer(4));
});
afterEach(() => { delete window.trackVisionModels; });

it('checks the backend version and reuses local weights without downloading', async () => {
    prepare.mockResolvedValue(new Uint8Array([1, 2]));
    const result = await loadBackendPositionModel();
    expect(apiService.get).toHaveBeenCalledWith('/ai-model/ultralytics', { name: 'track-position' });
    expect(prepare).toHaveBeenCalledWith(positionMetadata, undefined, 512);
    expect(apiService.getBinary).not.toHaveBeenCalled();
    expect(result!.metadata.classNames).toEqual(['curb', 'track']);
    expect(new Uint8Array(result!.bytes)).toEqual(new Uint8Array([1, 2]));
});

it('downloads missing weights from the backend and saves them before inference', async () => {
    prepare.mockResolvedValueOnce(null).mockResolvedValueOnce(new Uint8Array([7]));
    await loadBackendPositionModel();
    expect(apiService.getBinary).toHaveBeenCalledWith(metadata.downloadPath, { timeoutMs: 300000 });
    expect(prepare).toHaveBeenLastCalledWith(positionMetadata, new ArrayBuffer(4), 512);
});

it('coalesces simultaneous panel loads into one download and export', async () => {
    prepare.mockResolvedValueOnce(null).mockResolvedValueOnce(new Uint8Array([7]));
    const [first, second] = await Promise.all([loadBackendPositionModel(), loadBackendPositionModel()]);
    expect(first).toBe(second);
    expect(apiService.getBinary).toHaveBeenCalledTimes(1);
    expect(prepare).toHaveBeenCalledTimes(2);
});

it('rejects invalid model IDs without fetching any weights', async () => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: [{ ...positionCatalogueModel, _id: '../model' }] });
    await expect(loadBackendPositionModel()).rejects.toThrow('invalid Track Vision model metadata');
    expect(prepare).not.toHaveBeenCalled();
    expect(apiService.getBinary).not.toHaveBeenCalled();
});

it('preserves conversion errors and never downloads another model as a fallback', async () => {
    prepare.mockRejectedValue(new Error('Backend labels do not match the model'));
    await expect(loadBackendPositionModel()).rejects.toThrow('Backend labels do not match');
    expect(apiService.getBinary).not.toHaveBeenCalled();
});

it('loads the newest position checkpoint at its training resolution using the existing cache', async () => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: [
        { ...positionCatalogueModel, _id: 'd'.repeat(24), createdAt: '2026-10-14T12:00:00Z', metadata: { format: 'acla-yolo-position-v1', imgsz: 768 } },
        { ...catalogueModel, _id: 'c'.repeat(24) },
        positionCatalogueModel,
    ] });
    prepare.mockResolvedValue(new Uint8Array([9]));
    const [first, second] = await Promise.all([loadBackendPositionModel(), loadBackendPositionModel()]);
    expect(first).toBe(second);
    expect(first).toMatchObject({ inputSize: 512, metadata: { id: metadata.id, task: 'position' } });
    expect(apiService.get).toHaveBeenCalledWith('/ai-model/ultralytics', { name: 'track-position' });
    expect(prepare).toHaveBeenCalledWith(positionMetadata, undefined, 512);
    expect(apiService.getBinary).not.toHaveBeenCalled();
});

it('returns no shared checkpoint when only a separate segmentation model exists', async () => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: [catalogueModel] });
    await expect(loadBackendPositionModel()).resolves.toBeNull();
    expect(prepare).not.toHaveBeenCalled();
});

it.each(['acla-yolo-position-v1', 'acla-yolo-position-v2', 'future-position-v999', undefined])(
    'loads the newest position model regardless of its format label (%s)', async (format) => {
        const newer = { ...positionCatalogueModel, _id: 'e'.repeat(24), createdAt: '2026-10-16T19:22:34Z',
            metadata: { format, imgsz: 768 } };
        (apiService.get as jest.Mock).mockResolvedValue({ data: [positionCatalogueModel, newer] });
        prepare.mockResolvedValue(new Uint8Array([9]));
        await expect(loadBackendPositionModel()).resolves.toMatchObject({ inputSize: 768, metadata: { id: newer._id } });
        expect(prepare).toHaveBeenCalledWith(expect.objectContaining({ id: newer._id }), undefined, 768);
    });

it.each([undefined, '768', 0, 518, 4096])('identifies an unusable position input size (%s)', async (imgsz) => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: [{ ...positionCatalogueModel, metadata: { imgsz } }] });
    await expect(loadBackendPositionModel()).rejects.toThrow('metadata.imgsz');
    expect(prepare).not.toHaveBeenCalled();
});

it('preserves the published category vocabulary and order for position inference', async () => {
    const categories = { game: ['iracing', 'acc'], map_name: ['nurburgring_24h', 'Autodromo Nazionale Monza'] };
    (apiService.get as jest.Mock).mockResolvedValue({ data: [{ ...positionCatalogueModel,
        metadata: { format: 'arbitrary-future-version', imgsz: 768, categories } }] });
    prepare.mockResolvedValue(new Uint8Array([9]));
    await expect(loadBackendPositionModel()).resolves.toMatchObject({ categories });
});

describe.each([
    { task: 'position', model: positionCatalogueModel, load: () => loadBackendPositionModel() },
])('$task model selection', ({ model, load }) => {
    it('selects the latest upload date for the exact model name and task, regardless of catalogue order', async () => {
        const older = { ...model, _id: 'f'.repeat(24), createdAt: '2026-10-15T12:30:00Z', updatedAt: '2026-10-16T12:00:00Z' };
        const newer = { ...model, _id: 'b'.repeat(24), createdAt: '2026-10-15T09:00:00-04:00' };
        const otherName = { ...model, _id: 'c'.repeat(24), name: `${model.name}-v99`, createdAt: '2026-10-16T12:00:00Z' };
        const otherTask = { ...model, _id: 'd'.repeat(24), task: 'detect', createdAt: '2026-10-16T12:00:00Z' };
        prepare.mockResolvedValue(new Uint8Array([9]));
        for (const data of [[older, otherName, newer, otherTask], [otherTask, newer, otherName, older]]) {
            (apiService.get as jest.Mock).mockResolvedValue({ data });
            await expect(load()).resolves.toMatchObject({ metadata: { id: newer._id, name: model.name, task: model.task } });
            expect(prepare).toHaveBeenLastCalledWith(expect.objectContaining({
                id: newer._id, downloadPath: `/ai-model/ultralytics/${newer._id}/file`,
            }), undefined, expect.any(Number));
        }
    });

    it('breaks equal upload-date ties by model ID consistently', async () => {
        const tied = { ...model, _id: 'e'.repeat(24) };
        prepare.mockResolvedValue(new Uint8Array([9]));
        for (const data of [[model, tied], [tied, model]]) {
            (apiService.get as jest.Mock).mockResolvedValue({ data });
            await expect(load()).resolves.toMatchObject({ metadata: { id: tied._id } });
        }
    });

    it('checks for a newer upload with the same name on the next load', async () => {
        const newer = { ...model, _id: 'b'.repeat(24), createdAt: '2026-10-16T12:00:00Z' };
        (apiService.get as jest.Mock).mockResolvedValueOnce({ data: [model] }).mockResolvedValueOnce({ data: [model, newer] });
        prepare.mockResolvedValue(new Uint8Array([9]));
        await expect(load()).resolves.toMatchObject({ metadata: { id: model._id } });
        await expect(load()).resolves.toMatchObject({ metadata: { id: newer._id } });
        expect(apiService.get).toHaveBeenCalledTimes(2);
    });

    it.each([undefined, 'invalid-date'])('rejects an unverifiable upload date (%s)', async (createdAt) => {
        (apiService.get as jest.Mock).mockResolvedValue({ data: [{ ...model, createdAt }] });
        await expect(load()).rejects.toThrow('Invalid Track Vision model upload date');
        expect(prepare).not.toHaveBeenCalled();
    });
});

it('does not load unrelated model names', async () => {
    (apiService.get as jest.Mock).mockResolvedValue({ data: [{ ...positionCatalogueModel, name: 'other-position' }] });
    await expect(loadBackendPositionModel()).resolves.toBeNull();
    expect(prepare).not.toHaveBeenCalled();
});
