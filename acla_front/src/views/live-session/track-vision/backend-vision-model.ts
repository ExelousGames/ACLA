import apiService from 'services/api.service';

export interface BackendVisionModel {
    id: string;
    name: string;
    task: 'segment' | 'position';
    filename: string;
    classNames: string[];
    sizeBytes: number;
    sha256: string;
    downloadPath: string;
}

interface CatalogueVisionModel extends Omit<BackendVisionModel, 'id' | 'downloadPath'> {
    _id: string;
    createdAt: string;
    metadata: { imgsz: number; categories?: Record<string, string[]> };
}

declare global {
    interface Window {
        trackVisionModels?: {
            prepare(model: BackendVisionModel, bytes?: ArrayBuffer): Promise<Uint8Array | null>;
        };
    }
}

async function findLatestModel(name: string, task: BackendVisionModel['task']) {
    const { data } = await apiService.get<CatalogueVisionModel[]>('/ai-model/ultralytics', { name });
    if (!Array.isArray(data)) throw new Error('Invalid Track Vision model catalogue.');
    const candidates = data.filter((model) => model.name === name && model.task === task
        && typeof model.filename === 'string' && /\.onnx$/i.test(model.filename))
        .map((model) => ({ model, uploadedAt: Date.parse(model.createdAt) }));
    if (candidates.some(({ uploadedAt }) => !Number.isFinite(uploadedAt))) {
        throw new Error('Invalid Track Vision model upload date.');
    }
    // Compare upload timestamps, independent of catalogue order or model-name versions.
    candidates.sort((a, b) => b.uploadedAt - a.uploadedAt || b.model._id.localeCompare(a.model._id));
    return candidates[0]?.model;
}

function toMetadata(model: CatalogueVisionModel): BackendVisionModel {
    return { id: model._id, name: model.name, task: model.task, filename: model.filename,
        classNames: model.classNames, sha256: model.sha256, sizeBytes: model.sizeBytes,
        downloadPath: `/ai-model/ultralytics/${model._id}/file` };
}

async function prepare(metadata: BackendVisionModel) {
    const cache = window.trackVisionModels;
    if (!cache) throw new Error('Restart the Electron desktop app to enable local Track Vision models.');
    if (!/^[a-f\d]{24}$/i.test(metadata.id) || !Array.isArray(metadata.classNames) || !metadata.classNames.length
        || !metadata.classNames.every((label) => typeof label === 'string' && label.trim())
        || metadata.downloadPath !== `/ai-model/ultralytics/${metadata.id}/file`) {
        throw new Error('The backend returned invalid Track Vision model metadata.');
    }
    let bytes = await cache.prepare(metadata);
    if (!bytes) {
        const weights = await apiService.getBinary(metadata.downloadPath, { timeoutMs: 300000 });
        bytes = await cache.prepare(metadata, weights);
    }
    if (!bytes?.byteLength) throw new Error('Track Vision model preparation produced no weights.');
    return { bytes: new Uint8Array(bytes).buffer, metadata };
}

async function loadModel(name: string, task: BackendVisionModel['task']) {
    const model = await findLatestModel(name, task);
    if (!model) return null;
    const inputSize = model.metadata?.imgsz;
    // Format labels are descriptive, not a compatibility allowlist. The inference
    // runtime checks the downloaded ONNX graph's tensor interface.
    if (!Number.isInteger(inputSize) || inputSize < 32 || inputSize > 2048 || inputSize % 32 !== 0) {
        throw new Error(`${name} metadata.imgsz must be an integer from 32 to 2048, divisible by 32.`);
    }
    return { ...await prepare(toMetadata(model)), inputSize, categories: model.metadata?.categories };
}

let positionPending: ReturnType<typeof loadModel> | undefined;
export function loadBackendPositionModel() {
    if (!positionPending) positionPending = loadModel('track-position', 'position').finally(() => { positionPending = undefined; });
    return positionPending;
}

let segmentationPending: ReturnType<typeof loadModel> | undefined;
export function loadBackendSegmentationModel() {
    if (!segmentationPending) segmentationPending = loadModel('track-segments', 'segment').finally(() => { segmentationPending = undefined; });
    return segmentationPending;
}
