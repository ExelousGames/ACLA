import apiService from 'services/api.service';

export interface BackendVisionModel {
    id: string;
    name: string;
    task: 'segment';
    classNames: string[];
    sizeBytes: number;
    sha256: string;
    downloadPath: string;
}

declare global {
    interface Window {
        trackVisionModels?: {
            prepare(model: BackendVisionModel, bytes?: ArrayBuffer): Promise<Uint8Array | null>;
        };
    }
}

let pending: Promise<{ bytes: ArrayBuffer; metadata: BackendVisionModel }> | undefined;

async function load() {
    const cache = window.trackVisionModels;
    if (!cache) throw new Error('Restart the Electron desktop app to enable local Track Vision models.');
    try {
        const { data: metadata } = await apiService.get<BackendVisionModel>('/ai-model/ultralytics/track-vision');
        if (!metadata || !/^[a-f\d]{24}$/i.test(metadata.id) || metadata.task !== 'segment'
            || !Array.isArray(metadata.classNames) || !metadata.classNames.length
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
    } catch (error) {
        if (error instanceof Error) throw error;
        const failure = error as { message?: string; data?: { message?: string } };
        throw new Error(failure?.data?.message || failure?.message || 'Unable to retrieve the Track Vision model from the backend.');
    }
}

export function loadBackendVisionModel() {
    if (!pending) pending = load().finally(() => { pending = undefined; });
    return pending;
}
