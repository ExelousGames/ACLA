import apiService from 'services/api.service';
import { PythonShellOptions } from './pythonService';

const DEFAULT_MODEL_TYPE = 'imitation_learning';

interface ChunkedModelInitResponse {
    success: boolean;
    data: any;
    chunking: {
        sessionId: string;
        totalChunks: number;
        chunkSize: number;
        totalSize: number;
        totalSizeHuman: string;
    };
    message?: string;
}

interface ChunkResponse {
    success: boolean;
    sessionId: string;
    chunkIndex: number;
    totalChunks: number;
    data: string;
    isLastChunk: boolean;
    message?: string;
}

export interface DownloadModelOptions {
    trackName?: string;
    carName?: string;
    modelType?: string;
}

interface NormalizedModelOptions {
    trackName?: string;
    carName?: string;
    modelType: string;
}

interface ModelCacheEntry {
    data: any;
    refCount: number;
    options: NormalizedModelOptions;
}

const normalizeOptions = (options: DownloadModelOptions = {}): NormalizedModelOptions => ({
    trackName: options.trackName ?? undefined,
    carName: options.carName ?? undefined,
    modelType: options.modelType ?? DEFAULT_MODEL_TYPE
});

const buildCacheKey = (options: NormalizedModelOptions): string => {
    const trackPart = options.trackName ?? '*';
    const carPart = options.carName ?? '*';
    return `${options.modelType}::${trackPart}::${carPart}`;
};

const imitationModelCache = new Map<string, ModelCacheEntry>();

export interface ExpertPredictionResult {
    status: 'success' | 'error' | 'log';
    prediction?: Record<string, number>;
    metadata?: {
        telemetry_samples_used: number;
        has_normalized_positions: boolean;
    };
    normalized_positions?: number[];
    position_series?: Array<Record<string, number>>;
    message?: string;
    traceback?: string;
    logs?: string[];
    stage?: string;
}

const base64ToUint8Array = (base64: string): Uint8Array => {
    if (typeof base64 !== 'string') {
        throw new Error('Chunk payload is not a base64 string');
    }

    if (typeof window === 'undefined' || typeof window.atob !== 'function') {
        throw new Error('Base64 decoding is not supported in this environment');
    }

    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
};

const mergeBuffers = (arrays: Uint8Array[]): Uint8Array => {
    const totalLength = arrays.reduce((acc, arr) => acc + arr.length, 0);
    const merged = new Uint8Array(totalLength);
    let offset = 0;
    arrays.forEach((arr) => {
        merged.set(arr, offset);
        offset += arr.length;
    });
    return merged;
};

export const downloadActiveImitationModel = async (
    options: DownloadModelOptions = {}
): Promise<any> => {
    const { trackName, carName, modelType = DEFAULT_MODEL_TYPE } = options;

    const params: Record<string, string> = {};
    if (trackName) params.trackName = trackName;
    if (carName) params.carName = carName;

    const initResponse = await apiService.get<ChunkedModelInitResponse>(
        `/ai-model/active/${modelType}/prepare-chunked`,
        params
    );

    const initPayload = initResponse.data as unknown as ChunkedModelInitResponse;

    if (!initPayload?.success) {
        throw new Error(initPayload?.message || 'Failed to initialize model download');
    }

    const { sessionId, totalChunks } = initPayload.chunking;
    if (!sessionId || typeof totalChunks !== 'number' || totalChunks <= 0) {
        throw new Error('Invalid chunking metadata received from backend');
    }

    const chunkBuffers: Uint8Array[] = [];
    for (let index = 0; index < totalChunks; index += 1) {
        const chunkResponse = await apiService.get<ChunkResponse>(
            `/ai-model/active/chunked-data/${sessionId}/${index}`
        );
        const payload = chunkResponse.data as unknown as ChunkResponse;
        if (!payload?.success || !payload.data) {
            throw new Error(payload?.message || `Failed to download chunk #${index}`);
        }
        chunkBuffers.push(base64ToUint8Array(payload.data));
    }

    const merged = mergeBuffers(chunkBuffers);
    const jsonString = new TextDecoder().decode(merged);

    try {
        return JSON.parse(jsonString);
    } catch (error) {
        throw new Error(`Failed to parse model JSON: ${(error as Error).message}`);
    }
};

export const acquirePersistentImitationModel = async (
    options: DownloadModelOptions = {},
    forceRefresh = false
): Promise<{ cacheKey: string; data: any; options: NormalizedModelOptions }> => {
    const normalized = normalizeOptions(options);
    const cacheKey = buildCacheKey(normalized);

    if (forceRefresh) {
        imitationModelCache.delete(cacheKey);
    }

    let cacheEntry = imitationModelCache.get(cacheKey);

    if (!cacheEntry) {
        const data = await downloadActiveImitationModel(normalized);
        cacheEntry = {
            data,
            refCount: 0,
            options: normalized
        };
        imitationModelCache.set(cacheKey, cacheEntry);
    }

    cacheEntry.refCount += 1;

    return {
        cacheKey,
        data: cacheEntry.data,
        options: normalized
    };
};

export const releasePersistentImitationModel = (cacheKey: string): void => {
    const cacheEntry = imitationModelCache.get(cacheKey);
    if (!cacheEntry) {
        return;
    }

    cacheEntry.refCount -= 1;
    if (cacheEntry.refCount <= 0) {
        imitationModelCache.delete(cacheKey);
    }
};

export const writeTempJsonFile = async (
    content: unknown,
    prefix: string
): Promise<string> => {
    if (!window?.electronAPI?.writeTempFile) {
        throw new Error('Temporary file API is not available in this environment');
    }

    const payload = typeof content === 'string' ? content : JSON.stringify(content);
    const result = await window.electronAPI.writeTempFile({
        content: payload,
        prefix,
        extension: '.json'
    });

    if (!result?.success || !result.path) {
        throw new Error(result?.error || 'Failed to write temporary JSON file');
    }

    return result.path;
};

const deleteTempFileSafely = async (path: string | undefined) => {
    if (!path) return;
    try {
        if (window?.electronAPI?.deleteTempFile) {
            await window.electronAPI.deleteTempFile(path);
        }
    } catch (error) {
        console.warn('Failed to delete temp file', path, error);
    }
};

export const runExpertActionPrediction = async (
    modelData: any,
    telemetryData: any[]
): Promise<ExpertPredictionResult> => {
    if (!window?.electronAPI?.runPythonScript) {
        throw new Error('Python execution API is not available in this environment');
    }

    const modelFilePath = await writeTempJsonFile(modelData, 'imitation_model');
    const telemetryFilePath = await writeTempJsonFile(telemetryData, 'telemetry_samples');

    const pythonOptions: PythonShellOptions = {
        mode: 'text',
        pythonOptions: ['-u'],
        scriptPath: 'src/py-scripts',
        args: [modelFilePath, telemetryFilePath]
    };

    try {
        const { shellId } = await window.electronAPI.runPythonScript(
            'run_expert_actions_prediction.py',
            pythonOptions
        );

        return await new Promise<ExpertPredictionResult>((resolve, reject) => {
        let resolved = false;
        let lastMessage: ExpertPredictionResult | null = null;
        let removedListener = false;

        const removeListener = window.electronAPI.onPythonMessage((returnedShellId, message) => {
            if (returnedShellId !== shellId) return;
            if (!message) return;
            try {
                const parsed = JSON.parse(message) as ExpertPredictionResult;
                lastMessage = parsed;
                if (parsed.status === 'success' || parsed.status === 'error') {
                    resolved = true;
                    if (!removedListener) {
                        removeListener();
                        removedListener = true;
                    }
                    void deleteTempFileSafely(modelFilePath);
                    void deleteTempFileSafely(telemetryFilePath);
                    if (parsed.status === 'success') {
                        resolve(parsed);
                    } else {
                        reject(new Error(parsed.message || 'Python script reported an error'));
                    }
                }
            } catch (error) {
                console.warn('Non-JSON message from Python script', message, error);
            }
        });

        window.electronAPI.onPythonEnd((returnedShellId) => {
            if (returnedShellId !== shellId) return;
            if (!removedListener) {
                removeListener();
                removedListener = true;
            }
            void deleteTempFileSafely(modelFilePath);
            void deleteTempFileSafely(telemetryFilePath);
            if (resolved) {
                return;
            }
            if (lastMessage) {
                if (lastMessage.status === 'success') {
                    resolve(lastMessage);
                } else {
                    reject(new Error(lastMessage.message || 'Python script reported an error'));
                }
            } else {
                reject(new Error('Python script completed without returning data'));
            }
        });
    });
    } catch (error) {
        await deleteTempFileSafely(modelFilePath);
        await deleteTempFileSafely(telemetryFilePath);
        throw error;
    }
};
