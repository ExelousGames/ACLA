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
    modelType?: string;
}

interface NormalizedModelOptions {
    modelType: string;
}

interface ModelCacheEntry {
    data: any;
    refCount: number;
    options: NormalizedModelOptions;
}

const normalizeOptions = (options: DownloadModelOptions = {}): NormalizedModelOptions => ({
    modelType: options.modelType ?? DEFAULT_MODEL_TYPE
});

const buildCacheKey = (options: NormalizedModelOptions): string => options.modelType;

const imitationModelCache = new Map<string, ModelCacheEntry>();

export interface ExpertPredictionResult {
    status: 'success' | 'error' | 'log' | 'ready' | 'shutdown' | 'pong';
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
    request_id?: string;
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
    options: NormalizedModelOptions
): Promise<any> => {
    const { modelType } = options;

    const initResponse = await apiService.get<ChunkedModelInitResponse>(
        `/ai-model/active/${modelType}/prepare-chunked`
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

const STREAM_READY_TIMEOUT_MS = 10000;
const STREAM_READY_MAX_ATTEMPTS = 12;
const STREAM_REQUEST_TIMEOUT_MS = 15000;

interface PendingPrediction {
    resolve: (value: ExpertPredictionResult) => void;
    reject: (error: Error) => void;
    logs: string[];
    timeoutId: number;
}

export interface ExpertActionsRunner {
    predict: (telemetryData: any[]) => Promise<ExpertPredictionResult>;
    dispose: () => Promise<void>;
}

let predictionRequestSequence = 0;

const ensureElectronStreamingSupport = () => {
    if (
        !window?.electronAPI?.runPythonScript ||
        !window.electronAPI.sendMessageToPython ||
        typeof window.electronAPI.onPythonMessage !== 'function' ||
        typeof window.electronAPI.onPythonEnd !== 'function'
    ) {
        throw new Error('Python streaming API is not available in this environment');
    }
};

export const createExpertActionsRunner = async (
    modelData: any
): Promise<ExpertActionsRunner> => {
    ensureElectronStreamingSupport();

    const modelFilePath = await writeTempJsonFile(modelData, 'imitation_model');

    const pythonOptions: PythonShellOptions = {
        mode: 'text',
        pythonOptions: ['-u'],
        scriptPath: 'src/py-scripts',
        args: [modelFilePath, '--stream']
    };

    const { shellId } = await window.electronAPI.runPythonScript(
        'run_expert_actions_prediction.py',
        pythonOptions
    );

    const pending = new Map<string, PendingPrediction>();
    let isReady = false;
    let isClosed = false;
    let isDisposed = false;
    let removeMessageListener: (() => void) | null = null;
    let removeEndListener: (() => void) | null = null;

    let readyResolve: (() => void) | null = null;
    let readyReject: ((error: Error) => void) | null = null;

    const readyPromise = new Promise<void>((resolve, reject) => {
        readyResolve = resolve;
        readyReject = reject;
    });

    const clearReadyCallbacks = () => {
        readyResolve = null;
        readyReject = null;
    };

    let readyWatchdogId: number | null = null;
    let readyPingAttempts = 0;

    const stopReadyWatchdog = (resetAttempts = false) => {
        if (readyWatchdogId !== null) {
            window.clearTimeout(readyWatchdogId);
            readyWatchdogId = null;
        }
        if (resetAttempts) {
            readyPingAttempts = 0;
        }
    };

    const scheduleReadyWatchdog = () => {
        if (isClosed || isReady) {
            return;
        }

        stopReadyWatchdog(false);

        readyWatchdogId = window.setTimeout(() => {
            if (isClosed || isReady) {
                return;
            }

            if (readyPingAttempts >= STREAM_READY_MAX_ATTEMPTS) {
                handleSessionFailure(new Error('Timed out while waiting for expert prediction session to start'));
                return;
            }

            readyPingAttempts += 1;

            window.electronAPI.sendMessageToPython(shellId, JSON.stringify({
                action: 'ping',
                request_id: `ready-ping-${shellId}-${Date.now()}-${readyPingAttempts}`
            })).then((response) => {
                if (!response?.success) {
                    throw new Error(response?.error || 'Python session unavailable');
                }
            }).catch((error) => {
                handleSessionFailure(error instanceof Error ? error : new Error(String(error)));
            });

            scheduleReadyWatchdog();
        }, STREAM_READY_TIMEOUT_MS);
    };

    const cleanup = () => {
        if (removeMessageListener) {
            removeMessageListener();
            removeMessageListener = null;
        }
        if (removeEndListener) {
            removeEndListener();
            removeEndListener = null;
        }
        void deleteTempFileSafely(modelFilePath);
    };

    const handleSessionFailure = (error: Error) => {
        if (!isClosed) {
            isClosed = true;
        }

        stopReadyWatchdog(true);

        if (!isReady) {
            if (readyReject) {
                readyReject(error);
            }
            clearReadyCallbacks();
        }

        pending.forEach((entry) => {
            window.clearTimeout(entry.timeoutId);
            entry.reject(error);
        });
        pending.clear();
        cleanup();
    };

    readyPingAttempts = 0;
    scheduleReadyWatchdog();

    const handlePythonMessage = (returnedShellId: number, message: string) => {
        if (returnedShellId !== shellId || !message || isClosed) {
            return;
        }

        let parsed: ExpertPredictionResult & { request_id?: string };
        try {
            parsed = JSON.parse(message);
        } catch (error) {
            console.warn('Non-JSON message from Python script', message, error);
            return;
        }

        const requestKey = parsed.request_id ?? (parsed as any).requestId ?? null;
        const status = parsed.status;

        if (!isReady) {
            if (status === 'ready') {
                isReady = true;
                stopReadyWatchdog(true);
                if (readyResolve) {
                    readyResolve();
                }
                clearReadyCallbacks();
                return;
            }

            if (status === 'error') {
                handleSessionFailure(new Error(parsed.message || 'Failed to initialize expert prediction session'));
                return;
            }

            if (status === 'log') {
                readyPingAttempts = 0;
                scheduleReadyWatchdog();
                return;
            }
        }

        if (status === 'pong') {
            if (!isReady) {
                readyPingAttempts = 0;
                scheduleReadyWatchdog();
            }
            return;
        }

        if (status === 'log' && requestKey != null) {
            const pendingEntry = pending.get(String(requestKey));
            if (pendingEntry && parsed.message) {
                pendingEntry.logs.push(parsed.message);
            }
            return;
        }

        if (requestKey != null) {
            const key = String(requestKey);
            const pendingEntry = pending.get(key);
            if (pendingEntry) {
                window.clearTimeout(pendingEntry.timeoutId);

                if (status === 'success') {
                    pending.delete(key);
                    const aggregatedLogs = [...pendingEntry.logs];
                    if (Array.isArray(parsed.logs)) {
                        parsed.logs.forEach((log) => {
                            if (!aggregatedLogs.includes(log)) {
                                aggregatedLogs.push(log);
                            }
                        });
                    }

                    const payload: ExpertPredictionResult = {
                        ...parsed,
                        logs: aggregatedLogs.length > 0 ? aggregatedLogs : parsed.logs
                    };

                    pendingEntry.resolve(payload);
                    return;
                }

                if (status === 'error') {
                    pending.delete(key);
                    const error = new Error(parsed.message || 'Python script reported an error');
                    (error as Error & { detail?: ExpertPredictionResult }).detail = parsed;
                    pendingEntry.reject(error);
                    return;
                }

                if (status === 'shutdown') {
                    pending.delete(key);
                    pendingEntry.reject(new Error('Expert prediction session ended unexpectedly'));
                    return;
                }
            }
        }

        if (status === 'shutdown') {
            handleSessionFailure(new Error('Expert prediction session ended unexpectedly'));
            return;
        }

        if (status === 'error' && requestKey == null) {
            handleSessionFailure(new Error(parsed.message || 'Python session error'));
        }
    };

    removeMessageListener = window.electronAPI.onPythonMessage(handlePythonMessage);
    removeEndListener = window.electronAPI.onPythonEnd('expert-actions-service', (returnedShellId: number) => {
        if (returnedShellId !== shellId) {
            return;
        }

        handleSessionFailure(new Error('Expert prediction session terminated'));
    });

    try {
        await readyPromise;
    } catch (error) {
        throw error;
    }

    const predict = async (telemetryData: any[]): Promise<ExpertPredictionResult> => {
        if (isClosed) {
            throw new Error('Expert prediction session is not available');
        }

        if (!Array.isArray(telemetryData) || telemetryData.length === 0) {
            throw new Error('Telemetry data is empty');
        }

        if (pending.size > 0) {
            throw new Error('Another prediction request is already running');
        }

        const requestId = `req-${shellId}-${Date.now()}-${++predictionRequestSequence}`;

        return new Promise<ExpertPredictionResult>((resolve, reject) => {
            const timeoutId = window.setTimeout(() => {
                if (!pending.has(requestId)) {
                    return;
                }
                const timedOut = pending.get(requestId);
                if (timedOut) {
                    pending.delete(requestId);
                    timedOut.reject(new Error('Prediction timed out'));
                }
            }, STREAM_REQUEST_TIMEOUT_MS);

            const pendingEntry: PendingPrediction = {
                resolve,
                reject: (error: Error) => {
                    reject(error);
                },
                logs: [],
                timeoutId
            };

            pending.set(requestId, pendingEntry);

            window.electronAPI.sendMessageToPython(shellId, JSON.stringify({
                action: 'predict',
                request_id: requestId,
                telemetry: telemetryData
            })).then((response) => {
                if (!response?.success) {
                    throw new Error(response?.error || 'Python session unavailable');
                }
            }).catch((error) => {
                const normalizedError = error instanceof Error ? error : new Error(String(error));
                if (pending.delete(requestId)) {
                    window.clearTimeout(timeoutId);
                    pendingEntry.reject(normalizedError);
                }
                handleSessionFailure(normalizedError);
            });
        });
    };

    const dispose = async (): Promise<void> => {
        if (isDisposed) {
            return;
        }
        isDisposed = true;

        stopReadyWatchdog(true);

        if (isClosed) {
            cleanup();
            return;
        }

        isClosed = true;

        pending.forEach((entry, key) => {
            window.clearTimeout(entry.timeoutId);
            entry.reject(new Error('Expert prediction session disposed'));
            pending.delete(key);
        });

        try {
            await window.electronAPI.sendMessageToPython(shellId, JSON.stringify({
                action: 'shutdown',
                request_id: `shutdown-${shellId}`
            }));
        } catch (error) {
            console.warn('Failed to send shutdown message to expert prediction session', error);
        } finally {
            cleanup();
        }
    };

    return {
        predict,
        dispose
    };
};
