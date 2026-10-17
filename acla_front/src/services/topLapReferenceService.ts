import apiService from './api.service';
import { decodeBase64Bytes, decodeTopLapPickle } from './topLapReferencePickle';

export interface TopLapReferenceEntry {
    game: string;
    track: string;
    car: string;
    avg_grip_int: number;
    x: number[];
    y: number[][];
    target_features: string[];
}

interface ModelPreparation {
    success: boolean;
    message?: string;
    chunking?: { sessionId: string; totalChunks: number; totalSize: number };
}

interface ModelChunk {
    success: boolean;
    message?: string;
    sessionId: string;
    chunkIndex: number;
    totalChunks: number;
    data: string;
}

const invalidModel = (): never => { throw new Error('Invalid reference lap model'); };
const finiteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const decodeModel = (payload: any): TopLapReferenceEntry[] => {
    const store = payload?.top_lap_store;
    if (!store || typeof store !== 'object' || Array.isArray(store)) return invalidModel();
    return Object.values(store).map((encoded) => {
        if (typeof encoded !== 'string') return invalidModel();
        const entry = decodeTopLapPickle(encoded) as TopLapReferenceEntry | null;
        if (!entry || typeof entry.track !== 'string' || !entry.track
            || typeof entry.car !== 'string' || !entry.car
            || (entry.game !== undefined && (typeof entry.game !== 'string' || !entry.game))
            || !Number.isInteger(entry.avg_grip_int) || entry.avg_grip_int < 0 || entry.avg_grip_int > 6
            || !Array.isArray(entry.target_features) || !entry.target_features.every((name) => typeof name === 'string')
            || new Set(entry.target_features).size !== entry.target_features.length
            || !Array.isArray(entry.x) || !entry.x.length
            || !entry.x.every((position, index) => finiteNumber(position) && position >= 0 && position <= 1
                && (index === 0 || position > entry.x[index - 1]))
            || !Array.isArray(entry.y) || entry.y.length !== entry.x.length
            || !entry.y.every((row) => Array.isArray(row) && row.length === entry.target_features.length
                && row.every(finiteNumber))) return invalidModel();
        // Models published before game was recorded contain ACC laps only.
        return { ...entry, game: entry.game ?? 'acc' };
    });
};

const downloadModel = async (): Promise<TopLapReferenceEntry[]> => {
    const { data: preparation } = await apiService.get<ModelPreparation>(
        '/ai-model/active/top_lap_reference/prepare-chunked',
    );
    const chunking = preparation.chunking;
    if (!preparation.success || !chunking?.sessionId
        || !Number.isSafeInteger(chunking.totalChunks) || chunking.totalChunks < 1
        || !Number.isSafeInteger(chunking.totalSize) || chunking.totalSize < 1) {
        throw new Error(preparation.message || 'Unable to download reference lap model');
    }
    const { sessionId, totalChunks, totalSize } = chunking;
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const text: string[] = [];
    let receivedBytes = 0;
    // Bound concurrent requests and decode in order, including split UTF-8 characters.
    for (let start = 0; start < totalChunks; start += 4) {
        const chunks = await Promise.all(Array.from({ length: Math.min(4, totalChunks - start) }, async (_, index) => {
            const chunkIndex = start + index;
            const { data } = await apiService.get<ModelChunk>(
                `/ai-model/active/chunked-data/${encodeURIComponent(sessionId)}/${chunkIndex}`,
            );
            if (!data.success || data.sessionId !== sessionId || data.chunkIndex !== chunkIndex
                || data.totalChunks !== totalChunks || typeof data.data !== 'string') {
                throw new Error(data.message || 'Incomplete reference lap model download');
            }
            return decodeBase64Bytes(data.data);
        }));
        for (const chunk of chunks) {
            receivedBytes += chunk.length;
            text.push(decoder.decode(chunk, { stream: true }));
        }
    }
    if (receivedBytes !== totalSize) throw new Error('Incomplete reference lap model download');
    text.push(decoder.decode());
    return decodeModel(JSON.parse(text.join('')));
};

// Share the complete model across map instances and track/car changes. Failed
// transfers are discarded so Retry starts a fresh backend download session.
let modelRequest: Promise<TopLapReferenceEntry[]> | null = null;
export const loadTopLapReferenceModel = (): Promise<TopLapReferenceEntry[]> => {
    if (!modelRequest) {
        modelRequest = downloadModel().catch((error) => {
            modelRequest = null;
            throw error;
        });
    }
    return modelRequest;
};
