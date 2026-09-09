import apiService from 'services/api.service';

export interface TtsRequest {
    text: string;
    voice?: string;
    speed?: number;
    language?: string;
}

/** JSON-safe audio that can be retained with a component or sent to an overlay. */
export interface TtsPack {
    text: string;
    audioDataUrl: string;
    durationMs: number;
}

export const isTtsPack = (value: unknown): value is TtsPack => {
    if (!value || typeof value !== 'object') return false;
    const pack = value as TtsPack;
    return typeof pack.text === 'string' && Boolean(pack.text.trim())
        && typeof pack.audioDataUrl === 'string'
        && /^data:audio\/wav;base64,[A-Za-z0-9+/]+={0,2}$/.test(pack.audioDataUrl)
        && Number.isFinite(pack.durationMs) && pack.durationMs > 0;
};

const wavDurationMs = (bytes: Uint8Array): number => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tag = (offset: number) => String.fromCharCode(...Array.from(bytes.slice(offset, offset + 4)));
    if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
        throw new Error('The speech service did not return WAV audio.');
    }
    let byteRate = 0;
    let dataSize = 0;
    for (let offset = 12; offset + 8 <= bytes.length;) {
        const size = view.getUint32(offset + 4, true);
        if (offset + 8 + size > bytes.length) throw new Error('The speech audio is incomplete.');
        if (tag(offset) === 'fmt ' && size >= 16) byteRate = view.getUint32(offset + 16, true);
        if (tag(offset) === 'data') dataSize += size;
        offset += 8 + size + (size % 2);
    }
    if (!byteRate || !dataSize) throw new Error('The speech service returned empty audio.');
    return (dataSize / byteRate) * 1000;
};

export const synthesizeTts = async (request: TtsRequest, signal?: AbortSignal): Promise<TtsPack> => {
    const text = request.text.trim();
    if (!text || text.length > 4000) throw new Error('Speech text must contain 1 to 4000 characters.');
    if (signal?.aborted) throw new DOMException('Speech generation was cancelled.', 'AbortError');
    const response = await apiService.post<ArrayBuffer>('/user-ai-model/voice-synthesize', {
        ...request,
        text,
        speed: request.speed ?? 1,
        language: request.language ?? 'en-us',
    }, { responseType: 'arraybuffer', timeout: 120_000, signal });
    if (signal?.aborted) throw new DOMException('Speech generation was cancelled.', 'AbortError');
    const bytes = new Uint8Array(response.data);
    const durationMs = wavDurationMs(bytes);
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 32768) {
        binary += String.fromCharCode(...Array.from(bytes.subarray(offset, offset + 32768)));
    }
    return { text, audioDataUrl: `data:audio/wav;base64,${btoa(binary)}`, durationMs };
};

/** Prepare in order without overloading the backend speech model. */
export const synthesizeTtsPack = async (
    requests: readonly TtsRequest[],
    signal?: AbortSignal,
): Promise<TtsPack[]> => {
    const packs: TtsPack[] = [];
    for (const request of requests) packs.push(await synthesizeTts(request, signal));
    return packs;
};
