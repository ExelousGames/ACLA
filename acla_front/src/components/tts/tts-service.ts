import apiService from 'services/api.service';

export interface TtsRequest {
    text: string;
    voice?: string;
    speed?: number;
    language?: string;
}

/** Prepared audio retained by its main-window component owner. */
export interface TtsPack {
    text: string;
    audioDataUrl: string;
    durationMs: number;
    /** First audible frame through the end of the last; null means an entirely silent clip. */
    speechBounds?: { startMs: number; endMs: number } | null;
}

export const isTtsPack = (value: unknown): value is TtsPack => {
    if (!value || typeof value !== 'object') return false;
    const pack = value as TtsPack;
    return typeof pack.text === 'string' && Boolean(pack.text.trim())
        && typeof pack.audioDataUrl === 'string'
        && /^data:audio\/wav;base64,[A-Za-z0-9+/]+={0,2}$/.test(pack.audioDataUrl)
        && Number.isFinite(pack.durationMs) && pack.durationMs > 0
        && (pack.speechBounds == null || (Number.isFinite(pack.speechBounds.startMs)
            && Number.isFinite(pack.speechBounds.endMs) && pack.speechBounds.startMs >= 0
            && pack.speechBounds.endMs > pack.speechBounds.startMs && pack.speechBounds.endMs <= pack.durationMs));
};

const prepareWav = (bytes: Uint8Array, trimSilence: boolean): Pick<TtsPack, 'durationMs' | 'speechBounds'> & { bytes: Uint8Array } => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tag = (offset: number) => String.fromCharCode(...Array.from(bytes.slice(offset, offset + 4)));
    if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
        throw new Error('The speech service did not return WAV audio.');
    }
    let byteRate = 0;
    let dataSize = 0;
    let format = 0, channels = 0, sampleRate = 0, blockAlign = 0, bitsPerSample = 0, formatOffset = 0;
    const chunks: Array<{ offset: number; size: number }> = [];
    for (let offset = 12; offset + 8 <= bytes.length;) {
        const size = view.getUint32(offset + 4, true);
        if (offset + 8 + size > bytes.length) throw new Error('The speech audio is incomplete.');
        if (tag(offset) === 'fmt ' && size >= 16) {
            formatOffset = offset + 8;
            format = view.getUint16(offset + 8, true);
            channels = view.getUint16(offset + 10, true);
            sampleRate = view.getUint32(offset + 12, true);
            byteRate = view.getUint32(offset + 16, true);
            blockAlign = view.getUint16(offset + 20, true);
            bitsPerSample = view.getUint16(offset + 22, true);
        }
        if (tag(offset) === 'data') {
            dataSize += size;
            chunks.push({ offset: offset + 8, size });
        }
        offset += 8 + size + (size % 2);
    }
    if (!byteRate || !dataSize) throw new Error('The speech service returned empty audio.');
    const durationMs = dataSize * 1000 / byteRate;
    // The speech service emits PCM16 WAV. Other encodings retain natural-completion playback.
    if (format !== 1 || bitsPerSample !== 16) return { bytes, durationMs };
    if (!channels || !sampleRate || blockAlign !== channels * 2 || byteRate !== sampleRate * blockAlign
        || chunks.some(({ size }) => size % blockAlign !== 0)) throw new Error('The speech audio has invalid PCM frames.');
    const audible = (offset: number) => {
        for (let channel = 0; channel < channels; channel++) {
            if (Math.abs(view.getInt16(offset + channel * 2, true) / 32768) > 0.01) return true;
        }
        return false;
    };
    // Scan from each end, preserving every pause between the first and last spoken sounds.
    let firstFrame = 0;
    let startMs: number | undefined;
    for (const chunk of chunks) {
        for (let offset = chunk.offset; offset < chunk.offset + chunk.size; offset += blockAlign, firstFrame++) {
            if (audible(offset)) { startMs = firstFrame * 1000 / sampleRate; break; }
        }
        if (startMs !== undefined) break;
    }
    if (startMs === undefined) return { bytes, durationMs, speechBounds: null };
    let lastFrame = dataSize / blockAlign - 1;
    for (let index = chunks.length - 1; index >= 0; index--) {
        const chunk = chunks[index];
        for (let offset = chunk.offset + chunk.size - blockAlign; offset >= chunk.offset; offset -= blockAlign, lastFrame--) {
            if (!audible(offset)) continue;
            if (!trimSilence) return { bytes, durationMs, speechBounds: { startMs, endMs: (lastFrame + 1) * 1000 / sampleRate } };
            // Keep complete PCM frames, including all pauses between the audible endpoints.
            const trimmed = new Uint8Array(44 + (lastFrame - firstFrame + 1) * blockAlign);
            const header = new DataView(trimmed.buffer);
            const writeTag = (at: number, value: string) => trimmed.set(Array.from(value, (char) => char.charCodeAt(0)), at);
            writeTag(0, 'RIFF');
            header.setUint32(4, trimmed.length - 8, true);
            writeTag(8, 'WAVE');
            writeTag(12, 'fmt ');
            header.setUint32(16, 16, true);
            trimmed.set(bytes.subarray(formatOffset, formatOffset + 16), 20);
            writeTag(36, 'data');
            header.setUint32(40, trimmed.length - 44, true);
            let sourceByte = 0, targetByte = 44;
            for (const data of chunks) {
                const start = Math.max(0, firstFrame * blockAlign - sourceByte);
                const end = Math.min(data.size, (lastFrame + 1) * blockAlign - sourceByte);
                if (end > start) {
                    trimmed.set(bytes.subarray(data.offset + start, data.offset + end), targetByte);
                    targetByte += end - start;
                }
                sourceByte += data.size;
            }
            const audibleDurationMs = (lastFrame - firstFrame + 1) * 1000 / sampleRate;
            return { bytes: trimmed, durationMs: audibleDurationMs, speechBounds: { startMs: 0, endMs: audibleDurationMs } };
        }
    }
    return { bytes, durationMs, speechBounds: null };
};

export const createTtsPack = (text: string, audio: ArrayBuffer, trimSilence = false): TtsPack => {
    const { bytes, ...timing } = prepareWav(new Uint8Array(audio), trimSilence);
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 32768) {
        binary += String.fromCharCode(...Array.from(bytes.subarray(offset, offset + 32768)));
    }
    return { text, audioDataUrl: `data:audio/wav;base64,${btoa(binary)}`, ...timing };
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
    return createTtsPack(text, response.data);
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
