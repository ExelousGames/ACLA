/**
 * Neural TTS client (Phase 2).
 *
 * Calls the backend's `/user-ai-model/voice-synthesize` endpoint (which
 * forwards to the AI service's Kokoro engine) and plays the returned
 * WAV via an `HTMLAudioElement`. Replaces the browser-native
 * `window.speechSynthesis` for AI chat responses.
 *
 * Designed to coexist with the existing speechSynthesis path:
 *   - On success: returns a NeuralTtsPlayback whose `.ended` resolves on
 *     end and which exposes `.stop()` for interruption.
 *   - On failure: throws. Callers should fall back to speechSynthesis.
 */

import apiService from 'services/api.service';

export interface NeuralTtsOptions {
    /** Voice ID, e.g. "af_bella". Server picks a default if omitted. */
    voice?: string;
    /** 0.5 (slow) .. 2.0 (fast). Defaults to 1.0. */
    speed?: number;
    /** Language code, e.g. "en-us". */
    language?: string;
    /** Volume 0.0 .. 1.0 applied to the HTMLAudioElement. */
    volume?: number;
}

export interface NeuralTtsPlayback {
    /** Resolves when playback finishes naturally or is stopped. */
    ended: Promise<void>;
    /** Stop playback immediately. Resolves `ended`. */
    stop: () => void;
    /** Underlying audio element — exposed for advanced consumers. */
    audio: HTMLAudioElement;
}

/**
 * Synthesize `text` via Kokoro on the backend and play it through an
 * `HTMLAudioElement`. Returns a handle for stopping playback and a
 * promise that resolves when playback ends.
 *
 * Throws on network errors or non-2xx responses — caller should catch
 * and fall back to `window.speechSynthesis`.
 */
export async function speakWithNeuralTts(
    text: string,
    options: NeuralTtsOptions = {},
): Promise<NeuralTtsPlayback> {
    const cleanText = text.trim();
    if (!cleanText) {
        throw new Error('speakWithNeuralTts: text is empty');
    }

    // Hits NestJS → AI service → Kokoro. Returns audio/wav bytes.
    const wavBuffer = await apiService.postBinary(
        'user-ai-model/voice-synthesize',
        {
            text: cleanText,
            voice: options.voice,
            speed: options.speed ?? 1.0,
            language: options.language ?? 'en-us',
        },
        // Cold-start of the Kokoro model on the server can take 5-10s on
        // first request. After that, short phrases finish in <500ms.
        { timeoutMs: 60000 },
    );

    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);

    const audio = new Audio(url);
    audio.volume = options.volume ?? 0.9;

    // Build the `ended` promise eagerly so callers can `await` it.
    let resolveEnded!: () => void;
    let rejectEnded!: (err: unknown) => void;
    const ended = new Promise<void>((resolve, reject) => {
        resolveEnded = resolve;
        rejectEnded = reject;
    });

    const cleanup = () => {
        URL.revokeObjectURL(url);
    };

    audio.addEventListener('ended', () => {
        cleanup();
        resolveEnded();
    });
    audio.addEventListener('error', (event) => {
        cleanup();
        rejectEnded(new Error(`Audio playback failed: ${(event as ErrorEvent).message ?? 'unknown'}`));
    });

    const stop = () => {
        try {
            audio.pause();
            audio.currentTime = 0;
        } catch {
            // Ignore — caller just wants playback to stop.
        }
        cleanup();
        resolveEnded();
    };

    try {
        await audio.play();
    } catch (err) {
        cleanup();
        throw err;
    }

    return { ended, stop, audio };
}
