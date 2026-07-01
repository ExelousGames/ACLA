/**
 * Sequential gapless audio playback queue (Phase 2.5).
 *
 * Each Kokoro-synthesized sentence arrives as a base64 WAV blob on the
 * SSE stream. We don't have a single contiguous audio source we can pipe
 * into a MediaSource (different WAVs have their own headers), so the
 * simplest reliable approach is: play one HTMLAudioElement, when it ends,
 * play the next.
 *
 * The gap is typically <50ms on a warm browser — imperceptible for spoken
 * sentences. If users complain about choppiness later, we can swap to
 * AudioContext + decodeAudioData for sample-accurate gapless playback.
 */

export interface AudioStreamQueueOptions {
    volume?: number; // 0..1 — default 0.9
    onStart?: () => void; // fires when the first chunk starts playing
    onIdle?: () => void; // fires when the queue drains (no more pending audio)
    onError?: (err: unknown) => void;
}

export interface AudioStreamQueue {
    /** Decode a base64 WAV and append it to the playback queue. */
    enqueueBase64Wav: (b64: string) => void;
    /** Stop playback, clear pending, free resources. Safe to call multiple times. */
    stop: () => void;
    /** True if anything is currently playing or pending. */
    isActive: () => boolean;
}

export function createAudioStreamQueue(
    options: AudioStreamQueueOptions = {},
): AudioStreamQueue {
    const volume = options.volume ?? 0.9;

    // Pending blob URLs waiting to be played, in order.
    const pendingUrls: string[] = [];
    let currentAudio: HTMLAudioElement | null = null;
    let stopped = false;
    let hasStarted = false;

    const playNext = () => {
        if (stopped) return;
        if (pendingUrls.length === 0) {
            currentAudio = null;
            options.onIdle?.();
            return;
        }

        const url = pendingUrls.shift()!;
        const audio = new Audio(url);
        audio.volume = volume;
        currentAudio = audio;

        const cleanup = () => {
            URL.revokeObjectURL(url);
        };

        audio.addEventListener('ended', () => {
            cleanup();
            playNext();
        });
        audio.addEventListener('error', (event) => {
            cleanup();
            options.onError?.(event);
            // Don't halt the queue on one bad chunk — try the next.
            playNext();
        });

        audio.play().then(
            () => {
                if (!hasStarted) {
                    hasStarted = true;
                    options.onStart?.();
                }
            },
            (err) => {
                cleanup();
                options.onError?.(err);
                playNext();
            },
        );
    };

    return {
        enqueueBase64Wav: (b64: string) => {
            if (stopped) return;
            try {
                const bytes = base64ToUint8Array(b64);
                const blob = new Blob([bytes], { type: 'audio/wav' });
                const url = URL.createObjectURL(blob);
                pendingUrls.push(url);
                if (!currentAudio) {
                    playNext();
                }
            } catch (err) {
                options.onError?.(err);
            }
        },
        stop: () => {
            if (stopped) return;
            stopped = true;
            if (currentAudio) {
                try {
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                } catch { /* ignore */ }
                currentAudio = null;
            }
            // Revoke any still-queued blob URLs.
            for (const url of pendingUrls) {
                try { URL.revokeObjectURL(url); } catch { /* ignore */ }
            }
            pendingUrls.length = 0;
        },
        isActive: () => currentAudio !== null || pendingUrls.length > 0,
    };
}


/**
 * Convert a base64 string to a Uint8Array WITHOUT going through atob's
 * binary-string-to-charcode step, which is fine for WAV bytes but a bit
 * faster + clearer this way.
 */
function base64ToUint8Array(b64: string): Uint8Array {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        out[i] = binary.charCodeAt(i);
    }
    return out;
}
