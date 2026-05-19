/**
 * SSE client for /user-ai-model/ai-query/stream (Phase 2.5).
 *
 * We use fetch + ReadableStream rather than the browser's `EventSource`
 * because:
 *   - EventSource is GET-only; we need POST for the question body.
 *   - EventSource doesn't let us send Authorization headers.
 *   - We want a clean AbortController for "stop speaking" cancellation.
 *
 * Event protocol (matches acla_ai_service/app/services/voice/stream_events.py):
 *   token        — { type: "token", text }
 *   audio        — { type: "audio", sentence, wav_b64, voice? }
 *   tool_start   — { type: "tool_start", name, arguments }
 *   tool_end     — { type: "tool_end", name, ok, error? }
 *   done         — { type: "done", answer, side_products?, context?, messages? }
 *   error        — { type: "error", message, error_type }
 */

export interface StreamingChatHandlers {
    onToken?: (text: string) => void;
    onAudio?: (event: { sentence: string; wav_b64: string; voice?: string }) => void;
    onToolStart?: (event: { name: string; arguments: Record<string, unknown> }) => void;
    onToolEnd?: (event: { name: string; ok: boolean; error?: string }) => void;
    onDone?: (event: {
        answer: string;
        side_products?: Record<string, unknown>;
        context?: Record<string, unknown>;
        messages?: unknown[];
    }) => void;
    onError?: (event: { message: string; error_type: string }) => void;
}

export interface StreamingChatRequest {
    question: string;
    context?: Record<string, unknown>;
}

export interface StreamingChatSession {
    /** Promise that resolves when the stream ends naturally or is aborted. */
    done: Promise<void>;
    /** Abort the stream — triggers `onError` if mid-stream, no-op if already done. */
    abort: () => void;
}

/**
 * Open a streaming chat session. Events are delivered via the handlers
 * (no need for the caller to await tokens one at a time).
 *
 * Errors during streaming are reported via `onError` and resolve `done`
 * — they do NOT reject. This makes the call site simpler: a single
 * try/await around session.done suffices.
 */
export function streamChat(
    request: StreamingChatRequest,
    handlers: StreamingChatHandlers,
): StreamingChatSession {
    const controller = new AbortController();

    const baseUrl =
        'http://' +
        (process.env.REACT_APP_BACKEND_SERVER_IP as string) +
        ':' +
        (process.env.REACT_APP_BACKEND_PROXY_PORT as string);

    const url = `${baseUrl}/user-ai-model/ai-query/stream`;

    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
    };
    const token = localStorage.getItem('token');
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const done = (async () => {
        let response: Response;
        try {
            response = await fetch(url, {
                method: 'POST',
                headers,
                body: JSON.stringify(request),
                signal: controller.signal,
            });
        } catch (err) {
            if ((err as DOMException).name === 'AbortError') return;
            handlers.onError?.({
                message: (err as Error).message ?? 'Network error',
                error_type: 'NetworkError',
            });
            return;
        }

        if (!response.ok || !response.body) {
            handlers.onError?.({
                message: `HTTP ${response.status}: ${await safeReadText(response)}`,
                error_type: 'HttpError',
            });
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pendingBuffer = '';

        try {
            while (true) {
                const { value, done: streamDone } = await reader.read();
                if (streamDone) break;

                pendingBuffer += decoder.decode(value, { stream: true });

                // SSE events are separated by a blank line ("\n\n" or "\r\n\r\n").
                let separatorIdx: number;
                while ((separatorIdx = findEventBoundary(pendingBuffer)) >= 0) {
                    const rawEvent = pendingBuffer.slice(0, separatorIdx);
                    pendingBuffer = pendingBuffer.slice(separatorIdx + 2);

                    const dataLines = rawEvent
                        .split(/\r?\n/)
                        .filter((line) => line.startsWith('data:'))
                        .map((line) => line.slice(5).trimStart());

                    if (dataLines.length === 0) continue;
                    const payloadJson = dataLines.join('\n');

                    let payload: { type: string;[key: string]: unknown };
                    try {
                        payload = JSON.parse(payloadJson);
                    } catch (err) {
                        // Malformed event — log and continue; better than killing the stream.
                        console.warn('[streaming-chat] failed to parse SSE event:', payloadJson, err);
                        continue;
                    }

                    dispatchEvent(payload, handlers);
                }
            }
        } catch (err) {
            if ((err as DOMException)?.name === 'AbortError') return;
            handlers.onError?.({
                message: (err as Error).message ?? 'Stream read error',
                error_type: 'StreamReadError',
            });
        } finally {
            try { reader.releaseLock(); } catch { /* ignore */ }
        }
    })();

    return {
        done,
        abort: () => controller.abort(),
    };
}


// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function findEventBoundary(buffer: string): number {
    const a = buffer.indexOf('\n\n');
    const b = buffer.indexOf('\r\n\r\n');
    if (a < 0) return b;
    if (b < 0) return a;
    return Math.min(a, b);
}

async function safeReadText(response: Response): Promise<string> {
    try { return await response.text(); }
    catch { return '(no body)'; }
}

function dispatchEvent(
    payload: { type: string;[key: string]: any },
    handlers: StreamingChatHandlers,
): void {
    switch (payload.type) {
        case 'token':
            handlers.onToken?.(String(payload.text ?? ''));
            break;
        case 'audio':
            handlers.onAudio?.({
                sentence: String(payload.sentence ?? ''),
                wav_b64: String(payload.wav_b64 ?? ''),
                voice: payload.voice as string | undefined,
            });
            break;
        case 'tool_start':
            handlers.onToolStart?.({
                name: String(payload.name ?? ''),
                arguments: (payload.arguments as Record<string, unknown>) ?? {},
            });
            break;
        case 'tool_end':
            handlers.onToolEnd?.({
                name: String(payload.name ?? ''),
                ok: Boolean(payload.ok),
                error: payload.error as string | undefined,
            });
            break;
        case 'done':
            handlers.onDone?.({
                answer: String(payload.answer ?? ''),
                side_products: payload.side_products as Record<string, unknown> | undefined,
                context: payload.context as Record<string, unknown> | undefined,
                messages: payload.messages as unknown[] | undefined,
            });
            break;
        case 'error':
            handlers.onError?.({
                message: String(payload.message ?? 'unknown error'),
                error_type: String(payload.error_type ?? 'Error'),
            });
            break;
        default:
            // Forward-compat: ignore unknown event types rather than breaking.
            console.debug('[streaming-chat] unknown event type:', payload.type, payload);
    }
}
