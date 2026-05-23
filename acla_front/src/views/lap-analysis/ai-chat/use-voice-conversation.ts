/**
 * Voice conversation hook — racing engineer (Phase 1 of the racing-engineer
 * rebuild). Opens a WebSocket to the AI service's `/voice/stream` and runs
 * BOTH channels over the same connection:
 *
 * - **Binary frames** — raw PCM16 mic in / Kokoro TTS out. Same protocol
 *   as before.
 * - **Text frames** — JSON tool-relay messages. The backend LLM emits
 *   `{type:"tool_call",id,name,arguments}` frames; this hook dispatches
 *   them through a caller-supplied handler registry and replies with
 *   `{type:"tool_result",...}` or `{type:"tool_error",...}`. Long-running
 *   handlers (e.g. per-turn coaching) can also push
 *   `{type:"observation",data}` frames any time via `ctx.sendObservation`.
 *
 * See `acla_ai_service/app/voice/tool_relay.py` for the backend half of
 * this protocol.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import apiService from 'services/api.service';

export type VoiceConversationState =
    | 'idle'           // not connected
    | 'connecting'     // WS handshake in progress
    | 'listening'      // mic open, sending to server
    | 'speaking'       // server is sending us audio
    | 'error';

/** Context passed to every frontend tool handler. */
export interface ToolHandlerContext {
    /** Push an `observation` frame on the open WS. Safe to call from a
     *  background monitoring agent at any time. Becomes a synthetic user
     *  turn in the LLM's context on the backend. */
    sendObservation: (data: Record<string, unknown>) => void;
}

/** One frontend tool handler. Return value becomes the `tool_result`. Throw
 *  to emit a `tool_error`. */
export type FrontendToolHandler = (
    args: Record<string, unknown>,
    ctx: ToolHandlerContext,
) => Promise<unknown> | unknown;

export interface VoiceConversationOptions {
    /** Driving session id — required for backend tools that look up
     *  recent telemetry / lap data by session. */
    sessionId?: string;
    /** User id — required for backend tools that key off the logged-in
     *  user (e.g. saved preferences, history). */
    userId?: string;
    /** Map of frontend tool name → handler. The LLM picks which tools to
     *  call from its system prompt; the backend routes the call to this
     *  hook over the WS via a `tool_call` text frame; we dispatch by
     *  name. Missing handler → automatic `tool_error`. */
    toolHandlers?: Record<string, FrontendToolHandler>;
}

export interface VoiceConversation {
    state: VoiceConversationState;
    error: string | null;
    /** Start the session — opens mic + WS. Throws if user denies mic. */
    start: () => Promise<void>;
    /** Stop the session — closes mic, WS, audio playback. Idempotent. */
    stop: () => void;
}

export function useVoiceConversation(
    options: VoiceConversationOptions = {},
): VoiceConversation {
    const [state, setState] = useState<VoiceConversationState>('idle');
    const [error, setError] = useState<string | null>(null);

    // Hold refs to all the resources we need to tear down on stop().
    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const playbackContextRef = useRef<AudioContext | null>(null);
    const playbackQueueTimeRef = useRef<number>(0);

    /**
     * Open the backend voice WS through apiService — same baseURL + JWT
     * source as every REST call. `user_id` is derived server-side from
     * the JWT claim and isn't sent from here.
     */
    const openWs = useCallback((): WebSocket => {
        return apiService.openWebSocket('/voice/stream', {
            session_id: options.sessionId,
        });
    }, [options.sessionId]);

    // Always-fresh handler registry — updated as options.toolHandlers changes
    // without forcing the WS to reopen.
    const toolHandlersRef = useRef<Record<string, FrontendToolHandler>>(
        options.toolHandlers || {},
    );
    useEffect(() => {
        toolHandlersRef.current = options.toolHandlers || {};
    }, [options.toolHandlers]);

    const stop = useCallback(() => {
        // Tear down in reverse order of construction. All steps are idempotent.
        try { workletNodeRef.current?.disconnect(); } catch { /* ignore */ }
        workletNodeRef.current = null;

        try {
            micStreamRef.current?.getTracks().forEach((t) => t.stop());
        } catch { /* ignore */ }
        micStreamRef.current = null;

        try { audioContextRef.current?.close(); } catch { /* ignore */ }
        audioContextRef.current = null;

        try { playbackContextRef.current?.close(); } catch { /* ignore */ }
        playbackContextRef.current = null;
        playbackQueueTimeRef.current = 0;

        if (wsRef.current) {
            try {
                if (wsRef.current.readyState <= WebSocket.OPEN) {
                    wsRef.current.close(1000, 'client stop');
                }
            } catch { /* ignore */ }
        }
        wsRef.current = null;

        setState('idle');
    }, []);

    const start = useCallback(async () => {
        if (state !== 'idle' && state !== 'error') return;

        setError(null);
        setState('connecting');

        try {
            // --- 1. Request mic permission ---
            const micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
                video: false,
            });
            micStreamRef.current = micStream;

            // --- 2. Set up capture AudioContext + worklet ---
            // Capture at 16kHz mono to match the server's expected input rate
            // (see audio_in_sample_rate in pipecat_pipeline.py). Whisper STT
            // wants 16kHz natively, so doing the resample on the client saves
            // network bandwidth and avoids server-side resampling.
            //
            // Browsers MAY refuse non-default sample rates on some OSes — if
            // 16kHz isn't supported, fall back to the system default. The
            // server resamples in that case (slightly slower path).
            let captureContext: AudioContext;
            try {
                captureContext = new AudioContext({ sampleRate: 16000 });
            } catch {
                console.warn('[voice] 16kHz capture context rejected, using system default');
                captureContext = new AudioContext();
            }
            audioContextRef.current = captureContext;
            await captureContext.audioWorklet.addModule('/pcm-capture-worklet.js');

            const source = captureContext.createMediaStreamSource(micStream);
            const workletNode = new AudioWorkletNode(captureContext, 'pcm-capture');
            workletNodeRef.current = workletNode;
            source.connect(workletNode);
            // Do NOT connect workletNode to captureContext.destination — that
            // would echo the mic back to the speakers.

            // --- 3. Open WebSocket ---
            const ws = openWs();
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            // Hook up the worklet → WS pipe. We send raw PCM16 ArrayBuffers.
            // The Pipecat server's ProtobufFrameSerializer expects framed
            // protobuf, so this RAW PCM mode will only work if the server is
            // configured for it — see Phase 3 known limitations.
            workletNode.port.onmessage = (event) => {
                if (ws.readyState !== WebSocket.OPEN) return;
                try {
                    ws.send(event.data as ArrayBuffer);
                } catch (err) {
                    console.warn('[voice] send failed:', err);
                }
            };

            // --- 4. Set up playback AudioContext ---
            const playbackContext = new AudioContext({ sampleRate: 24000 });
            playbackContextRef.current = playbackContext;
            playbackQueueTimeRef.current = playbackContext.currentTime;

            ws.onopen = () => {
                setState('listening');
            };

            // ── Tool-relay text channel ────────────────────────────────────
            // Helpers that wrap the WS for tool handlers. Defined here so
            // they capture the live `ws` instance; not exposed externally.
            const sendText = (payload: object) => {
                if (ws.readyState !== WebSocket.OPEN) return;
                try { ws.send(JSON.stringify(payload)); }
                catch (err) { console.warn('[voice/tool-relay] send failed:', err); }
            };
            const toolCtx: ToolHandlerContext = {
                sendObservation: (data) => sendText({ type: 'observation', data }),
            };

            const handleToolCall = async (msg: {
                id?: string; name?: string; arguments?: Record<string, unknown>;
            }) => {
                const id = msg.id;
                const name = msg.name;
                if (!id || !name) {
                    console.warn('[voice/tool-relay] bad tool_call frame:', msg);
                    return;
                }
                const handler = toolHandlersRef.current[name];
                if (!handler) {
                    sendText({ type: 'tool_error', id, error: `no handler for '${name}'` });
                    return;
                }
                try {
                    const result = await handler(msg.arguments || {}, toolCtx);
                    sendText({
                        type: 'tool_result',
                        id,
                        result: result && typeof result === 'object' ? result : { value: result },
                    });
                } catch (err) {
                    const message = (err as Error)?.message || String(err);
                    sendText({ type: 'tool_error', id, error: message });
                }
            };

            ws.onmessage = (event) => {
                // Text frame → tool-relay channel. Binary frame → PCM audio.
                if (typeof event.data === 'string') {
                    let parsed: any;
                    try { parsed = JSON.parse(event.data); }
                    catch { console.warn('[voice/tool-relay] non-JSON text frame'); return; }
                    if (parsed?.type === 'tool_call') {
                        void handleToolCall(parsed);
                    } else {
                        console.warn('[voice/tool-relay] unknown text frame:', parsed?.type);
                    }
                    return;
                }
                if (!(event.data instanceof ArrayBuffer)) return;
                // Server sent raw PCM16 mono at the kokoro sample rate.
                queuePlayback(event.data, playbackContext);
                // Always set 'speaking' — setState is idempotent and the
                // closure-captured `state` value is stale here.
                setState((prev) => (prev === 'speaking' ? prev : 'speaking'));
            };

            ws.onerror = (event) => {
                console.error('[voice] WS error:', event);
                setError('Voice connection error');
                setState('error');
            };

            ws.onclose = (event) => {
                // closure-captured `state` is stale; use the setter form.
                setState((prev) => {
                    if (prev === 'idle') return prev;
                    if (event.code !== 1000) {
                        setError(`Voice connection closed (${event.code}): ${event.reason || 'unknown'}`);
                        return 'error';
                    }
                    return 'idle';
                });
            };
        } catch (err) {
            console.error('[voice] start failed:', err);
            setError((err as Error).message || 'Failed to start voice session');
            setState('error');
            stop();
        }
    }, [state, openWs, stop]);

    /**
     * Schedule a PCM16 chunk for gapless playback on the playback AudioContext.
     */
    const queuePlayback = (pcm16Buffer: ArrayBuffer, context: AudioContext) => {
        const int16 = new Int16Array(pcm16Buffer);
        if (int16.length === 0) return;

        // Convert to Float32 in [-1, 1].
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] < 0 ? int16[i] / 0x8000 : int16[i] / 0x7fff;
        }

        const audioBuffer = context.createBuffer(1, float32.length, context.sampleRate);
        audioBuffer.copyToChannel(float32, 0);

        const source = context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(context.destination);

        const now = context.currentTime;
        const startAt = Math.max(now, playbackQueueTimeRef.current);
        source.start(startAt);
        playbackQueueTimeRef.current = startAt + audioBuffer.duration;
    };

    // Auto-cleanup on unmount.
    useEffect(() => {
        return () => stop();
    }, [stop]);

    return { state, error, start, stop };
}
