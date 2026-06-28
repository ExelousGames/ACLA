/**
 * Voice conversation hook — racing engineer (Phase 1 of the racing-engineer
 * rebuild). Opens a WebSocket to the backend's `/voice/stream` and runs
 * BOTH channels over the same connection:
 *
 * - **Binary frames** — raw PCM16 mic in / assistant audio out. Same protocol
 *   as before.
 * - **Text frames** — JSON tool-relay messages. The backend emits
 *   `{type:"tool_call",id,name,arguments}` frames; this hook dispatches
 *   them through a caller-supplied handler registry and replies with
 *   `{type:"tool_result",...}` or `{type:"tool_error",...}`. Long-running
 *   handlers (e.g. per-turn coaching) can also push
 *   `{type:"observation",data:{text}}` frames any time via `ctx.sendObservation`.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import apiService from 'services/api.service';
import { buildFormattedObservationFrame } from './voice-observation-formatter';

export type VoiceConversationState =
    | 'idle'           // not connected
    | 'connecting'     // WS handshake in progress
    | 'listening'      // mic open, sending to server
    | 'speaking'       // server is sending us audio
    | 'error';

/** Context passed to every frontend tool handler. */
export interface ToolHandlerContext {
    /** Push an `observation` frame on the open WS. Safe to call from a
     *  background monitoring agent at any time. The frontend formats it
     *  before the backend injects it into the LLM context. */
    sendObservation: (data: Record<string, unknown>) => void;
}

/** One frontend tool handler. Return value becomes the `tool_result`. Throw
 *  to emit a `tool_error`. */
export type FrontendToolHandler = (
    args: Record<string, unknown>,
    ctx: ToolHandlerContext,
) => Promise<unknown> | unknown;

/** Capability shape for one frontend-implemented tool. Sent to the AI
 *  service on WS open so it can merge executable frontend parameter shapes
 *  with LLM-facing instructions from its external knowledge base. */
export interface FrontendToolSchema {
    name: string;
    description?: string;
    /** JSON-Schema-style `properties` object. */
    properties: Record<string, unknown>;
    required: string[];
}

export type AiSessionContext = Record<string, unknown>;

/** One event surfaced to the chat UI off the voice WS. The hook fires
 *  these via `onEvent` so the caller can append them to a message list. */
export type VoiceEvent =
    | { kind: 'user_transcript'; text: string; source?: 'voice' | 'typed' }
    | { kind: 'assistant_transcript'; text: string; emotion?: string }
    | { kind: 'observation'; data: Record<string, unknown> }
    | {
        kind: 'tool_event';
        name: string;
        title: string;
        status: 'started' | 'completed';
        arguments?: Record<string, unknown>;
        ok?: boolean;
        error?: string | null;
    };

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
    /** Capabilities for frontend-implemented tools. Sent to the AI service
     *  as the first text frame on WS open; the service supplies tool-use
     *  instructions from its external knowledge base. */
    frontendTools?: FrontendToolSchema[];
    /** QueryScope JSON Schema shape. Backend tools whose parameters reference
     *  a query scope (e.g. analyze_telemetry) consume this executable data
     *  shape from the WS handshake instead of re-declaring it in Python. */
    querySchemaScope?: object;
    /** Compact frontend view/session state injected into the backend system
     *  context before the LLM chooses tools. */
    sessionContext?: AiSessionContext;
    /** Fires for each transcript / tool event the backend sends. The
     *  caller is responsible for appending to its own message list. */
    onEvent?: (event: VoiceEvent) => void;
}

export interface VoiceConversation {
    state: VoiceConversationState;
    error: string | null;
    /** Current mic input level in [0, 1] — peak amplitude over the last
     *  ~66ms window. 0 while not capturing. Updates ~15Hz. Use this to
     *  render a volume meter so the user can confirm the mic is hot. */
    micLevel: number;
    micDisabled: boolean;
    /** Start the session — opens mic + WS. Throws if user denies mic. */
    start: () => Promise<void>;
    /** Stop the session — closes mic, WS, audio playback. Idempotent. */
    stop: () => void;
    setMicDisabled: (disabled: boolean) => void;
    /** Send a typed chat message over the WS. Returns false if no WS is
     *  open. The backend treats it as a synthetic user turn and runs
     *  the LLM (same path as a spoken turn). */
    sendUserText: (text: string) => boolean;
    /** Push a background observation into the open voice session. Returns
     *  false when the voice WebSocket is not ready. */
    sendObservation: (data: Record<string, unknown>) => boolean;
}

export function useVoiceConversation(
    options: VoiceConversationOptions = {},
): VoiceConversation {
    const [state, setState] = useState<VoiceConversationState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [micLevel, setMicLevel] = useState<number>(0);
    const [micDisabled, setMicDisabledState] = useState<boolean>(false);

    // Hold refs to all the resources we need to tear down on stop().
    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const playbackContextRef = useRef<AudioContext | null>(null);
    const playbackQueueTimeRef = useRef<number>(0);
    const playbackSerialRef = useRef<number>(0);
    const playbackIdleTimeoutRef = useRef<number | null>(null);
    const micDisabledRef = useRef(false);

    /**
     * Open the backend voice WS through apiService — same baseURL + JWT
     * source as every REST call. `user_id` is derived server-side from
     * the JWT claim and isn't sent from here.
     */
    const openWs = useCallback((): WebSocket => {
        const sessionMode = typeof options.sessionContext?.session_mode === 'string'
            ? options.sessionContext.session_mode
            : undefined;
        return apiService.openWebSocket('/voice/stream', {
            session_id: options.sessionId,
            session_mode: sessionMode,
        });
    }, [options.sessionContext, options.sessionId]);

    // Always-fresh handler registry — updated as options.toolHandlers changes
    // without forcing the WS to reopen.
    const toolHandlersRef = useRef<Record<string, FrontendToolHandler>>(
        options.toolHandlers || {},
    );
    useEffect(() => {
        toolHandlersRef.current = options.toolHandlers || {};
    }, [options.toolHandlers]);

    // Same pattern for onEvent — keeps closures fresh without re-opening WS.
    const onEventRef = useRef<((event: VoiceEvent) => void) | undefined>(
        options.onEvent,
    );
    useEffect(() => {
        onEventRef.current = options.onEvent;
    }, [options.onEvent]);

    const sessionContextRef = useRef<AiSessionContext | null>(
        options.sessionContext ?? null,
    );
    useEffect(() => {
        sessionContextRef.current = options.sessionContext ?? null;
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        try {
            ws.send(JSON.stringify({
                type: 'session_context',
                session_context: sessionContextRef.current,
            }));
        } catch (err) {
            console.warn('[voice] session_context update failed:', err);
        }
    }, [options.sessionContext]);

    const setMicDisabled = useCallback((disabled: boolean) => {
        micDisabledRef.current = disabled;
        setMicDisabledState(disabled);
        setMicLevel(0);
        try {
            micStreamRef.current?.getAudioTracks().forEach((track) => {
                track.enabled = !disabled;
            });
        } catch { /* ignore */ }
    }, []);

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
        playbackSerialRef.current += 1;
        if (playbackIdleTimeoutRef.current !== null) {
            window.clearTimeout(playbackIdleTimeoutRef.current);
            playbackIdleTimeoutRef.current = null;
        }

        if (wsRef.current) {
            try {
                if (wsRef.current.readyState <= WebSocket.OPEN) {
                    wsRef.current.close(1000, 'client stop');
                }
            } catch { /* ignore */ }
        }
        wsRef.current = null;

        setMicLevel(0);
        setState('idle');
    }, []);

    const start = useCallback(async () => {
        if (state !== 'idle' && state !== 'error') {
            return;
        }

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
            micStream.getAudioTracks().forEach((track) => {
                track.enabled = !micDisabledRef.current;
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

            try {
                await captureContext.audioWorklet.addModule('/pcm-capture-worklet.js');
            } catch (err) {
                console.error('[voice] failed to load pcm-capture-worklet.js — check that /pcm-capture-worklet.js is reachable:', err);
                throw err;
            }

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

            // Hook up the worklet → WS pipe. The worklet posts two kinds of
            // messages: { type:'pcm', buffer } (forwarded over the WS) and
            // { type:'level', rms, peak } (used to drive the mic meter so
            // the user can see whether their voice is registering).
            workletNode.port.onmessage = (event) => {
                const data = event.data;
                if (data && data.type === 'level') {
                    if (micDisabledRef.current) {
                        setMicLevel(0);
                        return;
                    }
                    // Prefer peak — it's what the user perceives as "am I
                    // talking right now". RMS is averaged and looks sleepy.
                    const lvl = typeof data.peak === 'number' ? data.peak : 0;
                    setMicLevel(lvl > 1 ? 1 : lvl < 0 ? 0 : lvl);
                    return;
                }
                if (!data || data.type !== 'pcm' || !data.buffer) return;
                if (micDisabledRef.current) return;
                if (ws.readyState !== WebSocket.OPEN) return;
                try {
                    ws.send(data.buffer as ArrayBuffer);
                } catch (err) {
                    console.warn('[voice] send failed:', err);
                }
            };

            // --- 4. Set up playback AudioContext ---
            const playbackContext = new AudioContext({ sampleRate: 24000 });
            playbackContextRef.current = playbackContext;
            playbackQueueTimeRef.current = playbackContext.currentTime;

            ws.onopen = () => {
                // First text frame on every voice session: hand the AI
                // service the frontend-implemented tool capability shapes.
                // The backend blocks the pipeline build until this arrives,
                // then enriches them with external knowledge-base tool
                // instructions before exposing them to the LLM.
                try {
                    ws.send(JSON.stringify({
                        type: 'frontend_info',
                        session_context: sessionContextRef.current,
                        tools: options.frontendTools || [],
                        query_scope_schema: options.querySchemaScope ?? null,
                    }));
                } catch (err) {
                    console.warn('[voice] frontend_info send failed:', err);
                }
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
                sendObservation: (data) => {
                    onEventRef.current?.({ kind: 'observation', data });
                    sendText(buildFormattedObservationFrame(data));
                },
            };

            const handleToolCall = async (msg: {
                id?: string; name?: string; arguments?: Record<string, unknown>;
            }) => {
                const id = msg.id;
                const name = msg.name;
                if (!id || !name) {
                    console.warn('[ai-tool] bad tool_call frame:', msg);
                    return;
                }
                const handler = toolHandlersRef.current[name];
                console.log('[ai-tool] ◀ tool_call received', {
                    id,
                    name,
                    arguments: msg.arguments,
                    handlerRegistered: !!handler,
                });
                if (!handler) {
                    console.warn(
                        '[ai-tool] no handler for', name,
                        '— available handlers:', Object.keys(toolHandlersRef.current),
                    );
                    sendText({ type: 'tool_error', id, error: `no handler for '${name}'` });
                    return;
                }
                try {
                    const result = await handler(msg.arguments || {}, toolCtx);
                    console.log('[ai-tool] ▶ tool_result', { id, name, result });
                    sendText({
                        type: 'tool_result',
                        id,
                        result: result && typeof result === 'object' ? result : { value: result },
                    });
                } catch (err) {
                    const message = (err as Error)?.message || String(err);
                    console.error('[ai-tool] ▶ tool_error', { id, name, error: message, err });
                    sendText({ type: 'tool_error', id, error: message });
                }
            };

            ws.onmessage = (event) => {
                // Text frame → tool-relay channel. Binary frame → PCM audio.
                if (typeof event.data === 'string') {
                    let parsed: any;
                    try { parsed = JSON.parse(event.data); }
                    catch { console.warn('[voice/tool-relay] non-JSON text frame:', event.data); return; }
                    if (parsed?.type === 'tool_call') {
                        void handleToolCall(parsed);
                    } else if (parsed?.type === 'user_transcript') {
                        const text = String(parsed.text || '').trim();
                        if (text) {
                            onEventRef.current?.({
                                kind: 'user_transcript',
                                text,
                                source: parsed.source === 'typed' ? 'typed' : 'voice',
                            });
                        }
                    } else if (parsed?.type === 'assistant_transcript') {
                        const text = String(parsed.text || '').trim();
                        if (text) {
                            const emotion = typeof parsed.emotion === 'string' ? parsed.emotion : undefined;
                            onEventRef.current?.({ kind: 'assistant_transcript', text, emotion });
                        }
                    } else if (parsed?.type === 'tool_event') {
                        const name = String(parsed.name || '');
                        if (name) {
                            onEventRef.current?.({
                                kind: 'tool_event',
                                name,
                                title: String(parsed.title || name),
                                status: parsed.status === 'completed' ? 'completed' : 'started',
                                arguments: parsed.arguments && typeof parsed.arguments === 'object'
                                    ? parsed.arguments
                                    : undefined,
                                ok: typeof parsed.ok === 'boolean' ? parsed.ok : undefined,
                                error: typeof parsed.error === 'string' ? parsed.error : null,
                            });
                        }
                    } else if (parsed?.type === 'error') {
                        // Backend explicit error (e.g. pipecat / faster-whisper
                        // not installed — see acla_ai_service/app/api/voice.py).
                        const msg = parsed.message || parsed.error_type || 'backend error';
                        console.error('[voice] backend error frame:', msg);
                        setError(msg);
                        setState('error');
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
                console.error('[voice] WS error event:', event);
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
    }, [
        state,
        openWs,
        stop,
        options.frontendTools,
        options.querySchemaScope,
    ]);

    /**
     * Schedule a PCM16 chunk for gapless playback on the playback AudioContext.
     */
    const queuePlayback = (pcm16Buffer: ArrayBuffer, context: AudioContext) => {
        const int16 = new Int16Array(pcm16Buffer);
        if (int16.length === 0) return;

        if (playbackIdleTimeoutRef.current !== null) {
            window.clearTimeout(playbackIdleTimeoutRef.current);
            playbackIdleTimeoutRef.current = null;
        }

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
        const serial = ++playbackSerialRef.current;
        source.onended = () => {
            if (serial !== playbackSerialRef.current) return;
            playbackIdleTimeoutRef.current = window.setTimeout(() => {
                playbackIdleTimeoutRef.current = null;
                if (serial !== playbackSerialRef.current) return;
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    playbackQueueTimeRef.current = context.currentTime;
                    setState((prev) => (prev === 'speaking' ? 'listening' : prev));
                }
            }, 160);
        };

        const now = context.currentTime;
        const startAt = Math.max(now, playbackQueueTimeRef.current);
        source.start(startAt);
        playbackQueueTimeRef.current = startAt + audioBuffer.duration;
    };

    /**
     * Send a typed chat message over the open voice WS. Backend treats it
     * as a synthetic user turn (same path as a spoken turn after STT).
     * Returns false if there's no open WS to send on.
     */
    const sendUserText = useCallback((text: string): boolean => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return false;
        const trimmed = text.trim();
        if (!trimmed) return false;
        try {
            ws.send(JSON.stringify({
                type: 'user_text',
                text: trimmed,
                session_context: sessionContextRef.current,
            }));
            return true;
        } catch (err) {
            console.warn('[voice] sendUserText failed:', err);
            return false;
        }
    }, []);

    const sendObservation = useCallback((data: Record<string, unknown>): boolean => {
        onEventRef.current?.({ kind: 'observation', data });
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return false;
        try {
            ws.send(JSON.stringify(buildFormattedObservationFrame(data)));
            return true;
        } catch (err) {
            console.warn('[voice] sendObservation failed:', err);
            return false;
        }
    }, []);

    // Auto-cleanup on unmount.
    useEffect(() => {
        return () => stop();
    }, [stop]);

    return {
        state,
        error,
        micLevel,
        micDisabled,
        start,
        stop,
        setMicDisabled,
        sendUserText,
        sendObservation,
    };
}
