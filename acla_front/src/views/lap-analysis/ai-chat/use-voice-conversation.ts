/**
 * Voice conversation hook (Phase 3).
 *
 * Opens a WebSocket to the AI service's /voice/stream, captures mic
 * audio via AudioWorklet, sends PCM16 frames upstream, and plays
 * incoming PCM16 audio frames via Web Audio.
 *
 * Phase 3a scope: this hook talks directly to the AI service WebSocket.
 * Auth-gated NestJS proxying is deferred to Phase 3b — acceptable for
 * an Electron desktop app where the renderer is already on the same
 * trust boundary as the backend.
 *
 * Wire format: this hook speaks Pipecat's protobuf frame protocol. The
 * server uses `ProtobufFrameSerializer`. We keep the wire-format details
 * inside this module so consumers see a simple "start/stop" API.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export type VoiceConversationState =
    | 'idle'           // not connected
    | 'connecting'     // WS handshake in progress
    | 'listening'      // mic open, sending to server
    | 'speaking'       // server is sending us audio
    | 'error';

export interface VoiceConversationOptions {
    /** Track name passed to the server as session context. */
    trackName?: string;
    /** Car name passed to the server as session context. */
    carName?: string;
    /** User ID (optional — server logs only for now). */
    userId?: string;
    /** Override base URL — defaults to the AI service env config. */
    baseUrl?: string;
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
     * Resolve the AI service WS URL. In dev the AI service is on
     * REACT_APP_AI_SERVICE_URL or falls back to REACT_APP_BACKEND_SERVER_IP:8000.
     */
    const resolveWsUrl = useCallback((): string => {
        if (options.baseUrl) return options.baseUrl;
        const explicit = process.env.REACT_APP_AI_SERVICE_WS_URL;
        if (explicit) return explicit;

        const host = process.env.REACT_APP_BACKEND_SERVER_IP || 'localhost';
        const port = process.env.REACT_APP_AI_SERVICE_PORT || '8000';
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const params = new URLSearchParams();
        if (options.trackName) params.set('track_name', options.trackName);
        if (options.carName) params.set('car_name', options.carName);
        if (options.userId) params.set('user_id', options.userId);
        const qs = params.toString();
        return `${proto}//${host}:${port}/voice/stream${qs ? '?' + qs : ''}`;
    }, [options.baseUrl, options.trackName, options.carName, options.userId]);

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
            const ws = new WebSocket(resolveWsUrl());
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

            ws.onmessage = (event) => {
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
    }, [state, resolveWsUrl, stop]);

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
