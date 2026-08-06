import { useCallback, useContext, useEffect, useRef } from 'react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { createPythonStreamSession, PythonStreamEvent, PythonStreamSession } from 'services/pythonStreaming';
import { RecordingState } from './recording-state';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';

const toAccStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) {
        return null;
    }

    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

const getStaticPayload = (data: Record<string, any>): Record<string, unknown> | null => {
    if (data.Static && typeof data.Static === 'object') {
        return data.Static;
    }

    const track = typeof data.Static_track === 'string' ? data.Static_track : undefined;
    const carModel = typeof data.Static_car_model === 'string' ? data.Static_car_model : undefined;
    if (!track && !carModel) {
        return null;
    }

    return {
        ...(track ? { track } : {}),
        ...(carModel ? { car_model: carModel } : {}),
    };
};

export default function LiveSessionDetectionManager() {
    const liveSession = useContext(LiveSessionContext);
    const liveSessionRef = useRef(liveSession);
    const sessionCheckingStreamRef = useRef<PythonStreamSession<Record<string, unknown>> | null>(null);
    const sessionCheckingStreamCleanupRef = useRef<(() => void) | null>(null);
    const sessionCheckingStreamGenerationRef = useRef(0);
    const sessionCheckingStreamStartingGenerationRef = useRef<number | null>(null);

    liveSessionRef.current = liveSession;

    const processCheckingSessionStreamUpdate = useCallback((event: PythonStreamEvent<Record<string, unknown>>) => {
        const ctx = liveSessionRef.current;
        if (!ctx || ctx.sessionGame !== 'acc' || !event) {
            return;
        }

        if (event.status === 'update') {
            const data = (event.data ?? {}) as Record<string, any>;
            const graphics = data.Graphics ?? {};
            const status = toAccStatus(graphics.status ?? data.Graphics_status);

            if (data && typeof data === 'object') {
                ctx.setCurrentTelemetry(data);
            }

            if (status !== null) {
                if (status === ACC_STATUS.ACC_LIVE) {
                    const staticPayload = getStaticPayload(data);
                    if (staticPayload) {
                        ctx.setStaticData(staticPayload);
                    }
                    ctx.transitionRecordingState({ type: 'sessionAvailable' });
                } else if (status === ACC_STATUS.ACC_OFF) {
                    ctx.transitionRecordingState({ type: 'sessionUnavailable' });
                }
                return;
            }

            if (data.checking === true || data.available === false) {
                ctx.transitionRecordingState({ type: 'sessionUnavailable' });
            }
        } else if (event.status === 'ready') {
            if (ctx.telemetryStatus == null) {
                ctx.transitionRecordingState({ type: 'sessionUnavailable' });
            }
        } else if (event.status === 'error') {
            console.error('ACC session checker error:', event.message ?? 'Unknown error', event.traceback ?? '');
        } else if (event.status === 'shutdown') {
            sessionCheckingStreamCleanupRef.current?.();
            sessionCheckingStreamCleanupRef.current = null;
            sessionCheckingStreamRef.current = null;
            sessionCheckingStreamStartingGenerationRef.current = null;
        }
    }, []);

    const stopSessionCheckingStream = useCallback(async ({ force = false } = {}) => {
        sessionCheckingStreamGenerationRef.current += 1;
        sessionCheckingStreamStartingGenerationRef.current = null;

        const cleanup = sessionCheckingStreamCleanupRef.current;
        sessionCheckingStreamCleanupRef.current = null;
        cleanup?.();

        const stream = sessionCheckingStreamRef.current;
        sessionCheckingStreamRef.current = null;

        if (!stream) {
            return;
        }

        try {
            await stream.dispose({ force });
        } catch (error) {
            console.warn('Failed to dispose ACC session checker stream', error);
        }
    }, []);

    const startSessionCheckingStream = useCallback(async () => {
        if (sessionCheckingStreamStartingGenerationRef.current !== null || sessionCheckingStreamRef.current) {
            return sessionCheckingStreamRef.current;
        }

        const generation = sessionCheckingStreamGenerationRef.current;
        sessionCheckingStreamStartingGenerationRef.current = generation;
        try {
            const stream = await createPythonStreamSession<Record<string, unknown>>({
                scriptName: 'ACCCheckAvailableSession.py',
                pythonOptions: { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [] },
                readyTimeoutMs: 8000
            });

            if (generation !== sessionCheckingStreamGenerationRef.current) {
                await stream.dispose({ force: true });
                return null;
            }

            sessionCheckingStreamRef.current = stream;
            sessionCheckingStreamCleanupRef.current = stream.onMessage(processCheckingSessionStreamUpdate);

            await stream.waitUntilReady();
            return stream;
        } catch (error) {
            if (generation !== sessionCheckingStreamGenerationRef.current) {
                return null;
            }
            console.error('Failed to start ACC session checker stream', error);
            await stopSessionCheckingStream({ force: true });
            throw error;
        } finally {
            if (sessionCheckingStreamStartingGenerationRef.current === generation) {
                sessionCheckingStreamStartingGenerationRef.current = null;
            }
        }
    }, [processCheckingSessionStreamUpdate, stopSessionCheckingStream]);

    const shouldMaintainSessionCheckingStream =
        liveSession.sessionGame === 'acc'
        && (
            liveSession.recordingState === RecordingState.CHECKING
            || liveSession.recordingState === RecordingState.HOLDING
            || liveSession.recordingState === RecordingState.RESUME_READY
        );

    useEffect(() => {
        let cancelled = false;

        const ensureStream = async () => {
            await Promise.resolve();
            if (cancelled) {
                return;
            }

            if (shouldMaintainSessionCheckingStream) {
                try {
                    await startSessionCheckingStream();
                } catch (error) {
                    if (!cancelled) {
                        console.error('Unable to ensure ACC session checker stream', error);
                    }
                }
            } else {
                if (liveSession.sessionGame !== 'acc') {
                    liveSessionRef.current?.transitionRecordingState({ type: 'sessionUnavailable' });
                }
                await stopSessionCheckingStream();
            }
        };

        void ensureStream();

        return () => {
            cancelled = true;
            void stopSessionCheckingStream({ force: true });
        };
    }, [liveSession.sessionGame, shouldMaintainSessionCheckingStream, startSessionCheckingStream, stopSessionCheckingStream]);

    return null;
}
