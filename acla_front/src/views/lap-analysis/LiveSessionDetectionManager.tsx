import { useCallback, useContext, useEffect, useRef } from 'react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { createPythonStreamSession, PythonStreamEvent, PythonStreamSession } from 'services/pythonStreaming';
import { AnalysisContext } from './analysis-context';
import { RecordingState } from './recording-state';

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
    const analysisContext = useContext(AnalysisContext);
    const analysisContextRef = useRef(analysisContext);
    const sessionCheckingStreamRef = useRef<PythonStreamSession<Record<string, unknown>> | null>(null);
    const sessionCheckingStreamCleanupRef = useRef<(() => void) | null>(null);
    const sessionCheckingStreamStartingRef = useRef(false);

    useEffect(() => {
        analysisContextRef.current = analysisContext;
    }, [analysisContext]);

    const processCheckingSessionStreamUpdate = useCallback((event: PythonStreamEvent<Record<string, unknown>>) => {
        const ctx = analysisContextRef.current;
        if (!ctx || !event) {
            return;
        }

        if (event.status === 'update') {
            const data = (event.data ?? {}) as Record<string, any>;
            const graphics = data.Graphics ?? {};
            const status = toAccStatus(graphics.status ?? data.Graphics_status);

            if (data && typeof data === 'object') {
                ctx.setLiveSessionData(data);
            }

            if (status !== null) {
                if (status === ACC_STATUS.ACC_LIVE) {
                    const staticPayload = getStaticPayload(data);
                    if (staticPayload) {
                        ctx.setRecordedSessionStaticsData(staticPayload);
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
            if (ctx.TelemetryDataLiveStatus == null) {
                ctx.transitionRecordingState({ type: 'sessionUnavailable' });
            }
        } else if (event.status === 'error') {
            console.error('ACC session checker error:', event.message ?? 'Unknown error', event.traceback ?? '');
        } else if (event.status === 'shutdown') {
            sessionCheckingStreamCleanupRef.current?.();
            sessionCheckingStreamCleanupRef.current = null;
            sessionCheckingStreamRef.current = null;
            sessionCheckingStreamStartingRef.current = false;
        }
    }, []);

    const stopSessionCheckingStream = useCallback(async ({ force = false } = {}) => {
        sessionCheckingStreamStartingRef.current = false;

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
        if (sessionCheckingStreamStartingRef.current || sessionCheckingStreamRef.current) {
            return sessionCheckingStreamRef.current;
        }

        sessionCheckingStreamStartingRef.current = true;
        try {
            const stream = await createPythonStreamSession<Record<string, unknown>>({
                scriptName: 'ACCCheckAvailableSession.py',
                pythonOptions: { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [] },
                readyTimeoutMs: 8000
            });

            sessionCheckingStreamRef.current = stream;
            sessionCheckingStreamCleanupRef.current = stream.onMessage(processCheckingSessionStreamUpdate);

            await stream.waitUntilReady();
            return stream;
        } catch (error) {
            console.error('Failed to start ACC session checker stream', error);
            await stopSessionCheckingStream({ force: true });
            throw error;
        } finally {
            sessionCheckingStreamStartingRef.current = false;
        }
    }, [processCheckingSessionStreamUpdate, stopSessionCheckingStream]);

    const shouldMaintainSessionCheckingStream =
        analysisContext.recordingState === RecordingState.CHECKING
        || analysisContext.recordingState === RecordingState.HOLDING
        || analysisContext.recordingState === RecordingState.RESUME_READY;

    useEffect(() => {
        let cancelled = false;

        const ensureStream = async () => {
            if (shouldMaintainSessionCheckingStream) {
                try {
                    await startSessionCheckingStream();
                } catch (error) {
                    if (!cancelled) {
                        console.error('Unable to ensure ACC session checker stream', error);
                    }
                }
            } else {
                await stopSessionCheckingStream();
            }
        };

        void ensureStream();

        return () => {
            cancelled = true;
            void stopSessionCheckingStream({ force: true });
        };
    }, [shouldMaintainSessionCheckingStream, startSessionCheckingStream, stopSessionCheckingStream]);

    return null;
}
