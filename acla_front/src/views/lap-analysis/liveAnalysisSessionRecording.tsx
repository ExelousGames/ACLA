import { Card, Flex, Box, IconButton, Heading, Grid, Text, Slider, Spinner, AlertDialog, Button } from '@radix-ui/themes';
import { useContext, useEffect, useRef, useState, useMemo, useCallback, JSX } from 'react';
import { AnalysisContext } from './analysis-context';
import { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto, RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useAuth } from 'hooks/AuthProvider';
import apiService from 'services/api.service';
import { PythonShellOptions } from 'services/pythonService';
import { createPythonStreamSession, PythonStreamEvent, PythonStreamSession } from 'services/pythonStreaming';

enum RecordingState {
    CHECKING = 'CHECKING', // checking for live session
    READY = 'READY', // find live session, ready to record
    RECORDING = 'RECORDING', // actively recording
    HOLDING = 'HOLDING', // paused because game paused, awaiting resume
    RESUME_READY = 'RESUME_READY', // live session detected again while paused
    UPLOAD_READY = 'UPLOAD_READY' // recording stopped, ready to upload
}

type StopReason = 'manual' | 'pause' | 'error' | 'complete';

const UPLOAD_CHUNK_SIZE = 5;
const POST_UPLOAD_RESET_DELAY_MS = 1200;
const POST_SUCCESS_DIALOG_CLOSE_MS = 800;

const PlayIcon = ({ size = 18 }: { size?: number }) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 30 30" width={size} height={size}>
        <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
    </svg>
);

const StopIcon = ({ size = 16 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 2C1.44772 2 1 2.44772 1 3V12C1 12.5523 1.44772 13 2 13H13C13.5523 13 14 12.5523 14 12V3C14 2.44772 13.5523 2 13 2H2ZM3 3H12V12H3V3Z" fill="currentColor" />
    </svg>
);

const UploadIcon = ({ size = 16 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" />
    </svg>
);

const PauseBadgeIcon = ({ size = 16 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M5 3C4.44772 3 4 3.44772 4 4V11C4 11.5523 4.44772 12 5 12C5.55228 12 6 11.5523 6 11V4C6 3.44772 5.55228 3 5 3ZM10 3C9.44772 3 9 3.44772 9 4V11C9 11.5523 9.44772 12 10 12C10.5523 12 11 11.5523 11 11V4C11 3.44772 10.5523 3 10 3Z" fill="currentColor" />
    </svg>
);

const toAccStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) {
        return null;
    }

    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

export default function LiveAnalysisSessionRecording() {
    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();
    const [state, setState] = useState<RecordingState>(RecordingState.CHECKING);
    const analysisContextRef = useRef(analysisContext);

    useEffect(() => {
        analysisContextRef.current = analysisContext;
    }, [analysisContext]);

    const TelemetryDataLiveStatus = analysisContext.TelemetryDataLiveStatus;
    const canRecord = state === RecordingState.READY || state === RecordingState.RESUME_READY;

    type RecordingEvent =
        | { type: 'sessionAvailable' }
        | { type: 'sessionUnavailable' }
        | { type: 'recordingStarted' }
        | { type: 'recordingResumed' }
        | { type: 'recordingStopped'; reason: StopReason }
        | { type: 'reset' };

    const transition = useCallback((event: RecordingEvent) => {

        setState((prev) => {
            switch (event.type) {
                case 'sessionAvailable':
                    if (prev === RecordingState.CHECKING) {
                        return RecordingState.READY;
                    }
                    if (prev === RecordingState.HOLDING) {
                        return RecordingState.RESUME_READY;
                    }
                    return prev;
                case 'sessionUnavailable':
                    if (prev === RecordingState.RESUME_READY) {
                        return RecordingState.HOLDING;
                    }
                    if (prev === RecordingState.RECORDING || prev === RecordingState.HOLDING || prev === RecordingState.UPLOAD_READY) {
                        return prev;
                    }
                    return RecordingState.CHECKING;
                case 'recordingStarted':
                    return RecordingState.RECORDING;
                case 'recordingResumed':
                    return prev === RecordingState.HOLDING || prev === RecordingState.RESUME_READY ? RecordingState.RECORDING : prev;
                case 'recordingStopped':
                    switch (event.reason) {
                        case 'pause':
                            return RecordingState.HOLDING;
                        case 'error':
                            return RecordingState.READY;
                        case 'manual':
                        case 'complete':
                            return RecordingState.UPLOAD_READY;
                        default:
                            return prev;
                    }
                case 'reset':
                    return RecordingState.CHECKING;
                default:
                    return prev;
            }
        });
    }, []);

    const recordingShellIdRef = useRef<number | null>(null);
    const pythonMessageCleanupRef = useRef<(() => void) | null>(null);
    const pythonEndCleanupRef = useRef<(() => void) | null>(null);
    const stopReasonRef = useRef<StopReason | null>(null);
    const startInFlightRef = useRef(false);
    const hasReceivedLiveSampleRef = useRef(false);
    const recordingFileInfoRef = useRef<{ folder: string; filename: string } | null>(null);

    const sessionCheckingStreamRef = useRef<PythonStreamSession<Record<string, unknown>> | null>(null);
    const sessionCheckingStreamCleanupRef = useRef<(() => void) | null>(null);
    const sessionCheckingStreamStartingRef = useRef(false);

    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [showRetryButton, setShowRetryButton] = useState(false);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const uploadInFlightRef = useRef(false);
    const hasRecordedData = analysisContext.recordedTelemetryDataCount > 0 && Boolean(analysisContext.recordedSessionDataFilePath);

    const uploadStatusLabel = isUploading
        ? 'Uploading...'
        : (state === RecordingState.HOLDING || state === RecordingState.RESUME_READY) && hasRecordedData
            ? 'Upload available (recording paused)'
            : hasRecordedData
                ? 'Ready to upload'
                : 'No data recorded';
    const uploadStatusColor = isUploading
        ? 'blue'
        : (state === RecordingState.HOLDING || state === RecordingState.RESUME_READY) && hasRecordedData
            ? 'amber'
            : hasRecordedData
                ? 'green'
                : 'gray';



    const applyStopOutcome = useCallback((reason: StopReason) => {
        if (pythonMessageCleanupRef.current) {
            pythonMessageCleanupRef.current();
            pythonMessageCleanupRef.current = null;
        }
        if (pythonEndCleanupRef.current) {
            pythonEndCleanupRef.current();
            pythonEndCleanupRef.current = null;
        }

        recordingShellIdRef.current = null;
        stopReasonRef.current = null;
        startInFlightRef.current = false;
        hasReceivedLiveSampleRef.current = false;

        switch (reason) {
            case 'pause': {
                transition({ type: 'recordingStopped', reason: 'pause' });
                break;
            }
            case 'error': {
                recordingFileInfoRef.current = null;
                analysisContext.clearRecordingSession();
                transition({ type: 'recordingStopped', reason: 'error' });
                break;
            }
            case 'manual': {
                transition({ type: 'recordingStopped', reason: 'manual' });
                break;
            }
            default: {
                transition({ type: 'recordingStopped', reason: 'complete' });
            }
        }
    }, [analysisContext, transition]);

    /**
     * Process updates from the ACC session checking stream
     * @param event PythonStreamEvent<Record<string, unknown>>
     * @returns void
     */
    const processCheckingSessionStreamUpdate = useCallback((event: PythonStreamEvent<Record<string, unknown>>) => {
        const ctx = analysisContextRef.current;
        if (!ctx || !event) {
            return;
        }

        if (event.status === 'update') {
            const data = (event.data ?? {}) as Record<string, any>;
            const graphics = (data as any).Graphics ?? {};
            const status = toAccStatus(graphics.status);

            if (status !== null) {
                if (status === ACC_STATUS.ACC_LIVE) {
                    if ((data as any).Static) {
                        ctx.setRecordedSessionStaticsData((data as any).Static);
                    }
                    transition({ type: 'sessionAvailable' });
                } else if (status === ACC_STATUS.ACC_PAUSE) {
                    // recorder will transition when the python process ends
                } else if (status === ACC_STATUS.ACC_OFF) {
                    transition({ type: 'sessionUnavailable' });
                }
            } else if (data.checking === true) {
                transition({ type: 'sessionUnavailable' });
            } else if (data.available === false) {
                transition({ type: 'sessionUnavailable' });
            }
        } else if (event.status === 'ready') {
            if (ctx.TelemetryDataLiveStatus == null) {
                transition({ type: 'sessionUnavailable' });
            }
        } else if (event.status === 'error') {
            console.error('ACC session checker error:', event.message ?? 'Unknown error', event.traceback ?? '');
        } else if (event.status === 'shutdown') {
            sessionCheckingStreamCleanupRef.current?.();
            sessionCheckingStreamCleanupRef.current = null;
            sessionCheckingStreamRef.current = null;
            sessionCheckingStreamStartingRef.current = false;
        }
    }, [transition]);

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
        state === RecordingState.CHECKING || state === RecordingState.HOLDING || state === RecordingState.RESUME_READY;

    useEffect(() => {
        let cancelled = false;

        const ensureStream = async () => {
            if (shouldMaintainSessionCheckingStream) {
                try {
                    // Start the session checking stream
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

    const stopRecordingProcess = useCallback(async (reason: StopReason) => {
        if (stopReasonRef.current && stopReasonRef.current !== 'complete') {
            return;
        }

        stopReasonRef.current = reason;

        if (state !== RecordingState.RECORDING) {
            applyStopOutcome(reason);
            return;
        }

        const shellId = recordingShellIdRef.current;
        if (shellId == null || !window?.electronAPI?.stopPythonScript) {
            applyStopOutcome(reason);
            return;
        }

        try {
            const result = await window.electronAPI.stopPythonScript(shellId);
            if (!result?.success) {
                applyStopOutcome('error');
            }
        } catch (error) {
            console.error('Failed to stop python script', error);
            applyStopOutcome('error');
        }
    }, [applyStopOutcome, state]);

    const determineStopReason = useCallback((): StopReason => {
        if (stopReasonRef.current) {
            return stopReasonRef.current;
        }

        const ctx = analysisContextRef.current;
        const liveStatus = ctx?.TelemetryDataLiveStatus ?? null;
        console.log('Determining stop reason, live status:', liveStatus);
        if (liveStatus === ACC_STATUS.ACC_PAUSE) {
            return 'pause';
        }

        if (liveStatus !== ACC_STATUS.ACC_LIVE) {
            return 'complete';
        }

        if (!hasReceivedLiveSampleRef.current) {
            return 'error';
        }

        return 'complete';
    }, []);


    const startRecording = useCallback(async ({ resumeExisting = false }: { resumeExisting?: boolean } = {}) => {
        if (!canRecord || startInFlightRef.current) {
            return;
        }

        startInFlightRef.current = true;

        analysisContext.setMap((analysisContext.recordedSessioStaticsData as any)?.track || analysisContext.mapSelected || 'Unknown Track');
        let folder = '../session_recording';
        let filename: string;

        if (resumeExisting && recordingFileInfoRef.current) {
            ({ folder, filename } = recordingFileInfoRef.current);
        } else {
            const now = new Date();
            filename = `acc_${now.getFullYear()}_${now.getMonth()}_${now.getDate()}_${now.getHours()}_${now.getMinutes()}_${now.getSeconds()}.csv`;
            recordingFileInfoRef.current = { folder, filename };
        }

        const options: PythonShellOptions = { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [folder, filename] };
        const script = 'ACCMemoryExtractor.py';

        if (!resumeExisting) {
            const newSessionName = `Racing Session ${new Date().toLocaleString()}`;
            analysisContext.setSession({
                session_name: newSessionName,
                SessionId: '',
                map: analysisContext.mapSelected || (analysisContext.recordedSessioStaticsData as any)?.track || 'Unknown Track',
                user_id: '',
                points: [],
                data: [],
                car: (analysisContext.recordedSessioStaticsData as any)?.car_model || 'Unknown Car'
            } as RacingSessionDetailedInfoDto as any);
        }

        hasReceivedLiveSampleRef.current = false;
        transition({ type: resumeExisting ? 'recordingResumed' : 'recordingStarted' });
        try {
            const { shellId } = await window.electronAPI.runPythonScript(script, options);
            recordingShellIdRef.current = shellId;
            stopReasonRef.current = null;

            const messageCleanup = window.electronAPI.onPythonMessage((incomingId: number, message: string) => {
                if (incomingId !== shellId) {
                    return;
                }
                try {
                    const obj = JSON.parse(message);
                    analysisContext.setLiveSessionData(obj);
                    void analysisContext.writeRecordedLiveSessionData(obj).catch(() => undefined);
                    hasReceivedLiveSampleRef.current = true;
                } catch { }
            });
            pythonMessageCleanupRef.current = messageCleanup;

            const removeEndListener = window.electronAPI.onPythonEnd('live-analysis-session-recording', (incomingId: number) => {
                if (incomingId !== shellId) {
                    return;
                }

                if (pythonMessageCleanupRef.current) {
                    pythonMessageCleanupRef.current();
                    pythonMessageCleanupRef.current = null;
                }

                removeEndListener();
                pythonEndCleanupRef.current = null;

                const reason = determineStopReason();
                applyStopOutcome(reason);
            });
            pythonEndCleanupRef.current = removeEndListener;
        } catch (error) {
            console.error('Failed to start recording session', error);
            applyStopOutcome('error');
        } finally {
            startInFlightRef.current = false;
        }
    }, [analysisContext, applyStopOutcome, canRecord, transition, uploadDialogOpen, determineStopReason]);

    useEffect(() => {
        if (state !== RecordingState.HOLDING) {
            return;
        }
        if (TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE) {
            return;
        }
        if (isUploading || uploadDialogOpen || uploadInFlightRef.current) {
            return;
        }
        if (startInFlightRef.current) {
            return;
        }
        if (!recordingFileInfoRef.current) {
            return;
        }

        void startRecording({ resumeExisting: true });
    }, [TelemetryDataLiveStatus, isUploading, startRecording, state, uploadDialogOpen]);

    const cleanupTelemetryFile = useCallback(async (filePath: string) => {
        try { const options: PythonShellOptions = { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [filePath] }; await window.electronAPI.runPythonScript('delete_telemetry_file.py', options); } catch { }
    }, []);

    const resetToChecking = useCallback(() => {
        analysisContext.clearRecordingSession();
        // Clear the current session to ensure a fresh one is created for the next recording
        analysisContext.setSession(null);
        recordingFileInfoRef.current = null;
        uploadInFlightRef.current = false;
        setUploadProgress(0); setUploadStatus(''); setUploadError(null); setShowRetryButton(false); setUploadDialogOpen(false); setIsUploading(false);
        transition({ type: 'reset' });
        hasReceivedLiveSampleRef.current = false;
        stopReasonRef.current = null;
    }, [analysisContext, transition]);

    const handleUpload = useCallback(async () => {
        if (uploadInFlightRef.current) return false;
        if (!hasRecordedData) { setUploadError('No telemetry data available for upload'); setShowRetryButton(false); return false; }
        if (!analysisContext.sessionSelected?.session_name || !analysisContext.mapSelected || !auth?.userEmail) { setUploadError('Missing required session or user information'); setShowRetryButton(false); return false; }
        uploadInFlightRef.current = true; setIsUploading(true); setUploadProgress(0); setUploadStatus('Reading telemetry data...'); setUploadError(null); setShowRetryButton(false);
        try {
            // Reserve progress ranges: 0-40% for reading, 40-90% for chunk upload, 90-100% finalize
            let estimatedTotal: number | null = null;
            let lastRead = 0;
            const data = await analysisContext.readRecordedSessionData((read, total) => {
                lastRead = read;
                if (total && total > 0) estimatedTotal = total;
                // If total known compute percentage otherwise logarithmic approximation
                let pct: number;
                if (estimatedTotal) {
                    pct = Math.min(read / estimatedTotal, 1) * 40; // scale into 0-40
                } else {
                    // Unknown total: approach 40% asymptotically
                    pct = 40 * (1 - Math.exp(-read / 500));
                }
                setUploadProgress(Math.max(0, Math.min(40, Math.floor(pct))));
            });
            if (!data || data.length === 0) throw new Error('No telemetry data found to upload');
            setUploadProgress(45); setUploadStatus(`Processing ${data.length} telemetry points...`);
            const chunks: any[] = []; for (let i = 0; i < data.length; i += UPLOAD_CHUNK_SIZE) chunks.push(data.slice(i, i + UPLOAD_CHUNK_SIZE));
            const metadata: UploadReacingSessionInitDto = { sessionName: analysisContext.sessionSelected.session_name, mapName: analysisContext.mapSelected, carName: analysisContext.recordedSessioStaticsData.car_model || 'Unknown Car', userId: auth?.userProfile.id || 'unknown' };
            setUploadProgress(50); setUploadStatus('Initializing upload...');
            const initResp = await apiService.post('/racing-session/upload/init', metadata); if (!initResp.data) throw new Error('Failed to initialize upload');
            const { uploadId } = initResp.data as UploadRacingSessionInitReturnDto;
            setUploadProgress(55); setUploadStatus(`Uploading ${chunks.length} chunks...`);
            for (let i = 0; i < chunks.length; i++) { const params = new URLSearchParams(); params.append('uploadId', uploadId); await apiService.post(`/racing-session/upload/chunk?${params.toString()}`, { chunk: chunks[i], chunkIndex: i }); const pct = Math.floor(55 + (i + 1) / chunks.length * 35); setUploadProgress(pct); setUploadStatus(`Uploading chunk ${i + 1} of ${chunks.length}...`); }
            setUploadProgress(92); setUploadStatus('Finalizing upload...');
            const final = new URLSearchParams(); final.append('uploadId', uploadId); await apiService.post(`/racing-session/upload/complete?${final.toString()}`, {});
            setUploadProgress(100); setUploadStatus('Upload completed successfully!');
            if (analysisContext.recordedSessionDataFilePath) await cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath);
            setTimeout(() => { setIsUploading(false); setTimeout(() => resetToChecking(), POST_SUCCESS_DIALOG_CLOSE_MS); }, POST_UPLOAD_RESET_DELAY_MS);
            uploadInFlightRef.current = false;
            return true;
        } catch (e) { setUploadError(e instanceof Error ? e.message : 'Upload failed'); setIsUploading(false); setShowRetryButton(true); uploadInFlightRef.current = false; return false; }
    }, [analysisContext, auth, cleanupTelemetryFile, resetToChecking, hasRecordedData]);

    const handleCancelUpload = useCallback(async () => { if (analysisContext.recordedSessionDataFilePath) await cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath); resetToChecking(); }, [analysisContext, cleanupTelemetryFile, resetToChecking]);
    const handleRetryUpload = useCallback(() => { setUploadError(null); setShowRetryButton(false); setUploadProgress(0); handleUpload(); }, [handleUpload]);

    const openUploadDialog = useCallback(() => {
        if (isUploading) {
            return;
        }
        setUploadDialogOpen(true);
    }, [isUploading]);

    const closeUploadDialog = useCallback(() => {
        if (isUploading) {
            return;
        }
        setUploadDialogOpen(false);
    }, [isUploading]);

    const handleDialogOpenChange = useCallback((open: boolean) => {
        if (!open && isUploading) {
            return;
        }
        setUploadDialogOpen(open);
    }, [isUploading]);

    const controlButtons = useMemo(() => {
        switch (state) {
            case RecordingState.CHECKING:
                return (
                    <Button radius="full" variant="outline" color="gray" disabled>
                        <Flex align="center" gap="2">
                            <Spinner size="1" />
                            <span>Looking for live session…</span>
                        </Flex>
                    </Button>
                );
            case RecordingState.READY:
                return (
                    <Button radius="full" color="blue" onClick={() => { if (canRecord) { void startRecording(); } }}>
                        <Flex align="center" gap="2">
                            <PlayIcon size={14} />
                            <span>Start Recording</span>
                        </Flex>
                    </Button>
                );
            case RecordingState.RECORDING:
                return (
                    <Button radius="full" color="red" onClick={() => { void stopRecordingProcess('manual'); }}>
                        <Flex align="center" gap="2">
                            <StopIcon size={14} />
                            <span>Stop Recording</span>
                        </Flex>
                    </Button>
                );
            case RecordingState.HOLDING:
            case RecordingState.RESUME_READY: {
                return (
                    <Flex align="center" gap="2">
                        <Button radius="full" variant="outline" color="blue" disabled={!canRecord || isUploading} onClick={() => { void startRecording({ resumeExisting: true }); }}>
                            <Flex align="center" gap="2">
                                <PlayIcon size={14} />
                                <span>Resume</span>
                            </Flex>
                        </Button>
                        <Button radius="full" color="green" onClick={openUploadDialog} disabled={!hasRecordedData || isUploading}>
                            <Flex align="center" gap="2">
                                <UploadIcon size={14} />
                                <span>Upload Session</span>
                            </Flex>
                        </Button>
                    </Flex>
                );
            }
            case RecordingState.UPLOAD_READY:
                return (
                    <Flex align="center" gap="2">
                        <Button radius="full" color="green" onClick={openUploadDialog} disabled={!hasRecordedData || isUploading}>
                            <Flex align="center" gap="2">
                                <UploadIcon size={14} />
                                <span>Upload Session</span>
                            </Flex>
                        </Button>
                        <Button radius="full" variant="outline" color="gray" onClick={() => { void handleCancelUpload(); closeUploadDialog(); }} disabled={isUploading}>
                            <span>Discard</span>
                        </Button>
                    </Flex>
                );
            default:
                return null;
        }
    }, [state, canRecord, startRecording, stopRecordingProcess, TelemetryDataLiveStatus, hasRecordedData, isUploading, openUploadDialog, handleCancelUpload, closeUploadDialog]);

    return (
        <Box position="absolute" left="0" right="0" bottom="0" mb="5" height="64px" style={{ borderRadius: '100px', boxShadow: 'var(--shadow-6)', marginLeft: 200, marginRight: 200 }}>
            <Flex height="100%" justify="between" position="relative">
                <Flex gap="4" align="center" p="3">

                    {controlButtons}
                    <AlertDialog.Root open={uploadDialogOpen} onOpenChange={handleDialogOpenChange}>
                        <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(e) => { if (isUploading) e.preventDefault(); }}>
                            <AlertDialog.Title>Upload Racing Session</AlertDialog.Title>
                            <AlertDialog.Description size="2">Upload your recorded racing session data.</AlertDialog.Description>
                            {(isUploading || showRetryButton || uploadError) && (
                                <Box my="4">
                                    {isUploading && (
                                        <>
                                            <Flex justify="between" mb="2"><Text size="2" weight="medium">{uploadStatus}</Text><Text size="2" color="gray">{uploadProgress}%</Text></Flex>
                                            <Box width="100%" height="8px" style={{ backgroundColor: 'var(--gray-a5)', borderRadius: 'var(--radius-2)', overflow: 'hidden' }}>
                                                <Box height="100%" style={{ width: `${uploadProgress}%`, backgroundColor: uploadError ? 'var(--red-9)' : 'var(--blue-9)', transition: 'width 0.3s ease' }} />
                                            </Box>
                                        </>
                                    )}
                                    {uploadError && <Text size="2" color="red" mt="2">{uploadError}</Text>}
                                    {showRetryButton && !isUploading && <Flex mt="2" gap="2"><Button size="1" variant="outline" onClick={handleRetryUpload}>Retry Upload</Button></Flex>}
                                </Box>
                            )}
                            <Card size="4">
                                <Heading as="h3" size="6" trim="start" mb="5">Session <Text as="div" size="3" weight="bold" color="blue">{analysisContext.sessionSelected?.session_name || 'Unknown Session'}</Text></Heading>
                                <Grid columns="2" gapX="4" gapY="5">
                                    <Box>
                                        <Text as="div" size="2" mb="1" color="gray">Map</Text>
                                        <Text as="div" size="3" mb="1" weight="bold">{analysisContext.mapSelected || 'Unknown Map'}</Text>
                                        <Text as="div" size="2">Practice session</Text>
                                    </Box>
                                    <Box>
                                        <Text as="div" size="2" mb="1" color="gray">Car</Text>
                                        <Text as="div" size="3" weight="bold">{analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car'}</Text>
                                    </Box>
                                    <Flex direction="column" gap="1" gridColumn="1 / -1">
                                        <Flex justify="between"><Text size="3" mb="1" weight="bold">Status</Text><Text size="2" color={uploadStatusColor}>{uploadStatusLabel}</Text></Flex>
                                    </Flex>
                                </Grid>
                            </Card>
                            <Flex gap="3" mt="4" justify="end">
                                {!isUploading && uploadProgress < 100 && (
                                    <>
                                        <Button variant="outline" color="red" onClick={() => { void handleCancelUpload(); closeUploadDialog(); }}>Cancel</Button>
                                        <Button onClick={() => { void handleUpload(); }} disabled={isUploading || !hasRecordedData}>Upload Session</Button>
                                    </>
                                )}
                                {isUploading && uploadProgress < 100 && (<Button variant="outline" disabled><Spinner size="1" />Uploading...</Button>)}
                                {!isUploading && uploadProgress === 100 && (<Button onClick={closeUploadDialog}>Close</Button>)}
                            </Flex>
                        </AlertDialog.Content>
                    </AlertDialog.Root>

                </Flex>
                <Flex align="center" gap="3">
                    <Flex align="center" gap="3">
                        <Box>
                            <Text size="1" as="div" weight="medium">Racing Map Name Here</Text>
                            <Text size="1" as="div" color="gray" mb="2">Practice Session</Text>
                            <Box position="relative" height="4px" width="320px" style={{ backgroundColor: 'var(--gray-a5)', borderRadius: 'var(--radius-1)' }}>
                                <Box position="absolute" height="4px" width="64px" style={{ borderRadius: 'var(--radius-1)', backgroundColor: 'var(--gray-a9)' }} />
                                <Box position="absolute" top="0" right="0" mt="-28px"><Text size="1" color="gray">0:58 / Lap 2</Text></Box>
                            </Box>
                        </Box>
                    </Flex>
                </Flex>
                <Flex align="center" gap="2" p="5">
                    <Slider defaultValue={[80]} variant="soft" color="gray" radius="full" size="2" style={{ width: 80 }} />
                </Flex>
            </Flex>
        </Box>
    );
}