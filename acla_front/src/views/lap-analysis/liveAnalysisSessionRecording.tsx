import { Card, Flex, Box, Heading, Grid, Text, Spinner, AlertDialog, Button } from '@radix-ui/themes';
import { useContext, useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import './liveAnalysisSessionRecording.css';
import { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto } from 'data/live-analysis/live-analysis-type';
import { useAuth } from 'hooks/AuthProvider';
import apiService from 'services/api.service';
import { RecordingState, StopReason } from './recording-state';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import type { LiveRecordingMetadata } from 'views/live-session/live-session-types';

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

type LiveAnalysisSessionRecordingProps = {
    recorderHostId?: string;
};

export default function LiveAnalysisSessionRecording({ recorderHostId }: LiveAnalysisSessionRecordingProps) {
    const analysisContext = useContext(LiveSessionContext);
    const auth = useAuth();
    const state = analysisContext.recordingState;
    const registerRecorderControl = analysisContext.registerRecorderControl;
    const analysisContextRef = useRef(analysisContext);

    useEffect(() => {
        analysisContextRef.current = analysisContext;
    }, [analysisContext]);

    const canRecord = analysisContext.sessionGame !== null
        && (state === RecordingState.READY || state === RecordingState.RESUME_READY);

    const startInFlightRef = useRef(false);
    const stopInFlightRef = useRef<Promise<void> | null>(null);

    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [showRetryButton, setShowRetryButton] = useState(false);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const [recordingUnavailable, setRecordingUnavailable] = useState<string | null>(null);
    const [recorderHost, setRecorderHost] = useState<HTMLElement | null>(null);

    useEffect(() => {
        if (!recorderHostId) {
            setRecorderHost(null);
            return;
        }
        setRecorderHost(document.getElementById(recorderHostId));
        return () => setRecorderHost(null);
    }, [analysisContext.sessionGame, recorderHostId]);

    const uploadInFlightRef = useRef(false);
    const restoredFileIsUploadable = analysisContext.recordingFileValidation
        ? analysisContext.recordingFileValidation.exists
            && analysisContext.recordingFileValidation.readable
            && analysisContext.recordingFileValidation.hasData
        : null;
    const hasRecordedData = Boolean(analysisContext.recordingFileKey)
        && (restoredFileIsUploadable ?? (
            analysisContext.recordedSampleCount > 0
            || state === RecordingState.RECORDING
        ));

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



    const stopRecordingProcess = useCallback(async (reason: StopReason) => {
        if (stopInFlightRef.current) return stopInFlightRef.current;
        stopInFlightRef.current = (async () => {
            try {
                await analysisContextRef.current.stopRecordingSession(reason);
            } finally {
                startInFlightRef.current = false;
                stopInFlightRef.current = null;
            }
        })();
        return stopInFlightRef.current;
    }, []);


    const startRecording = useCallback(async () => {
        if (!canRecord || startInFlightRef.current) {
            return;
        }

        startInFlightRef.current = true;
        const ctx = analysisContextRef.current;
        const sessionGame = ctx.sessionGame;
        if (!sessionGame) {
            startInFlightRef.current = false;
            return;
        }
        const rawTrackName = ctx.staticData?.Static_track || ctx.currentTelemetry?.Static_track;
        const rawCarName = ctx.staticData?.Static_car_model || ctx.currentTelemetry?.Static_car_model;
        const trackName = typeof rawTrackName === 'string' && rawTrackName ? rawTrackName : 'Unknown Track';
        const carName = typeof rawCarName === 'string' && rawCarName ? rawCarName : 'Unknown Car';
        const newSessionName = `Racing Session ${new Date().toLocaleString()}`;
        const metadata: LiveRecordingMetadata = {
            sessionName: newSessionName,
            mapName: trackName,
            carName,
            gameRecordedFrom: sessionGame,
        };
        setRecordingUnavailable(null);
        try {
            const result = await ctx.startRecordingSession(sessionGame);
            if (!result.ok) {
                if (result.error.type === 'unsupported-recording-game') {
                    setRecordingUnavailable('Live recording for this simulator is coming soon.');
                } else {
                    setRecordingUnavailable(result.error.message);
                }
            } else {
                ctx.setRecordingMetadata(metadata);
            }
        } catch (error) {
            console.error('Failed to start recording session', error);
            setRecordingUnavailable(error instanceof Error ? error.message : String(error));
        } finally {
            startInFlightRef.current = false;
        }
    }, [canRecord]);

    useEffect(() => {
        return () => {
            if (analysisContextRef.current.recordingActive) {
                void stopRecordingProcess('complete').catch(() => undefined);
            }
        };
    }, [stopRecordingProcess]);

    const cleanupTelemetryFile = useCallback(async (filePath: string) => {
        try {
            if (window.electronAPI?.deleteTempFile) {
                const result = await window.electronAPI.deleteTempFile(filePath);
                return result.success;
            }
            return false;
        } catch {
            return false;
        }
    }, []);

    const resetRecorderUi = useCallback(() => {
        uploadInFlightRef.current = false;
        setUploadProgress(0); setUploadStatus(''); setUploadError(null); setShowRetryButton(false); setUploadDialogOpen(false); setIsUploading(false);
        startInFlightRef.current = false;
        stopInFlightRef.current = null;
        setRecordingUnavailable(null);
    }, []);

    const returnToDetectionGate = useCallback(() => {
        resetRecorderUi();
        analysisContextRef.current.endLiveSession();
    }, [resetRecorderUi]);

    useEffect(() => {
        if (analysisContext.sessionGame === null) {
            resetRecorderUi();
        }
    }, [analysisContext.sessionGame, resetRecorderUi]);

    const handleUpload = useCallback(async () => {
        if (uploadInFlightRef.current) return false;
        const initialContext = analysisContextRef.current;
        const validationAllowsUpload = initialContext.recordingFileValidation
            ? initialContext.recordingFileValidation.exists
                && initialContext.recordingFileValidation.readable
                && initialContext.recordingFileValidation.hasData
            : (
                initialContext.recordedSampleCount > 0
                || initialContext.recordingState === RecordingState.RECORDING
            );
        const canAttemptUpload = Boolean(initialContext.recordingFileKey) && validationAllowsUpload;
        if (!canAttemptUpload) { setUploadError('No telemetry data available for upload'); setShowRetryButton(false); return false; }
        if (!initialContext.recordingMetadata?.sessionName || !initialContext.recordingMetadata?.mapName || !initialContext.sessionGame || !auth?.userEmail) { setUploadError('Missing required session or user information'); setShowRetryButton(false); return false; }
        uploadInFlightRef.current = true; setIsUploading(true); setUploadProgress(0); setUploadStatus('Preparing telemetry data...'); setUploadError(null); setShowRetryButton(false);
        try {
            if (initialContext.recordingState === RecordingState.RECORDING) {
                await stopRecordingProcess('manual');
            }
            const uploadContext = analysisContextRef.current;
            const metadata: UploadReacingSessionInitDto = {
                sessionName: uploadContext.recordingMetadata!.sessionName,
                mapName: uploadContext.recordingMetadata!.mapName,
                carName: uploadContext.recordingMetadata!.carName,
                userId: auth?.userProfile.id || 'unknown',
                game_recorded_from: uploadContext.recordingMetadata!.gameRecordedFrom,
            };
            setUploadProgress(5); setUploadStatus('Initializing upload...');
            const initResp = await apiService.post('/racing-session/upload/init', metadata); if (!initResp.data) throw new Error('Failed to initialize upload');
            const { uploadId } = initResp.data as UploadRacingSessionInitReturnDto;
            let chunkIndex = 0;
            let uploadedRows = 0;
            const uploadChunk = async (chunk: unknown[], index: number) => {
                let retries = 3;
                let success = false;
                while (retries > 0 && !success) {
                    try {
                        const params = new URLSearchParams();
                        params.append('uploadId', uploadId);
                        await apiService.post(`/racing-session/upload/chunk?${params.toString()}`, { chunk, chunkIndex: index });
                        success = true;
                    } catch (err) {
                        console.warn(`Chunk ${index} upload failed, retrying... (${retries} attempts left)`, err);
                        retries--;
                        if (retries === 0) throw err;
                        const retryDelayMs = 1000 * (4 - retries);
                        await new Promise(resolve => setTimeout(resolve, retryDelayMs));
                    }
                }
                uploadedRows += chunk.length;
                setUploadStatus(`Uploaded ${uploadedRows.toLocaleString()} telemetry points...`);
            };
            setUploadStatus('Reading and uploading telemetry data...');
            const summary = await uploadContext.streamRecordedTelemetry(
                async (rows) => {
                    const currentIndex = chunkIndex++;
                    await uploadChunk(rows, currentIndex);
                },
                (_rowsRead, _totalRows, bytesRead, totalBytes) => {
                    const pct = totalBytes > 0 ? 10 + Math.floor((bytesRead / totalBytes) * 75) : 10;
                    setUploadProgress(Math.max(10, Math.min(85, pct)));
                },
            );
            if (summary.rowCount === 0) throw new Error('No telemetry data found to upload');
            setUploadProgress(92); setUploadStatus('Finalizing upload...');
            const final = new URLSearchParams(); final.append('uploadId', uploadId); await apiService.post(`/racing-session/upload/complete?${final.toString()}`, {});
            setUploadProgress(100); setUploadStatus('Upload completed successfully!');
            if (uploadContext.recordingFileKey) await cleanupTelemetryFile(uploadContext.recordingFileKey);
            uploadContext.clearPersistedDraft?.();
            setTimeout(() => { setIsUploading(false); setTimeout(() => returnToDetectionGate(), POST_SUCCESS_DIALOG_CLOSE_MS); }, POST_UPLOAD_RESET_DELAY_MS);
            uploadInFlightRef.current = false;
            return true;
        } catch (e: any) {
            const errorMessage = e?.message || (e instanceof Error ? e.message : 'Upload failed');
            setUploadError(errorMessage);
            setIsUploading(false);
            setShowRetryButton(true);
            uploadInFlightRef.current = false;
            return false;
        }
    }, [auth, cleanupTelemetryFile, returnToDetectionGate, stopRecordingProcess]);

    const handleDiscardSession = useCallback(async () => {
        if (uploadInFlightRef.current) return;
        const discardContext = analysisContextRef.current;
        const fileKey = discardContext.recordingFileKey;
        if (discardContext.recordingState === RecordingState.RECORDING) {
            try {
                await stopRecordingProcess('manual');
            } catch { /* the main process has already torn down the failed pipeline */ }
        }
        if (fileKey) await cleanupTelemetryFile(fileKey);
        discardContext.clearPersistedDraft?.();
        returnToDetectionGate();
    }, [cleanupTelemetryFile, returnToDetectionGate, stopRecordingProcess]);
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

    useEffect(() => {
        const control = { openUploadFlow: openUploadDialog };
        registerRecorderControl(control);
        return () => registerRecorderControl(null);
    }, [openUploadDialog, registerRecorderControl]);

    const uploadDialog = (
        <AlertDialog.Root open={uploadDialogOpen} onOpenChange={handleDialogOpenChange}>
            <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(e) => { if (isUploading) e.preventDefault(); }}>
                <AlertDialog.Title>Finish Live Session</AlertDialog.Title>
                <AlertDialog.Description size="2">
                    Upload the recorded data, discard it, or keep the current session open.
                </AlertDialog.Description>
                {(isUploading || showRetryButton || uploadError || analysisContext.restorationError) && (
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
                        {analysisContext.restorationError && <Text size="2" color="red" mt="2">{analysisContext.restorationError}</Text>}
                        {showRetryButton && !isUploading && <Flex mt="2" gap="2"><Button size="1" variant="outline" onClick={handleRetryUpload}>Retry Upload</Button></Flex>}
                    </Box>
                )}
                <Card size="4">
                    <Heading as="h3" size="6" trim="start" mb="5">Session <Text as="div" size="3" weight="bold" color="blue">{analysisContext.recordingMetadata?.sessionName || 'Unknown Session'}</Text></Heading>
                    <Grid columns="2" gapX="4" gapY="5">
                        <Box>
                            <Text as="div" size="2" mb="1" color="gray">Map</Text>
                            <Text as="div" size="3" mb="1" weight="bold">{analysisContext.recordingMetadata?.mapName || analysisContext.staticData.Static_track || 'Unknown Map'}</Text>
                            <Text as="div" size="2">Practice session</Text>
                        </Box>
                        <Box>
                            <Text as="div" size="2" mb="1" color="gray">Car</Text>
                            <Text as="div" size="3" weight="bold">{analysisContext.recordingMetadata?.carName || analysisContext.staticData.Static_car_model || 'Unknown Car'}</Text>
                        </Box>
                        <Flex direction="column" gap="1" gridColumn="1 / -1">
                            <Flex justify="between"><Text size="3" mb="1" weight="bold">Status</Text><Text size="2" color={uploadStatusColor}>{uploadStatusLabel}</Text></Flex>
                        </Flex>
                    </Grid>
                </Card>
                <Flex gap="3" mt="4" justify="end">
                    {!isUploading && uploadProgress < 100 && (
                        <>
                            <Button variant="outline" color="gray" onClick={closeUploadDialog}>Keep Session</Button>
                            <Button variant="outline" color="red" onClick={() => { void handleDiscardSession(); }}>Discard Session</Button>
                            <Button onClick={() => { void handleUpload(); }} disabled={isUploading || !hasRecordedData}>Upload Session</Button>
                        </>
                    )}
                    {isUploading && uploadProgress < 100 && (<Button variant="outline" disabled><Spinner size="1" />Uploading...</Button>)}
                    {!isUploading && uploadProgress === 100 && (<Button onClick={closeUploadDialog}>Close</Button>)}
                </Flex>
            </AlertDialog.Content>
        </AlertDialog.Root>
    );

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
                    <Button radius="full" color="red" onClick={() => {
                        void stopRecordingProcess('manual').catch((error) => {
                            setRecordingUnavailable(error instanceof Error ? error.message : String(error));
                        });
                    }}>
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
                        <Button radius="full" variant="outline" color="blue" disabled={!canRecord || isUploading} onClick={() => { void startRecording(); }}>
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
                        <Button radius="full" variant="outline" color="gray" onClick={() => { void handleDiscardSession(); }} disabled={isUploading}>
                            <span>Discard</span>
                        </Button>
                    </Flex>
                );
            default:
                return null;
        }
    }, [state, canRecord, startRecording, stopRecordingProcess, hasRecordedData, isUploading, openUploadDialog, handleDiscardSession]);

    const isRecording = state === RecordingState.RECORDING;
    const isPaused = state === RecordingState.HOLDING || state === RecordingState.RESUME_READY;
    const channelLabel =
        state === RecordingState.CHECKING ? 'TELEMETRY · SCANNING' :
        state === RecordingState.READY ? 'TELEMETRY · LIVE' :
        state === RecordingState.RECORDING ? 'REC · LIVE' :
        state === RecordingState.HOLDING ? 'REC · PAUSED' :
        state === RecordingState.RESUME_READY ? 'REC · STANDBY' :
        state === RecordingState.UPLOAD_READY ? 'REC · STOPPED' :
        'TELEMETRY';
    const channelMod =
        isRecording ? 'live-recording-bar__channel--rec' :
        isPaused ? 'live-recording-bar__channel--paused' :
        state === RecordingState.READY ? 'live-recording-bar__channel--live' :
        state === RecordingState.UPLOAD_READY ? 'live-recording-bar__channel--stopped' :
        '';
    if (!recorderHost) return null;

    return createPortal(
        <Box className={`live-recording-bar ${isRecording ? 'live-recording-bar--rec' : ''}`} position="absolute" left="0" right="0" bottom="0" mb="5" height="64px" style={{ marginLeft: 'max(24px, 10%)', marginRight: 'max(24px, 10%)' }}>
            <Flex height="100%" align="center" position="relative" overflow="hidden" className="live-recording-bar__inner">
                <Flex gap="3" align="center" p="3" style={{ minWidth: 0, flex: 1 }}>

                    <div className={`live-recording-bar__channel ${channelMod}`}>
                        <span className="live-recording-bar__channel-dot" />
                        {channelLabel}
                    </div>

                    {controlButtons}
                    {recordingUnavailable && (
                        <Text size="2" color="amber">{recordingUnavailable}</Text>
                    )}
                    {uploadDialog}

                </Flex>
                <div className="live-recording-bar__status">
                    <div className="live-recording-bar__status-row">
                        <span className="live-recording-bar__status-label">MAP</span>
                        <span className="live-recording-bar__status-value">{analysisContext.recordingMetadata?.mapName || analysisContext.staticData.Static_track || '—'}</span>
                    </div>
                    <div className="live-recording-bar__status-row">
                        <span className="live-recording-bar__status-label">SAMPLES</span>
                        <span className="live-recording-bar__status-value live-recording-bar__status-value--mono">
                            {analysisContext.recordedSampleCount.toLocaleString()}
                        </span>
                    </div>
                </div>

            </Flex>
        </Box>,
        recorderHost,
    );
}
