import { Card, Flex, Box, IconButton, Heading, Grid, Text, Slider, Spinner, AlertDialog, Button } from '@radix-ui/themes';
import { useContext, useEffect, useRef, useState, useMemo, useCallback, JSX } from 'react';
import { AnalysisContext } from './session-analysis';
import { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto, RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useAuth } from 'hooks/AuthProvider';
import apiService from 'services/api.service';
import { PythonShellOptions } from 'services/pythonService';

enum RecordingState { CHECKING = 'CHECKING', READY = 'READY', RECORDING = 'RECORDING', UPLOAD_READY = 'UPLOAD_READY' }

const CHECK_SESSION_INTERVAL_MS = 2000;
const SESSION_CHECK_TIMEOUT_MS = 10000;
const UPLOAD_CHUNK_SIZE = 5;
const POST_UPLOAD_RESET_DELAY_MS = 1200;
const POST_SUCCESS_DIALOG_CLOSE_MS = 800;

export default function LiveAnalysisSessionRecording() {
    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();

    const [state, setState] = useState<RecordingState>(RecordingState.CHECKING);
    const isRecording = state === RecordingState.RECORDING;
    const canRecord = state === RecordingState.READY;

    const isCheckingRef = useRef(false);
    const lastCheckRef = useRef<number | null>(null);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [showRetryButton, setShowRetryButton] = useState(false);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const uploadInFlightRef = useRef(false);

    const icon = useMemo((): JSX.Element => {
        switch (state) {
            case RecordingState.CHECKING: return <Spinner size="3" />;
            case RecordingState.READY:
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" fill="currentcolor" viewBox="0 0 30 30" width="20" height="20" style={{ marginRight: -2 }}>
                        <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                    </svg>
                );
            case RecordingState.RECORDING:
                return <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 2C1.44772 2 1 2.44772 1 3V12C1 12.5523 1.44772 13 2 13H13C13.5523 13 14 12.5523 14 12V3C14 2.44772 13.5523 2 13 2H2ZM3 3H12V12H3V3Z" fill="currentColor" /></svg>;
            case RecordingState.UPLOAD_READY:
                return <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" /></svg>;
            default: return <Spinner size="3" />;
        }
    }, [state]);

    const stopInterval = useCallback(() => { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }, []);

    const sessionCheck = useCallback(async () => {
        if (state !== RecordingState.CHECKING || isCheckingRef.current) return;
        isCheckingRef.current = true; lastCheckRef.current = Date.now();
        const options: PythonShellOptions = { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [] };
        try {
            const { shellId } = await window.electronAPI.runPythonScript('ACCOneTimeMemoryExtractor.py', options);
            await new Promise<void>((resolve) => {
                let done = false;
                const timeout = setTimeout(() => { if (!done) { done = true; isCheckingRef.current = false; resolve(); } }, SESSION_CHECK_TIMEOUT_MS);
                const finish = () => { if (!done) { done = true; clearTimeout(timeout); isCheckingRef.current = false; resolve(); } };
                const handleMessage = (_id: number, message: string) => {
                    if (_id !== shellId) return;
                    try {
                        const obj = JSON.parse(message);
                        if (obj.Graphics?.status === ACC_STATUS.ACC_LIVE) {
                            analysisContext.setRecordedSessionStaticsData(obj.Static);
                            setState(RecordingState.READY);
                            stopInterval();
                        }
                    } catch { }
                    finish();
                };
                const handleEnd = (_id: number) => { if (_id === shellId) finish(); };
                window.electronAPI.OnPythonMessageOnce(handleMessage);
                window.electronAPI.onPythonEnd(handleEnd);
            });
        } catch { isCheckingRef.current = false; }
    }, [state, analysisContext, stopInterval]);

    const startInterval = useCallback(() => {
        if (intervalRef.current) return;
        intervalRef.current = setInterval(() => {
            if (state === RecordingState.CHECKING) {
                sessionCheck();
                if (lastCheckRef.current && Date.now() - lastCheckRef.current > SESSION_CHECK_TIMEOUT_MS * 1.5) isCheckingRef.current = false;
            }
        }, CHECK_SESSION_INTERVAL_MS);
    }, [sessionCheck, state]);

    useEffect(() => {
        if (state === RecordingState.CHECKING) { sessionCheck(); startInterval(); } else { stopInterval(); }
        return () => stopInterval();
    }, [state, sessionCheck, startInterval, stopInterval]);

    const startRecording = useCallback(async () => {
        if (!canRecord || !analysisContext.recordedSessioStaticsData) return;
        analysisContext.setMap(analysisContext.recordedSessioStaticsData.track || 'Unknown Track');
        const now = new Date();
        const filename = `acc_${now.getFullYear()}_${now.getMonth()}_${now.getDate()}_${now.getHours()}_${now.getMinutes()}_${now.getSeconds()}.csv`;
        const folder = '../session_recording';
        const options: PythonShellOptions = { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [folder, filename] };
        const script = 'ACCMemoryExtractor.py';
        analysisContext.setSession(prev => {
            if (!prev) {
                return { session_name: new Date().toString(), SessionId: '', map: analysisContext.mapSelected || 'Unknown Track', user_id: '', points: [], data: [], car: analysisContext.recordedSessioStaticsData.car_model || 'Unknown Car' } as RacingSessionDetailedInfoDto as any;
            }
            prev.session_name = new Date().toString();
            return prev;
        });
        setState(RecordingState.RECORDING);
        try {
            const { shellId } = await window.electronAPI.runPythonScript(script, options);
            const cleanup = window.electronAPI.onPythonMessage((incomingId: number, message: string) => {
                if (incomingId === shellId) {
                    try { const obj = JSON.parse(message); analysisContext.setLiveSessionData(obj); analysisContext.writeRecordedLiveSessionData(obj); } catch { }
                }
            });
            window.electronAPI.onPythonEnd((incomingId: number) => {
                if (incomingId === shellId) { setState(RecordingState.UPLOAD_READY); if (typeof cleanup === 'function') cleanup(); }
            });
        } catch { setState(RecordingState.READY); }
    }, [canRecord, analysisContext]);

    const manualStop = useCallback(() => { if (state === RecordingState.RECORDING) setState(RecordingState.UPLOAD_READY); }, [state]);

    const cleanupTelemetryFile = useCallback(async (filePath: string) => {
        try { const options: PythonShellOptions = { mode: 'text', pythonOptions: ['-u'], scriptPath: 'src/py-scripts', args: [filePath] }; await window.electronAPI.runPythonScript('delete_telemetry_file.py', options); } catch { }
    }, []);

    const resetToChecking = useCallback(() => {
        analysisContext.clearRecordingSession();
        uploadInFlightRef.current = false;
        setUploadProgress(0); setUploadStatus(''); setUploadError(null); setShowRetryButton(false); setUploadDialogOpen(false); setIsUploading(false);
        isCheckingRef.current = false; lastCheckRef.current = null; setState(RecordingState.CHECKING);
    }, [analysisContext]);

    const handleUpload = useCallback(async () => {
        if (uploadInFlightRef.current) return false;
        if (!analysisContext.sessionSelected?.session_name || !analysisContext.mapSelected || !auth?.userEmail) { setUploadError('Missing required session or user information'); return false; }
        uploadInFlightRef.current = true; setIsUploading(true); setUploadProgress(0); setUploadStatus('Reading telemetry data...'); setUploadError(null);
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
                    pct = Math.min( read / estimatedTotal, 1) * 40; // scale into 0-40
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
            return true;
        } catch (e) { setUploadError(e instanceof Error ? e.message : 'Upload failed'); setIsUploading(false); setShowRetryButton(true); uploadInFlightRef.current = false; return false; }
    }, [analysisContext, auth, cleanupTelemetryFile, resetToChecking]);

    const handleCancelUpload = useCallback(async () => { if (analysisContext.recordedSessionDataFilePath) await cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath); resetToChecking(); }, [analysisContext, cleanupTelemetryFile, resetToChecking]);
    const handleRetryUpload = useCallback(() => { setUploadError(null); setShowRetryButton(false); setUploadProgress(0); handleUpload(); }, [handleUpload]);

    return (
        <Box position="absolute" left="0" right="0" bottom="0" mb="5" height="64px" style={{ borderRadius: '100px', boxShadow: 'var(--shadow-6)', marginLeft: 200, marginRight: 200 }}>
            <Flex height="100%" justify="between" position="relative">
                <Flex gap="4" align="center" p="3">
                    {state !== RecordingState.UPLOAD_READY ? (
                        <IconButton radius="full" size="3" onClick={isRecording ? manualStop : startRecording} color={isRecording ? 'red' : 'blue'} disabled={state === RecordingState.CHECKING}>{icon}</IconButton>
                    ) : (
                        <AlertDialog.Root open={uploadDialogOpen} onOpenChange={(open) => { if (isUploading) return; setUploadDialogOpen(open); }}>
                            <AlertDialog.Trigger>
                                <IconButton radius="full" size="3" onClick={() => setUploadDialogOpen(true)}>
                                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" /></svg>
                                </IconButton>
                            </AlertDialog.Trigger>
                            <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(e) => { if (isUploading) e.preventDefault(); }}>
                                <AlertDialog.Title>Upload Racing Session</AlertDialog.Title>
                                <AlertDialog.Description size="2">Upload your recorded racing session data.</AlertDialog.Description>
                                {isUploading && (
                                    <Box my="4">
                                        <Flex justify="between" mb="2"><Text size="2" weight="medium">{uploadStatus}</Text><Text size="2" color="gray">{uploadProgress}%</Text></Flex>
                                        <Box width="100%" height="8px" style={{ backgroundColor: 'var(--gray-a5)', borderRadius: 'var(--radius-2)', overflow: 'hidden' }}>
                                            <Box height="100%" style={{ width: `${uploadProgress}%`, backgroundColor: uploadError ? 'var(--red-9)' : 'var(--blue-9)', transition: 'width 0.3s ease' }} />
                                        </Box>
                                        {uploadError && <Text size="2" color="red" mt="2">{uploadError}</Text>}
                                        {showRetryButton && <Flex mt="2" gap="2"><Button size="1" variant="outline" onClick={handleRetryUpload}>Retry Upload</Button></Flex>}
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
                                            <Flex justify="between"><Text size="3" mb="1" weight="bold">Status</Text><Text size="2" color={isUploading ? 'blue' : 'green'}>{isUploading ? 'Uploading...' : 'Ready to upload'}</Text></Flex>
                                        </Flex>
                                    </Grid>
                                </Card>
                                <Flex gap="3" mt="4" justify="end">
                                    {!isUploading && uploadProgress < 100 && (<><Button variant="outline" color="red" onClick={() => { handleCancelUpload(); setUploadDialogOpen(false); }}>Cancel</Button><Button onClick={() => handleUpload()} disabled={isUploading}>Upload Session</Button></>)}
                                    {isUploading && uploadProgress < 100 && (<Button variant="outline" disabled><Spinner size="1" />Uploading...</Button>)}
                                    {!isUploading && uploadProgress === 100 && (<Button onClick={() => setUploadDialogOpen(false)}>Close</Button>)}
                                </Flex>
                            </AlertDialog.Content>
                        </AlertDialog.Root>
                    )}
                    <Flex align="center" gap="4">
                        <IconButton color="gray" variant="ghost" radius="full" size="2">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30" fill="currentcolor" fillOpacity={0.7} width="20" height="20"><path d="M 20 4 L 20 7 L 8 7 C 4.6983746 7 2 9.6983746 2 13 A 1.0001 1.0001 0 1 0 4 13 C 4 10.779625 5.7796254 9 8 9 L 20 9 L 20 12 L 27 8 L 20 4 z M 26.984375 15.986328 A 1.0001 1.0001 0 0 0 26 17 C 26 19.220375 24.220375 21 22 21 L 10 21 L 10 18 L 3 22 L 10 26 L 10 23 L 22 23 C 25.301625 23 28 20.301625 28 17 A 1.0001 1.0001 0 0 0 26.984375 15.986328 z" /></svg>
                        </IconButton>
                    </Flex>
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