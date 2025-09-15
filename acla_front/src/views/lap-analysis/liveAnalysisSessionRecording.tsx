import { Card, Flex, Box, TextField, IconButton, Heading, Grid, Text, Slider, Avatar, Spinner, AlertDialog, Button } from '@radix-ui/themes';
import { JSX, useContext, useEffect, useRef, useState, useMemo, useCallback } from 'react';

import { Link } from 'react-router-dom';
import { CallbackFunction, PythonShellOptions } from 'services/pythonService';
import { AnalysisContext } from './session-analysis';
import { AllMapsBasicInfoListDto, MapOption, RacingSessionDetailedInfoDto, UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import { useAuth } from 'hooks/AuthProvider';
import { ACC_STATUS, ACCMemoeryTracks } from 'data/live-analysis/live-map-data';
import { Cross2Icon } from '@radix-ui/react-icons';
import { IpcRendererEvent } from 'electron';

// Button states enum
enum ButtonState {
    CHECKING = 'checking',
    READY_TO_START = 'ready_to_start',
    RECORDING = 'recording',
    UPLOAD_READY = 'upload_ready'
}


const LiveAnalysisSessionRecording = () => {
    // Constants
    const CHECK_SESSION_INTERVAL_MS = 2000;
    const SESSION_CHECK_TIMEOUT_MS = 10000;
    const UPLOAD_CHUNK_SIZE = 5;
    const POST_UPLOAD_RESET_DELAY_MS = 1200;
    const POST_SUCCESS_DIALOG_CLOSE_MS = 800;
    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();

    // Persistent ref for tracking if a session check is in progress (avoids re-renders)
    const isCheckingLiveSessionRef = useRef(false);
    const lastCheckStartRef = useRef<number | null>(null);

    //stores the state of a game session
    const [hasValidLiveSession, setValidLiveSession] = useState(ACC_STATUS.ACC_OFF);

    //store the recording state of a game recording
    const [isRecording, setIsRecording] = useState(false);

    //store the ending state of the game recording
    const [isRecordEnded, setIsRecorEnded] = useState(false);

    //store the shell id of the script for checking game live session
    const [checkSessionScriptShellId, setCheckSessionScriptShellId] = useState(-1);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // Button state management
    const [buttonState, setButtonState] = useState<ButtonState>(ButtonState.CHECKING);

    // Upload progress state
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState<string>('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [showRetryButton, setShowRetryButton] = useState(false);
    // Keep upload dialog open until upload completes
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

    // Centralized button component generator
    const getPrimaryButtonComponent = (state: ButtonState): JSX.Element => {
        switch (state) {
            case ButtonState.CHECKING:
                return <Spinner size="3" />;

            case ButtonState.READY_TO_START:
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" fill="currentcolor" viewBox="0 0 30 30" width="20" height="20" style={{ marginRight: -2 }}>
                        <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                    </svg>
                );

            case ButtonState.RECORDING:
                return (
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 2C1.44772 2 1 2.44772 1 3V12C1 12.5523 1.44772 13 2 13H13C13.5523 13 14 12.5523 14 12V3C14 2.44772 13.5523 2 13 2H2ZM3 3H12V12H3V3Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path>
                    </svg>
                );

            case ButtonState.UPLOAD_READY:
                return (
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path>
                    </svg>
                );

            default:
                return <Spinner size="3" />;
        }
    };

    const primaryButton = useMemo(() => getPrimaryButtonComponent(buttonState), [buttonState]);


    useEffect(() => {
        if (hasValidLiveSession !== ACC_STATUS.ACC_LIVE && !isCheckingLiveSessionRef.current) {
            startCheckingLiveSessionInterval();
        }
        return () => {
            stopCheckingLiveSessionInterval();
            if (analysisContext.recordedSessionDataFilePath && !isRecording) {
                cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath).catch(console.error);
            }
        };
        // intentionally omitting dependencies to mimic componentDidMount while using stable callbacks
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /**
     * check acc memory once and see if there is a valid session running
     */
    const CheckSessionValid = useCallback(async () => {

        if (isCheckingLiveSessionRef.current) {
            return;
        }
    isCheckingLiveSessionRef.current = true;
    lastCheckStartRef.current = Date.now();

        //setup for running a new python in main process
        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: 'src/py-scripts',
            args: []
        } as PythonShellOptions;

        //script to run
        const script = 'ACCOneTimeMemoryExtractor.py';

        try {
            //running the script in the main process (electron.js) instead this renderer process. we will wait for the result to comeback to onPythonMessage().
            const { shellId } = await window.electronAPI.runPythonScript(script, options);

            return new Promise<void>((resolve, reject) => {

                // Track if the promise has been resolved
                let isResolved = false;

                // Timeout to prevent hanging promises
                const timeout = setTimeout(() => {
                    if (!isResolved) {
                        isResolved = true;
                        isCheckingLiveSessionRef.current = false;
                        reject(new Error('Session check timeout'));
                    }
                }, SESSION_CHECK_TIMEOUT_MS); // timeout

                const cleanup = () => {
                    clearTimeout(timeout);
                };

                //create a function to handle the return of the python script
                const handleMessage = (returnedShellId: number, message: string) => {
                    //all scripts will call this function, we must identify the right id 
                    if (shellId === returnedShellId) {
                        try {
                            const obj = JSON.parse(message);

                            //if the script print out valid session map 
                            if (obj.Graphics && obj.Graphics.status === ACC_STATUS.ACC_LIVE) {
                                console.log("Found valid live session!");
                                //found a valid live session, stop the checking process
                                stopCheckingLiveSessionInterval();
                                setValidLiveSession(obj.Graphics.status);

                                //set primary button to start button
                                setButtonState(ButtonState.READY_TO_START);

                                //set the static data too, so we can use it later.
                                console.log("Setting static data:", obj);
                                analysisContext.setRecordedSessionStaticsData(obj.Static);
                                // Resolve early and clear flag
                                if (!isResolved) {
                                    isResolved = true;
                                    cleanup();
                                    isCheckingLiveSessionRef.current = false;
                                    resolve();
                                }
                            } else {
                                console.log("Session status is not live:", obj.Graphics?.status);
                                // Not live; allow interval to trigger another check
                                if (!isResolved) {
                                    isResolved = true;
                                    cleanup();
                                    isCheckingLiveSessionRef.current = false;
                                    resolve();
                                }
                            }
                        } catch (error) {
                            console.error('Error parsing session data:', error);
                            console.error('Raw message that failed to parse:', message);

                            // If we couldn't parse the session data, we should reject the promise
                            if (!isResolved) {
                                isResolved = true;
                                cleanup();
                                isCheckingLiveSessionRef.current = false;
                                reject(error);
                            }
                        }
                    }
                };

                // handle the event when python script for session recording is terminated
                const handleScriptEnd = (returnedShellId: number) => {
                    if (shellId === returnedShellId && !isResolved) {
                        isResolved = true;
                        cleanup();
                        isCheckingLiveSessionRef.current = false;
                        resolve();
                    }
                };

                setCheckSessionScriptShellId(shellId);

                // Set up listener for Python messages
                window.electronAPI.OnPythonMessageOnce(handleMessage);
                window.electronAPI.onPythonEnd(handleScriptEnd);
            });

        } catch (error) {
            console.error('Error running Python script:', error);
            isCheckingLiveSessionRef.current = false;
            throw error;
        }
    }, [analysisContext]);

    /**
     * run the python script and start record the session
     * @returns 
     */
    const StartRecording = useCallback(async () => {
        //if no valid live sesssion, we dont do anything
        if (hasValidLiveSession != ACC_STATUS.ACC_LIVE) {
            console.log("No valid live session, current status:", hasValidLiveSession);
            return;
        }
        console.debug('[Recording] StartRecording invoked');

        // Check if we have valid static data
        if (!analysisContext.recordedSessioStaticsData) {
            console.error("No static data available from session check");
            return;
        };

        //use the track name directly from the static data
        const trackname: string = analysisContext.recordedSessioStaticsData.track || "Unknown Track";
        const carname: string = analysisContext.recordedSessioStaticsData.car_model || "Unknown Car";
        analysisContext.setMap(trackname);

        const currentDate = new Date();
        const filename: string = `acc_${currentDate.getFullYear()}_${currentDate.getMonth()}_${currentDate.getDate()}_${currentDate.getHours()}_${currentDate.getMinutes()}_${currentDate.getSeconds()}.csv`;
        const folder: string = '../session_recording';

        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: 'src/py-scripts',
            args: [folder, filename]
        } as PythonShellOptions;
        const script = 'ACCMemoryExtractor.py';

        analysisContext.setSession((prev) => {
            if (!prev) {
                const newSession: RacingSessionDetailedInfoDto = {
                    session_name: new Date().toString(),
                    SessionId: '',
                    map: analysisContext.mapSelected || "Unknown Track",
                    user_id: '',
                    points: [],
                    data: [],
                    car: analysisContext.recordedSessioStaticsData.car_model || "Unknown Car"
                };
                return newSession
            }
            prev.session_name = new Date().toString()
            return prev;
        });

        // Set the primary button to the recording icon
        setButtonState(ButtonState.RECORDING);
    console.debug('[Recording] Button state -> RECORDING');

        try {
            //running the script in the main process (electron.js) instead this renderer process
            const { shellId } = await window.electronAPI.runPythonScript(script, options);

            const messageCleanup = window.electronAPI.onPythonMessage((incomingScriptShellId: number, message: string) => {

                if (shellId == incomingScriptShellId) { //check return result of recording script
                    try {
                        const obj = JSON.parse(message);
                        analysisContext.setLiveSessionData(obj);
                        // Write telemetry data to file instead of accumulating in memory
                        analysisContext.writeRecordedLiveSessionData(obj);

                    } catch (error) {
                        console.error("Error parsing Python message:", error);
                        console.error("Raw message:", message);
                    }
                }
            });

            window.electronAPI.onPythonEnd((incomingScriptShellId: number) => {
                if (shellId == incomingScriptShellId) {// session recording is terminated
                    console.log("Recording script ended, stopping recording...");
                    setIsRecording(false);
                    setIsRecorEnded(true);
                    setButtonState(ButtonState.UPLOAD_READY);
                    console.debug('[Recording] Transition to UPLOAD_READY');
                    // Clean up the message listener
                    if (typeof messageCleanup === 'function') {
                        messageCleanup();
                    }
                }
            });

            setIsRecording(true);

        } catch (error) {
            console.error("Error starting recording:", error);
            setButtonState(ButtonState.READY_TO_START); // Reset button state on error
            console.debug('[Recording] Error, reverting to READY_TO_START');
        }
    }, [hasValidLiveSession, analysisContext]);

    /**
     * start checking the valid live session in a interval
     */
    const startCheckingLiveSessionInterval = useCallback(() => {
        setButtonState(ButtonState.CHECKING);
    console.debug('[SessionCheck] Interval tick');

        //check every 2 sec by using a python script
        intervalRef.current = setInterval(async () => {
            // Prevent overlapping checks
            if (isCheckingLiveSessionRef.current) {
                return;
            }

            // If we're currently recording, check if the session is still live
            if (isRecording) {
                try {
                    await CheckSessionValid();
                    // If session is no longer live during recording, stop recording
                    if (hasValidLiveSession !== ACC_STATUS.ACC_LIVE) {
                        console.log("Session no longer live during recording, stopping...");
                        setIsRecording(false);
                        setIsRecorEnded(true);
                        setButtonState(ButtonState.UPLOAD_READY);
                        // Note: We don't clean up the file here as user may want to upload
                    }
                } catch (error) {
                    console.error("Error checking session during recording:", error);
                }
                return; // Skip normal session checking when recording
            }

            try {
                await CheckSessionValid();
                console.debug('[SessionCheck] CheckSessionValid executed while recording');
                console.debug('[SessionCheck] CheckSessionValid executed');
            } catch (error) {
                console.error("Error in CheckSessionValid:", error);
                // Don't stop the interval on error, just log it and continue
                isCheckingLiveSessionRef.current = false; // Reset the flag in case of error
            }
            // Stale guard: if flag stuck beyond timeout, reset
            if (lastCheckStartRef.current && Date.now() - lastCheckStartRef.current > SESSION_CHECK_TIMEOUT_MS * 1.5) {
                console.warn('Stale session check detected; resetting flag');
                isCheckingLiveSessionRef.current = false;
            }
        }, CHECK_SESSION_INTERVAL_MS);
    }, [CheckSessionValid, hasValidLiveSession, isRecording]);

    // Stop the interval
    const stopCheckingLiveSessionInterval = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    // Function to clean up telemetry file
    const cleanupTelemetryFile = async (filePath: string) => {
        try {
            const options = {
                mode: 'text',
                pythonOptions: ['-u'],
                scriptPath: 'src/py-scripts',
                args: [filePath]
            } as PythonShellOptions;

            await window.electronAPI.runPythonScript('delete_telemetry_file.py', options);
            console.log(`Cleaned up telemetry file: ${filePath}`);
        } catch (error) {
            console.error('Error cleaning up telemetry file:', error);
            // Don't throw error as cleanup failure shouldn't block the main flow
        }
    };

    //after a session is determined as terminated, and user selected to upload the data, we do it here
    const uploadInFlightRef = useRef(false);
    const handleUpload = useCallback(async () => {
        if (uploadInFlightRef.current) return false;
        if (!analysisContext.sessionSelected?.session_name || !analysisContext.mapSelected || !auth?.userEmail) {
            setUploadError('Missing required session or user information');
            return false;
        }
        uploadInFlightRef.current = true;
        setIsUploading(true);
        setUploadError(null);
        setUploadProgress(0);
        setUploadStatus('Reading telemetry data...');

        try {
            // Read telemetry data from file instead of memory
            console.log('Attempting to read telemetry data from:', analysisContext.recordedSessionDataFilePath);
            console.log('Recorded telemetry data count:', analysisContext.recordedTelemetryDataCount);
            const data = await analysisContext.readRecordedSessionData();
            console.log('Read telemetry data:', data?.length, 'points');

            if (!data || data.length === 0) {
                console.log('No telemetry data found. File path:', analysisContext.recordedSessionDataFilePath, 'Count:', analysisContext.recordedTelemetryDataCount);
                throw new Error('No telemetry data found to upload');
            }

            setUploadProgress(10);
            setUploadStatus(`Processing ${data.length} telemetry points...`);

            //send by chunks
            const chunks: any[] = [];
            const chunkSize = UPLOAD_CHUNK_SIZE;
            const metadata = {
                sessionName: analysisContext.sessionSelected?.session_name,
                mapName: analysisContext.mapSelected,
                carName: analysisContext.recordedSessioStaticsData.car_model || "Unknown Car",
                userId: auth?.userProfile.id || "unknown",
            } as UploadReacingSessionInitDto;

            //separate recorded data into chunks
            for (let i = 0; i < data.length; i += chunkSize) {
                chunks.push(data.slice(i, i + chunkSize));
            }

            setUploadProgress(20);
            setUploadStatus('Initializing upload...');

            // First send metadata
            const initResponse = await apiService.post('/racing-session/upload/init', metadata);

            if (!initResponse.data) {
                throw new Error('Failed to initialize upload');
            }
            const { uploadId } = initResponse.data as UploadRacingSessionInitReturnDto;

            setUploadProgress(30);
            setUploadStatus(`Uploading ${chunks.length} chunks...`);

            // Then send chunks with progress tracking
            for (let i = 0; i < chunks.length; i++) {
                const url = '/racing-session/upload/chunk';
                const params = new URLSearchParams();
                params.append('uploadId', uploadId);

                await apiService.post(`${url}?${params.toString()}`, {
                    chunk: chunks[i],
                    chunkIndex: i
                });

                // Update progress based on chunk completion
                const chunkProgress = Math.floor(40 + (i + 1) / chunks.length * 50);
                setUploadProgress(chunkProgress);
                setUploadStatus(`Uploading chunk ${i + 1} of ${chunks.length}...`);
            }

            setUploadProgress(90);
            setUploadStatus('Finalizing upload...');

            // Finalize
            const finalUrl = '/racing-session/upload/complete';
            const finalParams = new URLSearchParams();
            finalParams.append('uploadId', uploadId);

            await apiService.post(`${finalUrl}?${finalParams.toString()}`, {});

            setUploadProgress(100);
            setUploadStatus('Upload completed successfully!');

            // Clean up the telemetry file after successful upload
            if (analysisContext.recordedSessionDataFilePath) {
                await cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath);
            }

            // Clear the recording session state
            analysisContext.clearRecordingSession();

            // Wait a moment to show completion then reset
            setTimeout(() => {
                setIsUploading(false);
                setTimeout(() => {
                    setUploadDialogOpen(false);
                    reEnterCheckingValidSession();
                }, POST_SUCCESS_DIALOG_CLOSE_MS);
            }, POST_UPLOAD_RESET_DELAY_MS);

            return true;

        } catch (error) {
            console.error('Upload error:', error);
            setUploadError(error instanceof Error ? error.message : 'Upload failed');
            setIsUploading(false);
            setShowRetryButton(true);
            uploadInFlightRef.current = false;
            return false;
        }
    }, [analysisContext, auth, cleanupTelemetryFile]);

    function reEnterCheckingValidSession() {
        // Note: We don't clean up telemetry file here as it might be needed for upload

        // Reset all relevant state
        console.debug('[ReEnter] resetting session state');
        isCheckingLiveSessionRef.current = false;
        lastCheckStartRef.current = null;
        setIsRecording(false);
        setIsRecorEnded(false);
        setValidLiveSession(ACC_STATUS.ACC_OFF);
        setCheckSessionScriptShellId(-1);
        setButtonState(ButtonState.CHECKING);
        analysisContext.setSession(null);
        // Reset the telemetry file path for new session
        analysisContext.setRecordedSessionDataFilePath(null);

        // Reset upload states
        setIsUploading(false);
        setUploadProgress(0);
        setUploadStatus('');
        setUploadError(null);
        setShowRetryButton(false);

        // Stop any existing interval before starting a new one
        stopCheckingLiveSessionInterval();

        // Start checking again
        startCheckingLiveSessionInterval();
    }    // Function to manually stop recording
    const stopRecording = useCallback(() => {
        console.log("Manually stopping recording...");
        setIsRecording(false);
        setIsRecorEnded(true);
        setButtonState(ButtonState.UPLOAD_READY);
    console.debug('[Recording] Manual stop -> UPLOAD_READY');

        // Don't clear the recording session yet - we still need the file path for upload
        // analysisContext.clearRecordingSession(); // This will be called after upload or cancel
    }, []);

    // Function to handle cancel upload
    const handleCancelUpload = useCallback(async () => {
        // Clean up the telemetry file when canceling
        if (analysisContext.recordedSessionDataFilePath) {
            await cleanupTelemetryFile(analysisContext.recordedSessionDataFilePath);
        }

        // Clear the recording session state
        analysisContext.clearRecordingSession();

        setIsUploading(false);
        setUploadProgress(0);
        setUploadStatus('');
        setUploadError(null);
        setShowRetryButton(false);
        reEnterCheckingValidSession();
    }, [analysisContext]);

    // Function to retry upload
    const handleRetryUpload = useCallback(() => {
        setUploadError(null);
        setShowRetryButton(false);
        setUploadProgress(0);
        handleUpload();
    }, [handleUpload]);

    return (

        <Box
            position="absolute"
            left="0"
            right="0"
            bottom="0"
            mb="5"
            height="64px"
            style={{
                borderRadius: "100px",
                boxShadow: "var(--shadow-6)",
                marginLeft: 200,
                marginRight: 200,
            }}
        >

            <Flex height="100%" justify="between" position="relative">
                <Flex gap="4" align="center" p="3">
                    {!isRecordEnded ?
                        //if record hasnt been initilized or stil recording
                        <IconButton
                            radius="full"
                            size="3"
                            onClick={isRecording ? stopRecording : StartRecording}
                            color={isRecording ? "red" : "blue"}
                        >
                            {primaryButton}
                        </IconButton> :

                        //record ended
                        <AlertDialog.Root
                            open={uploadDialogOpen}
                            onOpenChange={(open) => {
                                if (isUploading) return; // prevent closing during upload
                                setUploadDialogOpen(open);
                            }}
                        >
                            <AlertDialog.Trigger>
                                <IconButton radius="full" size="3" onClick={() => setUploadDialogOpen(true)}>
                                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path></svg>
                                </IconButton>
                            </AlertDialog.Trigger>
                            <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(e) => { if (isUploading) { e.preventDefault(); } }}>
                                <AlertDialog.Title>Upload Racing Session</AlertDialog.Title>
                                <AlertDialog.Description size="2">
                                    Upload your recorded racing session data to the server for analysis and storage.
                                </AlertDialog.Description>

                                {/* Upload Progress Section */}
                                {isUploading && (
                                    <Box my="4">
                                        <Flex justify="between" mb="2">
                                            <Text size="2" weight="medium">{uploadStatus}</Text>
                                            <Text size="2" color="gray">{uploadProgress}%</Text>
                                        </Flex>
                                        <Box
                                            width="100%"
                                            height="8px"
                                            style={{
                                                backgroundColor: "var(--gray-a5)",
                                                borderRadius: "var(--radius-2)",
                                                overflow: "hidden"
                                            }}
                                        >
                                            <Box
                                                height="100%"
                                                style={{
                                                    width: `${uploadProgress}%`,
                                                    backgroundColor: uploadError ? "var(--red-9)" : "var(--blue-9)",
                                                    borderRadius: "var(--radius-2)",
                                                    transition: "width 0.3s ease"
                                                }}
                                            />
                                        </Box>
                                        {uploadError && (
                                            <Text size="2" color="red" mt="2">
                                                {uploadError}
                                            </Text>
                                        )}
                                        {showRetryButton && (
                                            <Flex mt="2" gap="2">
                                                <Button size="1" variant="outline" onClick={handleRetryUpload}>
                                                    Retry Upload
                                                </Button>
                                            </Flex>
                                        )}
                                    </Box>
                                )}

                                <Card size="4">
                                    <Heading as="h3" size="6" trim="start" mb="5">
                                        Session {" "}
                                        <Text as="div" size="3" weight="bold" color="blue">
                                            {analysisContext.sessionSelected?.session_name || "Unknown Session"}
                                        </Text>
                                    </Heading>

                                    <Grid columns="2" gapX="4" gapY="5">
                                        <Box>
                                            <Text as="div" size="2" mb="1" color="gray">
                                                Map
                                            </Text>
                                            <Text as="div" size="3" mb="1" weight="bold">
                                                {analysisContext.mapSelected || "Unknown Map"}
                                            </Text>
                                            <Text as="div" size="2">
                                                Practice session
                                            </Text>
                                        </Box>

                                        <Box>
                                            <Text as="div" size="2" mb="1" color="gray">
                                                Car
                                            </Text>
                                            <Text as="div" size="3" weight="bold">
                                                {analysisContext.recordedSessioStaticsData?.car_model || "Unknown Car"}
                                            </Text>
                                        </Box>

                                        <Flex direction="column" gap="1" gridColumn="1 / -1">
                                            <Flex justify="between">
                                                <Text size="3" mb="1" weight="bold">
                                                    Status
                                                </Text>
                                                <Text size="2" color={isUploading ? "blue" : "green"}>
                                                    {isUploading ? "Uploading..." : "Ready to upload"}
                                                </Text>
                                            </Flex>
                                        </Flex>
                                    </Grid>
                                </Card>

                                <Flex gap="3" mt="4" justify="end">
                                    {!isUploading && uploadProgress < 100 && (
                                        <>
                                            <Button
                                                variant="outline"
                                                color="red"
                                                onClick={() => {
                                                    handleCancelUpload();
                                                    setUploadDialogOpen(false);
                                                }}
                                            >
                                                Cancel
                                            </Button>
                                            <Button onClick={() => handleUpload()} disabled={isUploading}>
                                                Upload Session
                                            </Button>
                                        </>
                                    )}
                                    {isUploading && uploadProgress < 100 && (
                                        <Button variant="outline" disabled>
                                            <Spinner size="1" />
                                            Uploading...
                                        </Button>
                                    )}
                                    {!isUploading && uploadProgress === 100 && (
                                        <Button onClick={() => setUploadDialogOpen(false)}>
                                            Close
                                        </Button>
                                    )}
                                </Flex>
                            </AlertDialog.Content>
                        </AlertDialog.Root>


                    }
                    <Flex align="center" gap="4">
                        <IconButton
                            color="gray"
                            variant="ghost"
                            radius="full"
                            size="2"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 30 30"
                                fill="currentcolor"
                                fillOpacity={0.7}
                                width="20"
                                height="20"
                            >
                                <path d="M 20 4 L 20 7 L 8 7 C 4.6983746 7 2 9.6983746 2 13 A 1.0001 1.0001 0 1 0 4 13 C 4 10.779625 5.7796254 9 8 9 L 20 9 L 20 12 L 27 8 L 20 4 z M 26.984375 15.986328 A 1.0001 1.0001 0 0 0 26 17 C 26 19.220375 24.220375 21 22 21 L 10 21 L 10 18 L 3 22 L 10 26 L 10 23 L 22 23 C 25.301625 23 28 20.301625 28 17 A 1.0001 1.0001 0 0 0 26.984375 15.986328 z" />
                            </svg>
                        </IconButton>
                    </Flex>
                </Flex>

                <Flex align="center" gap="3">

                    <Flex align="center" gap="3">
                        <Box>
                            <Text size="1" as="div" weight="medium">
                                Racing Map Name Here
                            </Text>
                            <Text size="1" as="div" color="gray" mb="2">
                                Partice Session
                            </Text>

                            <Box
                                position="relative"
                                height="4px"
                                width="320px"
                                style={{
                                    backgroundColor: "var(--gray-a5)",
                                    borderRadius: "var(--radius-1)",
                                }}
                            >
                                <Box
                                    position="absolute"
                                    height="4px"
                                    width="64px"
                                    style={{
                                        borderRadius: "var(--radius-1)",
                                        backgroundColor: "var(--gray-a9)",
                                    }}
                                />
                                <Box position="absolute" top="0" right="0" mt="-28px">
                                    <Text size="1" color="gray">
                                        0:58 / Lap 2
                                    </Text>
                                </Box>
                            </Box>
                        </Box>
                    </Flex>


                </Flex>

                <Flex align="center" gap="2" p="5">
                    <Slider
                        defaultValue={[80]}
                        variant="soft"
                        color="gray"
                        radius="full"
                        size="2"
                        style={{ width: 80 }}
                    />

                </Flex>
            </Flex>
        </Box>


    );

};







export default LiveAnalysisSessionRecording;