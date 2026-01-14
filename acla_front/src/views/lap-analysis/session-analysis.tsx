import './session-analysis.css';

import {
    Box,
    Tabs
} from "@radix-ui/themes";

import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import SessionAnalysisSplit from './sessionAnalysis/session-analysis-split';
import { useEnvironment } from 'contexts/EnvironmentContext';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { PythonShellOptions } from 'services/pythonService';
import { createPythonStreamSession, PythonStreamEvent, PythonStreamSession } from 'services/pythonStreaming';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { AnalysisContext } from './analysis-context';

const normalizeAccStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) {
        return null;
    }

    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

const TELEMETRY_WRITE_TIMEOUT_MS = 6000;

type TelemetryWriterEvent = {
    status?: string;
    request_id?: string;
    message?: string;
    written?: number;
    [key: string]: unknown;
};

type PendingTelemetryWrite = {
    resolve: () => void;
    reject: (error: Error) => void;
    timeoutId: number;
};

const SessionAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [liveData, setLiveData] = useState({});
    const [TelemetryDataLiveStatus, setTelemetryDataLiveStatus] = useState<ACC_STATUS | null>(null);
    const [recordedSessioStaticsData, setRecordedSessionStaticsData] = useState({});
    const [recordedSessionDataFilePath, setRecordedSessionDataFilePath] = useState<string | null>(null);
    const [recordedTelemetryDataCount, setRecordedTelemetryDataCount] = useState<number>(0);
    const [activeVisualizations, setActiveVisualizations] = useState<VisualizationInstance[]>([]);
    const [latestGuidanceMessage, setLatestGuidanceMessage] = useState<string | null>(null);

    // Use ref to persist file path during recording to prevent state reset issues
    const recordingFilePathRef = useRef<string | null>(null);
    const writeQueueRef = useRef<Promise<void>>(Promise.resolve());
    const telemetryWriterSessionRef = useRef<PythonStreamSession<TelemetryWriterEvent> | null>(null);
    const telemetryWriterCleanupRef = useRef<(() => void) | null>(null);
    const telemetryWriterFilePathRef = useRef<string | null>(null);
    const telemetryWriterPendingRef = useRef<Map<string, PendingTelemetryWrite>>(new Map());
    const telemetryWriterSequenceRef = useRef(0);

    const environment = useEnvironment();

    const disposeTelemetryWriter = useCallback(async ({ force = false }: { force?: boolean } = {}) => {
        const cleanup = telemetryWriterCleanupRef.current;
        if (cleanup) {
            cleanup();
            telemetryWriterCleanupRef.current = null;
        }

        const session = telemetryWriterSessionRef.current;
        telemetryWriterSessionRef.current = null;
        telemetryWriterFilePathRef.current = null;

        for (const [requestId, pending] of Array.from(telemetryWriterPendingRef.current.entries())) {
            telemetryWriterPendingRef.current.delete(requestId);
            pending.reject(new Error('Telemetry writer disposed'));
        }

        if (session) {
            try {
                await session.dispose({ force });
            } catch (error) {
                console.warn('Failed to dispose telemetry writer session', error);
            }
        }
    }, []);

    const handleTelemetryWriterEvent = useCallback((event: PythonStreamEvent<TelemetryWriterEvent>) => {
        if (!event) {
            return;
        }

        const status = typeof event.status === 'string' ? event.status : '';
        const requestId = typeof event.request_id === 'string' ? event.request_id : undefined;

        if (status === 'ok' && requestId) {
            const pending = telemetryWriterPendingRef.current.get(requestId);
            if (pending) {
                telemetryWriterPendingRef.current.delete(requestId);
                pending.resolve();
            }
            return;
        }

        if (status === 'error') {
            const error = new Error(typeof event.message === 'string' ? event.message : 'Telemetry writer error');
            if (requestId) {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.reject(error);
                }
            } else {
                console.error('Telemetry writer emitted error without request id', event);
                for (const [pendingId, pending] of Array.from(telemetryWriterPendingRef.current.entries())) {
                    telemetryWriterPendingRef.current.delete(pendingId);
                    pending.reject(error);
                }
            }
            return;
        }

        if (status === 'shutdown') {
            if (requestId) {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.resolve();
                }
            }
            void disposeTelemetryWriter({ force: true });
            return;
        }
    }, [disposeTelemetryWriter]);

    const ensureTelemetryWriter = useCallback(async (filePath: string) => {
        if (telemetryWriterSessionRef.current && telemetryWriterFilePathRef.current === filePath) {
            const existingSession = telemetryWriterSessionRef.current;
            await existingSession.waitUntilReady();
            return existingSession;
        }

        await disposeTelemetryWriter({ force: true });

        try {
            const session = await createPythonStreamSession<TelemetryWriterEvent>({
                scriptName: 'append_telemetry_data.py',
                pythonOptions: {
                    mode: 'text',
                    pythonOptions: ['-u'],
                    scriptPath: 'src/py-scripts',
                    args: [filePath]
                },
                readyTimeoutMs: 8000
            });

            telemetryWriterSessionRef.current = session;
            telemetryWriterFilePathRef.current = filePath;
            telemetryWriterCleanupRef.current = session.onMessage(handleTelemetryWriterEvent);

            await session.waitUntilReady();
            return session;
        } catch (error) {
            await disposeTelemetryWriter({ force: true });
            throw error;
        }
    }, [disposeTelemetryWriter, handleTelemetryWriterEvent]);

    const finalizeRecordingWrites = useCallback(async () => {
        try {
            await writeQueueRef.current;
        } catch (error) {
            console.warn('Telemetry write queue rejected during finalization', error);
        } finally {
            writeQueueRef.current = Promise.resolve();
        }

        await disposeTelemetryWriter({ force: false });
    }, [disposeTelemetryWriter]);

    // File-based telemetry data functions
    const writeRecordedLiveSessionData = async (data: any): Promise<void> => {
        const enqueueWrite = async () => {
            let currentFilePath = recordingFilePathRef.current || recordedSessionDataFilePath;

            if (!currentFilePath) {
                const timestamp = new Date().getTime();
                const sessionId = sessionSelected?.SessionId || 'unknown';
                currentFilePath = `../session_recording/temp/telemetry_${sessionId}_${timestamp}.jsonl`;
                setRecordedSessionDataFilePath(currentFilePath);
                recordingFilePathRef.current = currentFilePath;
                setRecordedTelemetryDataCount(0);
            }

            const session = await ensureTelemetryWriter(currentFilePath);
            telemetryWriterSequenceRef.current += 1;
            const requestId = `telemetry-append-${Date.now()}-${telemetryWriterSequenceRef.current}`;

            let resolveAck!: () => void;
            let rejectAck!: (error: Error) => void;
            const ackPromise = new Promise<void>((resolve, reject) => {
                resolveAck = resolve;
                rejectAck = reject;
            });

            const timeoutId = window.setTimeout(() => {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.reject(new Error('Telemetry writer append timed out'));
                }
            }, TELEMETRY_WRITE_TIMEOUT_MS);

            telemetryWriterPendingRef.current.set(requestId, {
                resolve: () => {
                    window.clearTimeout(timeoutId);
                    resolveAck();
                },
                reject: (error: Error) => {
                    window.clearTimeout(timeoutId);
                    rejectAck(error);
                },
                timeoutId
            });

            try {
                await session.send('append', { data }, requestId);
            } catch (error) {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.reject(error instanceof Error ? error : new Error(String(error)));
                }
                throw error;
            }

            await ackPromise;
            setRecordedTelemetryDataCount(prev => prev + 1);
        };

        const nextWrite = writeQueueRef.current.then(enqueueWrite);

        writeQueueRef.current = nextWrite
            .catch((error) => {
                if (error instanceof Error && error.message === 'Telemetry writer disposed') {
                    return;
                }
                console.error('Telemetry write failed', error);
            })
            .then(() => undefined);

        return nextWrite.catch((error) => {
            if (error instanceof Error && error.message === 'Telemetry writer disposed') {
                return;
            }
            throw error;
        });
    };

    const readRecordedSessionData = async (onProgress?: (read: number, total: number | null, bytesRead?: number, totalBytes?: number) => void): Promise<any[]> => {
        const currentFilePath = recordingFilePathRef.current || recordedSessionDataFilePath;
        console.log('readRecordedSessionData called with file path:', currentFilePath);

        if (!currentFilePath) {
            console.log('No file path available for reading telemetry data');
            return [];
        }
        try {
            const options = {
                mode: 'text',
                pythonOptions: ['-u'],
                scriptPath: 'src/py-scripts',
                args: [currentFilePath]
            } as PythonShellOptions;

            const { shellId } = await window.electronAPI.runPythonScript('read_telemetry_data.py', options);

            return new Promise((resolve) => {
                let completeReceived = false;
                const allData: any[] = [];
                let removeMessageListener: (() => void) | null = null;
                let removeEndListener: (() => void) | null = null;

                const cleanup = () => {
                    if (removeMessageListener) {
                        removeMessageListener();
                        removeMessageListener = null;
                    }
                    if (removeEndListener) {
                        removeEndListener();
                        removeEndListener = null;
                    }
                };

                removeMessageListener = window.electronAPI.onPythonMessage((returnedShellId: number, message: string) => {
                    if (returnedShellId !== shellId) return;
                    try {
                        const obj = JSON.parse(message);
                        if (obj.type === 'progress') {
                            if (onProgress) onProgress(obj.read, obj.total ?? null, obj.bytesRead, obj.totalBytes);
                        } else if (obj.type === 'chunk') {
                            if (Array.isArray(obj.data)) {
                                allData.push(...obj.data);
                            }
                        } else if (obj.type === 'complete') {
                            completeReceived = true;
                            if (Array.isArray(obj.data)) {
                                allData.push(...obj.data);
                            }
                            console.log('Telemetry data complete. Points:', allData.length);
                            resolve(allData);
                            cleanup();
                        } else if (obj.type === 'error') {
                            console.error('Error from telemetry reader:', obj.message);
                        }
                    } catch (e) {
                        // Non JSON lines ignored
                    }
                });

                removeEndListener = window.electronAPI.onPythonEnd('session-analysis', (returnedShellId: number) => {
                    if (returnedShellId !== shellId) return;
                    if (!completeReceived) {
                        console.warn('Python process ended before complete event; returning collected data');
                        resolve(allData);
                        cleanup();
                    }
                });
            });
        } catch (error) {
            console.error('Error reading telemetry data from file:', error);
            return [];
        }
    };

    // Clear recording session (reset file paths and counters)
    const clearRecordingSession = (): void => {
        console.log('Clearing recording session');
        setRecordedSessionDataFilePath(null);
        recordingFilePathRef.current = null;
        setRecordedTelemetryDataCount(0);
        writeQueueRef.current = Promise.resolve();
        void disposeTelemetryWriter({ force: true });
    };

    // Function to send guidance messages to chat
    const sendGuidanceToChat = (message: string) => {
        setLatestGuidanceMessage(message);
    };

    useEffect(() => {
        if (!liveData || typeof liveData !== 'object') {
            return;
        }

        if (Object.keys(liveData).length === 0) {
            if (TelemetryDataLiveStatus !== null) {
                setTelemetryDataLiveStatus(null);
            }
            return;
        }

        const nextStatus = normalizeAccStatus((liveData as any)?.Graphics_status ?? (liveData as any)?.Graphics?.status);
        if (nextStatus !== null && nextStatus !== TelemetryDataLiveStatus) {
            setTelemetryDataLiveStatus(nextStatus);
        }
    }, [liveData, TelemetryDataLiveStatus]);
    //switch tab when a map or a session is selected
    useEffect(() => {
        if (mapSelected != null) {
            setActiveTab("sessionLists");
        }

        if (sessionSelected != null) {
            setActiveTab("session");
        }
    }, [mapSelected, sessionSelected]);


    //clean other tabs in situations
    useEffect(() => {

        //if current selected tab is Map tab
        if (activeTab == "mapLists") {
            setMap(null);
            setSession(null);
            return;
        }

        //if current tab is session list
        if (activeTab == "sessionLists") {
            setSession(null);
        }
    }, [activeTab]);

    useEffect(() => {
        return () => {
            void disposeTelemetryWriter({ force: true });
        };
    }, [disposeTelemetryWriter]);


    return (
        <AnalysisContext.Provider value={{
            mapSelected,
            sessionSelected,
            liveData,
            recordedSessionDataFilePath,
            recordedTelemetryDataCount,
            recordedSessioStaticsData,
            activeVisualizations,
            latestGuidanceMessage,
            setMap,
            setSession,
            setLiveSessionData: setLiveData,
            setRecordedSessionStaticsData,
            setRecordedSessionDataFilePath,
            writeRecordedLiveSessionData,
            readRecordedSessionData,
            finalizeRecordingWrites,
            clearRecordingSession,
            setActiveVisualizations,
            sendGuidanceToChat
            ,
            TelemetryDataLiveStatus
        }}>
            <Tabs.Root className="LiveAnalysisTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
                <Tabs.List className="live-analysis-tablists" justify="start">
                    <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                    {mapSelected == null ? "" : <Tabs.Trigger value="sessionLists">{mapSelected}</Tabs.Trigger>}
                    {sessionSelected == null ? "" : <Tabs.Trigger value="session">Session {sessionSelected.session_name}</Tabs.Trigger>}
                </Tabs.List>

                <Box className="live-analysis-container" >
                    <Tabs.Content className="TabContent" value="mapLists">
                        <MapList ></MapList>
                    </Tabs.Content>

                    <Tabs.Content className="TabContent" value="sessionLists">
                        <SessionList></SessionList>
                    </Tabs.Content>

                    <Tabs.Content className="TabContent" value="session">
                        <SessionAnalysisSplit></SessionAnalysisSplit>
                    </Tabs.Content>
                </Box >
            </Tabs.Root>
            {environment == 'electron' ? <LiveAnalysisSessionRecording></LiveAnalysisSessionRecording> : ''}
        </AnalysisContext.Provider>
    )
};

export default SessionAnalysis;