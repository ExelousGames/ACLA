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
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { AnalysisContext } from './analysis-context';

const normalizeAccStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) {
        return null;
    }

    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

const SessionAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [liveData, setLiveData] = useState({});
    const [liveStatus, setLiveStatus] = useState<ACC_STATUS | null>(null);
    const [recordedSessioStaticsData, setRecordedSessionStaticsData] = useState({});
    const [recordedSessionDataFilePath, setRecordedSessionDataFilePath] = useState<string | null>(null);
    const [recordedTelemetryDataCount, setRecordedTelemetryDataCount] = useState<number>(0);
    const [activeVisualizations, setActiveVisualizations] = useState<VisualizationInstance[]>([]);
    const [latestGuidanceMessage, setLatestGuidanceMessage] = useState<string | null>(null);

    // Use ref to persist file path during recording to prevent state reset issues
    const recordingFilePathRef = useRef<string | null>(null);

    const environment = useEnvironment();

    // File-based telemetry data functions
    const writeRecordedLiveSessionData = async (data: any): Promise<void> => {

        // Use the ref value if available, otherwise fall back to state
        const currentFilePath = recordingFilePathRef.current || recordedSessionDataFilePath;

        if (!currentFilePath) {
            // Generate a temporary file path for telemetry data
            const timestamp = new Date().getTime();
            const sessionId = sessionSelected?.SessionId || 'unknown';
            const filePath = `../session_recording/temp/telemetry_${sessionId}_${timestamp}.jsonl`;

            // Store in both state and ref
            setRecordedSessionDataFilePath(filePath);
            recordingFilePathRef.current = filePath;

            // Reset the data counter for new session
            setRecordedTelemetryDataCount(0);

            // Use the new path immediately for this write operation
            await writeToFile(filePath, data);
            return;
        }

        await writeToFile(currentFilePath, data);
    };

    const writeToFile = async (filePath: string, data: any): Promise<void> => {
        try {
            // Use Python script to append data to file (JSONL format for streaming)
            const options = {
                mode: 'text',
                pythonOptions: ['-u'],
                scriptPath: 'src/py-scripts',
                args: [filePath, JSON.stringify(data)]
            } as PythonShellOptions;
            // Create a simple Python script call to append data
            await window.electronAPI.runPythonScript('append_telemetry_data.py', options);

            // Increment the telemetry data counter
            setRecordedTelemetryDataCount(prev => prev + 1);
        } catch (error) {
            console.error('Error writing telemetry data to file:', error);
        }
    };

    const readRecordedSessionData = async (onProgress?: (read: number, total: number | null) => void): Promise<any[]> => {
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
                            if (onProgress) onProgress(obj.read, obj.total ?? null);
                        } else if (obj.type === 'complete') {
                            completeReceived = true;
                            if (Array.isArray(obj.data)) {
                                allData.push(...obj.data);
                                console.log('Telemetry data complete. Points:', obj.data.length);
                                resolve(allData);
                            } else {
                                resolve([]);
                            }
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
    };

    // Function to send guidance messages to chat
    const sendGuidanceToChat = (message: string) => {
        setLatestGuidanceMessage(message);
    };

    const updateLiveStatus = useCallback((status: ACC_STATUS | null) => {
        setLiveStatus(status);
    }, []);

    useEffect(() => {
        if (!liveData || typeof liveData !== 'object') {
            return;
        }

        const nextStatus = normalizeAccStatus((liveData as any)?.Graphics_status ?? (liveData as any)?.Graphics?.status);
        console.log('Next live status:', nextStatus);
        if (nextStatus !== null && nextStatus !== liveStatus) {
            setLiveStatus(nextStatus);
        }
    }, [liveData, liveStatus]);
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
            clearRecordingSession,
            setActiveVisualizations,
            sendGuidanceToChat
            ,
            liveStatus,
            setLiveStatus: updateLiveStatus
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