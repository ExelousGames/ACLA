import './session-analysis.css';

import {
    Box,
    Tabs
} from "@radix-ui/themes";

import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import React, { useEffect, useState, createContext, Dispatch, SetStateAction, useRef } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import SessionAnalysisSplit from './sessionAnalysis/session-analysis-split';
import { useEnvironment } from 'contexts/EnvironmentContext';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { PythonShellOptions } from 'services/pythonService';

//use interface when create a context, help prevent runtime error and type safe
interface AnalysisContextType {
    mapSelected: string | null,
    sessionSelected: RacingSessionDetailedInfoDto | null,
    /**
     * live data at runtime
     */
    liveData: any;
    recordedSessionDataFilePath: string | null;
    recordedTelemetryDataCount: number;
    recordedSessioStaticsData: any;
    /**
     * Active visualizations in the multi-info container
     */
    activeVisualizations: VisualizationInstance[];
    /**
     * Latest guidance message from ImitationGuidanceChart
     */
    latestGuidanceMessage: string | null;
    setMap: (map: string | null) => void;
    setSession: Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>;
    setLiveSessionData: (data: {}) => void;

    /**
     * Data that are initialized when the instance starts and never changes until the instance is closed.
     * @param data 
     * @returns 
     */
    setRecordedSessionStaticsData: (data: {}) => void;

    /**
     * Set the file path for recorded telemetry data
     */
    setRecordedSessionDataFilePath: (filePath: string | null) => void;

    /**
     * Write telemetry data to file
     */
    writeRecordedLiveSessionData: (data: any) => Promise<void>;

    /**
     * Read all recorded session data from file
     */
    readRecordedSessionData: () => Promise<any[]>;

    /**
     * Clear recording file path (call when recording stops)
     */
    clearRecordingSession: () => void;

    /**
     * Update active visualizations
     */
    setActiveVisualizations: Dispatch<SetStateAction<VisualizationInstance[]>>;

    /**
     * Send a guidance message from components to AI chat
     */
    sendGuidanceToChat: (message: string) => void;
};


//defined the structure here, pass down the props to child, must have init value here, otherwise createContext and useContext don't like it
export const AnalysisContext = createContext<AnalysisContextType>({
    mapSelected: '',
    sessionSelected: {} as RacingSessionDetailedInfoDto,
    liveData: {} as any,
    recordedSessionDataFilePath: null,
    recordedTelemetryDataCount: 0,
    recordedSessioStaticsData: {} as any,
    activeVisualizations: [],
    latestGuidanceMessage: null,
    setMap: (map: string | null) => { },
    setSession: ((value: RacingSessionDetailedInfoDto | null) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>,
    setLiveSessionData: (data: {}) => { },
    setRecordedSessionStaticsData: (data: {}) => { },
    setRecordedSessionDataFilePath: (filePath: string | null) => {
        console.warn('No provider for AnalysisContext');
    },
    writeRecordedLiveSessionData: async (data: any) => {
        console.warn('No provider for AnalysisContext');
    },
    readRecordedSessionData: async () => {
        console.warn('No provider for AnalysisContext');
        return [];
    },
    clearRecordingSession: () => {
        console.warn('No provider for AnalysisContext');
    },
    setActiveVisualizations: ((value: VisualizationInstance[]) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<VisualizationInstance[]>>,
    sendGuidanceToChat: (message: string) => {
        console.warn('No provider for AnalysisContext');
    },
});

const SessionAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [liveData, setLiveData] = useState({});
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

    const readRecordedSessionData = async (): Promise<any[]> => {
        const currentFilePath = recordingFilePathRef.current || recordedSessionDataFilePath;
        console.log('readRecordedSessionData called with file path:', currentFilePath);

        if (!currentFilePath) {
            console.log('No file path available for reading telemetry data');
            return [];
        } try {
            // Use Python script to read all data from file
            const options = {
                mode: 'text',
                pythonOptions: ['-u'],
                scriptPath: 'src/py-scripts',
                args: [currentFilePath]
            } as PythonShellOptions;

            console.log('Running read_telemetry_data.py with options:', options);
            const { shellId } = await window.electronAPI.runPythonScript('read_telemetry_data.py', options);

            return new Promise((resolve) => {
                // Set a timeout to avoid hanging forever
                const timeoutId = setTimeout(() => {
                    console.log('Timeout reading telemetry data');
                    resolve([]);
                }, 10000); // 10 second timeout

                window.electronAPI.onPythonMessage((returnedShellId: number, message: string) => {
                    if (shellId === returnedShellId) {
                        clearTimeout(timeoutId);
                        try {
                            console.log('Received telemetry data message:', message.substring(0, 200) + '...');
                            const data = JSON.parse(message);
                            console.log('Parsed telemetry data:', data.length, 'points');
                            resolve(data);
                        } catch (error) {
                            console.error('Error parsing telemetry data:', error);
                            console.error('Raw message:', message.substring(0, 500));
                            resolve([]);
                        }
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