import './session-analysis.css';

import {
    Box,
    Tabs
} from "@radix-ui/themes";

import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import React, { useEffect, useState, createContext, Dispatch, SetStateAction } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import SessionAnalysisMap from './sessionAnalysis/sessionAnalysisMap';
import { useEnvironment } from 'contexts/EnvironmentContext';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';

//use interface when create a context, help prevent runtime error and type safe
interface AnalysisContextType {
    mapSelected: string | null,
    sessionSelected: RacingSessionDetailedInfoDto | null,
    /**
     * live data at runtime
     */
    liveData: any;
    recordedSessionData: any[];
    recordedSessioStaticsData: any;
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
     * all the recored data
     */
    setRecordedSessionData: Dispatch<SetStateAction<any[]>>;
};


//defined the structure here, pass down the props to child, must have init value here, otherwise createContext and useContext don't like it
export const AnalysisContext = createContext<AnalysisContextType>({
    mapSelected: '',
    sessionSelected: {} as RacingSessionDetailedInfoDto,
    liveData: {} as any,
    recordedSessionData: [],
    recordedSessioStaticsData: {} as any,
    setMap: (map: string | null) => { },
    setSession: ((value: RacingSessionDetailedInfoDto | null) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>,
    setLiveSessionData: (data: {}) => { },
    setRecordedSessionStaticsData: (data: {}) => { },
    setRecordedSessionData: ((value: any[]) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<any[]>>,
});

const SessionAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [liveData, setLiveData] = useState({});
    const [recordedSessioStaticsData, setRecordedSessionStaticsData] = useState({});
    const [recordedSessionData, setRecordedSessionData] = useState<any[]>([]);
    const environment = useEnvironment();
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
        <AnalysisContext.Provider value={{ mapSelected: mapSelected, sessionSelected: sessionSelected, liveData: liveData, recordedSessionData: recordedSessionData, recordedSessioStaticsData: recordedSessioStaticsData, setMap, setSession, setLiveSessionData: setLiveData, setRecordedSessionStaticsData, setRecordedSessionData }}>
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
                        <SessionAnalysisMap></SessionAnalysisMap>
                    </Tabs.Content>
                </Box >
            </Tabs.Root>
            {environment == 'electron' ? <LiveAnalysisSessionRecording></LiveAnalysisSessionRecording> : ''}
        </AnalysisContext.Provider>
    )
};

export default SessionAnalysis;