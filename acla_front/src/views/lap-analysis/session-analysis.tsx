import './session-analysis.css';

import {
    Box,
    Tabs
} from "@radix-ui/themes";

import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import React, { useEffect, useState, createContext, Dispatch, SetStateAction } from 'react';
import { OptionSelected } from 'data/live-analysis/live-analysis-data';
import SessionAnalysisMap from './sessionAnalysis/sessionAnalysisMap';

interface AnalysisContextType {
    options: OptionSelected | null;

    /**
     * live data at runtime
     */
    liveSessionData: any;
    recordedSessionData: any[];
    setMap: (map: string) => void;
    setSession: (session: string) => void;
    setLiveSessionData: (data: {}) => void;

    /**
     * all the recored data
     */
    setRecordedSessionData: Dispatch<SetStateAction<any[]>>;
};

//defined the sturcture here, pass down the props to child, must have init value here, otherwise createContext and useContext don't like it
export const AnalysisContext = createContext<AnalysisContextType>({
    options: null,
    liveSessionData: {} as any,
    recordedSessionData: [],
    setMap: (map: string) => { },
    setSession: (session: string) => { },
    setLiveSessionData: (data: {}) => { },
    setRecordedSessionData: ((value: any[]) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<any[]>>,
});

const SessionAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [liveSessionData, setLiveSessionData] = useState({});
    const [recordedSessionData, setRecordedSessionData] = useState<any[]>([]);
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
        <AnalysisContext.Provider value={{ options: { mapOption: mapSelected, sessionOption: sessionSelected }, liveSessionData: liveSessionData, recordedSessionData: recordedSessionData, setMap, setSession, setLiveSessionData, setRecordedSessionData }}>
            <Tabs.Root className="LiveAnalysisTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
                <Tabs.List className="live-analysis-tablists" justify="start">
                    <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                    {mapSelected == null ? "" : <Tabs.Trigger value="sessionLists">{mapSelected}</Tabs.Trigger>}
                    {sessionSelected == null ? "" : <Tabs.Trigger value="session">Session {sessionSelected}</Tabs.Trigger>}
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
        </AnalysisContext.Provider>
    )
};

export default SessionAnalysis;