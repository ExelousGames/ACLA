import './live-analysis.css';

import {
    Avatar,
    Badge,
    Box,
    Button,
    Card,
    Checkbox,
    DropdownMenu,
    Flex,
    Grid,
    Heading,
    IconButton,
    Link,
    Separator,
    Strong,
    Switch,
    Text,
    TextField,
    Theme,
    Container,
    Tabs
} from "@radix-ui/themes";

import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import { useEffect, useState, createContext } from 'react';
import { MapOption, SessionOption } from 'data/live-analysis/live-analysis-data';
import SessionAnalysis from './sessionAnalysis/sessionAnalysis';

//pass down the props to child, must have init value here, otherwise createContext and useContext don't like it
export const AnalysisContext = createContext({
    mapContext: {
        mapSelected: {
            key: -1,
            datakey: -1,
            name: "",
            session_count: 0
        } as MapOption | null,
        setMap: (map: MapOption) => { }
    },
    sessionContext: {
        sessionSelected: {
            key: -1,
            datakey: -1,
            name: "",
            total_time: 0
        } as SessionOption | null,
        setSession: (session: SessionOption) => { }
    }

});

const LiveAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<MapOption | null>(null);
    const [sessionSelected, setSession] = useState<SessionOption | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');

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
        <AnalysisContext.Provider value={{ mapContext: { mapSelected, setMap }, sessionContext: { sessionSelected, setSession } }}>
            <Tabs.Root className="LiveAnalysisTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
                <Tabs.List className="live-analysis-tablists" justify="start">
                    <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                    {mapSelected == null ? "" : <Tabs.Trigger value="sessionLists">{mapSelected?.name}</Tabs.Trigger>}
                    {sessionSelected == null ? "" : <Tabs.Trigger value="session">Session {sessionSelected?.name}</Tabs.Trigger>}
                </Tabs.List>

                <Box className="live-analysis-container" >
                    <Tabs.Content className="TabContent" value="mapLists">
                        <MapList setMap={setMap}></MapList>
                    </Tabs.Content>

                    <Tabs.Content className="TabContent" value="sessionLists">
                        <SessionList></SessionList>
                    </Tabs.Content>

                    <Tabs.Content className="TabContent" value="session">
                        <SessionAnalysis></SessionAnalysis>
                    </Tabs.Content>
                </Box >
            </Tabs.Root>
        </AnalysisContext.Provider>
    )
};

export default LiveAnalysis;