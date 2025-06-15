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
import { MapOption, OptionSelected, SessionOption } from 'data/live-analysis/live-analysis-data';
import SessionAnalysis from './sessionAnalysis/sessionAnalysis';

//defined the sturcture here, pass down the props to child, must have init value here, otherwise createContext and useContext don't like it
export const AnalysisContext = createContext({
    analysisContext: {
        options: {
            mapOption: '',
            sessionOption: '',
        } as OptionSelected | null,
        setMap: (map: string) => { },
        setSession: (session: string) => { }
    },

});

const LiveAnalysis = () => {

    //must give state some init value otherwise createContext and useContext don't like it
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<string | null>(null);
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
        <AnalysisContext.Provider value={{ analysisContext: { options: { mapOption: mapSelected, sessionOption: sessionSelected }, setMap, setSession } }}>
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
                        <SessionAnalysis></SessionAnalysis>
                    </Tabs.Content>
                </Box >
            </Tabs.Root>
        </AnalysisContext.Provider>
    )
};

export default LiveAnalysis;