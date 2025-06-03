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
import { useEffect, useState } from 'react';

const LiveAnalysis = () => {

    const [mapSelected, setMap] = useState("");
    const [sessionSelected, setSession] = useState();


    return (
        <Tabs.Root defaultValue="mapLists">
            <Tabs.List justify="start">
                <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                {mapSelected == "" ? "" : <Tabs.Trigger value="sessionLists">{mapSelected}</Tabs.Trigger>}
                {mapSelected == "" ? "" : <Tabs.Trigger value="session">{sessionSelected}</Tabs.Trigger>}
            </Tabs.List>

            <Container pt="3" align='left'>
                <Tabs.Content value="mapLists">
                    <MapList setMap={setMap}></MapList>
                </Tabs.Content>

                <Tabs.Content value="sessionLists">
                    <SessionList></SessionList>
                </Tabs.Content>

                <Tabs.Content value="session">
                    <Text size="2">Edit your profile or update contact information.</Text>
                </Tabs.Content>
            </Container >
        </Tabs.Root>
    )
};

export default LiveAnalysis;