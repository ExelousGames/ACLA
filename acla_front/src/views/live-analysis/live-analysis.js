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
} from "@radix-ui/themes";
import MapList from './map-list/map-list';
import SessionList from './session-list/session-list';
import { Tabs } from "radix-ui";

const LiveAnalysis = () => {

    return (
        <Tabs.Root defaultValue="account">
            <Tabs.List>
                <Tabs.Trigger value="mapList">Account</Tabs.Trigger>
                <Tabs.Trigger value="sessionList">Documents</Tabs.Trigger>
                <Tabs.Trigger value="session">Settings</Tabs.Trigger>
            </Tabs.List>

            <Box pt="3">
                <Tabs.Content value="mapList">
                    <MapList></MapList>
                </Tabs.Content>

                <Tabs.Content value="sessionList">
                    <SessionList></SessionList>
                </Tabs.Content>

                <Tabs.Content value="session">
                    <Text size="2">Edit your profile or update contact information.</Text>
                </Tabs.Content>
            </Box>
        </Tabs.Root>

    )
};

export default LiveAnalysis;