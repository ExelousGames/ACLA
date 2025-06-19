import * as React from "react";
import { ScrollArea, Tabs } from "radix-ui";
import "./side-main-menu.css";
import { Box, Text, Flex, Container } from "@radix-ui/themes";
import SessionAnalysis from "views/lap-analysis/session-analysis";

const SideMainMenu = () => (
    <Tabs.Root className="TabsRoot" defaultValue="account">

        <ScrollArea.Root className="ScrollAreaRoot">

            <ScrollArea.Viewport className="ScrollAreaViewport">

                <Tabs.List className="TabsList">
                    <Tabs.Trigger className="TabsTrigger" value="analysis">Analysis</Tabs.Trigger>
                    <Tabs.Trigger className="TabsTrigger" value="sessionList">Documents</Tabs.Trigger>
                    <Tabs.Trigger className="TabsTrigger" value="session">Settings</Tabs.Trigger>
                </Tabs.List>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar className="ScrollAreaScrollbar" orientation="vertical">
                <ScrollArea.Thumb className="ScrollAreaThumb" />
            </ScrollArea.Scrollbar>
            <ScrollArea.Scrollbar className="ScrollAreaScrollbar" orientation="horizontal">
                <ScrollArea.Thumb className="ScrollAreaThumb" />
            </ScrollArea.Scrollbar>
            <ScrollArea.Corner className="ScrollAreaCorner" />
        </ScrollArea.Root>

        <Box className="Container" align='left'>
            <Tabs.Content className="TabsContent" value="analysis">
                <SessionAnalysis></SessionAnalysis>
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="sessionList">

            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="session">
                <Text size="2">Edit your profile or update contact information.</Text>
            </Tabs.Content>
        </Box >

    </Tabs.Root>



);

export default SideMainMenu;