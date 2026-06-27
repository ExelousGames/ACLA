import * as React from "react";
import { ScrollArea, Tabs } from "radix-ui";
import "./side-main-menu.css";
import { Box, Text, Button } from "@radix-ui/themes";
import SessionAnalysis from "views/lap-analysis/session-analysis";
import ProtectedComponent from "components/ProtectedComponent";
import UserSummary from "views/user-summary/user-summary";
import CircuitMaps from "views/circuit-maps/circuit-maps";

type SideMainMenuProps = {
    activeTab?: string;
    onTabChange?: (value: string) => void;
};

const SideMainMenu = ({ activeTab, onTabChange }: SideMainMenuProps) => {
    const tabsRootProps = activeTab !== undefined
        ? { value: activeTab, onValueChange: onTabChange }
        : { defaultValue: "analysis" };

    return (
        <Tabs.Root className="TabsRoot" {...tabsRootProps}>

        <ScrollArea.Root className="ScrollAreaRoot">

            <ScrollArea.Viewport className="ScrollAreaViewport">

                <Tabs.List className="TabsList">
                    <Tabs.Trigger className="TabsTrigger" value="analysis">Analysis</Tabs.Trigger>
                    <Tabs.Trigger className="TabsTrigger" value="userSummary">User Summary</Tabs.Trigger>
                    <Tabs.Trigger className="TabsTrigger" value="circuitMaps">Circuit Maps</Tabs.Trigger>
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

        <Box className="Container">
            <Tabs.Content className="TabsContent" value="analysis">
                <SessionAnalysis></SessionAnalysis>
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="userSummary">
                <UserSummary />
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="circuitMaps">
                <CircuitMaps />
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="adminPanel">
                <ProtectedComponent
                    requiredPermission={{ action: 'create', resource: 'user' }}
                    fallback="Admin access required"
                >
                    <Box p="4">
                        <Text>Admin Panel - Create Users</Text>
                        <Button>Create New User</Button>
                    </Box>
                </ProtectedComponent>
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="adminonly">
                <ProtectedComponent
                    requiredRole="admin"
                    fallback="Admin access required"
                >
                    <Box p="4">
                        <Text>Admin Only Section</Text>
                    </Box>
                </ProtectedComponent>
            </Tabs.Content>
        </Box >

        </Tabs.Root>
    );
};



export default SideMainMenu;
