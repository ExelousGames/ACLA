import * as React from "react";
import { ScrollArea, Tabs } from "radix-ui";
import "./side-main-menu.css";
import { Box, Text, Button } from "@radix-ui/themes";
import SessionAnalysis from "views/lap-analysis/session-analysis";
import ProtectedComponent from "components/ProtectedComponent";
import UserSummary from "views/user-summary/user-summary";
import CircuitMaps from "views/circuit-maps/circuit-maps";
import LiveSessionView from "views/live-session/LiveSessionView";
import { useEnvironment } from "contexts/EnvironmentContext";
import { getDashboardTabs, getDefaultDashboardTab } from "views/dashboard/dashboard-navigation";
import { AI_TOOL_COMPONENT_NAMES } from "contexts/AiToolComponentRefContext";

type SideMainMenuProps = {
    activeTab?: string;
    onTabChange?: (value: string) => void;
};

const SideMainMenu = ({ activeTab, onTabChange }: SideMainMenuProps) => {
    const environment = useEnvironment();
    const dashboardTabs = getDashboardTabs(environment);
    const tabsRootProps = activeTab !== undefined
        ? { value: activeTab, onValueChange: onTabChange }
        : { defaultValue: getDefaultDashboardTab(environment) };

    return (
        <Tabs.Root className="TabsRoot" {...tabsRootProps}>

        <ScrollArea.Root className="ScrollAreaRoot">

            <ScrollArea.Viewport className="ScrollAreaViewport">

                <Tabs.List className="TabsList">
                    {dashboardTabs.map((tab) => (
                        <Tabs.Trigger className="TabsTrigger" value={tab.value} key={tab.value}>
                            {tab.label}
                        </Tabs.Trigger>
                    ))}
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
            {environment === 'electron' ? (
                <Tabs.Content className="TabsContent" value="liveSession">
                    <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </Tabs.Content>
            ) : null}

            <Tabs.Content className="TabsContent" value="analysis">
                <SessionAnalysis name={AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS} />
            </Tabs.Content>

            <Tabs.Content className="TabsContent" value="userSummary">
                <UserSummary name={AI_TOOL_COMPONENT_NAMES.USER_SUMMARY} />
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
