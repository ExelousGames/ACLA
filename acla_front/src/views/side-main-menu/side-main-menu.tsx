import * as React from "react";
import { ScrollArea, Tabs, Tooltip } from "radix-ui";
import "./side-main-menu.css";
import { Box, Text, Button } from "@radix-ui/themes";
import {
    BarChartIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    DashboardIcon,
    GlobeIcon,
    LapTimerIcon,
    PersonIcon,
} from "@radix-ui/react-icons";
import SessionAnalysis from "views/lap-analysis/session-analysis";
import ProtectedComponent from "components/ProtectedComponent";
import UserSummary from "views/user-summary/user-summary";
import CircuitMaps from "views/circuit-maps/circuit-maps";
import LiveSessionView from "views/live-session/LiveSessionView";
import { useEnvironment } from "contexts/EnvironmentContext";
import {
    DASHBOARD_TABS,
    getDashboardTabs,
    getDefaultDashboardTab,
} from "views/dashboard/dashboard-navigation";
import { AI_TOOL_COMPONENT_NAMES } from "contexts/AiToolComponentRefContext";

type SideMainMenuProps = {
    activeTab?: string;
    onTabChange?: (value: string) => void;
};

const TAB_ICONS: Record<string, React.ComponentType<React.ComponentProps<typeof LapTimerIcon>>> = {
    [DASHBOARD_TABS.LIVE_SESSION]: LapTimerIcon,
    [DASHBOARD_TABS.ANALYSIS]: BarChartIcon,
    [DASHBOARD_TABS.USER_SUMMARY]: PersonIcon,
    [DASHBOARD_TABS.CIRCUIT_MAPS]: GlobeIcon,
};

const SideMainMenu = ({ activeTab, onTabChange }: SideMainMenuProps) => {
    const environment = useEnvironment();
    const dashboardTabs = getDashboardTabs(environment);
    const defaultTab = getDefaultDashboardTab(environment);
    const [isCollapsed, setIsCollapsed] = React.useState(false);
    const [uncontrolledTab, setUncontrolledTab] = React.useState(defaultTab);
    const selectedTab = activeTab ?? uncontrolledTab;
    const [openedTabs, setOpenedTabs] = React.useState<Set<string>>(
        () => new Set([selectedTab]),
    );
    React.useEffect(() => {
        setOpenedTabs((current) => current.has(selectedTab)
            ? current
            : new Set([...Array.from(current), selectedTab]));
    }, [selectedTab]);
    const isOpened = (value: string) => openedTabs.has(value) || selectedTab === value;
    const handleTabChange = (value: string) => {
        setOpenedTabs((current) => current.has(value)
            ? current
            : new Set([...Array.from(current), value]));
        if (activeTab === undefined) setUncontrolledTab(value);
        onTabChange?.(value);
    };

    return (
        <Tabs.Root
            className="TabsRoot"
            data-menu-collapsed={isCollapsed}
            value={selectedTab}
            onValueChange={handleTabChange}
        >
            <Tooltip.Provider delayDuration={250} skipDelayDuration={100}>
                <ScrollArea.Root className="ScrollAreaRoot" asChild>
                    <nav aria-label="Main navigation">
                        <div className="SideMenuHeader">
                            <Tooltip.Root disableHoverableContent>
                                <Tooltip.Trigger asChild>
                                    <button
                                        type="button"
                                        className="SideMenuToggle"
                                        aria-controls="dashboard-sidebar-navigation"
                                        aria-expanded={!isCollapsed}
                                        aria-label={isCollapsed ? "Expand main menu" : "Collapse main menu"}
                                        onClick={() => setIsCollapsed((collapsed) => !collapsed)}
                                    >
                                        {isCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
                                    </button>
                                </Tooltip.Trigger>
                                <Tooltip.Portal>
                                    <Tooltip.Content className="SideMenuTooltip" side="right" sideOffset={10}>
                                        {isCollapsed ? "Expand menu" : "Collapse menu"}
                                        <Tooltip.Arrow className="SideMenuTooltipArrow" />
                                    </Tooltip.Content>
                                </Tooltip.Portal>
                            </Tooltip.Root>
                        </div>

                        <ScrollArea.Viewport className="ScrollAreaViewport">
                            <Tabs.List className="TabsList" id="dashboard-sidebar-navigation">
                                {dashboardTabs.map((tab) => {
                                    const TabIcon = TAB_ICONS[tab.value] ?? DashboardIcon;

                                    return (
                                        <Tooltip.Root key={tab.value} disableHoverableContent>
                                            <Tooltip.Trigger asChild>
                                                <Tabs.Trigger
                                                    className="TabsTrigger"
                                                    value={tab.value}
                                                    aria-label={tab.label}
                                                >
                                                    <span className="TabsTriggerIcon" aria-hidden="true">
                                                        <TabIcon width={19} height={19} />
                                                    </span>
                                                    <span className="TabsTriggerLabel">{tab.label}</span>
                                                </Tabs.Trigger>
                                            </Tooltip.Trigger>
                                            <Tooltip.Portal>
                                                <Tooltip.Content className="SideMenuTooltip" side="right" sideOffset={10}>
                                                    {tab.label}
                                                    <Tooltip.Arrow className="SideMenuTooltipArrow" />
                                                </Tooltip.Content>
                                            </Tooltip.Portal>
                                        </Tooltip.Root>
                                    );
                                })}
                            </Tabs.List>
                        </ScrollArea.Viewport>
                        <ScrollArea.Scrollbar className="ScrollAreaScrollbar" orientation="vertical">
                            <ScrollArea.Thumb className="ScrollAreaThumb" />
                        </ScrollArea.Scrollbar>
                        <ScrollArea.Scrollbar className="ScrollAreaScrollbar" orientation="horizontal">
                            <ScrollArea.Thumb className="ScrollAreaThumb" />
                        </ScrollArea.Scrollbar>
                        <ScrollArea.Corner className="ScrollAreaCorner" />
                    </nav>
                </ScrollArea.Root>
            </Tooltip.Provider>

            <Box className="Container">
                {environment === 'electron' && isOpened('liveSession') ? (
                    <Tabs.Content className="TabsContent" value="liveSession" forceMount>
                        <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                    </Tabs.Content>
                ) : null}

                {isOpened('analysis') ? (
                    <Tabs.Content className="TabsContent" value="analysis" forceMount>
                        <SessionAnalysis name={AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS} />
                    </Tabs.Content>
                ) : null}

                {isOpened('userSummary') ? (
                    <Tabs.Content className="TabsContent" value="userSummary" forceMount>
                        <UserSummary name={AI_TOOL_COMPONENT_NAMES.USER_SUMMARY} />
                    </Tabs.Content>
                ) : null}

                {isOpened('circuitMaps') ? (
                    <Tabs.Content className="TabsContent" value="circuitMaps" forceMount>
                        <CircuitMaps />
                    </Tabs.Content>
                ) : null}

                {isOpened('adminPanel') ? <Tabs.Content className="TabsContent" value="adminPanel" forceMount>
                    <ProtectedComponent
                        requiredPermission={{ action: 'create', resource: 'user' }}
                        fallback="Admin access required"
                    >
                        <Box p="4">
                            <Text>Admin Panel - Create Users</Text>
                            <Button>Create New User</Button>
                        </Box>
                    </ProtectedComponent>
                </Tabs.Content> : null}

                {isOpened('adminonly') ? <Tabs.Content className="TabsContent" value="adminonly" forceMount>
                    <ProtectedComponent
                        requiredRole="admin"
                        fallback="Admin access required"
                    >
                        <Box p="4">
                            <Text>Admin Only Section</Text>
                        </Box>
                    </ProtectedComponent>
                </Tabs.Content> : null}
            </Box >

        </Tabs.Root>
    );
};



export default SideMainMenu;
