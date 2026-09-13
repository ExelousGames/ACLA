import { Environment } from 'utils/environment';

export const DASHBOARD_TABS = Object.freeze({
    LIVE_SESSION: 'liveSession',
    ANALYSIS: 'analysis',
    USER_SUMMARY: 'userSummary',
    CIRCUIT_MAPS: 'circuitMaps',
});

export interface DashboardTab {
    value: string;
    label: string;
}

const WEB_TABS: DashboardTab[] = [
    { value: DASHBOARD_TABS.ANALYSIS, label: 'Analysis' },
    { value: DASHBOARD_TABS.USER_SUMMARY, label: 'User Summary' },
    { value: DASHBOARD_TABS.CIRCUIT_MAPS, label: 'Circuit Maps' },
];

export const getDashboardTabs = (environment: Environment): DashboardTab[] => (
    environment === 'electron'
        ? [{ value: DASHBOARD_TABS.LIVE_SESSION, label: 'Live Session' }, ...WEB_TABS]
        : WEB_TABS
);

export const getDefaultDashboardTab = (environment: Environment): string => (
    environment === 'electron' ? DASHBOARD_TABS.LIVE_SESSION : DASHBOARD_TABS.ANALYSIS
);
