import { DASHBOARD_TABS, getDashboardTabs, getDefaultDashboardTab } from './dashboard-navigation';

describe('dashboard navigation', () => {
    it('puts Live Session first and selects it by default in Electron', () => {
        expect(getDashboardTabs('electron').map((tab) => tab.label)).toEqual([
            'Live Session',
            'Analysis',
            'User Summary',
            'Circuit Maps',
        ]);
        expect(getDefaultDashboardTab('electron')).toBe(DASHBOARD_TABS.LIVE_SESSION);
    });

    it('keeps web navigation unchanged', () => {
        expect(getDashboardTabs('web').map((tab) => tab.label)).toEqual([
            'Analysis',
            'User Summary',
            'Circuit Maps',
        ]);
        expect(getDefaultDashboardTab('web')).toBe(DASHBOARD_TABS.ANALYSIS);
    });
});
