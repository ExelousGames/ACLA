import React, { useState } from 'react';
import './MainDashboard.css';
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import HeaderMenu from 'views/header-menu/header-menu';
import { SessionAnalysisAssistant, SessionAnalysisProvider } from 'views/lap-analysis/session-analysis';
import { useEnvironment } from 'contexts/EnvironmentContext';
import LiveAnalysisSessionRecording from 'views/lap-analysis/liveAnalysisSessionRecording';
import LiveSessionDetectionManager from 'views/lap-analysis/LiveSessionDetectionManager';
import { LiveSessionProvider } from 'views/live-session/LiveSessionContext';
import { LIVE_SESSION_RECORDER_HOST_ID } from 'views/live-session/LiveSessionView';
import { DASHBOARD_TABS, getDefaultDashboardTab } from './dashboard-navigation';

const MainDashboard = ({ onTaskCreated }) => {
    const environment = useEnvironment();
    const authenticatedOwnerEmail = typeof window !== 'undefined'
        ? window.localStorage.getItem('username')
        : null;

    const [mainMenuTab, setMainMenuTab] = useState(() => getDefaultDashboardTab(environment));
    const assistantModeOverride = mainMenuTab === DASHBOARD_TABS.LIVE_SESSION
        ? 'live'
        : mainMenuTab === DASHBOARD_TABS.USER_SUMMARY
            ? 'user_summary'
            : undefined;

    return (
        <LiveSessionProvider ownerEmail={authenticatedOwnerEmail}>
            <SessionAnalysisProvider>
                {environment === 'electron' ? <LiveSessionDetectionManager /> : null}
                <div className="main-dashboard-container">
                <div className="main-dashboard-header">
                    <HeaderMenu />
                </div>

                <div className="main-dashboard-content">
                    <div className="main-dashboard-primary">
                        <SideMainMenu activeTab={mainMenuTab} onTabChange={setMainMenuTab} />
                    </div>
                    <SessionAnalysisAssistant assistantModeOverride={assistantModeOverride} />
                    {environment === 'electron' ? (
                        <LiveAnalysisSessionRecording
                            recorderHostId={mainMenuTab === DASHBOARD_TABS.LIVE_SESSION ? LIVE_SESSION_RECORDER_HOST_ID : undefined}
                        />
                    ) : null}
                </div>
                </div>
            </SessionAnalysisProvider>
        </LiveSessionProvider>
    );
};

export default MainDashboard;
