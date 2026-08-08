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
import { AiChatScreenProvider } from 'contexts/AiChatScreenContext';

const MainDashboard = ({ onTaskCreated }) => {
    const environment = useEnvironment();
    const authenticatedOwnerEmail = typeof window !== 'undefined'
        ? window.localStorage.getItem('username')
        : null;

    const [mainMenuTab, setMainMenuTab] = useState(() => getDefaultDashboardTab(environment));
    return (
        <LiveSessionProvider ownerEmail={authenticatedOwnerEmail}>
            <SessionAnalysisProvider>
                <AiChatScreenProvider>
                    {environment === 'electron' ? <LiveSessionDetectionManager /> : null}
                    <div className="main-dashboard-container">
                        <div className="main-dashboard-header">
                            <HeaderMenu />
                        </div>

                        <div className="main-dashboard-content">
                            <div className="main-dashboard-primary">
                                <SideMainMenu activeTab={mainMenuTab} onTabChange={setMainMenuTab} />
                            </div>
                            <SessionAnalysisAssistant />
                            {environment === 'electron' ? (
                                <LiveAnalysisSessionRecording
                                    recorderHostId={mainMenuTab === DASHBOARD_TABS.LIVE_SESSION ? LIVE_SESSION_RECORDER_HOST_ID : undefined}
                                />
                            ) : null}
                        </div>
                    </div>
                </AiChatScreenProvider>
            </SessionAnalysisProvider>
        </LiveSessionProvider>
    );
};

export default MainDashboard;
