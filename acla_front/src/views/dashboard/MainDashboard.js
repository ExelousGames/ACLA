import React, { useState } from 'react';
import './MainDashboard.css';
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import HeaderMenu from 'views/header-menu/header-menu';
import { SessionAnalysisAssistant, SessionAnalysisProvider } from 'views/lap-analysis/session-analysis';
import { useEnvironment } from 'contexts/EnvironmentContext';
import LiveAnalysisSessionRecording from 'views/lap-analysis/liveAnalysisSessionRecording';


const DASHBOARD_TABS = Object.freeze({
    ANALYSIS: 'analysis',
    USER_SUMMARY: 'userSummary',
    CIRCUIT_MAPS: 'circuitMaps',
});

const MainDashboard = ({ onTaskCreated }) => {
    const environment = useEnvironment();

    const [mainMenuTab, setMainMenuTab] = useState(DASHBOARD_TABS.ANALYSIS);
    const assistantModeOverride = mainMenuTab === DASHBOARD_TABS.USER_SUMMARY
        ? 'user_summary'
        : undefined;

    return (
        <SessionAnalysisProvider>
            <div className="main-dashboard-container">
                <div className="main-dashboard-header">
                    <HeaderMenu />
                </div>

                <div className={`main-dashboard-content ${environment === 'electron' ? 'has-recording-bar' : ''}`}>
                    <div className="main-dashboard-primary">
                        <SideMainMenu activeTab={mainMenuTab} onTabChange={setMainMenuTab} />
                    </div>
                    <SessionAnalysisAssistant assistantModeOverride={assistantModeOverride} />
                    {environment === 'electron' ? <LiveAnalysisSessionRecording /> : ''}
                </div>
            </div>
        </SessionAnalysisProvider>
    );
};

export default MainDashboard;
