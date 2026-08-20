import React, { useState } from 'react';
import './MainDashboard.css';
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import HeaderMenu from 'views/header-menu/header-menu';
import { SessionAnalysisProvider } from 'views/lap-analysis/session-analysis';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { LiveSessionProvider } from 'views/live-session/LiveSessionContext';
import { getDefaultDashboardTab } from './dashboard-navigation';
import { AiToolComponentRefProvider } from 'contexts/AiToolComponentRefContext';
import DashboardAssistant from './DashboardAssistant';

const MainDashboard = ({ onTaskCreated }) => {
    const environment = useEnvironment();
    const authenticatedOwnerEmail = typeof window !== 'undefined'
        ? window.localStorage.getItem('username')
        : null;

    const [mainMenuTab, setMainMenuTab] = useState(() => getDefaultDashboardTab(environment));
    return (
        <LiveSessionProvider ownerEmail={authenticatedOwnerEmail}>
            <SessionAnalysisProvider>
                <AiToolComponentRefProvider>
                    <div className="main-dashboard-container">
                        <div className="main-dashboard-header">
                            <HeaderMenu />
                        </div>

                        <div className="main-dashboard-content">
                            <div className="main-dashboard-primary">
                                <SideMainMenu activeTab={mainMenuTab} onTabChange={setMainMenuTab} />
                            </div>
                            <DashboardAssistant activeDashboardTab={mainMenuTab} />
                        </div>
                    </div>
                </AiToolComponentRefProvider>
            </SessionAnalysisProvider>
        </LiveSessionProvider>
    );
};

export default MainDashboard;
