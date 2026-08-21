import React, { useState } from 'react';
import './MainDashboard.css';
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import HeaderMenu from 'views/header-menu/header-menu';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { getDefaultDashboardTab } from './dashboard-navigation';
import { AiToolComponentRefProvider } from 'contexts/AiToolComponentRefContext';
import DashboardAssistant from './DashboardAssistant';

const MainDashboard = ({ onTaskCreated }) => {
    const environment = useEnvironment();
    const [mainMenuTab, setMainMenuTab] = useState(() => getDefaultDashboardTab(environment));
    return (
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
    );
};

export default MainDashboard;
