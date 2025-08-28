import React, { createContext, useState } from 'react';
import { useAuth } from "hooks/AuthProvider";
import './MainDashboard.css';
import {
    Avatar,
    Badge,
    Box,
    Button,
    Card,
    Checkbox,
    DropdownMenu,
    Flex,
    Grid,
    Heading,
    IconButton,
    Link,
    Separator,
    Strong,
    Switch,
    Text,
    TextField,
    Theme,
    Section,
} from "@radix-ui/themes";
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import { MainMenuOptions } from 'data/MainMenuOptions';
import HeaderMenu from 'views/header-menu/header-menu';


const MainMenuOptionSelectionContext = createContext();

const MainDashboard = ({ onTaskCreated }) => {
    const auth = useAuth();

    const [mainMenuOptionSelected, setMainMenuOption] = useState(MainMenuOptions.LIVE_ANALYSIS);

    return (
        <MainMenuOptionSelectionContext.Provider value={[mainMenuOptionSelected, setMainMenuOption]}>
            <div className="main-dashboard-container">
                <div className="main-dashboard-header">
                    <HeaderMenu />
                </div>

                <div className="main-dashboard-content">
                    <SideMainMenu />
                </div>
            </div>
        </MainMenuOptionSelectionContext.Provider>
    );
};

export default MainDashboard;