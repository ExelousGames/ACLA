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

        <MainMenuOptionSelectionContext value={[mainMenuOptionSelected, setMainMenuOption]}>
            <HeaderMenu></HeaderMenu>
            <SideMainMenu></SideMainMenu>

        </MainMenuOptionSelectionContext>

    );
};

export default MainDashboard;