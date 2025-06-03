import React, { createContext, useState } from 'react';
import axios from 'axios';
import { useAuth } from "hooks/AuthProvider";

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
} from "@radix-ui/themes";
import SideMainMenu from 'views/side-main-menu/side-main-menu';
import LiveAnalysis from 'views/live-analysis/live-analysis';
import { MainMenuOptions } from 'data/MainMenuOptions';


const MainMenuOptionSelectionContext = createContext();

const MainDashboard = ({ onTaskCreated }) => {
    const auth = useAuth();

    const [mainMenuOptionSelected, setMainMenuOption] = useState(MainMenuOptions.LIVE_ANALYSIS);
    return (

        <MainMenuOptionSelectionContext value={[mainMenuOptionSelected, setMainMenuOption]}>
            <Flex gap="3" rows="2" width="auto" direction="column">
                <Box className="Title" height="64px">Assetto Corsa Competizione Lap Analysis</Box>
                <SideMainMenu></SideMainMenu>
            </Flex>
        </MainMenuOptionSelectionContext>

    );
};

export default MainDashboard;