import React, { createContext, useState } from 'react';
import { useAuth } from "hooks/AuthProvider";
import ProtectedComponent from '../../components/ProtectedComponent';
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

            {/* Example of using ProtectedComponent for conditional rendering */}
            <ProtectedComponent
                requiredPermission={{ action: 'read', resource: 'menu' }}
                fallbackNavigation={"/login"}
            >
                <HeaderMenu></HeaderMenu>
                <SideMainMenu></SideMainMenu>
            </ProtectedComponent>

            {/* Example of using ProtectedComponent for conditional rendering */}
            <ProtectedComponent
                requiredPermission={{ action: 'create', resource: 'user' }}
                fallbackNavigation={<Text>You don't have permission to create users</Text>}
            >
                <Box p="4">
                    <Text>Admin Panel - Create Users</Text>
                    <Button>Create New User</Button>
                </Box>
            </ProtectedComponent>

            <ProtectedComponent
                requiredRole="admin"
                fallbackNavigation={<Text>Admin access required</Text>}
            >
                <Box p="4">
                    <Text>Admin Only Section</Text>
                </Box>
            </ProtectedComponent>
        </MainMenuOptionSelectionContext>
    );
};

export default MainDashboard;