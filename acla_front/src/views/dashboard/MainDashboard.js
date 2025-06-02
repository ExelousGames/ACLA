import React, { useState } from 'react';
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

const MainDashboard = ({ onTaskCreated }) => {
    const auth = useAuth();
    return (
        <div className="container">
            <div>
                <Grid gap="3" rows="repeat(2, 64px)" width="auto">
                    <SideMainMenu></SideMainMenu>
                    <LiveAnalysis></LiveAnalysis>
                </Grid>
            </div>
        </div >
    );
};

export default MainDashboard;