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

const MainDashboard = ({ onTaskCreated }) => {
    const auth = useAuth();
    return (
        <div className="container">
            <div>
                <h1>Welcome! {auth.user?.username}</h1>
                <button onClick={() => auth.logOut()} className="btn-submit">
                    logout
                </button>
            </div>
        </div>
    );
};

export default MainDashboard;