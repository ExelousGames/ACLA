import './header-menu.css';

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
    Container,
    Tabs
} from "@radix-ui/themes";
import { useAuth } from 'hooks/AuthProvider';


import { useEffect, useState, createContext } from 'react';


const HeaderMenu = () => {
    const auth = useAuth();

    const handleClick = (event: any) => {
        auth?.logout();
    };

    return (
        <div className='menu-header'>
            <Box className="MainDashboardTitle Title">Assetto Corsa Competizione Lap Analysis</Box>
            <DropdownMenu.Root>
                <DropdownMenu.Trigger>
                    <button className='header-menu-button'>
                        <Avatar src="" fallback="A" />
                        <DropdownMenu.TriggerIcon className='header-menu-trigger-icon' />
                    </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                    <DropdownMenu.Item >Profile</DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item shortcut="ctrl c" color="red" onClick={handleClick}>
                        Log out
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>
        </div>
    );

};

export default HeaderMenu;