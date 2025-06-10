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


import { useEffect, useState, createContext } from 'react';


const HeaderMenu = () => {

    return (
        <div className='menu-header'>
            <Box className="MainDashboardTitle Title">Assetto Corsa Competizione Lap Analysis</Box>
            <DropdownMenu.Root>
                <DropdownMenu.Trigger>
                    <Button className='header-menu-button'>
                        <Avatar
                            src=""
                            fallback="A"
                        />
                        <DropdownMenu.TriggerIcon />
                    </Button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                    <DropdownMenu.Item shortcut="⌘ E">Profile</DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item shortcut="⌘ ⌫" color="red">
                        Log out
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>
        </div>
    );

};

export default HeaderMenu;