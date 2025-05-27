import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import './create-user-info.css';

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

const CreateUserInfo = ({ onTaskCreated }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const serverIPandPort = process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT
    const server_url_header = 'http://' + serverIPandPort

    const handleSubmit = (e) => {
        e.preventDefault();
        axios.post(server_url_header + '/userinfo', { infoDto: { name: name } })
            .then(response => {
                setName('');
                setDescription('');
            })
            .catch(error => console.error('Error creating task:', error));
    };

    return (
        <div className = 'center'>
            <Box width="400px" height="64px">
                <Card size="4">
                            <Heading as="h3" size="6" trim="start" mb="5">
                                Login
                            </Heading>

                            <Box mb="5">
                                <Flex mb="1">
                                    <Text
                                        as="label"
                                        htmlFor="example-email-field" 
                                        size="2"
                                        weight="bold"
                                    >
                                        Email address
                                    </Text>
                                </Flex>
                                <TextField.Root
                                    placeholder="Enter your email"
                                    id="example-email-field"
                                />
                            </Box>

                            <Box mb="5" position="relative">
                                <Flex align="baseline" justify="between" mb="1">
                                    <Text
                                        as="label"
                                        size="2"
                                        weight="bold"
                                        htmlFor="example-password-field"
                                    >
                                        Password
                                    </Text>
                                </Flex>
                                <TextField.Root
                                    placeholder="Enter your password"
                                    id="example-password-field"
                                />
                            </Box>

                            <Flex mt="6" justify="end" gap="3">
                                <Button variant="outline">
                                    Create an account
                                </Button>
                                <Button>Sign in</Button>
                            </Flex>
                </Card>
            </Box>
        </div>
    );
};

export default CreateUserInfo;