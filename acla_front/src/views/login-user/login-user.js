import React, { useState } from 'react';
import axios from 'axios';
import './login-user.css';

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
    const [input, setInput] = useState({
        username: "",
        password: "",
    });

    const serverIPandPort = process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT
    const server_url_header = 'http://' + serverIPandPort

    const handleSubmit = (e) => {
        e.preventDefault();
        e.preventDefault();
        if (input.username !== "" && input.password !== "") {
            axios.post(server_url_header + '/userinfo', { infoDto: { name: name } })
                .then(response => {
                })
                .catch(error => console.error('Error creating task:', error));
        }
        alert("please provide a valid input");
    };

    const handleInput = (e) => {
        const { name, value } = e.target;
        setInput(values => ({ ...values, [name]: value }));
    };

    return (
        <form className='center' onSubmit={handleSubmit}>
            <Box width="400px" height="64px">
                <Card size="4">
                    <Heading as="h3" size="6" trim="start" mb="5">
                        Login
                    </Heading>

                    <Box mb="5">
                        <Flex mb="1">
                            <Text as="label" htmlFor="example-email-field" size="2" weight="bold" name="email" onChange={handleInput}>
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
                            <Text as="label" size="2" weight="bold" htmlFor="example-password-field" name="password" onChange={handleInput}>
                                Password
                            </Text>
                        </Flex>
                        <TextField.Root placeholder="Enter your password" id="example-password-field" />
                    </Box>

                    <Flex mt="6" justify="end" gap="3">
                        <Button variant="outline" type="submit">
                            Create an account
                        </Button>
                        <Button>Sign in</Button>
                    </Flex>
                </Card>
            </Box>
        </form>
    );
};

export default CreateUserInfo;