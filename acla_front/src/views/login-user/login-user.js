import React, { useState } from 'react';
import './login-user.css';
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

const LoginUser = () => {
    const [input, setInput] = useState({
        email: "",
        password: "",
    });

    //use Auth custom context
    const auth = useAuth();

    //when user tries to press login button
    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.email !== "" || input.password !== "") {
            auth.loginAction({ email: input.email, password: input.password });
            return
        }
        alert("please provide a valid input");
    };

    //check changes in the input
    const handleInput = (e) => {
        const { name, value } = e.target;
        //if we only called setInput [name]: value, this would remove others from the state. oldValues is the old value of the state
        setInput(oldValues => {
            return { ...oldValues, [name]: value }
        });
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
                            <Text as="label" htmlFor="example-email-field" size="2" weight="bold" >
                                Email address
                            </Text>
                        </Flex>
                        <TextField.Root placeholder="Enter your email" id="example-email-field" name="email" onChange={handleInput} />
                    </Box>

                    <Box mb="5" position="relative">
                        <Flex align="baseline" justify="between" mb="1">
                            <Text as="label" size="2" weight="bold" htmlFor="example-password-field" >
                                Password
                            </Text>
                        </Flex>
                        <TextField.Root placeholder="Enter your password" id="example-password-field" name="password" onChange={handleInput} />
                    </Box>

                    <Flex mt="6" justify="end" gap="3">
                        {/*
                        <Button variant="outline">
                            Create an account
                        </Button>
                        */}
                        <Button type='submit'>Sign in</Button>
                    </Flex>

                </Card>
            </Box>
        </form>
    );
};

export default LoginUser;