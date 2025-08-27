import React, { useState } from 'react';
import './register-user.css';
import {
    Box,
    Button,
    Card,
    Flex,
    Heading,
    Text,
    TextField,
} from "@radix-ui/themes";
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

// Register user component
const RegisterUser = () => {
    const [input, setInput] = useState({
        email: "",
        password: "",
        confirmPassword: "",
    });
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState("");
    const navigate = useNavigate();

    // Backend server API
    const serverIPandPort = process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT;
    const server_url_header = 'http://' + serverIPandPort;

    // Handle form submission
    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        if (!input.email || !input.password) {
            setMessage("Please provide email and password");
            return;
        }

        if (input.password !== input.confirmPassword) {
            setMessage("Passwords do not match");
            return;
        }

        setIsLoading(true);
        try {
            const response = await axios.post(server_url_header + '/userinfo/register', {
                email: input.email,
                password: input.password
            });

            setMessage("User registered successfully! Redirecting to login...");
            setTimeout(() => {
                navigate("/login");
            }, 2000);
        } catch (error) {
            console.error('Error during registration:', error);
            setMessage("Registration failed. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    // Handle input changes
    const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setInput(oldValues => {
            return { ...oldValues, [name]: value }
        });
    };

    return (
        <form className='center' onSubmit={handleSubmit}>
            <Box width="400px">
                <Card size="4">
                    <Heading as="h3" size="6" trim="start" mb="5">
                        Register
                    </Heading>

                    {message && (
                        <Box mb="3">
                            <Text size="2" color={message.includes("successfully") ? "green" : "red"}>
                                {message}
                            </Text>
                        </Box>
                    )}

                    <Box mb="5">
                        <Flex mb="1">
                            <Text as="label" htmlFor="register-email-field" size="2" weight="bold">
                                Email address
                            </Text>
                        </Flex>
                        <TextField.Root
                            placeholder="Enter your email"
                            id="register-email-field"
                            name="email"
                            type="email"
                            value={input.email}
                            onChange={handleInput}
                            required
                        />
                    </Box>

                    <Box mb="5">
                        <Flex mb="1">
                            <Text as="label" size="2" weight="bold" htmlFor="register-password-field">
                                Password
                            </Text>
                        </Flex>
                        <TextField.Root
                            placeholder="Enter your password"
                            id="register-password-field"
                            name="password"
                            type="password"
                            value={input.password}
                            onChange={handleInput}
                            required
                        />
                    </Box>

                    <Box mb="5">
                        <Flex mb="1">
                            <Text as="label" size="2" weight="bold" htmlFor="register-confirm-password-field">
                                Confirm Password
                            </Text>
                        </Flex>
                        <TextField.Root
                            placeholder="Confirm your password"
                            id="register-confirm-password-field"
                            name="confirmPassword"
                            type="password"
                            value={input.confirmPassword}
                            onChange={handleInput}
                            required
                        />
                    </Box>

                    <Flex mt="6" justify="end" gap="3">
                        <Button
                            variant="outline"
                            onClick={() => navigate("/login")}
                            type="button"
                        >
                            Back to Login
                        </Button>
                        <Button type='submit' disabled={isLoading}>
                            {isLoading ? "Registering..." : "Register"}
                        </Button>
                    </Flex>
                </Card>
            </Box>
        </form>
    );
};

export default RegisterUser;
