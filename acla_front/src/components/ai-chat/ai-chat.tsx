import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, Button, Card, Flex, Text, TextField, ScrollArea, Separator, Badge, Spinner } from '@radix-ui/themes';
import { PaperPlaneIcon, ChatBubbleIcon, PersonIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/session-analysis';

interface Message {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    isLoading?: boolean;
}

interface AiChatProps {
    sessionId?: string;
    title?: string;
}

const AiChat: React.FC<AiChatProps> = ({ sessionId, title = "AI Assistant" }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const analysisContext = useContext(AnalysisContext);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // Add welcome message when component mounts
        if (messages.length === 0) {
            const welcomeMessage: Message = {
                id: 'welcome',
                content: sessionId
                    ? "Hello! I'm your AI assistant. I can help you analyze your racing session data. What would you like to know?"
                    : "Hello! I'm your AI assistant. How can I help you today?",
                isUser: false,
                timestamp: new Date()
            };
            setMessages([welcomeMessage]);
        }
    }, [sessionId, messages.length]);

    useEffect(() => {
        const fetchImitationLearningGuidance = async () => {
            if (!analysisContext?.liveData) return;

            try {
                const response = await apiService.post('/imitation-learning-guidance', {
                    current_telemetry: analysisContext.liveData
                });
                // Handle the response here if needed
                console.log('Imitation learning guidance response:', response.data);
            } catch (error) {
                console.error('Error fetching imitation learning guidance:', error);
            }
        };

        fetchImitationLearningGuidance();
    }, [analysisContext?.liveData]);


    const handleSendMessage = async () => {
        if (!inputValue.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            content: inputValue.trim(),
            isUser: true,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);

        // Add loading message
        const loadingMessage: Message = {
            id: 'loading',
            content: 'Thinking...',
            isUser: false,
            timestamp: new Date(),
            isLoading: true
        };
        setMessages(prev => [...prev, loadingMessage]);

        try {
            let response;

            if (sessionId) {
                // Use racing session specific endpoint
                response = await apiService.post('/ai-model/ai-query', {
                    session_id: sessionId,
                    question: userMessage.content
                });
            } else {
                // Use general query endpoint
                response = await apiService.post('/ai/query', {
                    question: userMessage.content
                });
            }

            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                content: (response.data as any)?.answer || (response.data as any)?.response || "I'm sorry, I couldn't process your request.",
                isUser: false,
                timestamp: new Date()
            };

            // Remove loading message and add AI response
            setMessages(prev => prev.filter(msg => msg.id !== 'loading').concat(aiResponse));
        } catch (error) {
            console.error('Error sending message to AI:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: "I'm sorry, I encountered an error while processing your request. Please try again.",
                isUser: false,
                timestamp: new Date()
            };

            // Remove loading message and add error message
            setMessages(prev => prev.filter(msg => msg.id !== 'loading').concat(errorMessage));
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const getQuickQuestions = () => {
        if (!sessionId) return [];

        return [
            "What are my key performance metrics?",
            "Where can I improve my lap times?",
            "Analyze my cornering performance",
            "Show me sector-by-sector analysis"
        ];
    };

    const handleQuickQuestion = (question: string) => {
        setInputValue(question);
        inputRef.current?.focus();
    };


    const enable_limitation_guidance = async () => {


    };
    return (
        <Card className="ai-chat-container">
            <Flex direction="column" height="100%">
                {/* Header */}
                <Flex align="center" gap="2" p="3" style={{ borderBottom: '1px solid var(--gray-6)' }}>
                    <ChatBubbleIcon />
                    <Text size="4" weight="medium">{title}</Text>
                    {sessionId && <Badge variant="soft" color="blue">Session Analysis</Badge>}
                </Flex>

                {/* Messages Area */}
                <ScrollArea className="ai-chat-messages" style={{ flex: 1 }}>
                    <Flex direction="column" gap="3" p="3">
                        {messages.map((message) => (
                            <Flex
                                key={message.id}
                                direction="column"
                                align={message.isUser ? "end" : "start"}
                                gap="1"
                            >
                                <Flex align="center" gap="2">
                                    {!message.isUser && <PersonIcon />}
                                    <Text size="1" color="gray">
                                        {message.isUser ? 'You' : 'AI Assistant'}
                                    </Text>
                                    <Text size="1" color="gray">
                                        {message.timestamp.toLocaleTimeString()}
                                    </Text>
                                </Flex>
                                <Box
                                    className={`ai-chat-message ${message.isUser ? 'user' : 'ai'}`}
                                    style={{
                                        maxWidth: '80%',
                                        padding: '8px 12px',
                                        borderRadius: '12px',
                                        backgroundColor: message.isUser
                                            ? 'var(--accent-9)'
                                            : 'var(--gray-3)',
                                        color: message.isUser
                                            ? 'var(--accent-contrast)'
                                            : 'var(--gray-12)'
                                    }}
                                >
                                    {message.isLoading ? (
                                        <Flex align="center" gap="2">
                                            <Spinner size="1" />
                                            <Text size="2">{message.content}</Text>
                                        </Flex>
                                    ) : (
                                        <Text size="2" style={{ whiteSpace: 'pre-wrap' }}>
                                            {message.content}
                                        </Text>
                                    )}
                                </Box>
                            </Flex>
                        ))}
                        <div ref={messagesEndRef} />
                    </Flex>
                </ScrollArea>

                {/* Quick Questions */}
                {getQuickQuestions().length > 0 && messages.length <= 1 && (
                    <>
                        <Separator />
                        <Box p="3">
                            <Text size="2" color="gray" mb="2" style={{ display: 'block' }}>
                                Quick questions:
                            </Text>
                            <Flex direction="column" gap="2">
                                {getQuickQuestions().map((question, index) => (
                                    <Button
                                        key={index}
                                        variant="ghost"
                                        size="2"
                                        onClick={() => handleQuickQuestion(question)}
                                        style={{ justifyContent: 'flex-start' }}
                                    >
                                        {question}
                                    </Button>
                                ))}
                            </Flex>
                        </Box>
                    </>
                )}

                {/* Input Area */}
                <Separator />
                <Box p="3">
                    <Flex gap="2">
                        <TextField.Root
                            ref={inputRef}
                            placeholder="Ask me anything about your racing session..."
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyPress={handleKeyPress}
                            disabled={isLoading}
                            style={{ flex: 1 }}
                        />
                        <Button
                            onClick={handleSendMessage}
                            disabled={!inputValue.trim() || isLoading}
                            size="2"
                        >
                            <PaperPlaneIcon />
                        </Button>
                    </Flex>
                </Box>
            </Flex>
        </Card>
    );
};

export default AiChat;
