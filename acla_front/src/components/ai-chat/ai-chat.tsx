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
    functionCalls?: FunctionCall[];
    functionResults?: FunctionResult[];
}

interface FunctionCall {
    function: string;
    arguments: Record<string, any>;
}

interface FunctionResult {
    function: string;
    arguments: Record<string, any>;
    result: any;
    success: boolean;
    error?: string;
}

interface AiChatProps {
    sessionId?: string;
    title?: string;
}

const AiChat: React.FC<AiChatProps> = ({ sessionId, title = "AI Assistant" }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [debugMode, setDebugMode] = useState(false);
    const [imitationLearningEnabled, setImitationLearningEnabled] = useState(false);
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
            if (!analysisContext?.liveData || !imitationLearningEnabled) return;

            try {
                const response = await apiService.post('/racing-session/imitation-learning-guidance', {
                    current_telemetry: analysisContext.liveData,
                    guidance_type: "both", // "actions", "behavior", or "both"
                });
                // Handle the response here if needed
                console.log('Imitation learning guidance response:', response.data);
            } catch (error) {
                console.error('Error fetching imitation learning guidance:', error);
            }
        };

        fetchImitationLearningGuidance();
    }, [analysisContext?.liveData, imitationLearningEnabled]);


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

            // Use openai general natural language ai query endpoint
            response = await apiService.post('ai-model/ai-query', {
                question: userMessage.content,
                sessionId: sessionId,
            });

            const responseData = response.data as any;
            let aiResponseContent = responseData?.answer || responseData?.response || "I'm sorry, I couldn't process your request.";
            let functionCalls: FunctionCall[] = [];
            let functionResults: FunctionResult[] = [];
            console.log('AI response data:', responseData);

            // Check if the response contains function calls
            if (responseData?.function_calls && Array.isArray(responseData.function_calls)) {

                try {
                    // Parse function calls from the response
                    functionCalls = responseData.function_calls.map((fc: any) => ({
                        function: fc.function || fc.name,
                        arguments: typeof fc.arguments === 'string' ? JSON.parse(fc.arguments) : fc.arguments
                    }));

                    // Execute all function calls
                    for (const functionCall of functionCalls) {
                        const result = await executeFunctionCall(functionCall);
                        functionResults.push(result);
                    }

                } catch (parseError) {
                    console.error('Error parsing function calls:', parseError);
                    aiResponseContent += '\n\n*Note: Function calls were detected but could not be parsed properly.*';
                }
            }

            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                content: aiResponseContent,
                isUser: false,
                timestamp: new Date(),
                functionCalls: functionCalls.length > 0 ? functionCalls : undefined,
                functionResults: functionResults.length > 0 ? functionResults : undefined
            };

            // Remove loading messages and add AI response
            setMessages(prev => prev.filter(msg => !msg.id.includes('loading') && !msg.id.includes('executing')).concat(aiResponse));
        } catch (error) {
            console.error('Error sending message to AI:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: "I'm sorry, I encountered an error while processing your request. Please try again.",
                isUser: false,
                timestamp: new Date()
            };

            // Remove loading messages and add error message
            setMessages(prev => prev.filter(msg => !msg.id.includes('loading') && !msg.id.includes('executing')).concat(errorMessage));
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
            "Show me sector-by-sector analysis",
            "Get detailed telemetry data for this session",
            "Compare this session with my previous ones",
            "Help me follow the optimal racing line"
        ];
    };

    const handleQuickQuestion = (question: string) => {
        setInputValue(question);
        inputRef.current?.focus();
    };

    // Function execution system
    const executeFunctionCall = async (functionCall: FunctionCall): Promise<FunctionResult> => {
        try {
            console.log(`Executing function: ${functionCall.function} with args:`, functionCall.arguments);

            const result = await findAndExecuteFunction(functionCall.function, functionCall.arguments);

            return {
                function: functionCall.function,
                arguments: functionCall.arguments,
                result,
                success: true
            };
        } catch (error) {
            console.error(`Error executing function ${functionCall.function}:`, error);
            return {
                function: functionCall.function,
                arguments: functionCall.arguments,
                result: null,
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error'
            };
        }
    };

    // Define available functions that can be called
    const findAndExecuteFunction = async (functionName: string, args: Record<string, any>): Promise<any> => {
        // Add session context to function arguments if available and not already provided
        const sessionIdToUse = args.session_id ||
            sessionId ||
            analysisContext?.sessionSelected?.SessionId;

        switch (functionName) {
            case 'get_session_analysis':
                return await apiService.post('/racing-session/detailed-info', {
                    id: sessionIdToUse
                });

            case 'get_telemetry_data':
                return await apiService.post('/racing-session/telemetry', {
                    session_id: sessionIdToUse,
                    data_types: args.data_types || ['speed', 'acceleration']
                });

            case 'compare_lap_times':
                return await apiService.post('/racing-session/compare', {
                    session_ids: args.session_ids,
                    metrics: args.metrics || ['lap_times']
                });

            case 'get_performance_insights':
                return await apiService.post('/ai/performance-analysis', {
                    session_id: sessionIdToUse,
                    analysis_type: args.analysis_type || 'comprehensive'
                });

            case 'follow_expert_line':
                return await apiService.post('/ai/expert-line-guidance', {
                    session_id: sessionIdToUse,
                    data_types: args.data_types || ['speed', 'acceleration', 'braking', 'steering']
                });

            case 'enable_guide_user_racing':
                // Enable the continuous imitation learning guidance
                setImitationLearningEnabled(true);

                return {
                    status: 'Imitation learning guidance enabled - now continuously monitoring telemetry data',
                    enabled: true
                };

            case 'disable_guide_user_racing':
                // Disable the continuous imitation learning guidance
                setImitationLearningEnabled(false);

                return {
                    status: 'Imitation learning guidance disabled - no longer monitoring telemetry data',
                    enabled: false
                };

            case 'disable_ui_component':
                // Handle UI updates locally
                if (args.component === 'chart' && analysisContext) {
                    // Trigger chart update through context
                    console.log('Updating UI component:', args);
                    return { success: true, message: 'UI updated successfully' };
                }
                return { success: false, message: 'UI component not found or not supported' };

            case 'get_available_functions':
                // Return list of available functions
                return {
                    functions: [
                        'get_session_analysis',
                        'get_telemetry_data',
                        'compare_lap_times',
                        'get_performance_insights',
                        'follow_expert_line',
                        'get_imitation_learning_guidance',
                        'update_ui_component'
                    ],
                    session_context: !!sessionId,
                    analysis_context: !!analysisContext,
                    current_session: sessionIdToUse
                };

            default:
                throw new Error(`Unknown function: ${functionName}`);
        }
    };

    // Utility function to format function arguments for display
    const formatFunctionArgs = (args: Record<string, any>): string => {
        return Object.entries(args).map(([key, value]) => {
            if (typeof value === 'object') {
                return `${key}: ${JSON.stringify(value)}`;
            }
            return `${key}: ${value}`;
        }).join(', ');
    };


    return (
        <Card className="ai-chat-container">
            <Flex direction="column" height="100%">
                {/* Header */}
                <Flex align="center" justify="between" p="3" style={{ borderBottom: '1px solid var(--gray-6)' }}>
                    <Flex align="center" gap="2">
                        <ChatBubbleIcon />
                        <Text size="4" weight="medium">{title}</Text>
                        {sessionId && <Badge variant="soft" color="blue">Session Analysis</Badge>}
                    </Flex>
                    <Flex align="center" gap="2">
                        <Button
                            variant="ghost"
                            size="1"
                            onClick={() => setDebugMode(!debugMode)}
                            color={debugMode ? "blue" : "gray"}
                        >
                            Debug
                        </Button>
                    </Flex>
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
                                        <>
                                            <Text size="2" style={{ whiteSpace: 'pre-wrap' }}>
                                                {message.content}
                                            </Text>

                                            {/* Show function execution indicator (always visible) */}
                                            {!debugMode && message.functionResults && message.functionResults.length > 0 && (
                                                <Box mt="2">
                                                    <Badge
                                                        variant="soft"
                                                        color={message.functionResults.every(r => r.success) ? "green" : "orange"}
                                                        size="1"
                                                    >
                                                        {message.functionResults.every(r => r.success)
                                                            ? `${message.functionResults.length} command(s) executed successfully`
                                                            : `${message.functionResults.filter(r => r.success).length}/${message.functionResults.length} commands executed`
                                                        }
                                                    </Badge>
                                                </Box>
                                            )}

                                            {/* Display function calls if present and debug mode is on */}
                                            {debugMode && message.functionCalls && message.functionCalls.length > 0 && (
                                                <Box mt="2" p="2" style={{
                                                    backgroundColor: 'var(--gray-2)',
                                                    borderRadius: '6px',
                                                    border: '1px solid var(--gray-6)'
                                                }}>
                                                    <Text size="1" weight="bold" color="gray">
                                                        Function Calls Executed:
                                                    </Text>
                                                    {message.functionCalls.map((fc, index) => (
                                                        <Box key={index} mt="1">
                                                            <Text size="1" color="blue">
                                                                {fc.function}({formatFunctionArgs(fc.arguments)})
                                                            </Text>
                                                        </Box>
                                                    ))}
                                                </Box>
                                            )}

                                            {/* Display function results if present and debug mode is on */}
                                            {debugMode && message.functionResults && message.functionResults.length > 0 && (
                                                <Box mt="2" p="2" style={{
                                                    backgroundColor: message.functionResults.some(r => !r.success)
                                                        ? 'var(--red-2)'
                                                        : 'var(--green-2)',
                                                    borderRadius: '6px',
                                                    border: `1px solid ${message.functionResults.some(r => !r.success)
                                                        ? 'var(--red-6)'
                                                        : 'var(--green-6)'}`
                                                }}>
                                                    <Text size="1" weight="bold" color="gray">
                                                        Function Results:
                                                    </Text>
                                                    {message.functionResults.map((fr, index) => (
                                                        <Box key={index} mt="1">
                                                            <Text size="1" color={fr.success ? "green" : "red"}>
                                                                {fr.function}: {fr.success ? "✓ Success" : "✗ Error"}
                                                                {fr.error && ` - ${fr.error}`}
                                                            </Text>
                                                        </Box>
                                                    ))}
                                                </Box>
                                            )}
                                        </>
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
