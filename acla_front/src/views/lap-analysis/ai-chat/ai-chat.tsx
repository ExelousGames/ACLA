import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, Button, Card, Flex, Text, TextField, ScrollArea, Separator, Badge, Spinner, IconButton } from '@radix-ui/themes';
import { PaperPlaneIcon, ChatBubbleIcon, PersonIcon } from '@radix-ui/react-icons';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { detectEnvironment } from 'utils/environment';
import { createAiCommandRegistry, VISUALIZATION_COMMAND_FUNCTIONS } from './ai-command-registry';
import { speakWithNeuralTts, NeuralTtsPlayback } from './neural-tts';
import { useVoiceConversation, FrontendToolHandler, VoiceEvent } from './use-voice-conversation';

type MessageKind = 'chat' | 'tool';

interface Message {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    isLoading?: boolean;
    functionCalls?: FunctionCall[];
    functionResults?: FunctionResult[];
    // Phase 2.5 — true when this AI response already streamed its own audio
    // (Kokoro chunks via SSE). The auto-speak effect skips these so we don't
    // re-synthesize the whole answer.
    streamedAudio?: boolean;
    /** Default 'chat' — text bubble. 'tool' renders the distinct
     *  tool-call box (different background + readable title). */
    kind?: MessageKind;
    /** Tool-call metadata when kind === 'tool'. */
    tool?: {
        name: string;
        title: string;
        status: 'started' | 'completed';
        ok?: boolean;
        error?: string | null;
    };
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

    // Loading and mode states
    const [isLoading, setIsLoading] = useState(false);
    const [debugMode, setDebugMode] = useState(false);
    const [TrackGuideEnabled, setTrackGuideEnabled] = useState(false);

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');

    // Text-to-speech states. Neural TTS (Kokoro) is the only path; we
    // optimistically assume it's available and flip this to false on first
    // failure so the UI can show "not available" instead of retrying.
    const [neuralTtsAvailable, setNeuralTtsAvailable] = useState(true);
    const [isTextToSpeechEnabled, setIsTextToSpeechEnabled] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    // Active neural-TTS playback handle (Phase 2 — Kokoro via /voice-synthesize).
    const currentNeuralPlaybackRef = useRef<NeuralTtsPlayback | null>(null);
    // Mirrors neuralTtsAvailable for read access inside async closures that
    // would otherwise see a stale state value.
    const neuralTtsDisabledRef = useRef<boolean>(false);
    const analysisContext = useContext(AnalysisContext);

    // Racing engineer voice conversation. The hook owns mic, WS, and
    // audio playback; it ALSO multiplexes the tool-relay text channel on
    // the same WS — frontend tools listed below are reachable from the
    // backend LLM via JSON text frames.
    //
    // What each handler does:
    //   - get_session_info: returns whatever track/car the analysis view
    //     has cached. Returned to the LLM whenever it asks (it never sees
    //     these from connect-time params anymore).
    //   - get_recent_telemetry / start_per_turn_coaching: this view
    //     reviews recorded sessions rather than live ones, so the live
    //     buffer is empty. We return a clean "no live telemetry in this
    //     view" error — the engineer's anti-hallucination rule then
    //     verbalizes "I can't see your telemetry right now" rather than
    //     fabricating numbers. A future live-driving view can plug in
    //     real implementations without touching the backend.
    const handleVoiceEvent = (event: VoiceEvent) => {
        if (event.kind === 'user_transcript') {
            setMessages(prev => prev
                .filter(m => !m.isLoading)
                .concat({
                    id: generateUniqueId('user-voice'),
                    content: event.text,
                    isUser: true,
                    timestamp: new Date(),
                    kind: 'chat',
                }));
            return;
        }
        if (event.kind === 'assistant_transcript') {
            setMessages(prev => prev
                .filter(m => !m.isLoading)
                .concat({
                    id: generateUniqueId('ai-voice'),
                    content: event.text,
                    isUser: false,
                    timestamp: new Date(),
                    kind: 'chat',
                    streamedAudio: true, // audio already played via WS PCM
                }));
            return;
        }
        if (event.kind === 'tool_event') {
            setMessages(prev => {
                // If this tool started recently and is now completed, update
                // the existing row rather than appending a new one.
                if (event.status === 'completed') {
                    for (let i = prev.length - 1; i >= 0; i--) {
                        const m = prev[i];
                        if (m.kind === 'tool' && m.tool?.name === event.name && m.tool?.status === 'started') {
                            const next = prev.slice();
                            next[i] = {
                                ...m,
                                tool: {
                                    ...m.tool,
                                    status: 'completed',
                                    ok: event.ok,
                                    error: event.error ?? null,
                                },
                            };
                            return next;
                        }
                    }
                }
                return prev.concat({
                    id: generateUniqueId('tool'),
                    content: event.title,
                    isUser: false,
                    timestamp: new Date(),
                    kind: 'tool',
                    tool: {
                        name: event.name,
                        title: event.title,
                        status: event.status,
                        ok: event.ok,
                        error: event.error ?? null,
                    },
                });
            });
            return;
        }
    };

    const voiceConversation = useVoiceConversation({
        sessionId,
        onEvent: handleVoiceEvent,
        toolHandlers: {
            get_session_info: ((): FrontendToolHandler => async () => ({
                track: analysisContext?.recordedSessioStaticsData?.track || '',
                car: analysisContext?.recordedSessioStaticsData?.car || '',
                user_id: sessionId || '',
            }))(),
            get_recent_telemetry: ((): FrontendToolHandler => async () => ({
                error: 'no_live_telemetry_in_this_view',
            }))(),
            start_per_turn_coaching: ((): FrontendToolHandler => async () => ({
                error: 'no_live_telemetry_in_this_view',
            }))(),
            stop_per_turn_coaching: ((): FrontendToolHandler => async () => ({
                status: 'stopped',
            }))(),
        },
    });

    const buildVisualizationAssistantContext = () => {
        const visualizationContext = visualizationController.getVisualizationAssistantContext();
        return {
            ...visualizationContext,
            commandFunctions: VISUALIZATION_COMMAND_FUNCTIONS
        };
    };

    // Utility function to generate unique message IDs
    const generateUniqueId = (prefix: string = 'msg') => {
        return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    };

    const addStatusMessage = (type: string, content: string) => {
        const message: Message = {
            id: generateUniqueId(type),
            content,
            isUser: false,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, message]);
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    /**
     * Strip markdown so the TTS engine doesn't read "asterisk-asterisk bold".
     */
    const cleanTextForSpeech = (text: string): string => {
        return text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove markdown bold
            .replace(/\*(.*?)\*/g, '$1') // Remove markdown italic
            .replace(/```[\s\S]*?```/g, '') // Remove code blocks
            .replace(/`(.*?)`/g, '$1') // Remove inline code
            .replace(/https?:\/\/[^\s]+/g, 'link') // Replace URLs with "link"
            .replace(/[#]+\s*/g, '') // Remove markdown headers
            .replace(/\n+/g, '. ') // Replace newlines with periods
            .replace(/\s+/g, ' ') // Normalize whitespace
            .trim();
    };

    /**
     * Speak text using neural TTS (Kokoro via /voice-synthesize).
     * Throws if unavailable; caller marks TTS unavailable for the session.
     */
    const speakWithNeural = async (
        cleanText: string,
        options?: { isGuidance?: boolean },
    ): Promise<void> => {
        if (currentNeuralPlaybackRef.current) {
            currentNeuralPlaybackRef.current.stop();
            currentNeuralPlaybackRef.current = null;
        }

        setIsSpeaking(true);
        const playback = await speakWithNeuralTts(cleanText, {
            // Slightly faster delivery for on-track guidance cues.
            speed: options?.isGuidance ? 1.15 : 1.0,
            volume: 0.9,
        });
        currentNeuralPlaybackRef.current = playback;

        try {
            await playback.ended;
        } finally {
            if (currentNeuralPlaybackRef.current === playback) {
                currentNeuralPlaybackRef.current = null;
            }
            setIsSpeaking(false);
        }
    };

    const speakText = (text: string, options?: { isGuidance?: boolean }) => {
        if (!isTextToSpeechEnabled || neuralTtsDisabledRef.current) {
            return;
        }

        const cleanText = cleanTextForSpeech(text);
        if (!cleanText) return;

        speakWithNeural(cleanText, options).catch((err) => {
            console.warn('[AI Chat] Neural TTS failed; marking unavailable for this session:', err);
            neuralTtsDisabledRef.current = true;
            setNeuralTtsAvailable(false);
            setIsSpeaking(false);
        });
    };

    const stopSpeaking = () => {
        // Phase 2 — single-WAV neural TTS playback. (The Phase 2.5 SSE
        // streaming refs were removed in the racing-engineer rebuild —
        // voice WS audio is owned by useVoiceConversation now.)
        if (currentNeuralPlaybackRef.current) {
            currentNeuralPlaybackRef.current.stop();
            currentNeuralPlaybackRef.current = null;
        }
        setIsSpeaking(false);
    };

    const toggleTextToSpeech = () => {
        const newState = !isTextToSpeechEnabled;
        setIsTextToSpeechEnabled(newState);

        // Save preference to localStorage
        localStorage.setItem('ai-chat-tts-enabled', newState.toString());

        if (!newState && isSpeaking) {
            stopSpeaking();
        }

        // Show feedback message
        const statusMessage = newState ? 'Text-to-speech enabled' : 'Text-to-speech disabled';
        addStatusMessage('tts-toggle', statusMessage);
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Automatically speak new AI messages
    useEffect(() => {
        if (!isTextToSpeechEnabled || messages.length === 0) return;

        const lastMessage = messages[messages.length - 1];

        // Only speak AI messages (not user messages) and not loading messages
        if (!lastMessage.isUser && !lastMessage.isLoading && lastMessage.content) {
            // Don't speak the welcome message on first load
            if (lastMessage.id === 'welcome' && messages.length === 1) return;

            // Phase 2.5 — streaming responses already played their own
            // sentence-chunked audio via SSE. Don't double-speak.
            if (lastMessage.streamedAudio) return;

            // Determine if this is a guidance message
            const isGuidanceMessage = lastMessage.id.includes('guidance');

            // Add a small delay to ensure the message is rendered
            setTimeout(() => {
                speakText(lastMessage.content, { isGuidance: isGuidanceMessage });
            }, 300);
        }
    }, [messages, isTextToSpeechEnabled]);

    // Listen for guidance messages from ImitationGuidanceChart
    const lastProcessedGuidanceRef = useRef<string>('');
    const lastGuidanceTimestampRef = useRef<number>(0);
    useEffect(() => {
        if (analysisContext?.latestGuidanceMessage &&
            analysisContext.latestGuidanceMessage !== lastProcessedGuidanceRef.current) {

            // Throttle guidance messages to avoid spam (max 1 per 2 seconds)
            const now = Date.now();
            if (now - lastGuidanceTimestampRef.current < 2000) {
                return;
            }

            const guidanceMessage: Message = {
                id: generateUniqueId('guidance'),
                content: analysisContext.latestGuidanceMessage,
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, guidanceMessage]);
            lastProcessedGuidanceRef.current = analysisContext.latestGuidanceMessage;
            lastGuidanceTimestampRef.current = now;
        }
    }, [analysisContext?.latestGuidanceMessage]);

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
            if (!analysisContext?.liveData || !TrackGuideEnabled) return;

            try {

                // Handle the response here if needed
            } catch (error) {
                console.error('Error fetching imitation learning guidance:', error);
            }
        };

        fetchImitationLearningGuidance();
    }, [analysisContext?.liveData, TrackGuideEnabled]);

    // Auto-manage imitation guidance chart visibility
    useEffect(() => {
        if (!TrackGuideEnabled) {
            // Remove all auto-managed imitation guidance charts when disabled
            const existingCharts = visualizationController.getCurrentInstances();
            existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                    visualizationController.executeCommand({
                        action: 'remove',
                        id: chart.id
                    });
                    console.log('Auto-manage: removed imitation guidance chart:', chart.id);
                }
            });
        }
    }, [TrackGuideEnabled, sessionId]);

    // Cleanup auto-managed charts when component unmounts
    useEffect(() => {
        return () => {
            // Remove auto-managed imitation guidance charts on unmount
            const existingCharts = visualizationController.getCurrentInstances();
            existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                    visualizationController.executeCommand({
                        action: 'remove',
                        id: chart.id
                    });
                    console.log('Cleanup: removed auto-managed imitation guidance chart:', chart.id);
                }
            });
        };
    }, []);


    // One-time setup: detect environment for the Desktop Mode badge and
    // restore the user's TTS-enabled preference. Neural TTS availability is
    // determined lazily on the first speak attempt, not up front.
    useEffect(() => {
        setEnvironment(detectEnvironment());

        const savedTtsEnabled = localStorage.getItem('ai-chat-tts-enabled');
        if (savedTtsEnabled === 'true') {
            setIsTextToSpeechEnabled(true);
        }

        return () => {
            stopSpeaking();
        };
    }, []);

    // Ctrl+Space (or Cmd+Space) — stop ongoing TTS playback. The voice WS
    // is closed via the dedicated mic toggle in the toolbar, not a hotkey.
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if ((event.ctrlKey || event.metaKey) && event.code === 'Space' && isSpeaking) {
                event.preventDefault();
                stopSpeaking();
                addStatusMessage('speech-stop', 'Text-to-speech stopped.');
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isSpeaking]);

    const handleSendMessage = async () => {
        const text = inputValue.trim();
        if (!text || isLoading) return;

        // The voice WS is the single chat surface. Backend echoes a
        // user_transcript frame for typed input, so we don't append the
        // user message locally — handleVoiceEvent will when the echo
        // arrives. This keeps voice and text turns rendered identically.
        const sent = voiceConversation.sendUserText(text);
        if (!sent) {
            setMessages(prev => prev.concat({
                id: generateUniqueId('ai'),
                content: 'Click the mic to start a voice session first — text chat runs on the same connection.',
                isUser: false,
                timestamp: new Date(),
                kind: 'chat',
            }));
            return;
        }
        setInputValue('');
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputValue(e.target.value);
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    // Function execution system
    const executeFunctionCall = async (functionCall: FunctionCall, responseData: any): Promise<FunctionResult> => {
        try {
            console.log(`Executing function: ${functionCall.function} with args:`, functionCall.arguments);

            const result = await findAndExecuteFunction(functionCall.function, functionCall.arguments, responseData);

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
    const findAndExecuteFunction = async (functionName: string, args: Record<string, any>, responseData: any): Promise<any> => {
        const registry = createAiCommandRegistry({
            sessionId,
            analysisContext,
            startTrackGuide,
            setTrackGuideEnabled
        });

        const handler = registry[functionName];
        if (!handler) {
            throw new Error(`Unknown function: ${functionName}`);
        }

        return await handler(args, responseData);
    };

    const startTrackGuide = (responseData: any) => {


        // Add imitation guidance chart with the track data

        visualizationController.openVisualization('imitation-guidance-chart', {}, {
            title: 'AI Track Guidance',
            autoUpdate: true
        });


        setTrackGuideEnabled(true);
    }


    const formatFunctionArgs = (args: Record<string, any>): string => {
        return Object.entries(args).map(([key, value]) => {
            if (typeof value === 'object') {
                return `${key}: ${JSON.stringify(value)}`;
            }
            return `${key}: ${value}`;
        }).join(', ');
    };


    // Mic input level meter — five vertical bars driven by
    // voiceConversation.micLevel (0..1). Visible only while the voice
    // session is open so the user gets immediate feedback that the mic is
    // actually picking up their voice.
    const MicMeter: React.FC<{ level: number; active: boolean }> = ({ level, active }) => {
        if (!active) return null;
        const segments = 5;
        // Apply a small visual gain — most quiet-but-audible speech sits
        // around peak 0.2-0.4, so a 1:1 mapping would feel dead.
        const lit = Math.max(0, Math.min(segments, Math.round(level * segments * 1.8)));
        return (
            <Flex gap="1" align="end" style={{ height: 24, paddingInline: 4 }} title={`Mic level: ${Math.round(level * 100)}%`}>
                {Array.from({ length: segments }, (_, i) => {
                    const isLit = i < lit;
                    const color = i >= segments - 1
                        ? 'var(--red-9)'
                        : i >= segments - 2
                            ? 'var(--amber-9)'
                            : 'var(--green-9)';
                    return (
                        <Box
                            key={i}
                            style={{
                                width: 3,
                                height: 6 + i * 3,
                                borderRadius: 1,
                                backgroundColor: isLit ? color : 'var(--gray-6)',
                                transition: 'background-color 80ms linear',
                            }}
                        />
                    );
                })}
            </Flex>
        );
    };

    // Speaker Icon Component
    const SpeakerIcon = () => (
        <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
        </svg>
    );

    /**
     * Extracts JSON object from a string that may contain markdown code blocks or other text
     * @param text - The string that may contain JSON
     * @returns Parsed JSON object or null if no valid JSON found
     */
    const extractJsonFromString = (text: string): any => {
        if (!text || typeof text !== 'string') {
            return null;
        }

        try {
            // First, try to parse the entire string as JSON (handles case where it's just {})
            return JSON.parse(text.trim());
        } catch {
            // If that fails, look for JSON within the string
        }

        // Remove markdown code blocks (```json ... ``` or ``` ... ```
        let cleanedText = text.replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1');

        // Try to find JSON object boundaries
        const jsonPatterns = [
            // Look for objects starting with { and ending with }
            /\{[\s\S]*\}/,
            // Look for arrays starting with [ and ending with ]
            /\[[\s\S]*\]/
        ];

        for (const pattern of jsonPatterns) {
            const match = cleanedText.match(pattern);
            if (match) {
                try {
                    const jsonStr = match[0].trim();
                    return JSON.parse(jsonStr);
                } catch (parseError) {
                    console.warn('Failed to parse extracted JSON:', parseError);
                    continue;
                }
            }
        }

        // Try to find multiple JSON objects in the string
        const lines = cleanedText.split('\n');
        let jsonStart = -1;
        let braceCount = 0;
        let jsonContent = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            for (let j = 0; j < line.length; j++) {
                const char = line[j];

                if (char === '{') {
                    if (jsonStart === -1) {
                        jsonStart = i;
                        jsonContent = '';
                    }
                    braceCount++;
                    jsonContent += char;
                } else if (char === '}' && jsonStart !== -1) {
                    braceCount--;
                    jsonContent += char;

                    if (braceCount === 0) {
                        // Found complete JSON object
                        try {
                            return JSON.parse(jsonContent);
                        } catch (parseError) {
                            console.warn('Failed to parse found JSON object:', parseError);
                            jsonStart = -1;
                            jsonContent = '';
                        }
                    }
                } else if (jsonStart !== -1) {
                    jsonContent += char;
                }
            }

            if (jsonStart !== -1 && i < lines.length - 1) {
                jsonContent += '\n';
            }
        }

        console.warn('No valid JSON found in string:', text);
        return null;
    };

    // Add this helper function to validate and structure guidance data
    const parseGuidanceData = (text: string): { throttle_guidance?: string[], brake_guidance?: string[], steering_guidance?: string[] } | null => {
        const jsonData = extractJsonFromString(text);

        if (!jsonData || typeof jsonData !== 'object') {
            return null;
        }

        // Validate the structure matches expected guidance format
        const guidanceData: any = {};

        if (Array.isArray(jsonData.throttle_guidance)) {
            guidanceData.throttle_guidance = jsonData.throttle_guidance;
        }

        if (Array.isArray(jsonData.brake_guidance)) {
            guidanceData.brake_guidance = jsonData.brake_guidance;
        }

        if (Array.isArray(jsonData.steering_guidance)) {
            guidanceData.steering_guidance = jsonData.steering_guidance;
        }

        // Return null if no valid guidance arrays found
        if (Object.keys(guidanceData).length === 0) {
            return null;
        }
        return guidanceData;
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
                        {environment === 'electron' && (
                            <Badge variant="soft" color="green" size="1">
                                Desktop Mode
                            </Badge>
                        )}
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
                        {!neuralTtsAvailable && (
                            <Badge variant="soft" color="orange" size="1">
                                Text-to-speech not available
                            </Badge>
                        )}
                        {voiceConversation.error && (
                            <Badge variant="soft" color="red" size="1">
                                Voice error
                            </Badge>
                        )}
                        {isTextToSpeechEnabled && (
                            <Badge variant="soft" color="green" size="1">
                                TTS On
                            </Badge>
                        )}
                    </Flex>
                </Flex>

                {/* Messages Area */}
                <ScrollArea className="ai-chat-messages" style={{ flex: 1 }}>
                    <Flex direction="column" gap="3" p="3">
                        {messages.map((message) => {
                            // Tool-call messages get their own distinct box.
                            if (message.kind === 'tool' && message.tool) {
                                const t = message.tool;
                                const isError = t.status === 'completed' && t.ok === false;
                                const isRunning = t.status === 'started';
                                const bg = isError ? 'var(--red-2)' : isRunning ? 'var(--amber-2)' : 'var(--blue-2)';
                                const bd = isError ? 'var(--red-7)' : isRunning ? 'var(--amber-7)' : 'var(--blue-7)';
                                const fg = isError ? 'var(--red-12)' : isRunning ? 'var(--amber-12)' : 'var(--blue-12)';
                                return (
                                    <Flex key={message.id} direction="column" align="start" gap="1">
                                        <Flex align="center" gap="2">
                                            <Text size="1" color="gray">Tool</Text>
                                            <Text size="1" color="gray">
                                                {message.timestamp.toLocaleTimeString()}
                                            </Text>
                                        </Flex>
                                        <Box
                                            style={{
                                                maxWidth: '80%',
                                                padding: '8px 12px',
                                                borderRadius: '8px',
                                                backgroundColor: bg,
                                                color: fg,
                                                border: `1px dashed ${bd}`,
                                            }}
                                        >
                                            <Flex align="center" gap="2">
                                                {isRunning && <Spinner size="1" />}
                                                <Text size="2" weight="medium">{t.title}</Text>
                                            </Flex>
                                            {isError && t.error && (
                                                <Text size="1" color="red" style={{ display: 'block', marginTop: 4 }}>
                                                    {t.error}
                                                </Text>
                                            )}
                                            {debugMode && (
                                                <Text size="1" color="gray" style={{ display: 'block', marginTop: 4 }}>
                                                    {t.name}
                                                </Text>
                                            )}
                                        </Box>
                                    </Flex>
                                );
                            }
                            return (
                            <Flex
                                key={message.id}
                                direction="column"
                                align={message.isUser ? "end" : "start"}
                                gap="1"
                            >
                                <Flex align="center" gap="2">
                                    {!message.isUser && <PersonIcon />}
                                    <Text size="1" color="gray">
                                        {message.isUser
                                            ? 'You'
                                            : message.id.includes('guidance')
                                                ? '🎯 Live Track Guidance'
                                                : 'AI Assistant'
                                        }
                                    </Text>
                                    {/* Show speaking indicator for AI messages */}
                                    {!message.isUser && isTextToSpeechEnabled && isSpeaking && (
                                        <Flex align="center" gap="1">
                                            <SpeakerIcon />
                                            <Text size="1" color="blue">Speaking...</Text>
                                        </Flex>
                                    )}
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
                                            : message.id.includes('guidance')
                                                ? 'var(--green-3)'  // Special styling for guidance messages
                                                : 'var(--gray-3)',
                                        color: message.isUser
                                            ? 'var(--accent-contrast)'
                                            : message.id.includes('guidance')
                                                ? 'var(--green-12)'  // Special color for guidance messages
                                                : 'var(--gray-12)',
                                        border: message.id.includes('guidance') ? '1px solid var(--green-7)' : 'none'
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

                                            {/* Individual message speech control for AI messages */}
                                            {!message.isUser && neuralTtsAvailable && !message.isLoading && (
                                                <Flex justify="end" mt="1">
                                                    <Button
                                                        variant="ghost"
                                                        size="1"
                                                        onClick={() => speakText(message.content, { isGuidance: message.id.includes('guidance') })}
                                                        disabled={isSpeaking}
                                                        title="Speak this message"
                                                        style={{ opacity: 0.7 }}
                                                    >
                                                        <SpeakerIcon />
                                                    </Button>
                                                </Flex>
                                            )}

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
                            );
                        })}
                        <div ref={messagesEndRef} />
                    </Flex>
                </ScrollArea>

                {/* Input Area */}
                <Separator />
                <Box p="3">
                    <Flex gap="2">
                        <TextField.Root
                            placeholder={
                                voiceConversation.state === 'listening' || voiceConversation.state === 'speaking'
                                    ? 'Type a message to the engineer…'
                                    : 'Click the mic to start a session, then type or talk.'
                            }
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            disabled={isLoading}
                            style={{ flex: 1 }}
                        />
                        {/* Text-to-speech controls next to microphone */}
                        {neuralTtsAvailable && (
                            <IconButton
                                onClick={isSpeaking ? stopSpeaking : toggleTextToSpeech}
                                disabled={isLoading}
                                size="2"
                                variant={isTextToSpeechEnabled ? "solid" : "ghost"}
                                color={isSpeaking ? "red" : isTextToSpeechEnabled ? "green" : "gray"}
                                title={
                                    isSpeaking
                                        ? "Stop speaking"
                                        : isTextToSpeechEnabled
                                            ? "Disable auto text-to-speech"
                                            : "Enable auto text-to-speech"
                                }
                            >
                                {isSpeaking ? "🔇" : isTextToSpeechEnabled ? "🔊" : "🔇"}
                            </IconButton>
                        )}
                        {/* Mic level meter — shows the user that their voice is reaching the system. */}
                        <MicMeter
                            level={voiceConversation.micLevel}
                            active={voiceConversation.state === 'listening' || voiceConversation.state === 'speaking'}
                        />
                        {/* Phase 3 — Talk to coach (full voice conversation via Pipecat). */}
                        <IconButton
                            onClick={() => {
                                if (voiceConversation.state === 'idle' || voiceConversation.state === 'error') {
                                    voiceConversation.start().catch((err) => {
                                        console.error('Voice conversation failed to start:', err);
                                    });
                                } else {
                                    voiceConversation.stop();
                                }
                            }}
                            disabled={isLoading || voiceConversation.state === 'connecting'}
                            size="2"
                            variant={voiceConversation.state === 'idle' || voiceConversation.state === 'error' ? 'ghost' : 'solid'}
                            color={
                                voiceConversation.state === 'error'
                                    ? 'orange'
                                    : voiceConversation.state === 'speaking'
                                        ? 'blue'
                                        : voiceConversation.state === 'listening'
                                            ? 'red'
                                            : 'gray'
                            }
                            title={
                                voiceConversation.state === 'error'
                                    ? `Voice error: ${voiceConversation.error}. Click to retry.`
                                    : voiceConversation.state === 'connecting'
                                        ? 'Connecting to voice coach...'
                                        : voiceConversation.state === 'listening'
                                            ? 'Listening — click to end voice session'
                                            : voiceConversation.state === 'speaking'
                                                ? 'Coach speaking — click to end voice session'
                                                : 'Talk to coach (full voice conversation)'
                            }
                        >
                            {voiceConversation.state === 'idle' || voiceConversation.state === 'error' ? '💬' : '🎙️'}
                        </IconButton>

                        <Button
                            onClick={() => handleSendMessage()}
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
