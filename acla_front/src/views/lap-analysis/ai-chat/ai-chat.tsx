import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, Button, Card, Flex, Text, TextField, ScrollArea, Separator, Badge, Spinner, IconButton } from '@radix-ui/themes';
import { PaperPlaneIcon, ChatBubbleIcon, PersonIcon } from '@radix-ui/react-icons';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { detectEnvironment } from 'utils/environment';
import { createAiCommandRegistry, VISUALIZATION_COMMAND_FUNCTIONS } from './ai-command-registry';
import { speakWithNeuralTts, NeuralTtsPlayback } from './neural-tts';
import { useVoiceConversation, FrontendToolHandler } from './use-voice-conversation';

// Type declarations for Web Speech API
declare global {
    interface Window {
        SpeechRecognition: typeof SpeechRecognition;
        webkitSpeechRecognition: typeof SpeechRecognition;
    }
}

interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    abort(): void;
    onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
    onend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
}

interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
    resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
    error: string;
    message: string;
}

interface SpeechRecognitionResultList {
    length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
    length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
    isFinal: boolean;
}

interface SpeechRecognitionAlternative {
    transcript: string;
    confidence: number;
}

declare var SpeechRecognition: {
    prototype: SpeechRecognition;
    new(): SpeechRecognition;
};

interface Message {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    isLoading?: boolean;
    functionCalls?: FunctionCall[];
    functionResults?: FunctionResult[];
    isVoiceInput?: boolean;
    // Phase 2.5 — true when this AI response already streamed its own audio
    // (Kokoro chunks via SSE). The auto-speak effect skips these so we don't
    // re-synthesize the whole answer.
    streamedAudio?: boolean;
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

    // Simplified recording state management
    const [recording, setRecording] = useState({
        error: null as string | null,
        transcript: '',
        // idle means not actively recording or processing, completed means recording completed and finished processing.
        status: 'inactive' as 'inactive' | 'initing' | 'idle' | 'listening' | 'detected' | 'processing' | 'completed' | 'error'
    });

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');
    const [speechRecognition, setSpeechRecognition] = useState<SpeechRecognition | null>(null);
    const [electronSpeechAvailable, setElectronSpeechAvailable] = useState(false);

    // Text-to-speech states. Neural TTS (Kokoro) is the only path; we
    // optimistically assume it's available and flip this to false on first
    // failure so the UI can show "not available" instead of retrying.
    const [neuralTtsAvailable, setNeuralTtsAvailable] = useState(true);
    const [isTextToSpeechEnabled, setIsTextToSpeechEnabled] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);

    // Helper functions for recording state management
    const isUninteractableState = recording.status === 'initing' || recording.status === 'processing' || recording.status === 'listening';
    const isRecordingCompleted = recording.status === 'completed';
    const isVoiceActive = recording.status === 'listening' || recording.status === 'initing' || recording.status === 'processing';


    const updateRecording = (updates: Partial<typeof recording>) => {
        console.log('🎤 Recording state update:', updates);
        setRecording(prev => {
            const newState = { ...prev, ...updates };
            console.log('🎤 New recording state:', newState);
            return newState;
        });
    };

    const resetRecording = (clearTranscript = false) => {
        setRecording(prev => ({
            ...prev,
            error: null,
            status: 'idle',
            ...(clearTranscript && {
                transcript: ''
            })
        }));
    };

    const startRecording = () => {
        console.log('🎤 Starting recording - setting status to listening');
        updateRecording({
            error: null,
            status: 'initing',
            transcript: ''
        });
    };

    const stopRecording = (transcript = '') => {
        console.log('🎤 Stopping recording - setting status to idle');
        updateRecording({
            status: 'idle',
            transcript
        });
    };

    const setRecordingError = (error: string) => {
        console.log('🎤 Recording error:', error);
        updateRecording({
            error,
            status: 'error'
        });
    };

    // Simplified refs - combine timeouts into one object
    const timeoutRefs = useRef({
        silence: null as NodeJS.Timeout | null,
        voice: null as NodeJS.Timeout | null
    });
    const lastStartAttemptRef = useRef<number>(0);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
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
    const voiceConversation = useVoiceConversation({
        sessionId,
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

    // Simplified utility functions
    const resetRecordingState = (clearTranscripts = false) => {
        resetRecording(clearTranscripts);
    };

    const clearAllTimeouts = () => {
        Object.values(timeoutRefs.current).forEach(timeout => {
            if (timeout) {
                clearTimeout(timeout);
            }
        });
        timeoutRefs.current = { silence: null, voice: null };
    };

    const addStatusMessage = (type: string, content: string) => {
        const message: Message = {
            id: generateUniqueId(type),
            content: `🎤 ${content}`,
            isUser: false,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, message]);
    };

    const handleRecordingError = (error: string, shouldShowMessage = true) => {
        setRecordingError(error);
        clearAllTimeouts();
        if (shouldShowMessage) {
            addStatusMessage('error', error);
        }
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


    // Initialize speech recognition
    useEffect(() => {
        const currentEnvironment = detectEnvironment();
        setEnvironment(currentEnvironment);

        // Load saved TTS preference — neural TTS availability is determined
        // lazily on the first speak attempt, not up front.
        const savedTtsEnabled = localStorage.getItem('ai-chat-tts-enabled');
        if (savedTtsEnabled === 'true') {
            setIsTextToSpeechEnabled(true);
        }

        let cleanup: (() => void) | undefined;

        const initialize = async () => {
            if (currentEnvironment === 'electron') {
                // Check if Electron speech recognition is available
                cleanup = await initializeElectronSpeechRecognition();
            } else {
                // Initialize web speech recognition for browser
                initializeWebSpeechRecognition();
            }
        };

        initialize().catch(error => {
            console.error('Error initializing speech recognition:', error);
        });

        // Return cleanup function
        return () => {
            if (cleanup) {
                cleanup();
            }
            // Clean up text-to-speech
            stopSpeaking();
        };
    }, []);

    const initializeElectronSpeechRecognition = async () => {
        try {
            if (window.electronAPI && window.electronAPI.isSpeechRecognitionAvailable) {

                // Check availability
                const available = await window.electronAPI.isSpeechRecognitionAvailable();
                setElectronSpeechAvailable(available);

                if (available) {
                    // Set up event listeners for speech recognition
                    const statusUnsubscribe = window.electronAPI.onSpeechRecognitionStatus((status) => {
                        console.log('Speech recognition status:', status);

                        // Handle enhanced status updates
                        if (status.status === 'listening') {
                            updateRecording({ status: 'listening', error: null });

                        } else if (status.status === 'speech_detected') {
                            updateRecording({ status: 'detected', error: null });
                        } else if (status.status === 'processing') {
                            updateRecording({ status: 'processing', error: null });
                        }
                        else if (status.status === 'calibrating') {
                            updateRecording({ status: 'initing', error: null });
                        }
                        else if (status.status === 'ready') {
                            updateRecording({ status: 'idle', error: null });
                        }
                        else if (status.status === 'initing') {
                            updateRecording({ status: 'initing', error: null });
                        }
                        else if (status.status === 'idle') {
                            updateRecording({ status: 'idle', error: null });
                        }

                    });

                    const completeUnsubscribe = window.electronAPI.onSpeechRecognitionComplete((result) => {
                        console.log('Speech recognition complete:', result);
                        handleElectronSpeechResult(result);
                    });

                    // Return cleanup function to be called in useEffect cleanup
                    return () => {
                        console.log('Cleaning up Electron speech recognition listeners');
                        statusUnsubscribe();
                        completeUnsubscribe();
                    };
                } else {
                    console.warn('Electron speech recognition not available');
                    return () => { }; // Return empty cleanup function
                }
            } else {
                console.warn('Electron API not available');
                return () => { }; // Return empty cleanup function
            }
        } catch (error) {
            console.error('Error checking Electron speech recognition availability:', error);
            setElectronSpeechAvailable(false);
            return () => { }; // Return empty cleanup function
        }
    };

    const handleElectronSpeechResult = (result: { success: boolean, transcript?: string, error?: string, method?: string, confidence?: number, enhanced?: boolean }) => {
        resetRecording();
        clearAllTimeouts();

        if (result.success && result.transcript?.trim()) {
            const transcript = cleanTranscript(result.transcript.trim());

            if (transcript.length > 0) {
                setInputValue(transcript);
                updateRecording({ status: 'completed' });

                // Show recognition quality feedback
                if (result.enhanced && result.confidence !== undefined) {
                    const qualityMessage = result.confidence > 0.8 ? 'High quality' :
                        result.confidence > 0.6 ? 'Good quality' : 'Basic quality';
                    const methodName = result.method === 'whisper' ? 'Whisper AI' :
                        result.method === 'google' ? 'Google' :
                            result.method === 'sphinx' ? 'Offline' : 'Unknown';

                    addStatusMessage('quality', `${qualityMessage} recognition using ${methodName} (${Math.round((result.confidence || 0) * 100)}% confidence)`);
                }

                if (inputRef.current) {
                    inputRef.current.focus();
                }
            } else {
                addStatusMessage('empty-transcript', 'No speech detected or transcript was empty. Please try speaking more clearly.');
            }
        } else if (result.error) {
            handleRecordingError(result.error + (result.enhanced ? ' (Enhanced mode)' : ''));
        } else {
            addStatusMessage('no-speech', 'Recording completed but no speech was detected. Please try again.');
        }
    };

    const initializeWebSpeechRecognition = () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported in this browser');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            console.log('Voice recognition started');
            updateRecording({
                error: null,
                transcript: ''
            });
        };

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let interim = '';
            let final = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                const transcript = result[0].transcript;

                if (result.isFinal) {
                    final += transcript;
                } else {
                    interim += transcript;
                }
            }

            // Update the transcript in our recording state
            const currentTranscript = recording.transcript + final;
            updateRecording({
                transcript: currentTranscript
            });

            if (final) {
                if (timeoutRefs.current.silence) {
                    clearTimeout(timeoutRefs.current.silence);
                }

                timeoutRefs.current.silence = setTimeout(() => {
                    if (recognition && recording.status === 'listening') {
                        recognition.stop();
                    }
                }, 2000);
            }

            const combinedTranscript = (currentTranscript + interim).trim();
            setInputValue(combinedTranscript);
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error('Voice recognition error:', event.error, event.message);
            resetRecording(true);
            clearAllTimeouts();

            const errorMessages = {
                'network': 'Network error: Please check your internet connection and try again.',
                'not-allowed': 'Microphone access denied. Please allow microphone permissions and try again.',
                'no-speech': 'No speech detected. Please speak clearly and try again.',
                'audio-capture': 'Audio capture failed. Please check your microphone and try again.',
                'service-not-allowed': 'Speech recognition service not allowed. Please check browser settings.',
                'aborted': null // Don't show message for intentional aborts
            };

            const errorMessage = errorMessages[event.error as keyof typeof errorMessages] ||
                `Voice recognition error: ${event.error}. Please try speaking more clearly.`;

            if (errorMessage) {
                handleRecordingError(errorMessage);
            }
        };

        recognition.onend = () => {
            console.log('Voice recognition ended');
            const currentTranscript = cleanTranscript(recording.transcript.trim());
            resetRecording();
            clearAllTimeouts();

            if (currentTranscript && currentTranscript.length > 0) {
                setInputValue(currentTranscript);
                updateRecording({ status: 'completed' });

                setTimeout(() => {
                    if (inputRef.current) {
                        inputRef.current.focus();
                        inputRef.current.setSelectionRange(currentTranscript.length, currentTranscript.length);
                    }
                }, 100);
            } else {
                addStatusMessage('no-speech-web', 'Recording completed but no clear speech was detected. Please try again.');
            }
        };

        setSpeechRecognition(recognition);
    };    // Cleanup effect
    useEffect(() => {
        return () => {

            clearAllTimeouts();
            try {
                if (speechRecognition && (recording.status === 'listening' || recording.status === 'processing' || recording.status === 'initing')) {
                    speechRecognition.abort();
                }
            } catch (error) {
                console.warn('Error aborting speech recognition during cleanup:', error);
            } finally {
                resetRecording(true);
            }

            // Cleanup text-to-speech
            stopSpeaking();
        };
    }, [speechRecognition, recording.status]);

    const forceStopVoiceRecording = () => {
        console.log('Force stopping voice recording...');
        clearAllTimeouts();

        try {
            if (speechRecognition) {
                speechRecognition.abort();
            }
            if (window.electronAPI?.stopSpeechRecognition) {
                window.electronAPI.stopSpeechRecognition().catch(console.warn);
            }
        } catch (error) {
            console.warn('Error during force stop:', error);
        } finally {
            resetRecording(true);
        }
    };

    // Add keyboard shortcut to force stop recording (Escape key)
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && isUninteractableState) {
                event.preventDefault();
                forceStopVoiceRecording();
                addStatusMessage('escape-stop', 'Voice recording stopped by Escape key.');
            }
            // Add keyboard shortcut to stop speech (Ctrl+Space or Cmd+Space)
            if ((event.ctrlKey || event.metaKey) && event.code === 'Space' && isSpeaking) {
                event.preventDefault();
                stopSpeaking();
                addStatusMessage('speech-stop', 'Text-to-speech stopped.');
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isUninteractableState, isSpeaking]);

    // Debug and sync utilities (simplified)
    useEffect(() => {
        const debugRecordingState = () => ({
            recording,
            speechRecognition: !!speechRecognition,
            environment,
            electronSpeechAvailable
        });

        (window as any).debugVoiceRecording = debugRecordingState;

        // Periodic state sync check
        const interval = setInterval(() => {
            if (recording.status === 'listening' && !speechRecognition) {
                console.warn('State sync issue detected: recording status is listening but no speech recognition');
                resetRecording();
                setRecordingError('Recording state was reset due to sync issue');
            }
        }, 5000);

        return () => {
            delete (window as any).debugVoiceRecording;
            clearInterval(interval);
        };
    }, [recording, speechRecognition, environment, electronSpeechAvailable]);

    const startVoiceRecording = async () => {
        // Debounce mechanism
        const now = Date.now();
        if (now - lastStartAttemptRef.current < 1000) {
            console.warn('Voice recording start attempt too soon, ignoring');
            return;
        }
        lastStartAttemptRef.current = now;

        if (recording.status === 'listening') {
            console.warn('Voice recording already in progress');
            return;
        }

        try {
            startRecording();

            // Stop any existing speech recognition first
            if (speechRecognition) {
                try { speechRecognition.stop(); } catch (e) { console.warn('Error stopping previous recognition:', e); }
            }

            clearAllTimeouts();

            if (environment === 'electron') {
                await startElectronSpeechRecognition();
            } else {
                await startWebSpeechRecognition();
            }
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            const errorMessage = error instanceof Error ? error.message : 'Failed to start voice recognition';
            handleRecordingError(errorMessage);
            addStatusMessage('start-error', errorMessage);
        }
    };

    const startElectronSpeechRecognition = async () => {
        if (!electronSpeechAvailable || !window.electronAPI?.startSpeechRecognition) {
            throw new Error('Electron speech recognition not available');
        }

        if (recording.status === 'listening') {
            console.warn('Electron speech recognition already in progress');
            return;
        }

        // Stop any existing recognition
        try {
            if (window.electronAPI.stopSpeechRecognition) {
                await window.electronAPI.stopSpeechRecognition();
            }
        } catch (e) {
            console.warn('Error stopping previous Electron speech recognition:', e);
        }

        setInputValue('');

        // Forces recording to stop after 30 seconds maximum
        timeoutRefs.current.voice = setTimeout(async () => {
            if (recording.status === 'listening') {
                await stopVoiceRecording();
                addStatusMessage('timeout', 'Voice recording timed out after 30 seconds.');
            }
        }, 30000);

        const result = await window.electronAPI.startSpeechRecognition();
        if (!result.success) {
            resetRecording();
            clearAllTimeouts();
            throw new Error(result.error || 'Failed to start speech recognition');
        }
    };

    const startWebSpeechRecognition = async () => {
        if (!speechRecognition) {
            throw new Error('Web speech recognition not available');
        }

        if (recording.status === 'listening') {
            console.warn('Web speech recognition already in progress');
            return;
        }

        // Check microphone permissions
        if (navigator.permissions) {
            try {
                const permission = await navigator.permissions.query({ name: 'microphone' as PermissionName });
                if (permission.state === 'denied') {
                    throw new Error('Microphone access denied. Please allow microphone permissions in your browser settings.');
                }
            } catch (permError) {
                console.warn('Permission check failed:', permError);
            }
        }

        // Stop any ongoing recognition
        try {
            speechRecognition.stop();
            await new Promise(resolve => setTimeout(resolve, 100));
        } catch (e) {
            console.warn('Error stopping previous web speech recognition:', e);
        }

        setInputValue('');
        updateRecording({ transcript: '' });

        // Set timeout for 30 seconds
        timeoutRefs.current.voice = setTimeout(() => {
            if (speechRecognition && recording.status === 'listening') {
                speechRecognition.stop();
                addStatusMessage('timeout', 'Voice recording timed out after 30 seconds.');
            }
        }, 30000);

        try {
            speechRecognition.start();
        } catch (error) {
            resetRecording();
            clearAllTimeouts();
            throw error;
        }
    };

    const stopVoiceRecording = async () => {
        clearAllTimeouts();

        try {
            if (environment === 'electron') {
                await stopElectronSpeechRecognition();
            } else {
                stopWebSpeechRecognition();
            }
        } catch (error) {
            console.error('Error in stopVoiceRecording:', error);
        } finally {
            resetRecording();
        }
    };

    const stopElectronSpeechRecognition = async () => {
        if (!window.electronAPI?.stopSpeechRecognition) return;

        try {
            const result = await window.electronAPI.stopSpeechRecognition();
            if (!result.success) {
                console.warn('Failed to stop speech recognition:', result.error);
            }
        } catch (error) {
            console.error('Error stopping Electron speech recognition:', error);
            addStatusMessage('stop-error', error instanceof Error ? error.message : 'Failed to stop speech recognition');
        }
    };

    const stopWebSpeechRecognition = () => {
        if (speechRecognition) {
            try {
                speechRecognition.stop();
                resetRecording();
                clearAllTimeouts();
            } catch (error) {
                console.error('Error stopping web speech recognition:', error);
                setRecordingError('Failed to stop recording');
            }
        } else {
            resetRecording();
        }
    };
    /**
     * Text-chat submission path. Removed in the racing-engineer rebuild:
     * the legacy `/naturallanguagequery` and `/user-ai-model/ai-query`
     * endpoints (and the SSE flow that consumed them) were deleted along
     * with this client logic. The voice WS is now the single chat surface;
     * the text input below shows a hint pointing users at the mic button.
     *
     * A future change can re-enable text input by sending it as a
     * synthetic user message over the voice WS text channel.
     */
    const sendToAI = async (_messageContent: string): Promise<void> => {
        const notice: Message = {
            id: generateUniqueId('ai'),
            content:
                "Text chat is being migrated to the voice WS — click the " +
                "mic button to talk to the engineer for now.",
            isUser: false,
            timestamp: new Date(),
        };
        setMessages((prev) =>
            prev
                .filter((msg) => !msg.id.includes('loading') && !msg.id.includes('executing'))
                .concat(notice),
        );
        setIsLoading(false);
    };

    const handleSendMessage = async () => {
        if (!inputValue.trim() || isLoading) return;

        // Check if this message came from voice input
        const actualIsFromVoice = isRecordingCompleted;

        const userMessage: Message = {
            id: generateUniqueId('user'),
            content: inputValue.trim(),
            isUser: true,
            timestamp: new Date(),
            isVoiceInput: actualIsFromVoice
        };

        setMessages(prev => [...prev, userMessage]);
        const messageContent = inputValue.trim();
        setInputValue('');
        updateRecording({ status: 'idle' }); // Clear the voice input status

        // Set loading state
        setIsLoading(true);

        // Add loading message
        const loadingMessage: Message = {
            id: generateUniqueId('loading'),
            content: 'Thinking...',
            isUser: false,
            timestamp: new Date(),
            isLoading: true
        };
        setMessages(prev => [...prev, loadingMessage]);

        await sendToAI(messageContent);
    };

    // handle user input change
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        setInputValue(newValue);

        // If user is typing manually (not just backspacing), clear the voice status
        if (newValue.length > inputValue.length && isRecordingCompleted) {
            updateRecording({ status: 'idle' });
        }
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


    // Utility function to clean and improve transcript quality
    const cleanTranscript = (transcript: string): string => {
        return transcript
            .trim()
            // Remove extra whitespace
            .replace(/\s+/g, ' ')
            // Capitalize first letter of sentences
            .replace(/(^|\. )([a-z])/g, (match, prefix, letter) => prefix + letter.toUpperCase())
            // Fix common speech recognition errors in racing context
            .replace(/\bcar\b/gi, 'car')
            .replace(/\bturn\b/gi, 'turn')
            .replace(/\bspeed\b/gi, 'speed')
            .replace(/\bbrake\b/gi, 'brake')
            .replace(/\brace\b/gi, 'race')
            .replace(/\blap\b/gi, 'lap')
            .replace(/\btrack\b/gi, 'track');
    };
    const formatFunctionArgs = (args: Record<string, any>): string => {
        return Object.entries(args).map(([key, value]) => {
            if (typeof value === 'object') {
                return `${key}: ${JSON.stringify(value)}`;
            }
            return `${key}: ${value}`;
        }).join(', ');
    };


    // Microphone Icon Component
    const MicrophoneIcon = () => (
        <svg
            className="mic-icon"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M12 1a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
            <path d="M19 11v1a7 7 0 0 1-14 0v-1" />
            <line x1="12" y1="20" x2="12" y2="24" />
            <line x1="8" y1="24" x2="16" y2="24" />
        </svg>
    );

    // Recording Animation Microphone Icon
    const RecordingMicrophoneIcon = () => (
        <svg
            className="recording-mic-icon"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M12 1a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
            <path d="M19 11v1a7 7 0 0 1-14 0v-1" />
            <line x1="12" y1="20" x2="12" y2="24" />
            <line x1="8" y1="24" x2="16" y2="24" />
        </svg>
    );

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
                        {!speechRecognition && !electronSpeechAvailable && (
                            <Badge variant="soft" color="orange" size="1">
                                Voice input not supported
                            </Badge>
                        )}
                        {!neuralTtsAvailable && (
                            <Badge variant="soft" color="orange" size="1">
                                Text-to-speech not available
                            </Badge>
                        )}
                        {recording.error && (
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
                                        {message.isUser
                                            ? 'You'
                                            : message.id.includes('guidance')
                                                ? '🎯 Live Track Guidance'
                                                : 'AI Assistant'
                                        }
                                    </Text>
                                    {message.isVoiceInput && (
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ color: 'var(--accent-9)' }}>
                                            <path d="M12 1a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
                                            <path d="M19 11v1a7 7 0 0 1-14 0v-1" />
                                            <line x1="12" y1="20" x2="12" y2="24" />
                                            <line x1="8" y1="24" x2="16" y2="24" />
                                        </svg>
                                    )}
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
                        ))}
                        <div ref={messagesEndRef} />
                    </Flex>
                </ScrollArea>

                {/* Input Area */}
                <Separator />
                <Box p="3">
                    <Flex gap="2">
                        <TextField.Root
                            ref={inputRef}
                            placeholder={
                                isUninteractableState
                                    ? `🎤 ${recording.status === 'listening' ? 'Recording...' : recording.status === 'processing' ? 'Processing...' : 'Initializing...'} (Press Escape or click Force Stop to cancel)`
                                    : isRecordingCompleted
                                        ? "Voice input ready - press Enter to send or continue editing..."
                                        : "Ask me anything about your racing session..."
                            }
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            disabled={isLoading || isUninteractableState}
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

                        {(speechRecognition || electronSpeechAvailable) && (
                            <IconButton
                                onClick={isUninteractableState ? stopVoiceRecording : startVoiceRecording}
                                disabled={isLoading}
                                size="2"
                                variant={isUninteractableState ? "solid" : "ghost"}
                                color={isUninteractableState ? "red" : recording.error ? "orange" : "gray"}
                                title={
                                    recording.error
                                        ? `Voice error: ${recording.error}. Click to retry.`
                                        : isUninteractableState
                                            ? `Recording ${recording.status} - Click to stop or press Escape`
                                            : `Start voice recording (${environment === 'electron' ? 'Local' : 'Web'} mode)`
                                }
                                style={{
                                    ...(isUninteractableState && {
                                        backgroundColor: 'var(--red-9)',
                                        color: 'white'
                                    })
                                }}
                            >
                                {isUninteractableState ? <RecordingMicrophoneIcon /> : <MicrophoneIcon />}
                            </IconButton>
                        )}
                        <Button
                            onClick={() => handleSendMessage()}
                            disabled={!inputValue.trim() || isLoading || isUninteractableState}
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
