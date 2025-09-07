import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, Button, Card, Flex, Text, TextField, ScrollArea, Separator, Badge, Spinner, IconButton } from '@radix-ui/themes';
import { PaperPlaneIcon, ChatBubbleIcon, PersonIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/session-analysis';
import { visualizationController } from 'views/lap-analysis/visualization';
import { detectEnvironment } from 'utils/environment';

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
    const [isRecording, setIsRecording] = useState(false);
    const [speechRecognition, setSpeechRecognition] = useState<SpeechRecognition | null>(null);
    const [voiceError, setVoiceError] = useState<string | null>(null);
    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');
    const [electronSpeechAvailable, setElectronSpeechAvailable] = useState(false);
    const [isCurrentInputFromVoice, setIsCurrentInputFromVoice] = useState(false);
    const [interimTranscript, setInterimTranscript] = useState('');
    const [finalTranscript, setFinalTranscript] = useState('');
    const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const voiceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const isRecordingRef = useRef<boolean>(false); // Ref to track recording state reliably
    const lastStartAttemptRef = useRef<number>(0); // Debounce mechanism
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
                    track_name: analysisContext.recordedSessioStaticsData?.track || "unknown",
                    car_name: analysisContext.recordedSessioStaticsData?.car_model || "unknown",
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

    // Auto-manage imitation guidance chart visibility
    useEffect(() => {
        const imitationChartId = 'imitation-guidance-chart-auto';

        if (imitationLearningEnabled) {
            // Check if chart already exists to avoid duplicates
            const existingCharts = visualizationController.getCurrentInstances();
            const chartExists = existingCharts.some(chart =>
                chart.type === 'imitation-guidance-chart' ||
                chart.id === imitationChartId
            );

            if (!chartExists) {
                // Add the imitation guidance chart
                visualizationController.executeCommand({
                    action: 'add',
                    type: 'imitation-guidance-chart',
                    data: {
                        sessionId: sessionId,
                        autoManaged: true // Flag to indicate this was auto-added
                    },
                    config: {
                        title: 'AI Driving Guidance',
                        autoUpdate: true
                    }
                });

                console.log('Auto-added imitation guidance chart');
            }
        } else {
            // Remove auto-managed imitation guidance charts
            const existingCharts = visualizationController.getCurrentInstances();
            existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                    visualizationController.executeCommand({
                        action: 'remove',
                        id: chart.id
                    });
                    console.log('Auto-removed imitation guidance chart:', chart.id);
                }
            });
        }
    }, [imitationLearningEnabled, sessionId]);

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
        };
    }, []);

    const initializeElectronSpeechRecognition = async () => {
        try {
            if (window.electronAPI && window.electronAPI.isSpeechRecognitionAvailable) {
                const available = await window.electronAPI.isSpeechRecognitionAvailable();
                setElectronSpeechAvailable(available);

                if (available) {
                    // Set up event listeners for speech recognition
                    const statusUnsubscribe = window.electronAPI.onSpeechRecognitionStatus((status) => {
                        console.log('Speech recognition status:', status);

                        // Handle enhanced status updates
                        if (status.status === 'listening') {
                            setVoiceError(null);
                            if (status.enhanced) {
                                const listeningMessage: Message = {
                                    id: `listening-${Date.now()}`,
                                    content: `🎤 Enhanced listening mode activated${status.whisper ? ' with Whisper AI' : ''}...`,
                                    isUser: false,
                                    timestamp: new Date()
                                };
                                setMessages(prev => [...prev, listeningMessage]);
                            }
                        } else if (status.status === 'speech_detected') {
                            // Show speech detection feedback
                            const speechMessage: Message = {
                                id: `speech-detected-${Date.now()}`,
                                content: '🎤 Speech detected, continuing to listen...',
                                isUser: false,
                                timestamp: new Date()
                            };
                            setMessages(prev => [...prev, speechMessage]);
                        } else if (status.status === 'processing') {
                            const processingMessage: Message = {
                                id: `processing-${Date.now()}`,
                                content: `🎤 Processing audio${status.chunks_collected ? ` (${status.chunks_collected} segments)` : ''}...`,
                                isUser: false,
                                timestamp: new Date()
                            };
                            setMessages(prev => [...prev, processingMessage]);
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
        setIsRecording(false);
        isRecordingRef.current = false;

        // Clear timeout
        if (voiceTimeoutRef.current) {
            clearTimeout(voiceTimeoutRef.current);
            voiceTimeoutRef.current = null;
        }

        if (result.success && result.transcript && result.transcript.trim()) {
            const transcript = cleanTranscript(result.transcript.trim());
            console.log('Electron voice recognition result:', {
                transcript,
                method: result.method,
                confidence: result.confidence,
                enhanced: result.enhanced
            });

            // Only set input if we have meaningful content
            if (transcript.length > 0) {
                // Set the input value with the transcribed text, but don't auto-send
                setInputValue(transcript);
                setIsCurrentInputFromVoice(true);

                // Show recognition quality feedback
                if (result.enhanced && result.confidence !== undefined) {
                    const qualityMessage = result.confidence > 0.8 ? 'High quality' :
                        result.confidence > 0.6 ? 'Good quality' : 'Basic quality';
                    const methodName = result.method === 'whisper' ? 'Whisper AI' :
                        result.method === 'google' ? 'Google' :
                            result.method === 'sphinx' ? 'Offline' : 'Unknown';

                    const qualityInfo: Message = {
                        id: `quality-${Date.now()}`,
                        content: `🎤 ${qualityMessage} recognition using ${methodName} (${Math.round((result.confidence || 0) * 100)}% confidence)`,
                        isUser: false,
                        timestamp: new Date()
                    };
                    setMessages(prev => [...prev, qualityInfo]);
                }

                // Focus the input field so user can see the transcribed text
                if (inputRef.current) {
                    inputRef.current.focus();
                }
            } else {
                // Handle empty transcript
                const emptyMessage: Message = {
                    id: `empty-transcript-${Date.now()}`,
                    content: '🎤 No speech detected or transcript was empty. Please try speaking more clearly.',
                    isUser: false,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, emptyMessage]);
            }
        } else if (!result.success && result.error) {
            // Handle error
            setVoiceError(result.error);
            const errorMessage: Message = {
                id: `electron-error-${Date.now()}`,
                content: `🎤 ${result.error}${result.enhanced ? ' (Enhanced mode)' : ''}`,
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        } else if (result.success && (!result.transcript || !result.transcript.trim())) {
            // Handle success but empty transcript
            const noSpeechMessage: Message = {
                id: `no-speech-${Date.now()}`,
                content: '🎤 Recording completed but no speech was detected. Please try again.',
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, noSpeechMessage]);
        }
    };

    const initializeWebSpeechRecognition = () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.continuous = false; // Change to false to prevent auto-restart issues
            recognition.interimResults = true; // Show interim results for better user feedback

            // Always use English for voice recognition
            recognition.lang = 'en-US';

            console.log(`Initializing speech recognition with language: ${recognition.lang}`);

            recognition.onstart = () => {
                console.log('Voice recognition started');
                setVoiceError(null); // Clear any previous errors
                setInterimTranscript('');
                setFinalTranscript('');
            };

            recognition.onresult = (event: SpeechRecognitionEvent) => {
                let interim = '';
                let final = '';

                // Process all results from the current recognition session
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    const transcript = result[0].transcript;

                    if (result.isFinal) {
                        final += transcript;
                    } else {
                        interim += transcript;
                    }
                }

                // Update interim results for real-time feedback
                setInterimTranscript(interim);

                // If we have final results, process them
                if (final) {
                    setFinalTranscript(prev => prev + final);
                    setInterimTranscript(''); // Clear interim when we get final results

                    // Reset silence timeout when we get speech
                    if (silenceTimeoutRef.current) {
                        clearTimeout(silenceTimeoutRef.current);
                    }

                    // Set a silence timeout to automatically finish recognition
                    // if no speech is detected for 2 seconds after final result
                    silenceTimeoutRef.current = setTimeout(() => {
                        if (recognition && isRecording) {
                            recognition.stop();
                        }
                    }, 2000);
                }

                // Update input field with both final and interim text
                const combinedTranscript = (finalTranscript + final + interim).trim();
                setInputValue(combinedTranscript);

                console.log('Voice recognition - Final:', final, 'Interim:', interim, 'Combined:', combinedTranscript);
            };

            recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
                console.error('Voice recognition error:', event.error, event.message);
                setIsRecording(false);
                isRecordingRef.current = false;

                // Clear timeouts
                if (voiceTimeoutRef.current) {
                    clearTimeout(voiceTimeoutRef.current);
                    voiceTimeoutRef.current = null;
                }
                if (silenceTimeoutRef.current) {
                    clearTimeout(silenceTimeoutRef.current);
                    silenceTimeoutRef.current = null;
                }

                // Reset transcript states on error
                setInterimTranscript('');
                setFinalTranscript('');

                // Handle different types of errors
                let errorMessage = '';
                let shouldShowMessage = true;

                switch (event.error) {
                    case 'network':
                        errorMessage = 'Network error: Please check your internet connection and try again.';
                        setVoiceError('Network error');
                        break;
                    case 'not-allowed':
                        errorMessage = 'Microphone access denied. Please allow microphone permissions and try again.';
                        setVoiceError('Permission denied');
                        break;
                    case 'no-speech':
                        errorMessage = 'No speech detected. Please speak clearly and try again.';
                        setVoiceError('No speech detected');
                        // Still show message for no-speech as it helps user understand
                        break;
                    case 'audio-capture':
                        errorMessage = 'Audio capture failed. Please check your microphone and try again.';
                        setVoiceError('Audio capture failed');
                        break;
                    case 'service-not-allowed':
                        errorMessage = 'Speech recognition service not allowed. Please check browser settings.';
                        setVoiceError('Service not allowed');
                        break;
                    case 'aborted':
                        // Don't show error message for intentional aborts
                        shouldShowMessage = false;
                        setVoiceError(null);
                        break;
                    default:
                        errorMessage = `Voice recognition error: ${event.error}. Please try speaking more clearly.`;
                        setVoiceError(event.error);
                }

                // Add error message to chat if appropriate
                if (shouldShowMessage && errorMessage) {
                    const errorChatMessage: Message = {
                        id: `error-${Date.now()}`,
                        content: `🎤 ${errorMessage}`,
                        isUser: false,
                        timestamp: new Date()
                    };
                    setMessages(prev => [...prev, errorChatMessage]);
                }
            };

            recognition.onend = () => {
                console.log('Voice recognition ended');
                setIsRecording(false);
                isRecordingRef.current = false;

                // Clear timeouts
                if (voiceTimeoutRef.current) {
                    clearTimeout(voiceTimeoutRef.current);
                    voiceTimeoutRef.current = null;
                }
                if (silenceTimeoutRef.current) {
                    clearTimeout(silenceTimeoutRef.current);
                    silenceTimeoutRef.current = null;
                }

                // Finalize the transcript
                const completeFinalTranscript = cleanTranscript((finalTranscript + interimTranscript).trim());
                if (completeFinalTranscript && completeFinalTranscript.length > 0) {
                    setInputValue(completeFinalTranscript);
                    setIsCurrentInputFromVoice(true);

                    // Focus the input field and position cursor at the end
                    setTimeout(() => {
                        if (inputRef.current) {
                            inputRef.current.focus();
                            inputRef.current.setSelectionRange(completeFinalTranscript.length, completeFinalTranscript.length);
                        }
                    }, 100);

                    console.log('Voice recognition completed with transcript:', completeFinalTranscript);
                } else {
                    // Handle case where no meaningful speech was detected
                    const noSpeechMessage: Message = {
                        id: `no-speech-web-${Date.now()}`,
                        content: '🎤 Recording completed but no clear speech was detected. Please try again.',
                        isUser: false,
                        timestamp: new Date()
                    };
                    setMessages(prev => [...prev, noSpeechMessage]);
                    console.log('Voice recognition ended with no meaningful transcript');
                }

                // Reset transcript states
                setInterimTranscript('');
                setFinalTranscript('');
            };

            setSpeechRecognition(recognition);
        } else {
            console.warn('Speech recognition not supported in this browser');
        }
    };    // Cleanup effect
    useEffect(() => {
        return () => {
            // Clear voice timeouts if component unmounts
            if (voiceTimeoutRef.current) {
                clearTimeout(voiceTimeoutRef.current);
                voiceTimeoutRef.current = null;
            }
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }

            // Force stop any ongoing recording and reset state
            try {
                if (speechRecognition && (isRecording || isRecordingRef.current)) {
                    speechRecognition.abort(); // Use abort() instead of stop() for immediate termination
                }
            } catch (error) {
                console.warn('Error aborting speech recognition during cleanup:', error);
            } finally {
                // Always reset state during cleanup
                setIsRecording(false);
                isRecordingRef.current = false;
            }
        };
    }, [speechRecognition, isRecording]);

    // Add a force stop function for emergency situations
    const forceStopVoiceRecording = () => {
        console.log('Force stopping voice recording...');

        // Clear all timeouts
        if (voiceTimeoutRef.current) {
            clearTimeout(voiceTimeoutRef.current);
            voiceTimeoutRef.current = null;
        }
        if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }

        // Force stop all recognition
        try {
            if (speechRecognition) {
                speechRecognition.abort(); // Use abort for immediate stop
            }
            if (window.electronAPI?.stopSpeechRecognition) {
                window.electronAPI.stopSpeechRecognition().catch(console.warn);
            }
        } catch (error) {
            console.warn('Error during force stop:', error);
        } finally {
            // Always reset state
            setIsRecording(false);
            isRecordingRef.current = false;
            setInterimTranscript('');
            setVoiceError(null);
        }
    };

    // Add keyboard shortcut to force stop recording (Escape key)
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && (isRecording || isRecordingRef.current)) {
                event.preventDefault();
                forceStopVoiceRecording();

                // Show user feedback
                const escapeMessage: Message = {
                    id: `escape-stop-${Date.now()}`,
                    content: '🎤 Voice recording stopped by Escape key.',
                    isUser: false,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, escapeMessage]);
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [isRecording]);

    // Debug function to check current recording state
    const debugRecordingState = () => {
        const state = {
            isRecording,
            isRecordingRef: isRecordingRef.current,
            speechRecognition: !!speechRecognition,
            environment,
            electronSpeechAvailable,
            voiceError,
            interimTranscript,
            finalTranscript,
            hasVoiceTimeout: !!voiceTimeoutRef.current,
            hasSilenceTimeout: !!silenceTimeoutRef.current,
        };
        console.log('Voice Recording Debug State:', state);
        return state;
    };

    // Expose debug function to window for easy access
    useEffect(() => {
        (window as any).debugVoiceRecording = debugRecordingState;
        return () => {
            delete (window as any).debugVoiceRecording;
        };
    }, [isRecording, speechRecognition, voiceError, interimTranscript, finalTranscript]);

    // Periodic state sync check - ensures UI and actual speech recognition stay in sync
    useEffect(() => {
        const interval = setInterval(() => {
            // If we think we're recording but have no active speech recognition, fix it
            if ((isRecording || isRecordingRef.current) && !speechRecognition) {
                console.warn('State sync issue detected: recording state true but no speech recognition');
                setIsRecording(false);
                isRecordingRef.current = false;
                setVoiceError('Recording state was reset due to sync issue');
            }
        }, 5000); // Check every 5 seconds

        return () => clearInterval(interval);
    }, [isRecording, speechRecognition]);

    const startVoiceRecording = async () => {
        // Debounce mechanism - prevent calls within 1 second of each other
        const now = Date.now();
        if (now - lastStartAttemptRef.current < 1000) {
            console.warn('Voice recording start attempt too soon after previous attempt, ignoring');
            return;
        }
        lastStartAttemptRef.current = now;

        // Prevent multiple simultaneous recordings using both state and ref
        if (isRecording || isRecordingRef.current) {
            console.warn('Voice recording already in progress, ignoring new request');
            return;
        }

        try {
            // Set both state and ref immediately
            setIsRecording(true);
            isRecordingRef.current = true;

            // Clear any previous errors
            setVoiceError(null);

            // Stop any existing speech recognition first
            if (speechRecognition) {
                try {
                    speechRecognition.stop();
                } catch (e) {
                    console.warn('Error stopping previous speech recognition:', e);
                }
            }

            // Clear any existing timeouts
            if (voiceTimeoutRef.current) {
                clearTimeout(voiceTimeoutRef.current);
                voiceTimeoutRef.current = null;
            }
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }

            if (environment === 'electron') {
                await startElectronSpeechRecognition();
            } else {
                await startWebSpeechRecognition();
            }
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            // Always reset state on error
            setIsRecording(false);
            isRecordingRef.current = false;

            // Clear timeouts on error
            if (voiceTimeoutRef.current) {
                clearTimeout(voiceTimeoutRef.current);
                voiceTimeoutRef.current = null;
            }
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }

            const errorMessage = error instanceof Error ? error.message : 'Failed to start voice recognition';
            setVoiceError(errorMessage);

            const chatErrorMessage: Message = {
                id: `start-error-${Date.now()}`,
                content: `🎤 ${errorMessage}`,
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, chatErrorMessage]);
        }
    };

    const startElectronSpeechRecognition = async () => {
        if (!electronSpeechAvailable || !window.electronAPI?.startSpeechRecognition) {
            throw new Error('Electron speech recognition not available');
        }

        // Double-check we're not already recording using ref
        if (isRecordingRef.current) {
            console.warn('Electron speech recognition already in progress');
            return;
        }

        // Stop any existing Electron speech recognition first
        try {
            if (window.electronAPI.stopSpeechRecognition) {
                await window.electronAPI.stopSpeechRecognition();
            }
        } catch (e) {
            console.warn('Error stopping previous Electron speech recognition:', e);
        }

        setInputValue(''); // Clear current input

        // Set a timeout to stop recording after 30 seconds
        voiceTimeoutRef.current = setTimeout(async () => {
            if (isRecordingRef.current) {
                await stopVoiceRecording();
                const timeoutMessage: Message = {
                    id: `timeout-${Date.now()}`,
                    content: '🎤 Voice recording timed out after 30 seconds.',
                    isUser: false,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, timeoutMessage]);
            }
        }, 30000);

        const result = await window.electronAPI.startSpeechRecognition();
        if (!result.success) {
            setIsRecording(false); // Reset state on failure
            isRecordingRef.current = false;
            if (voiceTimeoutRef.current) {
                clearTimeout(voiceTimeoutRef.current);
                voiceTimeoutRef.current = null;
            }
            throw new Error(result.error || 'Failed to start speech recognition');
        }
    };

    const startWebSpeechRecognition = async () => {
        if (!speechRecognition) {
            throw new Error('Web speech recognition not available');
        }

        // Double-check we're not already recording using ref
        if (isRecordingRef.current) {
            console.warn('Web speech recognition already in progress');
            return;
        }

        // Check for microphone permissions
        if (navigator.permissions) {
            try {
                const permission = await navigator.permissions.query({ name: 'microphone' as PermissionName });
                if (permission.state === 'denied') {
                    throw new Error('Microphone access denied. Please allow microphone permissions in your browser settings.');
                }
            } catch (permError) {
                console.warn('Permission check failed:', permError);
                // Continue anyway, let speech recognition handle it
            }
        }

        // Stop any ongoing recognition first
        try {
            speechRecognition.stop();
            // Wait a short moment for cleanup
            await new Promise(resolve => setTimeout(resolve, 100));
        } catch (e) {
            console.warn('Error stopping previous web speech recognition:', e);
        }

        setInputValue(''); // Clear current input
        setInterimTranscript(''); // Reset interim transcript
        setFinalTranscript(''); // Reset final transcript

        // Set a longer timeout to stop recording after 30 seconds for better user experience
        voiceTimeoutRef.current = setTimeout(() => {
            if (speechRecognition && isRecordingRef.current) {
                speechRecognition.stop();
                const timeoutMessage: Message = {
                    id: `timeout-${Date.now()}`,
                    content: '🎤 Voice recording timed out after 30 seconds.',
                    isUser: false,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, timeoutMessage]);
            }
        }, 30000);

        try {
            speechRecognition.start();
        } catch (error) {
            setIsRecording(false); // Reset state on failure
            isRecordingRef.current = false;
            if (voiceTimeoutRef.current) {
                clearTimeout(voiceTimeoutRef.current);
                voiceTimeoutRef.current = null;
            }
            throw error;
        }
    };

    const stopVoiceRecording = async () => {
        // Clear timeout first
        if (voiceTimeoutRef.current) {
            clearTimeout(voiceTimeoutRef.current);
            voiceTimeoutRef.current = null;
        }

        // Always attempt to stop, even if state says we're not recording
        // This handles cases where state gets out of sync
        try {
            if (environment === 'electron') {
                await stopElectronSpeechRecognition();
            } else {
                stopWebSpeechRecognition();
            }
        } catch (error) {
            console.error('Error in stopVoiceRecording:', error);
        } finally {
            // Always reset state regardless of what happened above
            setIsRecording(false);
            isRecordingRef.current = false;
            setInterimTranscript('');
            // Note: Don't reset finalTranscript here as it might contain useful data
        }
    };

    const stopElectronSpeechRecognition = async () => {
        if (!window.electronAPI?.stopSpeechRecognition) return;

        try {
            const result = await window.electronAPI.stopSpeechRecognition();

            if (!result.success) {
                console.warn('Failed to stop speech recognition:', result.error);
            }

            // The actual result will be handled by the event listener
            // in handleElectronSpeechResult, so we don't need to do anything here

        } catch (error) {
            console.error('Error stopping Electron speech recognition:', error);
            setIsRecording(false);
            isRecordingRef.current = false;
            const errorMessage: Message = {
                id: `stop-error-${Date.now()}`,
                content: `🎤 Error: ${error instanceof Error ? error.message : 'Failed to stop speech recognition'}`,
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        }
    };

    const stopWebSpeechRecognition = () => {
        if (speechRecognition) {
            try {
                speechRecognition.stop();
                // Force state reset immediately for better UX
                setIsRecording(false);
                isRecordingRef.current = false;

                // Clear timeouts
                if (voiceTimeoutRef.current) {
                    clearTimeout(voiceTimeoutRef.current);
                    voiceTimeoutRef.current = null;
                }
                if (silenceTimeoutRef.current) {
                    clearTimeout(silenceTimeoutRef.current);
                    silenceTimeoutRef.current = null;
                }
            } catch (error) {
                console.error('Error stopping web speech recognition:', error);
                // Force state reset even on error
                setIsRecording(false);
                isRecordingRef.current = false;
                setVoiceError('Failed to stop recording');
            }
        } else {
            // If no speechRecognition object, still reset state
            setIsRecording(false);
            isRecordingRef.current = false;
        }
    };
    const sendToAI = async (messageContent: string) => {
        try {
            let response;

            // Use openai general natural language ai query endpoint
            response = await apiService.post('ai-model/ai-query', {
                question: messageContent,
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

    const handleSendMessage = async (isFromVoice = false) => {
        if (!inputValue.trim() || isLoading) return;

        // Use the voice input flag if not explicitly provided
        const actualIsFromVoice = isFromVoice || isCurrentInputFromVoice;

        const userMessage: Message = {
            id: Date.now().toString(),
            content: inputValue.trim(),
            isUser: true,
            timestamp: new Date(),
            isVoiceInput: actualIsFromVoice
        };

        setMessages(prev => [...prev, userMessage]);
        const messageContent = inputValue.trim();
        setInputValue('');
        setIsCurrentInputFromVoice(false); // Clear the voice input flag
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

        await sendToAI(messageContent);
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        setInputValue(newValue);

        // If user is typing manually (not just backspacing), clear the voice flag
        if (newValue.length > inputValue.length) {
            setIsCurrentInputFromVoice(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage(false);
        }
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
                    status: 'Imitation learning guidance enabled - now continuously monitoring telemetry data and displaying AI guidance chart',
                    enabled: true,
                    chartAdded: true
                };

            case 'disable_guide_user_racing':
                // Disable the continuous imitation learning guidance
                setImitationLearningEnabled(false);

                return {
                    status: 'Imitation learning guidance disabled - no longer monitoring telemetry data and guidance chart removed',
                    enabled: false,
                    chartRemoved: true
                };

            case 'disable_ui_component':
                // Handle UI updates locally
                if (args.component === 'chart' && analysisContext) {
                    // Trigger chart update through context
                    console.log('Updating UI component:', args);
                    return { success: true, message: 'UI updated successfully' };
                }
                return { success: false, message: 'UI component not found or not supported' };

            case 'add_imitation_guidance_chart':
                // Manually add imitation guidance chart
                const chartAdded = visualizationController.executeCommand({
                    action: 'add',
                    type: 'imitation-guidance-chart',
                    data: {
                        sessionId: sessionIdToUse,
                        manuallyAdded: true
                    },
                    config: {
                        title: args.title || 'AI Driving Guidance',
                        autoUpdate: args.autoUpdate !== false
                    }
                });

                return {
                    success: chartAdded,
                    message: chartAdded
                        ? 'Imitation guidance chart added successfully'
                        : 'Failed to add imitation guidance chart',
                    chartType: 'imitation-guidance-chart'
                };

            case 'remove_imitation_guidance_chart':
                // Remove specific imitation guidance chart or all of them
                const charts = visualizationController.getCurrentInstances();
                const imitationCharts = charts.filter(chart => chart.type === 'imitation-guidance-chart');

                let removedCount = 0;
                if (args.chartId) {
                    // Remove specific chart
                    const removed = visualizationController.executeCommand({
                        action: 'remove',
                        id: args.chartId
                    });
                    if (removed) removedCount = 1;
                } else {
                    // Remove all imitation guidance charts
                    imitationCharts.forEach(chart => {
                        const removed = visualizationController.executeCommand({
                            action: 'remove',
                            id: chart.id
                        });
                        if (removed) removedCount++;
                    });
                }

                return {
                    success: removedCount > 0,
                    message: `Removed ${removedCount} imitation guidance chart(s)`,
                    removedCount
                };

            case 'get_available_functions':
                // Return list of available functions
                return {
                    functions: [
                        'get_session_analysis',
                        'get_telemetry_data',
                        'compare_lap_times',
                        'get_performance_insights',
                        'follow_expert_line',
                        'enable_guide_user_racing',
                        'disable_guide_user_racing',
                        'add_imitation_guidance_chart',
                        'remove_imitation_guidance_chart',
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
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ animation: 'pulse 1s infinite' }}
        >
            <path d="M12 1a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
            <path d="M19 11v1a7 7 0 0 1-14 0v-1" />
            <line x1="12" y1="20" x2="12" y2="24" />
            <line x1="8" y1="24" x2="16" y2="24" />
        </svg>
    );


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
                        {voiceError && (
                            <Badge variant="soft" color="red" size="1">
                                Voice error
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
                                        {message.isUser ? 'You' : 'AI Assistant'}
                                    </Text>
                                    {message.isVoiceInput && (
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ color: 'var(--accent-9)' }}>
                                            <path d="M12 1a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
                                            <path d="M19 11v1a7 7 0 0 1-14 0v-1" />
                                            <line x1="12" y1="20" x2="12" y2="24" />
                                            <line x1="8" y1="24" x2="16" y2="24" />
                                        </svg>
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

                {/* Input Area */}
                <Separator />
                <Box p="3">
                    {/* Recording status indicator */}
                    {isRecording && (
                        <Flex align="center" gap="2" mb="2" p="2"
                            style={{
                                backgroundColor: 'var(--red-2)',
                                borderRadius: '6px',
                                border: '1px solid var(--red-6)'
                            }}>
                            <RecordingMicrophoneIcon />
                            <Text size="2" color="red" weight="medium">
                                Recording in progress...
                                {interimTranscript && (
                                    <Text size="1" color="gray" ml="2">
                                        "{interimTranscript.slice(0, 50)}{interimTranscript.length > 50 ? '...' : ''}"
                                    </Text>
                                )}
                            </Text>
                            <Button
                                size="1"
                                variant="soft"
                                color="red"
                                onClick={forceStopVoiceRecording}
                                title="Force stop recording (or press Escape)"
                            >
                                Force Stop
                            </Button>
                        </Flex>
                    )}

                    <Flex gap="2">
                        <TextField.Root
                            ref={inputRef}
                            placeholder={
                                isRecording
                                    ? "🎤 Recording... (Press Escape or click Force Stop to cancel)"
                                    : isCurrentInputFromVoice
                                        ? "Voice input ready - press Enter to send or continue editing..."
                                        : "Ask me anything about your racing session..."
                            }
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            disabled={isLoading || isRecording}
                            style={{ flex: 1 }}
                        />
                        {(speechRecognition || electronSpeechAvailable) && (
                            <IconButton
                                onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
                                disabled={isLoading}
                                size="2"
                                variant={isRecording ? "solid" : "ghost"}
                                color={isRecording ? "red" : voiceError ? "orange" : "gray"}
                                title={
                                    voiceError
                                        ? `Voice error: ${voiceError}. Click to retry.`
                                        : isRecording
                                            ? "Recording active - Click to stop or press Escape"
                                            : `Start voice recording (${environment === 'electron' ? 'Local' : 'Web'} mode)`
                                }
                            >
                                {isRecording ? <RecordingMicrophoneIcon /> : <MicrophoneIcon />}
                            </IconButton>
                        )}
                        <Button
                            onClick={() => handleSendMessage(false)}
                            disabled={!inputValue.trim() || isLoading || isRecording}
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
