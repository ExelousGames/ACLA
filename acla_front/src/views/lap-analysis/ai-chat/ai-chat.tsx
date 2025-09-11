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

    // Text-to-speech states
    const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null);
    const [isTextToSpeechEnabled, setIsTextToSpeechEnabled] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
    const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);

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
    const speechQueueRef = useRef<string[]>([]);
    const currentUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

    const analysisContext = useContext(AnalysisContext);

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

    // Text-to-speech utility functions
    const initializeTextToSpeech = () => {
        if ('speechSynthesis' in window) {
            const synth = window.speechSynthesis;
            setSpeechSynthesis(synth);

            // Load available voices
            const loadVoices = () => {
                const voices = synth.getVoices();
                setAvailableVoices(voices);

                // Select a default voice (prefer English voices)
                const englishVoices = voices.filter(voice => voice.lang.startsWith('en'));
                const defaultVoice = englishVoices.find(voice => voice.default) || 
                                   englishVoices.find(voice => voice.name.includes('Google')) ||
                                   englishVoices[0] || 
                                   voices[0];
                
                if (defaultVoice) {
                    setSelectedVoice(defaultVoice);
                }
            };

            // Load voices immediately
            loadVoices();

            // Also load voices when they become available (some browsers load them asynchronously)
            synth.onvoiceschanged = loadVoices;

            return true;
        }
        return false;
    };

    const speakText = (text: string, options?: { isGuidance?: boolean }) => {
        if (!speechSynthesis || !isTextToSpeechEnabled || !selectedVoice) {
            return;
        }

        // Stop current speech if speaking
        if (isSpeaking) {
            speechSynthesis.cancel();
            setIsSpeaking(false);
        }

        // Clean text for better speech synthesis
        const cleanText = text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove markdown bold
            .replace(/\*(.*?)\*/g, '$1') // Remove markdown italic
            .replace(/```[\s\S]*?```/g, '') // Remove code blocks
            .replace(/`(.*?)`/g, '$1') // Remove inline code
            .replace(/https?:\/\/[^\s]+/g, 'link') // Replace URLs with "link"
            .replace(/[#]+\s*/g, '') // Remove markdown headers
            .replace(/\n+/g, '. ') // Replace newlines with periods
            .replace(/\s+/g, ' ') // Normalize whitespace
            .trim();

        if (!cleanText) return;

        // Split very long messages into chunks to prevent browser timeout
        const maxLength = 200;
        if (cleanText.length > maxLength) {
            const sentences = cleanText.split(/[.!?]+/).filter(s => s.trim().length > 0);
            let currentChunk = '';
            
            for (const sentence of sentences) {
                if ((currentChunk + sentence).length > maxLength && currentChunk) {
                    speechQueueRef.current.push(currentChunk.trim() + '.');
                    currentChunk = sentence;
                } else {
                    currentChunk += (currentChunk ? '. ' : '') + sentence;
                }
            }
            
            if (currentChunk.trim()) {
                speechQueueRef.current.push(currentChunk.trim() + '.');
            }
            
            // Start speaking the first chunk
            if (speechQueueRef.current.length > 0) {
                const firstChunk = speechQueueRef.current.shift();
                if (firstChunk) {
                    speakText(firstChunk, options);
                }
            }
            return;
        }

        const utterance = new SpeechSynthesisUtterance(cleanText);
        currentUtteranceRef.current = utterance;

        utterance.voice = selectedVoice;
        utterance.rate = options?.isGuidance ? 1.1 : 0.9; // Slightly faster for guidance
        utterance.pitch = options?.isGuidance ? 1.1 : 1.0; // Slightly higher pitch for guidance
        utterance.volume = 0.8;

        utterance.onstart = () => {
            setIsSpeaking(true);
        };

        utterance.onend = () => {
            setIsSpeaking(false);
            currentUtteranceRef.current = null;
            
            // Process queue if there are more messages
            if (speechQueueRef.current.length > 0) {
                const nextText = speechQueueRef.current.shift();
                if (nextText) {
                    setTimeout(() => speakText(nextText), 100);
                }
            }
        };

        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event);
            setIsSpeaking(false);
            currentUtteranceRef.current = null;
        };

        speechSynthesis.speak(utterance);
    };

    const stopSpeaking = () => {
        if (speechSynthesis && isSpeaking) {
            speechSynthesis.cancel();
            setIsSpeaking(false);
            currentUtteranceRef.current = null;
            speechQueueRef.current = []; // Clear queue
        }
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

        // Initialize text-to-speech
        const ttsAvailable = initializeTextToSpeech();
        if (ttsAvailable) {
            console.log('Text-to-speech initialized successfully');
            
            // Load saved TTS preference
            const savedTtsEnabled = localStorage.getItem('ai-chat-tts-enabled');
            if (savedTtsEnabled === 'true') {
                setIsTextToSpeechEnabled(true);
            }
        } else {
            console.warn('Text-to-speech not available in this environment');
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
    const sendToAI = async (messageContent: string) => {
        try {
            let response;

            // Use openai general natural language ai query endpoint
            response = await apiService.post('user-ai-model/ai-query', {
                question: messageContent,
                context: {
                    session_id: sessionId || '',
                    trackname: analysisContext?.recordedSessioStaticsData?.track || '',
                },
            });

            const responseData = response.data as any;
            let aiResponseContent = responseData?.payload?.answer || responseData?.answer || responseData?.response || "I'm sorry, I couldn't process your request.";
            let functionCalls: FunctionCall[] = [];
            let functionResults: FunctionResult[] = [];
            console.log('AI response data:', responseData);

            // Check if the response contains track guidance data in side_products
            if (responseData?.payload?.side_products?.track_detail_for_guide) {
                try {
                    // Call startTrackGuide with the complete response data
                    startTrackGuide(responseData);
                    console.log('Track guidance enabled from AI response');
                } catch (error) {
                    console.error('Error enabling track guidance:', error);
                    aiResponseContent += '\n\n*Note: Track guidance data was received but could not be processed properly.*';
                }
            }

            // Check if the response contains function calls (legacy support)
            if (responseData?.function_calls && Array.isArray(responseData.function_calls)) {

                try {
                    // Parse function calls from the response
                    functionCalls = responseData.function_calls.map((fc: any) => ({
                        function: fc.function || fc.name,
                        arguments: typeof fc.arguments === 'string' ? JSON.parse(fc.arguments) : fc.arguments
                    }));

                    // Execute all function calls
                    for (const functionCall of functionCalls) {
                        const result = await executeFunctionCall(functionCall, responseData);
                        functionResults.push(result);
                    }

                } catch (parseError) {
                    console.error('Error parsing function calls:', parseError);
                    aiResponseContent += '\n\n*Note: Function calls were detected but could not be parsed properly.*';
                }
            }

            const aiResponse: Message = {
                id: generateUniqueId('ai'),
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
                id: generateUniqueId('error'),
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

            case 'track_detail_for_guide':
                // Enable the continuous imitation learning guidance
                startTrackGuide(responseData);

                return {
                    status: 'Imitation learning guidance enabled - now continuously monitoring telemetry data and displaying AI guidance chart',
                    enabled: true,
                    chartAdded: true
                };

            case 'disable_guide_user_racing':
                // Disable the continuous imitation learning guidance
                setTrackGuideEnabled(false);

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

    const startTrackGuide = (responseData: any) => {

        const preloadSentences = parseGuidanceData(responseData.payload.answer);
        const trackData = responseData.payload?.side_products?.track_detail_for_guide;

        // Add imitation guidance chart with the track data
        if (preloadSentences && trackData) {
            visualizationController.executeCommand({
                action: 'add',
                type: 'imitation-guidance-chart',
                data: {
                    preloadSentences: preloadSentences,
                    trackData: trackData,
                    sessionId: sessionId,
                    autoManaged: true
                },
                config: {
                    title: 'AI Track Guidance',
                    autoUpdate: true
                }
            });
        }

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
                        {/* Text-to-speech controls */}
                        {speechSynthesis && (
                            <>
                                <Button
                                    variant="ghost"
                                    size="1"
                                    onClick={toggleTextToSpeech}
                                    color={isTextToSpeechEnabled ? "green" : "gray"}
                                    title={isTextToSpeechEnabled ? "Disable text-to-speech" : "Enable text-to-speech"}
                                >
                                    🔊 TTS
                                </Button>
                                {isSpeaking && (
                                    <Button
                                        variant="ghost"
                                        size="1"
                                        onClick={stopSpeaking}
                                        color="red"
                                        title="Stop speaking"
                                    >
                                        🔇 Stop
                                    </Button>
                                )}
                            </>
                        )}
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
                        {!speechSynthesis && (
                            <Badge variant="soft" color="orange" size="1">
                                Text-to-speech not supported
                            </Badge>
                        )}
                        {recording.error && (
                            <Badge variant="soft" color="red" size="1">
                                Voice error
                            </Badge>
                        )}
                        {isTextToSpeechEnabled && (
                            <Badge variant="soft" color="green" size="1">
                                TTS On ({selectedVoice?.name?.split(' ')[0] || 'Default'})
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
                                            {!message.isUser && speechSynthesis && !message.isLoading && (
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
