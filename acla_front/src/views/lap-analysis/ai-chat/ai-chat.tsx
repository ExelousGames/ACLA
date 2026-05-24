import React, { useState, useRef, useEffect, useContext, useMemo } from 'react';
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

const QUICK_PROMPTS = [
    "How's my fuel?",
    "Best line through T3?",
    "Pit or stay out?",
    "Who's behind me?",
    "Bring it home safe",
];

const formatClock = (d: Date) =>
    `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;

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

    // Live clock for the transcript header (matches landing page vibe).
    const [clock, setClock] = useState(formatClock(new Date()));
    useEffect(() => {
        const id = setInterval(() => setClock(formatClock(new Date())), 1000);
        return () => clearInterval(id);
    }, []);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messagesScrollRef = useRef<HTMLDivElement>(null);
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
                    streamedAudio: true,
                }));
            // Broadcast to the floating pill overlay (separate Electron window).
            // 'storage' events fire in other same-origin BrowserWindows but not
            // in the window that writes — perfect one-way fanout.
            try {
                const pillText = event.text
                    .replace(/\*\*(.*?)\*\*/g, '$1')
                    .replace(/\*(.*?)\*/g, '$1')
                    .replace(/`(.*?)`/g, '$1')
                    .replace(/\s+/g, ' ')
                    .trim()
                    .slice(0, 280);
                if (pillText) {
                    localStorage.setItem('acla-pill-msg', JSON.stringify({
                        text: pillText,
                        ts: Date.now(),
                    }));
                }
            } catch { /* ignore storage write failures */ }
            return;
        }
        if (event.kind === 'tool_event') {
            setMessages(prev => {
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

    // Utility function to generate unique message IDs
    const generateUniqueId = (prefix: string = 'msg') => {
        return `${prefix}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
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
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/```[\s\S]*?```/g, '')
            .replace(/`(.*?)`/g, '$1')
            .replace(/https?:\/\/[^\s]+/g, 'link')
            .replace(/[#]+\s*/g, '')
            .replace(/\n+/g, '. ')
            .replace(/\s+/g, ' ')
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
        if (currentNeuralPlaybackRef.current) {
            currentNeuralPlaybackRef.current.stop();
            currentNeuralPlaybackRef.current = null;
        }
        setIsSpeaking(false);
    };

    const toggleTextToSpeech = () => {
        const newState = !isTextToSpeechEnabled;
        setIsTextToSpeechEnabled(newState);

        localStorage.setItem('ai-chat-tts-enabled', newState.toString());

        if (!newState && isSpeaking) {
            stopSpeaking();
        }

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

        if (!lastMessage.isUser && !lastMessage.isLoading && lastMessage.content) {
            if (lastMessage.id === 'welcome' && messages.length === 1) return;
            if (lastMessage.streamedAudio) return;

            const isGuidanceMessage = lastMessage.id.includes('guidance');

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
            const existingCharts = visualizationController.getCurrentInstances();
            existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                    visualizationController.executeCommand({
                        action: 'remove',
                        id: chart.id
                    });
                }
            });
        }
    }, [TrackGuideEnabled, sessionId]);

    useEffect(() => {
        return () => {
            const existingCharts = visualizationController.getCurrentInstances();
            existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                    visualizationController.executeCommand({
                        action: 'remove',
                        id: chart.id
                    });
                }
            });
        };
    }, []);

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

    // Ctrl+Space (or Cmd+Space) — stop ongoing TTS playback.
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

    const handleSendMessage = async (override?: string) => {
        const text = (override ?? inputValue).trim();
        if (!text || isLoading) return;

        // The voice WS is the single chat surface. Backend echoes a
        // user_transcript frame for typed input, so we don't append the
        // user message locally — handleVoiceEvent will when the echo arrives.
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

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    // Function execution system — preserved for future code paths that still
    // call into the registry (function-call indicators in messages).
    const executeFunctionCall = async (functionCall: FunctionCall, responseData: any): Promise<FunctionResult> => {
        try {
            const result = await findAndExecuteFunction(functionCall.function, functionCall.arguments, responseData);
            return { function: functionCall.function, arguments: functionCall.arguments, result, success: true };
        } catch (error) {
            return {
                function: functionCall.function,
                arguments: functionCall.arguments,
                result: null,
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
            };
        }
    };

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
        visualizationController.openVisualization('imitation-guidance-chart', {}, {
            title: 'AI Track Guidance',
            autoUpdate: true
        });
        setTrackGuideEnabled(true);
    };

    const formatFunctionArgs = (args: Record<string, any>): string => {
        return Object.entries(args).map(([key, value]) => {
            if (typeof value === 'object') return `${key}: ${JSON.stringify(value)}`;
            return `${key}: ${value}`;
        }).join(', ');
    };

    // ── Voice state → mic panel display ─────────────────────────────
    const vState = voiceConversation.state;
    const voiceActive = vState === 'listening' || vState === 'speaking';
    const channelLabel =
        vState === 'idle' ? 'CH-1 · OFFLINE' :
        vState === 'connecting' ? 'CH-1 · CONNECTING' :
        vState === 'error' ? 'CH-1 · ERROR' :
        'CH-1 · OPEN';
    const channelMod =
        vState === 'idle' ? 'ai-chat__mic-channel--idle' :
        vState === 'error' ? 'ai-chat__mic-channel--error' :
        '';
    const coreMod =
        vState === 'idle' || vState === 'connecting' ? 'ai-chat__mic-core--idle' :
        vState === 'error' ? 'ai-chat__mic-core--error' :
        '';
    const statusTop =
        vState === 'idle' ? 'TAP MIC' :
        vState === 'connecting' ? 'CONNECTING' :
        vState === 'speaking' ? 'ACLA' :
        vState === 'listening' ? 'DRIVER' :
        'VOICE';
    const statusBottom =
        vState === 'idle' ? 'TO START' :
        vState === 'connecting' ? '…' :
        vState === 'speaking' ? 'RESPONDING' :
        vState === 'listening' ? 'LISTENING' :
        vState === 'error' ? 'RETRY' :
        'IDLE';
    const statusMod =
        vState === 'idle' || vState === 'connecting' ? 'ai-chat__mic-status--idle' :
        vState === 'error' ? 'ai-chat__mic-status--error' :
        '';

    const toggleVoice = () => {
        if (vState === 'idle' || vState === 'error') {
            voiceConversation.start().catch((err) => {
                console.error('Voice conversation failed to start:', err);
            });
        } else {
            voiceConversation.stop();
        }
    };

    // Wave bars: driver real mic level when listening so the bars visually
    // confirm we're picking up audio; otherwise CSS-only decorative animation.
    const waveBars = useMemo(
        () => Array.from({ length: 24 }, (_, i) => ({
            delay: `${(i % 6) * 0.08}s`,
            duration: `${0.7 + (i % 5) * 0.1}s`,
        })),
        []
    );
    const liveLevels = useMemo(() => {
        // Stable per-bar response curve so adjacent bars don't all jump in sync.
        return Array.from({ length: 24 }, (_, i) => {
            const phase = (i / 24) * Math.PI * 2;
            return 0.55 + 0.45 * Math.abs(Math.sin(phase + Date.now() / 200));
        });
    }, [voiceConversation.micLevel]);
    const useLiveBars = vState === 'listening';

    return (
        <div className="ai-chat">
            <div className="ai-chat__grid-bg" aria-hidden="true" />

            {/* Header */}
            <div className="ai-chat__header">
                <span className="ai-chat__eyebrow">
                    <span className="ai-chat__eyebrow-dot" />
                    {title}
                </span>
                <div className="ai-chat__header-meta">
                    {sessionId && <span className="ai-chat__chip ai-chat__chip--blue">Session</span>}
                    {environment === 'electron' && (
                        <span className="ai-chat__chip ai-chat__chip--green">Desktop</span>
                    )}
                    {!neuralTtsAvailable && (
                        <span className="ai-chat__chip ai-chat__chip--amber">TTS Unavailable</span>
                    )}
                    {voiceConversation.error && (
                        <span className="ai-chat__chip ai-chat__chip--red" title={voiceConversation.error}>
                            Voice Error
                        </span>
                    )}
                    {isTextToSpeechEnabled && (
                        <span className="ai-chat__chip ai-chat__chip--green">TTS On</span>
                    )}
                    <button
                        type="button"
                        className="ai-chat__chip-btn"
                        onClick={() => setDebugMode(!debugMode)}
                        aria-pressed={debugMode}
                    >
                        Debug
                    </button>
                </div>
            </div>

            {/* Stage: mic panel + transcript */}
            <div className="ai-chat__stage">
                <aside className="ai-chat__mic-panel">
                    <div className="ai-chat__mic-head">
                        <span className={`ai-chat__mic-channel ${channelMod}`}>
                            <span className="ai-chat__eyebrow-dot" />
                            {channelLabel}
                        </span>
                        <span>VOICE LINK</span>
                    </div>

                    <div className="ai-chat__mic-visual">
                        {voiceActive && (
                            <>
                                <span className="ai-chat__mic-ring" />
                                <span className="ai-chat__mic-ring" />
                                <span className="ai-chat__mic-ring" />
                            </>
                        )}
                        <button
                            type="button"
                            className={`ai-chat__mic-core ${coreMod}`}
                            onClick={toggleVoice}
                            disabled={vState === 'connecting'}
                            title={
                                vState === 'error' ? `Voice error: ${voiceConversation.error}. Click to retry.` :
                                vState === 'connecting' ? 'Connecting…' :
                                voiceActive ? 'Click to end voice session' :
                                'Click to start voice session'
                            }
                            aria-label="Toggle voice session"
                        >
                            <svg viewBox="0 0 48 48" width="36" height="36" fill="none">
                                <rect x="18" y="6" width="12" height="22" rx="6"
                                    stroke="var(--lp-green)" strokeWidth="2" fill="rgba(0,230,118,0.08)" />
                                <path d="M10 22c0 7.7 6.3 14 14 14s14-6.3 14-14"
                                    stroke="var(--lp-green)" strokeWidth="2" strokeLinecap="round" />
                                <line x1="24" y1="36" x2="24" y2="42" stroke="var(--lp-green)" strokeWidth="2" />
                                <line x1="17" y1="42" x2="31" y2="42"
                                    stroke="var(--lp-green)" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        </button>
                    </div>

                    <div className={`ai-chat__mic-status ${statusMod}`}>
                        {statusTop}
                        <b>{statusBottom}</b>
                    </div>

                    <div
                        className={`ai-chat__mic-wave ${useLiveBars ? 'ai-chat__mic-wave--live' : vState === 'idle' ? 'ai-chat__mic-wave--idle' : ''}`}
                        aria-hidden="true"
                    >
                        {waveBars.map((b, i) => {
                            if (useLiveBars) {
                                const lvl = Math.min(1, voiceConversation.micLevel * 1.8 * liveLevels[i]);
                                return (
                                    <span
                                        key={i}
                                        className="ai-chat__mic-wave-bar"
                                        style={{ height: `${Math.max(8, lvl * 100)}%`, transition: 'height 80ms linear' }}
                                    />
                                );
                            }
                            return (
                                <span
                                    key={i}
                                    className="ai-chat__mic-wave-bar"
                                    style={{ animationDelay: b.delay, animationDuration: b.duration }}
                                />
                            );
                        })}
                    </div>

                    <div className="ai-chat__mic-hint">
                        Push <kbd>PTT</kbd> or say <kbd>&ldquo;Hey ACLA&rdquo;</kbd><br />
                        No menus. No screens. Just talk.
                    </div>

                    <div className="ai-chat__mic-controls">
                        {neuralTtsAvailable && (
                            <button
                                type="button"
                                className={`ai-chat__btn ${isTextToSpeechEnabled ? 'ai-chat__btn--primary' : ''} ${isSpeaking ? 'ai-chat__btn--danger' : ''}`}
                                onClick={isSpeaking ? stopSpeaking : toggleTextToSpeech}
                                disabled={isLoading}
                                title={
                                    isSpeaking ? 'Stop speaking' :
                                    isTextToSpeechEnabled ? 'Disable auto text-to-speech' :
                                    'Enable auto text-to-speech'
                                }
                                style={{ padding: '6px 12px', fontSize: '10px' }}
                            >
                                {isSpeaking ? 'STOP TTS' : isTextToSpeechEnabled ? 'TTS ON' : 'TTS OFF'}
                            </button>
                        )}
                    </div>
                </aside>

                <section className="ai-chat__transcript">
                    <div className="ai-chat__transcript-head">
                        <span className="ai-chat__transcript-title">
                            <span className="ai-chat__eyebrow-dot" />
                            LIVE TRANSCRIPT
                        </span>
                        <span className="ai-chat__transcript-time">{clock}</span>
                    </div>

                    <div className="ai-chat__msgs" ref={messagesScrollRef}>
                        {messages.map((message) => {
                            // Tool-call messages
                            if (message.kind === 'tool' && message.tool) {
                                const t = message.tool;
                                const isError = t.status === 'completed' && t.ok === false;
                                const isRunning = t.status === 'started';
                                const mod = isError ? 'ai-chat__tool--error'
                                    : isRunning ? 'ai-chat__tool--running'
                                    : 'ai-chat__tool--ok';
                                return (
                                    <div key={message.id}>
                                        <div className={`ai-chat__tool ${mod}`}>
                                            <span className="ai-chat__tool-icon">
                                                {isRunning ? '⟳' : isError ? '✕' : '✓'}
                                            </span>
                                            <span>{t.title}</span>
                                            <span className="ai-chat__tool-stamp">
                                                {message.timestamp.toLocaleTimeString()}
                                            </span>
                                        </div>
                                        {isError && t.error && (
                                            <div className="ai-chat__tool-detail" style={{ color: 'var(--lp-red)' }}>
                                                {t.error}
                                            </div>
                                        )}
                                        {debugMode && (
                                            <div className="ai-chat__tool-detail">{t.name}</div>
                                        )}
                                    </div>
                                );
                            }

                            const isGuidance = message.id.includes('guidance');
                            const role: 'driver' | 'acla' | 'guidance' = message.isUser
                                ? 'driver'
                                : isGuidance ? 'guidance' : 'acla';
                            const avatarLabel = role === 'driver' ? 'YOU' : role === 'guidance' ? '🎯' : 'AI';
                            const whoLabel = role === 'driver' ? 'YOU'
                                : role === 'guidance' ? 'LIVE GUIDANCE'
                                : 'ACLA';

                            return (
                                <div key={message.id} className={`ai-chat__msg ai-chat__msg--${role}`}>
                                    <div className="ai-chat__msg-avatar">{avatarLabel}</div>
                                    <div className="ai-chat__msg-body">
                                        <div className="ai-chat__msg-meta">
                                            <span className="ai-chat__msg-who">{whoLabel}</span>
                                            <span className="ai-chat__msg-stamp">
                                                {message.timestamp.toLocaleTimeString()}
                                            </span>
                                            {!message.isUser && isTextToSpeechEnabled && isSpeaking && (
                                                <span className="ai-chat__msg-stamp" style={{ color: 'var(--lp-green)' }}>
                                                    SPEAKING…
                                                </span>
                                            )}
                                        </div>

                                        {message.isLoading ? (
                                            <div className="ai-chat__typing">
                                                <span className="ai-chat__typing-dot" />
                                                <span className="ai-chat__typing-dot" />
                                                <span className="ai-chat__typing-dot" />
                                            </div>
                                        ) : (
                                            <>
                                                <div className="ai-chat__msg-text">{message.content}</div>

                                                {!message.isUser && neuralTtsAvailable && (
                                                    <button
                                                        type="button"
                                                        className="ai-chat__msg-speaker-btn"
                                                        onClick={() => speakText(message.content, { isGuidance })}
                                                        disabled={isSpeaking}
                                                        title="Speak this message"
                                                    >
                                                        🔊 SPEAK
                                                    </button>
                                                )}

                                                {!debugMode && message.functionResults && message.functionResults.length > 0 && (
                                                    <div style={{ marginTop: 6 }}>
                                                        <span className={`ai-chat__chip ${message.functionResults.every(r => r.success) ? 'ai-chat__chip--green' : 'ai-chat__chip--amber'}`}>
                                                            {message.functionResults.every(r => r.success)
                                                                ? `${message.functionResults.length} cmd ok`
                                                                : `${message.functionResults.filter(r => r.success).length}/${message.functionResults.length} cmd ok`}
                                                        </span>
                                                    </div>
                                                )}

                                                {debugMode && message.functionCalls && message.functionCalls.length > 0 && (
                                                    <div className="ai-chat__debug">
                                                        <span className="ai-chat__debug-title">Function Calls</span>
                                                        {message.functionCalls.map((fc, i) => (
                                                            <span key={i} className="ai-chat__debug-row">
                                                                {fc.function}({formatFunctionArgs(fc.arguments)})
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}

                                                {debugMode && message.functionResults && message.functionResults.length > 0 && (
                                                    <div className="ai-chat__debug">
                                                        <span className="ai-chat__debug-title">Function Results</span>
                                                        {message.functionResults.map((fr, i) => (
                                                            <span
                                                                key={i}
                                                                className={`ai-chat__debug-row ${fr.success ? 'ai-chat__debug-row--ok' : 'ai-chat__debug-row--error'}`}
                                                            >
                                                                {fr.function}: {fr.success ? '✓ ok' : '✕ err'}
                                                                {fr.error ? ` — ${fr.error}` : ''}
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                        <div ref={messagesEndRef} />
                    </div>
                </section>
            </div>

            {/* Pills */}
            <div className="ai-chat__pills">
                {QUICK_PROMPTS.map(p => (
                    <button
                        key={p}
                        type="button"
                        className="ai-chat__pill"
                        onClick={() => handleSendMessage(p)}
                        disabled={isLoading}
                    >
                        {p}
                    </button>
                ))}
            </div>

            {/* Input row */}
            <div className="ai-chat__input-row">
                <input
                    className="ai-chat__input"
                    placeholder={
                        voiceActive
                            ? 'Type a message to the engineer…'
                            : 'Click the mic to start a session, then type or talk.'
                    }
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                />
                <button
                    type="button"
                    className="ai-chat__btn ai-chat__btn--primary"
                    onClick={() => handleSendMessage()}
                    disabled={!inputValue.trim() || isLoading}
                    title="Send"
                >
                    SEND
                </button>
            </div>
        </div>
    );
};

export default AiChat;
