import React, { useState, useRef, useEffect, useContext, useMemo, useCallback } from 'react';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { useUserSummary } from 'contexts/UserSummaryContext';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { detectEnvironment } from 'utils/environment';
import apiService from 'services/api.service';
import {
    createAiCommandRegistry,
    startAgentRuntime,
} from './ai-command-registry';
import { getCornersForTrack } from 'views/lap-analysis/session-intelligence/track-corners';
import type { CornerDefinition } from 'views/lap-analysis/session-intelligence/types';
import type {
    AgentSessionInfo,
    AgentSessionMode,
    AgentSessionStartResult,
    AgentSessionStopResult,
    LivePerformanceAnalystState,
    OpportunityAgentState,
} from './ai-command-registry';
import { useVoiceConversation, VoiceEvent } from './use-voice-conversation';
import { AiMapDisplayPayload } from './AiMapToolDisplay';
import AiMessageDisplay, { type AiChatDisplayMessage } from './AiMessageDisplay';
import BaselineProgressDisplay from './BaselineProgressDisplay';
import ProcedurePlanDisplay from './ProcedurePlanDisplay';
import {
    advanceProcedurePlan,
    buildProcedurePlan,
    getProcedurePlanUpdateKey,
    isProcedurePlanClearEvent,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
    type ProcedurePlan,
    type ProcedurePlanRequest,
} from './ai-chat-plan';
import {
    BaselineCollectionTracker,
} from './BaselineCollectionTracker';
import LiveRangeTracker, {
    type LiveRangeTrackerHandle,
    type LiveRangeTrackerState,
} from './LiveRangeTracker';
import { useBaselineCollectionRuntime } from './BaselineCollectionRuntime';
import {
    getToolEnvelopeError,
    isToolOutputEnvelope,
    type ToolOutputEnvelope,
} from './ai-tool-base';

type AiChatSessionMode = 'live' | 'recorded' | 'user_summary';

const EMOTIONS = ['idle', 'sad', 'vibing', 'scared', 'waiting', 'hearing'] as const;
type Emotion = typeof EMOTIONS[number];
const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const EMOTION_TAG_RE = /^\[([a-z]+)\]\s*/;
const MAX_OVERTAKE_AGENT_ROWS = 300;
const TRANSCRIPT_BOTTOM_THRESHOLD_PX = 48;
type ChatLlmModelOption = {
    value: string;
    label: string;
};

const CHAT_LLM_MODEL_OPTIONS: ChatLlmModelOption[] = [
    {
        value: 'openai:gpt-5.5',
        label: 'GPT-5.5',
    },
    {
        value: 'openai:gpt-4.1',
        label: 'GPT-4.1',
    },
    {
        value: 'hosted:qwen/qwen3-32b',
        label: 'Qwen3 32B',
    },
    {
        value: 'hosted:llama-3.3-70b-versatile',
        label: 'Llama 3.3 70B',
    },
];
const DEFAULT_CHAT_LLM_MODEL_OPTION = CHAT_LLM_MODEL_OPTIONS[0];
const getChatLlmModelOption = (value: string) =>
    CHAT_LLM_MODEL_OPTIONS.find((option) => option.value === value)
    || DEFAULT_CHAT_LLM_MODEL_OPTION;

function extractEmotion(text: string): { emotion: Emotion | null; cleanText: string } {
    const m = text.match(EMOTION_TAG_RE);
    if (m && (EMOTIONS as readonly string[]).includes(m[1])) {
        return { emotion: m[1] as Emotion, cleanText: text.slice(m[0].length) };
    }
    return { emotion: null, cleanText: text };
}

type MessageKind = AiChatDisplayMessage['kind'];

interface Message {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    isLoading?: boolean;
    /** Default 'chat' — text bubble. 'tool' renders the distinct
     *  tool-call box (different background + readable title). */
    kind?: MessageKind;
    /** Tool-call metadata when kind === 'tool'. */
    tool?: {
        runId?: string;
        name: string;
        title: string;
        status: 'started' | 'completed';
        ok?: boolean;
        error?: string | null;
        result?: unknown;
    };
    mapDisplay?: AiMapDisplayPayload;
}


interface AiChatProps {
    sessionId?: string;
    sessionMode?: AiChatSessionMode;
    title?: string;
}

const formatClock = (d: Date) =>
    `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;

const OverlayIcon = ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <rect x="1.5" y="3.5" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3" />
        <rect x="5.5" y="6.5" width="9" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.3" fill="currentColor" fillOpacity="0.18" />
    </svg>
);

const getNormalizedCarPos = (telemetry: Record<string, any> | null): number | undefined => {
    if (!telemetry) return undefined;
    const keys = [
        'Graphics_normalized_car_position',
        'graphics_normalized_car_position',
        'normalized_car_position',
        'car_position',
    ];
    for (const key of keys) {
        if (key in telemetry) {
            const value = Number(telemetry[key]);
            if (Number.isFinite(value)) return value;
        }
    }
    return undefined;
};

const crossedNormalizedPosition = (
    lastPos: number,
    currentPos: number,
    targetPos: number,
): boolean => {
    if (currentPos >= lastPos) {
        return lastPos < targetPos && currentPos >= targetPos;
    }
    return lastPos < targetPos || currentPos >= targetPos;
};

const normalizeCornerNameForKnowledge = (cornerName: string): string =>
    cornerName.replace(/^T\d+\s+/i, '').trim();

const getTrackNameForGuide = (
    liveData: Record<string, any>,
): string | undefined =>
    typeof liveData.Static_track === 'string' && liveData.Static_track
        ? liveData.Static_track
        : undefined;

const countSummaryTracks = (summary: Record<string, any>): number => {
    const sessionAnalysis = summary?.sessionAnalysis;
    const practiceTracks = sessionAnalysis?.practice?.tracks;
    const analyzerTracks = sessionAnalysis?.tracks;
    const rootTracks = summary?.tracks;
    const tracks = practiceTracks || analyzerTracks || rootTracks;
    return tracks && typeof tracks === 'object' && !Array.isArray(tracks)
        ? Object.keys(tracks).length
        : 0;
};

const getContextDescription = (sessionMode: AiChatSessionMode): string => {
    if (sessionMode === 'recorded') {
        return 'Selected recorded session with saved playback, AI analysis, and session metadata.';
    }
    if (sessionMode === 'user_summary') {
        return 'User summary view with aggregate practice history.';
    }
    return 'Live session with streaming telemetry, event log, and live coaching context.';
};

const createClientSessionId = (prefix: string): string =>
    `${prefix}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getBaselineToolEventResult = (envelope: ToolOutputEnvelope) => {
    const payload = isRecord(envelope.ui_output) ? envelope.ui_output : {};
    const rawProgress = Number(payload.progress_percent ?? envelope.progress_percent ?? 0);

    return {
        progress_percent: Number.isFinite(rawProgress) ? rawProgress : 0,
        status: typeof payload.status === 'string' ? payload.status : envelope.status,
        car: typeof payload.car === 'string' ? payload.car : null,
        track: typeof payload.track === 'string' ? payload.track : null,
        message: typeof payload.message === 'string'
            ? payload.message
            : envelope.message ?? 'Baseline collection updated.',
    };
};

const getPlanToolArguments = (request: ProcedurePlanRequest): Record<string, unknown> => {
    if (!isRecord(request.payload)) return {};
    const nested = request.payload.arguments || request.payload.args || request.payload.parameters;
    return isRecord(nested)
        ? nested
        : request.payload;
};

const getPlanToolRunKey = (
    plan: ProcedurePlan,
    request: ProcedurePlanRequest,
): string => `${plan.currentStep}:${request.name || ''}:${JSON.stringify(request.payload ?? null)}`;

const getAgentDisplayName = (agentMode?: AgentSessionMode | null): string => {
    if (agentMode === 'track_guide') return 'Track Guide';
    if (agentMode === 'overtake') return 'Overtake';
    if (agentMode === 'live_performance_analyst') return 'Live Analyst';
    return 'Agent';
};

const findTriggeredCorners = (
    corners: CornerDefinition[],
    lastPos: number,
    currentPos: number,
): CornerDefinition[] =>
    corners.filter((corner) => crossedNormalizedPosition(lastPos, currentPos, corner.guideFrom ?? corner.from));

const extractCornerKnowledgeMessage = (raw: any): string | null => {
    if (raw?.status === 'unsupported' && typeof raw.message === 'string') {
        return raw.message;
    }

    const knowledge = raw?.track_knowledge;
    if (!knowledge || knowledge.error) return null;

    const detail = knowledge.corner_detail;

    if (typeof detail === 'string' && detail.trim()) {
        return detail.trim();
    }
    if (Array.isArray(detail) && detail.length > 0) {
        return detail.join('. ');
    }
    return null;
};

const AiChat: React.FC<AiChatProps> = ({ sessionId, sessionMode = 'live', title = "AI Assistant" }) => {
    const [mainMessages, setMainMessages] = useState<Message[]>([]);
    const [agentMessages, setAgentMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const mainClientSessionIdRef = useRef<string>(createClientSessionId('main'));
    const [activeAgentSession, setActiveAgentSession] = useState<AgentSessionInfo | null>(null);

    // Loading and mode states
    const [isLoading] = useState(false);
    const [debugMode, setDebugMode] = useState(false);
    const [TrackGuideEnabled, setTrackGuideEnabled] = useState(false);
    const [livePerformanceAnalystEnabled, setLivePerformanceAnalystEnabled] = useState(false);
    const [procedurePlan, setProcedurePlanState] = useState<ProcedurePlan | null>(null);

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');
    const [floatingChatOpen, setFloatingChatOpen] = useState(false);
    const [selectedChatLlmModel, setSelectedChatLlmModel] = useState(
        DEFAULT_CHAT_LLM_MODEL_OPTION.value,
    );

    // Emotion GIF settings — keyed by Emotion, values are data URLs.
    const [emotionGifs, setEmotionGifs] = useState<Partial<Record<Emotion, string>>>(() => {
        try { return JSON.parse(localStorage.getItem(EMOTION_GIFS_KEY) || '{}'); }
        catch { return {}; }
    });
    const [showEmoteSettings, setShowEmoteSettings] = useState(false);

    // Live clock for the transcript header (matches landing page vibe).
    const [clock, setClock] = useState(formatClock(new Date()));
    useEffect(() => {
        const id = setInterval(() => setClock(formatClock(new Date())), 1000);
        return () => clearInterval(id);
    }, []);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messagesScrollRef = useRef<HTMLDivElement>(null);
    const shouldAutoScrollMessagesRef = useRef(true);
    const analysisContext = useContext(AnalysisContext);
    const {
        userSummary,
        userSummaryLoading,
        userSummaryError,
    } = useUserSummary();
    const {
        getLabelName,
        getCategoryLabels,
    } = useAiLabels();
    const {
        getCircuitMapById,
        getCircuitMapByTrack,
    } = useCircuitMaps();
    const opportunityForecastRowsRef = useRef<Record<string, any>[]>([]);
    const opportunityAgentStateRef = useRef<OpportunityAgentState>({
        intervalId: null,
        inFlight: false,
        lastAlertKey: null,
        lastAlertAt: 0,
    });
    const livePerformanceAnalystStateRef = useRef<LivePerformanceAnalystState>({
        intervalId: null,
        inFlight: false,
        enabled: false,
        lastObservationKey: null,
        lastObservationAt: 0,
        lastSpokenAt: 0,
    });
    const trackGuideLastPosRef = useRef<number | undefined>(undefined);
    const trackGuideTriggeredRef = useRef<Set<string>>(new Set());
    const trackGuideRunTokenRef = useRef(0);
    const activeAgentTagsRef = useRef<string[]>([]);
    const activeAgentSessionRef = useRef<AgentSessionInfo | null>(null);
    const agentVoiceStopRef = useRef<() => void>(() => undefined);
    const mainVoiceStopRef = useRef<() => void>(() => undefined);
    const agentAutoStartSessionIdRef = useRef<string | null>(null);
    const procedurePlanRef = useRef<ProcedurePlan | null>(null);
    const procedurePlanOptedOutRef = useRef(false);
    const planToolRunsRef = useRef<Set<string>>(new Set());
    const lastBroadcastedProcedurePlanKeyRef = useRef<string | null>(null);
    const liveRangeTrackerRef = useRef<LiveRangeTrackerHandle | null>(null);

    useEffect(() => {
        activeAgentSessionRef.current = activeAgentSession;
    }, [activeAgentSession]);

    const messages = activeAgentSession ? agentMessages : mainMessages;
    const setFocusedMessages = useCallback((
        updater: React.SetStateAction<Message[]>,
    ) => {
        if (activeAgentSessionRef.current) {
            setAgentMessages(updater);
            return;
        }
        setMainMessages(updater);
    }, []);
    const setMessages = setFocusedMessages;

    useEffect(() => {
        const liveData = analysisContext?.liveData as Record<string, any> | null;
        if (!liveData || Object.keys(liveData).length === 0) {
            return;
        }
        opportunityForecastRowsRef.current = [
            ...opportunityForecastRowsRef.current,
            liveData,
        ].slice(-MAX_OVERTAKE_AGENT_ROWS);
    }, [analysisContext?.liveData]);

    // Utility function to generate unique message IDs
    const generateUniqueId = useCallback((prefix: string = 'msg') => {
        return `${prefix}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    }, []);

    const broadcastPillPayload = useCallback((
        payload: {
            kind: 'message' | 'tool' | 'baseline' | 'map' | 'plan' | 'range';
            text?: string;
            data?: unknown;
            emotion?: Emotion | null;
            tags?: string[];
            name?: string;
        },
    ) => {
        try {
            localStorage.setItem('acla-pill-msg', JSON.stringify({
                ...payload,
                ts: Date.now(),
                emotion: payload.emotion ?? undefined,
                tags: payload.tags ?? activeAgentTagsRef.current,
            }));
        } catch { /* ignore storage write failures */ }
    }, []);

    const broadcastPillMessage = useCallback((text: string, options: { emotion?: Emotion | null; tags?: string[]; name?: string } = {}) => {
        try {
            const pillText = text
                .replace(/\*\*(.*?)\*\*/g, '$1')
                .replace(/\*(.*?)\*/g, '$1')
                .replace(/`(.*?)`/g, '$1')
                .replace(/\s+/g, ' ')
                .trim()
                .slice(0, 280);
            if (pillText || options.tags !== undefined) {
                broadcastPillPayload({
                    kind: 'message',
                    text: pillText,
                    name: options.name,
                    emotion: options.emotion ?? undefined,
                    tags: options.tags ?? activeAgentTagsRef.current,
                });
            }
        } catch { /* ignore storage write failures */ }
    }, [broadcastPillPayload]);

    const displayMapInChat = useCallback((display: AiMapDisplayPayload) => {
        const fallbackText = display.status === 'unavailable'
            ? 'Map is not available'
            : display.note || display.title || display.map?.circuit_name || 'Map';
        broadcastPillPayload({
            kind: 'map',
            text: fallbackText,
            data: display,
        });
        setMessages(prev => prev
            .filter(m => !m.isLoading)
            .concat({
                id: generateUniqueId('map'),
                content: fallbackText,
                isUser: false,
                timestamp: new Date(),
                kind: 'chat',
                mapDisplay: display,
            }));
    }, [broadcastPillPayload, generateUniqueId, setMessages]);

    const setAgentTag = useCallback((tag: string, active: boolean) => {
        const current = activeAgentTagsRef.current;
        const next = active
            ? Array.from(new Set([...current, tag]))
            : current.filter((item) => item !== tag);
        if (next.length === current.length && next.every((item, index) => item === current[index])) {
            return;
        }
        activeAgentTagsRef.current = next;
        broadcastPillMessage('', { tags: next });
    }, [broadcastPillMessage]);

    useEffect(() => {
        activeAgentTagsRef.current = [];
        broadcastPillMessage('', { tags: [] });
    }, [broadcastPillMessage]);

    const setTrackGuideAgentEnabled = useCallback((enabled: boolean) => {
        if (!enabled) {
            trackGuideRunTokenRef.current += 1;
        }
        setTrackGuideEnabled(enabled);
    }, []);

    const setLivePerformanceAnalystAgentEnabled = useCallback((enabled: boolean) => {
        livePerformanceAnalystStateRef.current.enabled = enabled;
        setLivePerformanceAnalystEnabled(enabled);
    }, []);

    const setProcedurePlan = useCallback((plan: ProcedurePlan | null) => {
        if (!plan || isProcedurePlanStartEvent(plan.sourceEvent)) {
            planToolRunsRef.current.clear();
        }
        procedurePlanRef.current = plan;
        setProcedurePlanState(plan);
        if (!plan) {
            lastBroadcastedProcedurePlanKeyRef.current = null;
            return;
        }

        const planKey = getProcedurePlanUpdateKey(plan);
        if (planKey === lastBroadcastedProcedurePlanKeyRef.current) {
            return;
        }
        lastBroadcastedProcedurePlanKeyRef.current = planKey;

        broadcastPillPayload({
            kind: 'plan',
            text: plan.requests[plan.currentStep]?.title || plan.goal,
            data: plan,
        });
    }, [broadcastPillPayload]);

    const advanceProcedurePlanStep = useCallback((reason?: string) => {
        const current = procedurePlanRef.current;
        if (!current) {
            return { status: 'unavailable', error: 'no_procedure_plan' };
        }

        const result = advanceProcedurePlan(current, reason);
        setProcedurePlan(result.plan);

        return result;
    }, [setProcedurePlan]);

    const clearProcedurePlan = useCallback(() => {
        setProcedurePlan(null);
    }, [setProcedurePlan]);

    const setProcedurePlanRequestStatus = useCallback((
        index: number,
        status: ProcedurePlanRequest['status'],
        detail?: string,
    ) => {
        const current = procedurePlanRef.current;
        if (!current || !current.requests[index]) return;

        const next: ProcedurePlan = {
            ...current,
            requests: current.requests.map((request, requestIndex) => (
                requestIndex === index
                    ? {
                        ...request,
                        status,
                        detail: detail ?? request.detail,
                    }
                    : request
            )),
        };
        setProcedurePlan(next);
    }, [setProcedurePlan]);

    const optOutProcedurePlan = useCallback(() => {
        procedurePlanOptedOutRef.current = true;
        setProcedurePlan(null);
    }, [setProcedurePlan]);

    // Racing engineer voice conversation. The hook owns mic, WS, and
    // audio playback; it ALSO multiplexes the tool-relay text channel on
    // the same WS — frontend tools listed below are reachable from the
    // backend LLM via JSON text frames.
    const handleSessionVoiceEvent = useCallback((event: VoiceEvent, target: 'main' | 'agent') => {
        const setTargetMessages = target === 'agent' ? setAgentMessages : setMainMessages;
        if (event.kind === 'user_transcript') {
            if (isProcedurePlanOptOutRequest(event.text)) {
                optOutProcedurePlan();
            }
            setTargetMessages(prev => prev
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
            // Backend strips the [emotion] tag before sending the transcript,
            // but fall back to frontend parsing for robustness.
            const { emotion, cleanText } = event.emotion
                ? { emotion: event.emotion as Emotion, cleanText: event.text }
                : extractEmotion(event.text);
            setTargetMessages(prev => prev
                .filter(m => !m.isLoading)
                .concat({
                    id: generateUniqueId('ai-voice'),
                    content: cleanText,
                    isUser: false,
                    timestamp: new Date(),
                    kind: 'chat',
                }));
            // Broadcast to the floating pill overlay (separate Electron window).
            // 'storage' events fire in other same-origin BrowserWindows but not
            // in the window that writes — perfect one-way fanout.
            broadcastPillMessage(cleanText, {
                emotion,
                name: target === 'agent' ? getAgentDisplayName(activeAgentSessionRef.current?.agentMode) : undefined,
            });
            return;
        }
        if (event.kind === 'observation') {
            const sourceEvent = typeof event.data.event === 'string' ? event.data.event : undefined;
            if (isProcedurePlanClearEvent(sourceEvent)) {
                clearProcedurePlan();
                return;
            }
            const plan = buildProcedurePlan(event.data);
            if (plan) {
                if (isProcedurePlanStartEvent(plan.sourceEvent)) {
                    procedurePlanOptedOutRef.current = false;
                }
                if (procedurePlanOptedOutRef.current) {
                    return;
                }
                setProcedurePlan(plan);
            }
            return;
        }
        if (event.kind === 'tool_event') {
            console.log(`[ai-tool] tool_event ${event.status}`, {
                name: event.name,
                title: event.title,
                status: event.status,
                arguments: event.arguments,
                result: event.result,
                ok: event.ok,
                error: event.error,
            });
            broadcastPillPayload({
                kind: 'tool',
                text: event.title,
                data: {
                    runId: event.runId,
                    name: event.name,
                    title: event.title,
                    status: event.status,
                    ok: event.ok,
                    error: event.error ?? null,
                    result: event.result,
                },
            });
            setTargetMessages(prev => {
                for (let i = prev.length - 1; i >= 0; i--) {
                    const m = prev[i];
                    const matchesRun = event.runId
                        ? m.tool?.runId === event.runId
                        : m.tool?.name === event.name;
                    if (m.kind === 'tool' && m.tool && matchesRun) {
                        const next = prev.slice();
                        next[i] = {
                            ...m,
                            content: event.title,
                            tool: {
                                ...m.tool,
                                title: event.title,
                                status: event.status,
                                ok: event.ok,
                                error: event.error ?? null,
                                result: event.result,
                            },
                        };
                        return next;
                    }
                }
                return prev.concat({
                    id: generateUniqueId('tool'),
                    content: event.title,
                    isUser: false,
                    timestamp: new Date(),
                    kind: 'tool',
                    tool: {
                        runId: event.runId,
                        name: event.name,
                        title: event.title,
                        status: event.status,
                        ok: event.ok,
                        error: event.error ?? null,
                        result: event.result,
                    },
                });
            });
            return;
        }
    }, [
        broadcastPillMessage,
        broadcastPillPayload,
        clearProcedurePlan,
        generateUniqueId,
        optOutProcedurePlan,
        setProcedurePlan,
    ]);

    const handleMainVoiceEvent = useCallback((event: VoiceEvent) => {
        handleSessionVoiceEvent(event, 'main');
    }, [handleSessionVoiceEvent]);

    const handleAgentVoiceEvent = useCallback((event: VoiceEvent) => {
        handleSessionVoiceEvent(event, 'agent');
    }, [handleSessionVoiceEvent]);

    const handleBaselineToolOutput = useCallback((envelope: ToolOutputEnvelope) => {
        const envelopeError = getToolEnvelopeError(envelope);
        if (!envelope.final && !envelopeError) {
            return;
        }

        handleSessionVoiceEvent({
            kind: 'tool_event',
            runId: envelope.run_id,
            name: envelope.tool_name,
            title: envelope.message || 'Collect live baseline',
            status: envelope.final ? 'completed' : 'started',
            result: getBaselineToolEventResult(envelope),
            ok: !envelopeError,
            error: envelopeError,
        }, activeAgentSessionRef.current ? 'agent' : 'main');
    }, [handleSessionVoiceEvent]);

    const baselineCollectionRuntime = useBaselineCollectionRuntime({
        onToolOutput: handleBaselineToolOutput,
    });
    const baselineCollectionEnabled = baselineCollectionRuntime.enabled;
    const baselineCollectionTag = baselineCollectionRuntime.tag;
    const setBaselineCollectionEnabled = baselineCollectionRuntime.setEnabled;
    const getBaselineCollectionTag = baselineCollectionRuntime.getTag;
    const getBaselineLapRecord = baselineCollectionRuntime.getLapRecord;
    const getBaselineToolOutput = baselineCollectionRuntime.getToolOutput;
    const subscribeBaselineToolOutput = baselineCollectionRuntime.subscribeToolOutput;
    const restartBaselineCollection = baselineCollectionRuntime.restart;

    useEffect(() => {
        if (!baselineCollectionEnabled || !baselineCollectionTag) return;
        broadcastPillPayload({
            kind: 'baseline',
            text: baselineCollectionTag.detail,
            data: baselineCollectionTag,
        });
    }, [baselineCollectionEnabled, baselineCollectionTag, broadcastPillPayload]);

    const handleLiveRangeTrackerStateChange = useCallback((tracker: LiveRangeTrackerState | null) => {
        if (!tracker || tracker.ranges.length === 0) return;
        broadcastPillPayload({
            kind: 'range',
            text: `${tracker.ranges.length} tracked range${tracker.ranges.length === 1 ? '' : 's'}`,
            data: tracker,
        });
    }, [broadcastPillPayload]);

    const startTrackGuide = useCallback(() => {
        trackGuideRunTokenRef.current += 1;
        setTrackGuideEnabled(true);
    }, []);

    const resolvedSessionId = sessionId || (analysisContext?.sessionSelected as Record<string, any> | null)?.SessionId;

    const aiSessionContext = useMemo(() => {
        const selectedSession = analysisContext?.sessionSelected as Record<string, any> | null;
        const liveData = analysisContext?.liveData as Record<string, any> | null;
        const liveDataKeys = liveData ? Object.keys(liveData).length : 0;
        const summaryTrackCount = countSummaryTracks(userSummary || {});
        const summaryLoaded = !userSummaryLoading && !userSummaryError && summaryTrackCount > 0;
        const recordedAiAnalysis = analysisContext?.recordedAiAnalysis;
        const recordedAnalysisResult = recordedAiAnalysis?.result;
        const recordedPlaybackSummary = analysisContext?.recordedPlaybackSummary;
        const liveSnapshot = sessionMode === 'live'
            ? analysisContext?.sessionIntelligence?.getLiveSessionSnapshot?.()
            : null;
        const activeAgentModes = [
            ...(TrackGuideEnabled ? ['track_guide'] : []),
            ...(opportunityAgentStateRef.current.intervalId ? ['overtake'] : []),
            ...(livePerformanceAnalystEnabled ? ['live_performance_analyst'] : []),
        ];

        const context = {
            assistant_surface: 'lap_analysis_ai_chat',
            conversation_role: 'main',
            client_session_id: mainClientSessionIdRef.current,
            active_agent_session: activeAgentSession
                ? {
                    client_session_id: activeAgentSession.clientSessionId,
                    agent_mode: activeAgentSession.agentMode,
                    status: activeAgentSession.status,
                }
                : null,
            context_kind: sessionMode,
            context_description: getContextDescription(sessionMode),
            session_mode: sessionMode,
            session_id: resolvedSessionId || null,
            active_tab: analysisContext?.activeTab || null,
            selected_map_id: analysisContext?.mapSelected || selectedSession?.map || null,
            agent_modes: {
                active: activeAgentModes,
            },
            procedure_plan: procedurePlan
                ? {
                    goal: procedurePlan.goal,
                    requests: procedurePlan.requests,
                    current_request: procedurePlan.currentStep,
                    current_request_text: procedurePlan.requests[procedurePlan.currentStep]?.title || null,
                }
                : null,
            live_session_type: liveSnapshot?.live_session_type ?? 'unknown',
            track: liveSnapshot?.track || liveData?.Static_track || null,
            car: liveSnapshot?.car || liveData?.Static_car_model || null,
            current_lap: liveSnapshot?.current_lap ?? null,
            normalized_position: liveSnapshot?.normalized_position ?? getNormalizedCarPos(liveData),
            completed_laps: liveSnapshot?.completed_laps ?? null,
            sample_count: liveSnapshot?.sample_count ?? 0,
            capabilities: {
                live_session: sessionMode === 'live',
                recorded_session: sessionMode === 'recorded',
                ...(sessionMode === 'user_summary' ? { user_summary: summaryLoaded } : {}),
            },
            selected_session: selectedSession
                ? {
                    id: selectedSession.SessionId || null,
                    name: selectedSession.session_name || null,
                    map: selectedSession.map || null,
                    car: selectedSession.car || null,
                }
                : null,
            telemetry: {
                live_available: sessionMode === 'live' && Boolean(analysisContext?.sessionIntelligence),
                latest_sample_present: liveDataKeys > 0,
                latest_sample_key_count: liveDataKeys,
                live_status: analysisContext?.TelemetryDataLiveStatus ?? null,
                recorded_file_loaded: Boolean(analysisContext?.recordedSessionDataFilePath),
                recorded_sample_count: recordedPlaybackSummary?.sampleCount
                    ?? analysisContext?.recordedTelemetryDataCount
                    ?? 0,
            },
            recorded_session: {
                ai_analysis: {
                    status: recordedAiAnalysis?.status || 'idle',
                    message: recordedAiAnalysis?.message || null,
                    session_id: recordedAiAnalysis?.sessionId || null,
                    segment_count: recordedAnalysisResult?.segment_count ?? 0,
                    samples_analyzed: recordedAnalysisResult?.samples_analyzed ?? 0,
                    result_ready: Boolean(recordedAnalysisResult),
                },
                playback: {
                    session_id: recordedPlaybackSummary?.sessionId || null,
                    sample_count: recordedPlaybackSummary?.sampleCount ?? 0,
                    duration_seconds: recordedPlaybackSummary?.durationSeconds ?? 0,
                    playback_index: recordedPlaybackSummary?.playbackIndex ?? 0,
                    playback_time_seconds: recordedPlaybackSummary?.playbackTimeSeconds ?? 0,
                    active_segment: recordedPlaybackSummary?.activeSegment ?? null,
                },
            },
        };

        if (sessionMode !== 'user_summary') {
            return context;
        }

        return {
            ...context,
            user_summary: {
                loaded: summaryLoaded,
                loading: userSummaryLoading,
                error: userSummaryError || null,
                track_count: summaryTrackCount,
            },
        };
    }, [
        analysisContext?.TelemetryDataLiveStatus,
        analysisContext?.activeTab,
        analysisContext?.liveData,
        analysisContext?.mapSelected,
        analysisContext?.recordedAiAnalysis,
        analysisContext?.recordedPlaybackSummary,
        analysisContext?.recordedSessionDataFilePath,
        analysisContext?.recordedTelemetryDataCount,
        analysisContext?.sessionIntelligence,
        analysisContext?.sessionSelected,
        activeAgentSession,
        livePerformanceAnalystEnabled,
        procedurePlan,
        resolvedSessionId,
        sessionMode,
        TrackGuideEnabled,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const inactiveAgentToolHandlers = useMemo(() => ({}), []);
    const getProcedurePlan = useCallback(() => procedurePlanRef.current, []);
    const getOpportunityTelemetryRows = useCallback(() => opportunityForecastRowsRef.current, []);
    const getMissingLiveRangeTrackerResult = useCallback(() => ({
        status: 'error' as const,
        tracker: null,
        error: 'live_range_tracker_unavailable',
        message: 'Live range tracker UI is not mounted.',
    }), []);
    const setLiveRangeTracker = useCallback((args: Record<string, unknown>) => (
        liveRangeTrackerRef.current?.setTracker(args) ?? getMissingLiveRangeTrackerResult()
    ), [getMissingLiveRangeTrackerResult]);
    const updateLiveRangeTracker = useCallback((args: Record<string, unknown>) => (
        liveRangeTrackerRef.current?.updateTracker(args) ?? getMissingLiveRangeTrackerResult()
    ), [getMissingLiveRangeTrackerResult]);
    const getLiveRangeTracker = useCallback(() => (
        liveRangeTrackerRef.current?.getTracker() ?? getMissingLiveRangeTrackerResult()
    ), [getMissingLiveRangeTrackerResult]);

    const resetLivePerformanceAnalystRuntime = useCallback(() => {
        const analystAgent = livePerformanceAnalystStateRef.current;
        if (analystAgent.intervalId) {
            clearInterval(analystAgent.intervalId);
        }
        analystAgent.intervalId = null;
        analystAgent.inFlight = false;
        analystAgent.enabled = false;
        analystAgent.lastObservationKey = null;
        analystAgent.lastObservationAt = 0;
        analystAgent.lastSpokenAt = 0;
        analystAgent.analysisSessionId = null;
        analysisContext?.sessionIntelligence?.clearFocusSection?.();
        setLivePerformanceAnalystAgentEnabled(false);
        setBaselineCollectionEnabled(false);
        procedurePlanOptedOutRef.current = false;
        clearProcedurePlan();
    }, [analysisContext?.sessionIntelligence, clearProcedurePlan, setBaselineCollectionEnabled, setLivePerformanceAnalystAgentEnabled]);

    const resetOvertakeRuntime = useCallback(() => {
        const opportunityAgent = opportunityAgentStateRef.current;
        if (opportunityAgent.intervalId) {
            clearInterval(opportunityAgent.intervalId);
        }
        opportunityAgent.intervalId = null;
        opportunityAgent.inFlight = false;
        opportunityAgent.lastAlertKey = null;
        opportunityAgent.lastAlertAt = 0;
    }, []);

    const resetAgentRuntimes = useCallback(() => {
        setTrackGuideAgentEnabled(false);
        resetOvertakeRuntime();
        resetLivePerformanceAnalystRuntime();
    }, [
        resetLivePerformanceAnalystRuntime,
        resetOvertakeRuntime,
        setTrackGuideAgentEnabled,
    ]);

    const startAgentSession = useCallback((
        agentMode: AgentSessionMode,
        args: Record<string, any> = {},
    ): AgentSessionStartResult => {
        if (sessionMode !== 'live') {
            return {
                status: 'error',
                conversation_role: 'agent',
                agent_mode: agentMode,
                error: 'non_live_context_live_tools_unavailable',
                message: 'Agent sessions are only available in live session mode.',
            };
        }

        const existing = activeAgentSessionRef.current;
        if (existing && existing.agentMode === agentMode && existing.status !== 'stopped') {
            if (existing.status === 'error') {
                agentAutoStartSessionIdRef.current = null;
            }
            setActiveAgentSession({ ...existing, status: existing.status === 'error' ? 'starting' : existing.status });
            return {
                status: 'already_running',
                conversation_role: 'agent',
                agent_mode: agentMode,
                agent_session_id: existing.clientSessionId,
                parent_client_session_id: existing.parentClientSessionId,
            };
        }

        agentVoiceStopRef.current?.();
        mainVoiceStopRef.current?.();
        resetAgentRuntimes();

        const clientSessionId = createClientSessionId(`agent-${agentMode}`);
        const nextSession: AgentSessionInfo = {
            sessionRole: 'agent',
            clientSessionId,
            parentClientSessionId: mainClientSessionIdRef.current,
            agentMode,
            status: 'starting',
        };
        activeAgentSessionRef.current = nextSession;
        setActiveAgentSession(nextSession);
        setAgentMessages([]);
        setAgentTag(getAgentDisplayName(agentMode), true);
        broadcastPillMessage('', {
            name: getAgentDisplayName(agentMode),
            tags: [getAgentDisplayName(agentMode)],
        });

        return {
            status: 'started',
            conversation_role: 'agent',
            agent_mode: agentMode,
            agent_session_id: clientSessionId,
            parent_client_session_id: mainClientSessionIdRef.current,
        };
    }, [
        broadcastPillMessage,
        resetAgentRuntimes,
        sessionMode,
        setAgentTag,
    ]);

    const stopAgentSession = useCallback((
        agentSessionId?: string | null,
    ): AgentSessionStopResult => {
        const current = activeAgentSessionRef.current;
        if (!current || (agentSessionId && current.clientSessionId !== agentSessionId)) {
            return {
                status: 'not_running',
                conversation_role: 'agent',
                agent_mode: current?.agentMode,
                agent_session_id: agentSessionId || current?.clientSessionId || null,
            };
        }

        setActiveAgentSession({ ...current, status: 'stopping' });
        agentVoiceStopRef.current?.();
        resetAgentRuntimes();
        setActiveAgentSession(null);
        activeAgentSessionRef.current = null;
        agentAutoStartSessionIdRef.current = null;
        setAgentTag(getAgentDisplayName(current.agentMode), false);
        broadcastPillMessage('', { tags: [] });

        return {
            status: 'stopped',
            conversation_role: 'agent',
            agent_mode: current.agentMode,
            agent_session_id: current.clientSessionId,
        };
    }, [broadcastPillMessage, resetAgentRuntimes, setAgentTag]);

    const toolHandlers = useMemo(() => createAiCommandRegistry({
        sessionId: resolvedSessionId,
        sessionMode,
        conversationRole: 'main',
        activeAgentSession,
        analysisContext,
        sessionIntelligence: analysisContext?.sessionIntelligence,
        opportunityAgentState: opportunityAgentStateRef.current,
        livePerformanceAnalystState: livePerformanceAnalystStateRef.current,
        startTrackGuide,
        setTrackGuideEnabled: setTrackGuideAgentEnabled,
        setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
        setBaselineCollectionEnabled,
        restartBaselineCollection,
        advanceProcedurePlanStep,
        getBaselineCollectionTag,
        getBaselineLapRecord,
        getBaselineToolOutput,
        subscribeBaselineToolOutput,
        getProcedurePlan,
        clearProcedurePlan,
        setProcedurePlan,
        setAgentTagActive: setAgentTag,
        startAgentSession,
        stopAgentSession,
        getOpportunityTelemetryRows,
        userSummary,
        userSummaryLoading,
        userSummaryError,
        getLabelName,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        setLiveRangeTracker,
        updateLiveRangeTracker,
        getLiveRangeTracker,
        displayMap: displayMapInChat,
    }), [
        activeAgentSession,
        advanceProcedurePlanStep,
        analysisContext,
        clearProcedurePlan,
        displayMapInChat,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLiveRangeTracker,
        getLabelName,
        getOpportunityTelemetryRows,
        getBaselineCollectionTag,
        getBaselineLapRecord,
        getBaselineToolOutput,
        subscribeBaselineToolOutput,
        getProcedurePlan,
        restartBaselineCollection,
        resolvedSessionId,
        sessionMode,
        setAgentTag,
        setBaselineCollectionEnabled,
        setLiveRangeTracker,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startAgentSession,
        startTrackGuide,
        stopAgentSession,
        updateLiveRangeTracker,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const selectedChatLlmModelOption = getChatLlmModelOption(selectedChatLlmModel);
    const voiceConversation = useVoiceConversation({
        sessionId: resolvedSessionId,
        conversationRole: 'main',
        clientSessionId: mainClientSessionIdRef.current,
        chatLlmModel: selectedChatLlmModelOption.value,
        sessionContext: aiSessionContext,
        onEvent: handleMainVoiceEvent,
        toolHandlers,
    });
    const agentSessionContext = useMemo(() => (
        activeAgentSession
            ? {
                ...aiSessionContext,
                assistant_surface: 'lap_analysis_ai_chat_agent',
                conversation_role: 'agent',
                client_session_id: activeAgentSession.clientSessionId,
                parent_client_session_id: activeAgentSession.parentClientSessionId,
                agent_mode: activeAgentSession.agentMode,
                agent_session: {
                    client_session_id: activeAgentSession.clientSessionId,
                    parent_client_session_id: activeAgentSession.parentClientSessionId,
                    agent_mode: activeAgentSession.agentMode,
                    status: activeAgentSession.status,
                },
            }
            : null
    ), [activeAgentSession, aiSessionContext]);

    const agentToolHandlers = useMemo(() => createAiCommandRegistry({
        sessionId: resolvedSessionId,
        sessionMode,
        conversationRole: 'agent',
        activeAgentSession,
        analysisContext,
        sessionIntelligence: analysisContext?.sessionIntelligence,
        opportunityAgentState: opportunityAgentStateRef.current,
        livePerformanceAnalystState: livePerformanceAnalystStateRef.current,
        startTrackGuide,
        setTrackGuideEnabled: setTrackGuideAgentEnabled,
        setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
        setBaselineCollectionEnabled,
        restartBaselineCollection,
        advanceProcedurePlanStep,
        getBaselineCollectionTag,
        getBaselineLapRecord,
        getBaselineToolOutput,
        subscribeBaselineToolOutput,
        getProcedurePlan,
        clearProcedurePlan,
        setProcedurePlan,
        setAgentTagActive: setAgentTag,
        stopAgentSession,
        getOpportunityTelemetryRows,
        userSummary,
        userSummaryLoading,
        userSummaryError,
        getLabelName,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        setLiveRangeTracker,
        updateLiveRangeTracker,
        getLiveRangeTracker,
        displayMap: displayMapInChat,
    }), [
        activeAgentSession,
        advanceProcedurePlanStep,
        analysisContext,
        clearProcedurePlan,
        displayMapInChat,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLiveRangeTracker,
        getLabelName,
        getOpportunityTelemetryRows,
        getBaselineCollectionTag,
        getBaselineLapRecord,
        getBaselineToolOutput,
        subscribeBaselineToolOutput,
        getProcedurePlan,
        restartBaselineCollection,
        resolvedSessionId,
        sessionMode,
        setAgentTag,
        setBaselineCollectionEnabled,
        setLiveRangeTracker,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startTrackGuide,
        stopAgentSession,
        updateLiveRangeTracker,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const agentVoiceConversation = useVoiceConversation({
        sessionId: resolvedSessionId,
        conversationRole: 'agent',
        clientSessionId: activeAgentSession?.clientSessionId,
        parentClientSessionId: activeAgentSession?.parentClientSessionId,
        agentMode: activeAgentSession?.agentMode,
        chatLlmModel: selectedChatLlmModelOption.value,
        sessionContext: agentSessionContext || undefined,
        onEvent: handleAgentVoiceEvent,
        toolHandlers: activeAgentSession ? agentToolHandlers : inactiveAgentToolHandlers,
    });
    const sendAgentVoiceObservation = agentVoiceConversation.sendObservation;

    useEffect(() => {
        mainVoiceStopRef.current = voiceConversation.stop;
    }, [voiceConversation.stop]);

    useEffect(() => {
        agentVoiceStopRef.current = agentVoiceConversation.stop;
    }, [agentVoiceConversation.stop]);

    const activeAgentSessionId = activeAgentSession?.clientSessionId;
    const activeAgentSessionStatus = activeAgentSession?.status;
    const agentVoiceState = agentVoiceConversation.state;
    const startAgentVoiceConversation = agentVoiceConversation.start;

    useEffect(() => {
        if (!activeAgentSessionId) {
            agentAutoStartSessionIdRef.current = null;
            return;
        }
        if (activeAgentSessionStatus !== 'starting') return;
        if (agentVoiceState !== 'idle' && agentVoiceState !== 'error') return;
        if (agentAutoStartSessionIdRef.current === activeAgentSessionId) return;

        agentAutoStartSessionIdRef.current = activeAgentSessionId;
        startAgentVoiceConversation().catch((err) => {
            console.error('Agent voice conversation failed to start:', err);
            setActiveAgentSession((current) => current
                ? { ...current, status: 'error' }
                : current);
        });
    }, [
        activeAgentSessionId,
        activeAgentSessionStatus,
        agentVoiceState,
        startAgentVoiceConversation,
    ]);

    useEffect(() => {
        if (!activeAgentSession) return;
        if (agentVoiceConversation.state !== 'listening' && agentVoiceConversation.state !== 'speaking') return;
        if (activeAgentSession.status === 'starting') {
            const next = { ...activeAgentSession, status: 'active' as const };
            activeAgentSessionRef.current = next;
            setActiveAgentSession(next);
        }
        const shouldStartRuntime = (
            (activeAgentSession.agentMode === 'track_guide' && !TrackGuideEnabled)
            || (activeAgentSession.agentMode === 'overtake' && !opportunityAgentStateRef.current.intervalId)
            || (activeAgentSession.agentMode === 'live_performance_analyst' && !livePerformanceAnalystStateRef.current.enabled)
        );
        if (!shouldStartRuntime) return;

        window.setTimeout(() => {
            const current = activeAgentSessionRef.current;
            if (!current || current.clientSessionId !== activeAgentSession.clientSessionId) return;
            if (
                (current.agentMode === 'track_guide' && TrackGuideEnabled)
                || (current.agentMode === 'overtake' && opportunityAgentStateRef.current.intervalId)
                || (current.agentMode === 'live_performance_analyst' && livePerformanceAnalystStateRef.current.enabled)
            ) {
                return;
            }

            void startAgentRuntime(current.agentMode, {
                sessionId: resolvedSessionId,
                sessionMode,
                conversationRole: 'agent',
                activeAgentSession: current,
                analysisContext,
                sessionIntelligence: analysisContext?.sessionIntelligence,
                opportunityAgentState: opportunityAgentStateRef.current,
                livePerformanceAnalystState: livePerformanceAnalystStateRef.current,
                startTrackGuide,
                setTrackGuideEnabled: setTrackGuideAgentEnabled,
                setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
                setBaselineCollectionEnabled,
                restartBaselineCollection,
                advanceProcedurePlanStep,
                getBaselineCollectionTag,
                getBaselineLapRecord,
                getBaselineToolOutput,
                subscribeBaselineToolOutput,
                getProcedurePlan,
                clearProcedurePlan,
                setProcedurePlan,
                setAgentTagActive: setAgentTag,
                stopAgentSession,
                getOpportunityTelemetryRows,
                userSummary,
                userSummaryLoading,
                userSummaryError,
                getLabelName,
                getCategoryLabels,
                getCircuitMapById,
                getCircuitMapByTrack,
                setLiveRangeTracker,
                updateLiveRangeTracker,
                getLiveRangeTracker,
                displayMap: displayMapInChat,
            }, {}, {
                sendObservation: agentVoiceConversation.sendObservation,
            });
        }, 0);
    }, [
        activeAgentSession,
        TrackGuideEnabled,
        advanceProcedurePlanStep,
        analysisContext,
        clearProcedurePlan,
        displayMapInChat,
        agentVoiceConversation.state,
        agentVoiceConversation.sendObservation,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLiveRangeTracker,
        getLabelName,
        getOpportunityTelemetryRows,
        getBaselineCollectionTag,
        getBaselineLapRecord,
        getBaselineToolOutput,
        subscribeBaselineToolOutput,
        getProcedurePlan,
        restartBaselineCollection,
        resolvedSessionId,
        sessionMode,
        setAgentTag,
        setBaselineCollectionEnabled,
        setLiveRangeTracker,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startTrackGuide,
        stopAgentSession,
        updateLiveRangeTracker,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    useEffect(() => {
        const sessionIntelligence = analysisContext?.sessionIntelligence;
        if (sessionMode !== 'live' || !sessionIntelligence || !activeAgentSession) return;

        return sessionIntelligence.onLiveAnalystObservation((observation) => {
            if (!livePerformanceAnalystStateRef.current.enabled) return;
            sendAgentVoiceObservation(observation);
        });
    }, [
        activeAgentSession,
        analysisContext?.sessionIntelligence,
        sendAgentVoiceObservation,
        sessionMode,
    ]);

    const activeVoiceConversation = activeAgentSession ? agentVoiceConversation : voiceConversation;
    const vState = activeVoiceConversation.state;
    const voiceActive = vState === 'listening' || vState === 'speaking';
    const modelPickerDisabled = voiceActive || vState === 'connecting';
    const micDisabled = activeVoiceConversation.micDisabled;
    const sendActiveVoiceObservation = activeVoiceConversation.sendObservation;
    const sendLiveRangeTrackerObservation = useCallback((
        data: Record<string, unknown>,
    ) => sendActiveVoiceObservation(data), [sendActiveVoiceObservation]);
    const canOpenFloatingChat = typeof window !== 'undefined'
        && Boolean((window as any).electronAPI?.openFloatingChat);

    useEffect(() => {
        if (!procedurePlan) return;
        const request = procedurePlan.requests[procedurePlan.currentStep];
        if (!request || request.type !== 'tool_call' || !request.name) return;
        if (request.status === 'complete' || request.status === 'failed' || request.status === 'skipped') return;
        if (activeVoiceConversation.state !== 'listening' && activeVoiceConversation.state !== 'speaking') return;

        const runKey = getPlanToolRunKey(procedurePlan, request);
        if (planToolRunsRef.current.has(runKey)) return;
        planToolRunsRef.current.add(runKey);

        const runId = `plan-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        const requestIndex = procedurePlan.currentStep;
        setProcedurePlanRequestStatus(requestIndex, 'running');

        activeVoiceConversation.executeToolCall({
            id: runId,
            name: request.name,
            title: request.title,
            arguments: getPlanToolArguments(request),
        }).then((result) => {
            if (!result) {
                planToolRunsRef.current.delete(runKey);
                setProcedurePlanRequestStatus(
                    requestIndex,
                    'blocked',
                    'Start the active AI session before running this tool.',
                );
                return;
            }
            const envelope = isToolOutputEnvelope(result.result) ? result.result : null;
            if (!envelope) {
                setProcedurePlanRequestStatus(
                    requestIndex,
                    'failed',
                    'Tool did not return a standard output envelope.',
                );
                return;
            }
            const envelopeError = getToolEnvelopeError(envelope);
            if (!result.ok || envelopeError) {
                setProcedurePlanRequestStatus(
                    requestIndex,
                    'failed',
                    result.error || envelopeError || 'Tool failed.',
                );
                return;
            }
            if (!envelope.final) {
                setProcedurePlanRequestStatus(
                    requestIndex,
                    'blocked',
                    envelope.message || 'Tool has not produced a final output yet.',
                );
                return;
            }
            advanceProcedurePlanStep(envelope.message || `tool ${request.name} completed`);
        }).catch((error) => {
            setProcedurePlanRequestStatus(
                requestIndex,
                'failed',
                (error as Error)?.message || 'Tool failed.',
            );
        });
    }, [
        activeVoiceConversation,
        activeVoiceConversation.state,
        advanceProcedurePlanStep,
        procedurePlan,
        setProcedurePlanRequestStatus,
    ]);

    useEffect(() => {
        if (sessionMode === 'live') {
            return;
        }

        if (activeAgentSessionRef.current) {
            stopAgentSession(activeAgentSessionRef.current.clientSessionId);
        }
        setTrackGuideAgentEnabled(false);
        const opportunityAgent = opportunityAgentStateRef.current;
        if (opportunityAgent.intervalId) {
            clearInterval(opportunityAgent.intervalId);
            opportunityAgent.intervalId = null;
        }
        opportunityAgent.inFlight = false;
        opportunityAgent.lastAlertKey = null;
        opportunityAgent.lastAlertAt = 0;
        const analystAgent = livePerformanceAnalystStateRef.current;
        if (analystAgent.intervalId) {
            clearInterval(analystAgent.intervalId);
            analystAgent.intervalId = null;
        }
        analystAgent.inFlight = false;
        analystAgent.enabled = false;
        analystAgent.lastObservationKey = null;
        analystAgent.lastObservationAt = 0;
        analystAgent.lastSpokenAt = 0;
        setLivePerformanceAnalystAgentEnabled(false);
        setBaselineCollectionEnabled(false);
        procedurePlanOptedOutRef.current = false;
        clearProcedurePlan();
        setAgentTag('Track Guide', false);
        setAgentTag('Overtake', false);
        setAgentTag('Live Analyst', false);
    }, [clearProcedurePlan, sessionMode, setAgentTag, setBaselineCollectionEnabled, setLivePerformanceAnalystAgentEnabled, setTrackGuideAgentEnabled, stopAgentSession]);

    const toggleFloatingChat = useCallback(async () => {
        const api = (window as any).electronAPI;
        if (!api?.openFloatingChat) return;
        try {
            if (floatingChatOpen) {
                await api.closeFloatingChat();
                setFloatingChatOpen(false);
            } else {
                await api.openFloatingChat();
                setFloatingChatOpen(true);
            }
        } catch (err) {
            console.warn('Failed to toggle floating chat:', err);
        }
    }, [floatingChatOpen]);

    useEffect(() => {
        const api = (window as any).electronAPI;
        if (!api?.onFloatingChatClosed) return;
        const unsubscribe = api.onFloatingChatClosed(() => setFloatingChatOpen(false));
        if (api.isFloatingChatOpen) {
            api.isFloatingChatOpen()
                .then((open: boolean) => setFloatingChatOpen(Boolean(open)))
                .catch(() => undefined);
        }
        return () => { try { unsubscribe?.(); } catch { /* ignore */ } };
    }, []);

    const addGuidanceMessage = useCallback((content: string) => {
        const message: Message = {
            id: generateUniqueId('guidance'),
            content,
            isUser: false,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, message]);
    }, [generateUniqueId, setMessages]);

    const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
        messagesEndRef.current?.scrollIntoView({ behavior });
    }, []);

    const handleMessagesScroll = useCallback(() => {
        const el = messagesScrollRef.current;
        if (!el) return;
        const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        shouldAutoScrollMessagesRef.current = distanceFromBottom <= TRANSCRIPT_BOTTOM_THRESHOLD_PX;
    }, []);

    const handleGifUpload = (emotion: Emotion, e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            const dataUrl = reader.result as string;
            const next = { ...emotionGifs, [emotion]: dataUrl };
            setEmotionGifs(next);
            localStorage.setItem(EMOTION_GIFS_KEY, JSON.stringify(next));
        };
        reader.readAsDataURL(file);
        e.target.value = '';
    };

    const handleGifRemove = (emotion: Emotion) => {
        const next = { ...emotionGifs };
        delete next[emotion];
        setEmotionGifs(next);
        localStorage.setItem(EMOTION_GIFS_KEY, JSON.stringify(next));
    };

    useEffect(() => {
        if (shouldAutoScrollMessagesRef.current) {
            scrollToBottom();
        }
    }, [messages, scrollToBottom]);

    // Listen for guidance messages from ImitationGuidanceChart
    const lastProcessedGuidanceRef = useRef<string>('');
    const lastGuidanceTimestampRef = useRef<number>(0);
    useEffect(() => {
        if (!TrackGuideEnabled) {
            if (analysisContext?.latestGuidanceMessage) {
                lastProcessedGuidanceRef.current = analysisContext.latestGuidanceMessage;
            }
            return;
        }

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
    }, [analysisContext?.latestGuidanceMessage, generateUniqueId, setMessages, TrackGuideEnabled]);

    useEffect(() => {
        const liveData = analysisContext?.liveData as Record<string, any> | null;
        if (!TrackGuideEnabled) {
            trackGuideRunTokenRef.current += 1;
            trackGuideLastPosRef.current = undefined;
            trackGuideTriggeredRef.current.clear();
            return;
        }
        if (!liveData || Object.keys(liveData).length === 0) return;

        const currentPos = getNormalizedCarPos(liveData);
        const lastPos = trackGuideLastPosRef.current;
        if (currentPos === undefined) return;
        trackGuideLastPosRef.current = currentPos;
        if (lastPos === undefined) return;

        const trackName = getTrackNameForGuide(liveData);
        if (!trackName) return;

        const triggeredCorners = findTriggeredCorners(
            getCornersForTrack(trackName),
            lastPos,
            currentPos,
        );
        if (triggeredCorners.length === 0) return;

        const lap = Number(
            liveData.Graphics_completed_laps
            ?? liveData.Graphics_completed_lap
            ?? 0
        );
        triggeredCorners.forEach((triggeredCorner) => {
            const triggerPosition = triggeredCorner.guideFrom ?? triggeredCorner.from;
            const triggerKey = `${lap}:${triggerPosition}:${triggeredCorner.name}`;
            if (trackGuideTriggeredRef.current.has(triggerKey)) return;

            trackGuideTriggeredRef.current.add(triggerKey);
            const guideToken = trackGuideRunTokenRef.current;

            apiService.post('/racing-session/track-corner-knowledge', {
                track_name: trackName,
                corner_name: normalizeCornerNameForKnowledge(triggeredCorner.name),
                normalized_position: triggerPosition,
                trigger_position: triggerPosition,
                current_telemetry: liveData,
            }).then((response) => {
                if (guideToken !== trackGuideRunTokenRef.current) return;
                const message = extractCornerKnowledgeMessage(response.data);
                if (message) {
                    addGuidanceMessage(message);
                }
            }).catch((error) => {
                if (guideToken !== trackGuideRunTokenRef.current) return;
                const errorDetail = error?.data?.message || error?.data?.detail;
                if (
                    error?.status === 404
                    && typeof errorDetail === 'string'
                    && (
                        errorDetail.includes('not in corpus')
                        || (errorDetail.includes('corner') && errorDetail.includes('not found'))
                    )
                ) {
                    addGuidanceMessage("Track guide doesn't support the current track right now.");
                    return;
                }
                console.warn('Track guide agent knowledge request failed:', error);
            });
        });
    }, [addGuidanceMessage, analysisContext?.liveData, TrackGuideEnabled]);

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
        const opportunityAgent = opportunityAgentStateRef.current;
        const analystAgent = livePerformanceAnalystStateRef.current;
        return () => {
            if (opportunityAgent.intervalId) {
                clearInterval(opportunityAgent.intervalId);
                opportunityAgent.intervalId = null;
            }
            if (analystAgent.intervalId) {
                clearInterval(analystAgent.intervalId);
                analystAgent.intervalId = null;
            }

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
    }, []);

    const handleSendMessage = async (override?: string) => {
        const text = (override ?? inputValue).trim();
        if (!text || isLoading) return;
        if (isProcedurePlanOptOutRequest(text)) {
            optOutProcedurePlan();
        }

        // The voice WS is the single chat surface. Backend echoes a
        // user_transcript frame for typed input, so we don't append the
        // user message locally — handleVoiceEvent will when the echo arrives.
        const sent = activeVoiceConversation.sendUserText(text);
        if (!sent) {
            setMessages(prev => prev.concat({
                id: generateUniqueId('ai'),
                ...(activeAgentSession
                    ? { content: `Start the ${getAgentDisplayName(activeAgentSession.agentMode)} connection first. Agent chat runs on its own session.` }
                    : sessionMode === 'recorded'
                    ? { content: 'Start the assistant connection first. Recorded session context will be sent with the request.' }
                    : sessionMode === 'user_summary'
                        ? { content: 'Start the assistant connection first. User summary context will be sent with the request.' }
                        : { content: 'Click the mic to start a live voice session first - text chat runs on the same connection.' }),
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

    // ── Voice state → mic panel display ─────────────────────────────
    const sessionModeLabel = activeAgentSession
        ? getAgentDisplayName(activeAgentSession.agentMode)
        : sessionMode === 'recorded'
        ? 'Recorded Session'
        : sessionMode === 'user_summary'
            ? 'User Summary'
            : 'Live Session';
    const transcriptLabel = activeAgentSession
        ? `${getAgentDisplayName(activeAgentSession.agentMode).toUpperCase()} TRANSCRIPT`
        : sessionMode === 'recorded'
        ? 'RECORDED TRANSCRIPT'
        : sessionMode === 'user_summary'
            ? 'SUMMARY TRANSCRIPT'
            : 'LIVE TRANSCRIPT';

    const channelLabel =
        vState === 'idle' ? 'CH-1 · OFFLINE' :
        vState === 'connecting' ? 'CH-1 · CONNECTING' :
        vState === 'error' ? 'CH-1 · ERROR' :
        micDisabled ? 'CH-1 · MIC OFF' :
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
        micDisabled ? 'MIC' :
        vState === 'speaking' ? 'ACLA' :
        vState === 'listening' ? 'DRIVER' :
        'VOICE';
    const statusBottom =
        vState === 'idle' ? 'TO START' :
        vState === 'connecting' ? '…' :
        micDisabled ? 'DISABLED' :
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
            activeVoiceConversation.start().catch((err) => {
                console.error('Voice conversation failed to start:', err);
            });
        } else {
            activeVoiceConversation.stop();
        }
    };

    const toggleMicDisabled = () => {
        activeVoiceConversation.setMicDisabled(!micDisabled);
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
            return 0.55 + 0.45 * Math.abs(Math.sin(phase));
        });
    }, []);
    const useLiveBars = vState === 'listening' && !micDisabled;

    return (
        <div className="ai-chat">
            <BaselineCollectionTracker
                {...baselineCollectionRuntime.trackerProps}
                liveData={analysisContext?.liveData as Record<string, any> | null}
                sessionMode={sessionMode}
            />
            <div className="ai-chat__grid-bg" aria-hidden="true" />

            {/* Header */}
            <div className="ai-chat__header">
                <span className="ai-chat__eyebrow">
                    <span className="ai-chat__eyebrow-dot" />
                    {title}
                </span>
                <div className="ai-chat__header-meta">
                    <span className="ai-chat__chip ai-chat__chip--blue">{sessionModeLabel}</span>
                    <select
                        className="ai-chat__model-select"
                        value={selectedChatLlmModel}
                        onChange={(event) => setSelectedChatLlmModel(event.target.value)}
                        disabled={modelPickerDisabled}
                        aria-label="Chat LLM model"
                        title={modelPickerDisabled
                            ? 'End the current voice session before changing models'
                            : 'Choose the model for the next voice chat session'}
                    >
                        {CHAT_LLM_MODEL_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                    {environment === 'electron' && (
                        <span className="ai-chat__chip ai-chat__chip--green">Desktop</span>
                    )}
                    {activeVoiceConversation.error && (
                        <span className="ai-chat__chip ai-chat__chip--red" title={activeVoiceConversation.error}>
                            Voice Error
                        </span>
                    )}
                    {activeAgentSession && (
                        <span className="ai-chat__chip ai-chat__chip--amber">
                            Main Paused
                        </span>
                    )}
                    {activeAgentSession && (
                        <button
                            type="button"
                            className="ai-chat__chip-btn ai-chat__chip-btn--red"
                            onClick={() => stopAgentSession(activeAgentSession.clientSessionId)}
                            title="End the focused agent session"
                        >
                            End Agent
                        </button>
                    )}
                    <button
                        type="button"
                        className={`ai-chat__chip-btn ${micDisabled ? 'ai-chat__chip-btn--red' : ''}`}
                        onClick={toggleMicDisabled}
                        disabled={isLoading}
                        aria-pressed={micDisabled}
                        title={micDisabled ? 'Enable microphone capture' : 'Disable microphone capture'}
                    >
                        {micDisabled ? 'Mic Off' : 'Mic On'}
                    </button>
                    {canOpenFloatingChat && (
                        <button
                            type="button"
                            className={`ai-chat__chip-btn ai-chat__chip-btn--icon ${floatingChatOpen ? 'ai-chat__chip-btn--green' : ''}`}
                            onClick={() => { void toggleFloatingChat(); }}
                            aria-pressed={floatingChatOpen}
                            title={floatingChatOpen
                                ? 'Close the always-on-top AI chat overlay'
                                : 'Open the always-on-top AI chat overlay (visible over the game in borderless windowed mode)'}
                        >
                            <OverlayIcon size={14} />
                            <span>{floatingChatOpen ? 'Overlay On' : 'AI Overlay'}</span>
                        </button>
                    )}
                    <button
                        type="button"
                        className="ai-chat__chip-btn"
                        onClick={() => setDebugMode(!debugMode)}
                        aria-pressed={debugMode}
                    >
                        Debug
                    </button>
                    <button
                        type="button"
                        className="ai-chat__chip-btn"
                        onClick={() => setShowEmoteSettings(!showEmoteSettings)}
                        aria-pressed={showEmoteSettings}
                        title="Emotion GIF settings"
                    >
                        Emotes
                    </button>
                </div>
            </div>

            {/* Emotion GIF settings panel */}
            {showEmoteSettings && (
                <div className="ai-chat__emote-settings">
                    <div className="ai-chat__emote-settings-title">Emotion GIFs</div>
                    {EMOTIONS.map(em => (
                        <div key={em} className="ai-chat__emote-row">
                            <span className="ai-chat__emote-label">[{em}]</span>
                            {emotionGifs[em] && (
                                <img
                                    src={emotionGifs[em]}
                                    alt={em}
                                    className="ai-chat__emote-preview"
                                />
                            )}
                            <label className="ai-chat__btn ai-chat__btn--blue" style={{ cursor: 'pointer' }}>
                                {emotionGifs[em] ? 'Change' : 'Add GIF'}
                                <input
                                    type="file"
                                    accept=".gif,image/gif"
                                    style={{ display: 'none' }}
                                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleGifUpload(em, e)}
                                />
                            </label>
                            {emotionGifs[em] && (
                                <button
                                    type="button"
                                    className="ai-chat__btn"
                                    onClick={() => handleGifRemove(em)}
                                >
                                    Remove
                                </button>
                            )}
                        </div>
                    ))}
                </div>
            )}

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
                            className={`ai-chat__mic-core ${coreMod} ${micDisabled ? 'ai-chat__mic-core--muted' : ''}`}
                            onClick={toggleVoice}
                            disabled={vState === 'connecting'}
                            title={
                                vState === 'error' ? `Voice error: ${activeVoiceConversation.error}. Click to retry.` :
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
                        className={`ai-chat__mic-wave ${useLiveBars ? 'ai-chat__mic-wave--live' : (vState === 'idle' || micDisabled) ? 'ai-chat__mic-wave--idle' : ''}`}
                        aria-hidden="true"
                    >
                        {waveBars.map((b, i) => {
                            if (useLiveBars) {
                                const lvl = Math.min(1, activeVoiceConversation.micLevel * 1.8 * liveLevels[i]);
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

                </aside>

                <section className="ai-chat__transcript">
                    <div className="ai-chat__transcript-head">
                        <span className="ai-chat__transcript-title">
                            <span className="ai-chat__eyebrow-dot" />
                            {transcriptLabel}
                        </span>
                        <span className="ai-chat__transcript-time">{clock}</span>
                    </div>

                    {baselineCollectionEnabled && baselineCollectionTag && (
                        <BaselineProgressDisplay tag={baselineCollectionTag} />
                    )}

                    <LiveRangeTracker
                        ref={liveRangeTrackerRef}
                        liveData={analysisContext?.liveData as Record<string, any> | null}
                        sessionMode={sessionMode}
                        sessionIntelligence={analysisContext?.sessionIntelligence}
                        sendObservation={sendLiveRangeTrackerObservation}
                        resolveLabel={getLabelName}
                        onStateChange={handleLiveRangeTrackerStateChange}
                    />

                    {procedurePlan && (
                        <ProcedurePlanDisplay plan={procedurePlan} onClear={clearProcedurePlan} />
                    )}

                    <div className="ai-chat__msgs" ref={messagesScrollRef} onScroll={handleMessagesScroll}>
                        {messages.map((message) => (
                            <AiMessageDisplay
                                key={message.id}
                                message={message}
                                debugMode={debugMode}
                                assistantAvatarLabel={activeAgentSession ? 'LA' : 'AI'}
                                assistantWhoLabel={activeAgentSession ? getAgentDisplayName(activeAgentSession.agentMode).toUpperCase() : 'ACLA'}
                            />
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                </section>
            </div>

            {/* Input row */}
            <div className="ai-chat__input-row">
                <input
                    className="ai-chat__input"
                    placeholder={
                        activeAgentSession
                            ? `Talk to ${getAgentDisplayName(activeAgentSession.agentMode)}.`
                            : voiceActive
                            ? 'Type a message to the engineer…'
                            : sessionMode === 'recorded'
                                ? 'Ask about this recording.'
                                : sessionMode === 'user_summary'
                                    ? 'Ask about your summary.'
                                    : 'Ask about the live session.'
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
