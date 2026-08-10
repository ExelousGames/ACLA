import React, { useState, useRef, useEffect, useContext, useMemo, useCallback } from 'react';
import './ai-chat.css';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { useUserSummary } from 'contexts/UserSummaryContext';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
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
import { useVoiceConversation, type ToolResultFrame, type VoiceEvent } from './use-voice-conversation';
import { AiMapDisplayPayload } from './AiMapToolDisplay';
import AiMessageDisplay, { type AiChatDisplayMessage } from './AiMessageDisplay';
import ProcedurePlanDisplay from './ProcedurePlanDisplay';
import {
    advanceProcedurePlan,
    buildProcedurePlan,
    isProcedurePlanClearEvent,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
    type ProcedurePlan,
    type ProcedurePlanRequest,
} from './ai-chat-plan';
import { LiveRangeTodoListDisplay } from 'views/live-session/LiveRangeTodoList';
import type { JsonValue, LiveRangeTodoEventCallbackContext } from 'views/live-session/live-range-todo-list-types';
import type {
    BaselineCollectionHandle,
    BaselineCollectionTag,
} from 'views/live-session/BaselineCollection';
import { createLiveRangeTodoAiAdapter } from './live-range-todo-ai-adapter';
import {
    getToolEnvelopeError,
    getToolEnvelopeUiOutput,
    isToolOutputEnvelope,
    type ToolOutputEnvelope,
} from './ai-tool-base';
import { isLiveSessionAiAvailable, RecordingState } from 'views/lap-analysis/recording-state';
import {
    resolveAssistantRecordedSessionId,
    resolveRegisteredAssistantIdentity,
} from 'views/lap-analysis/assistant-session-mode';
import type { AssistantActiveScreen } from 'views/lap-analysis/assistant-session-mode';
import {
    AI_TOOL_COMPONENT_NAMES,
    NamedAiToolComponentHandle,
    useAiToolComponentRefs,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    overlayDisplayClient,
    overlaySessionClient,
} from 'views/floating-chat/overlay-display-client';
import type {
    OverlayDisplayType,
    OverlayPresentationSession,
    OverlayShellMetadata,
    OverlaySnapshotByType,
    OverlayUpsertOptions,
} from 'views/floating-chat/overlay-display-types';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import {
    asRecord,
    buildPracticeTrackSummaryViews,
} from 'views/user-summary/user-summary-model';

type AiChatSessionMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';
type AiChatJsonPrimitive = string | number | boolean | null;
type AiChatJsonValue =
    | AiChatJsonPrimitive
    | AiChatJsonValue[]
    | { [key: string]: AiChatJsonValue };

const toAiChatJsonValue = (value: unknown): AiChatJsonValue => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (Array.isArray(value)) return value.map(toAiChatJsonValue);
    if (value && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value).flatMap(([key, item]) => (
                typeof item === 'undefined' || typeof item === 'function' || typeof item === 'symbol'
                    ? []
                    : [[key, toAiChatJsonValue(item)]]
            )),
        );
    }
    return null;
};

const toAiChatJsonRecord = (
    value: Record<string, unknown>,
): Record<string, AiChatJsonValue> => toAiChatJsonValue(value) as Record<string, AiChatJsonValue>;

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
    name: string;
    activeScreen: AssistantActiveScreen;
}

export interface AiChatHandle extends NamedAiToolComponentHandle {
    getSessionMode(): AiChatSessionMode;
    getRecordingState(): RecordingState | null;
    startAgentSession(agentMode: AgentSessionMode, args?: Record<string, any>): AgentSessionStartResult;
    stopAgentSession(agentSessionId?: string | null): AgentSessionStopResult;
    startTrackGuide(): void;
    setTrackGuideEnabled(enabled: boolean): void;
    setLivePerformanceAnalystEnabled(enabled: boolean): void;
    advanceProcedurePlanStep(reason?: string): any;
    getProcedurePlan(): ProcedurePlan | null;
    clearProcedurePlan(): void;
    setProcedurePlan(plan: ProcedurePlan | null): void;
    setAgentTagActive(tag: string, active: boolean): void;
    getOpportunityTelemetryRows(): Record<string, any>[];
    getOpportunityAgentState(): OpportunityAgentState;
    getLivePerformanceAnalystState(): LivePerformanceAnalystState;
    getLabelName(labelId: string): string | undefined;
    getCategoryLabels(category: string): string[];
    getCircuitMapById(id: string): ReturnType<ReturnType<typeof useCircuitMaps>['getCircuitMapById']>;
    getCircuitMapByTrack: ReturnType<typeof useCircuitMaps>['getCircuitMapByTrack'];
    displayMap(display: AiMapDisplayPayload): void;
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

const getContextDescription = (sessionMode: AiChatSessionMode): string => {
    if (sessionMode === 'front_desk') {
        return 'Front desk assistant for general navigation, onboarding, and high-level help before a session is selected.';
    }
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
    const uiOutput = getToolEnvelopeUiOutput(envelope);
    const payload = isRecord(uiOutput) ? uiOutput : {};
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

const AiChat: React.FC<AiChatProps> = ({
    name,
    activeScreen,
}) => {
    const {
        sessionId,
        sessionMode,
        title,
    } = resolveRegisteredAssistantIdentity(activeScreen);
    const { directory: componentRefs, revision: componentRefRevision } = useAiToolComponentRefs();
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
    const [baselineCollectionTag, setBaselineCollectionTag] = useState<BaselineCollectionTag | null>(null);
    const [procedurePlan, setProcedurePlanState] = useState<ProcedurePlan | null>(null);

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');
    const [floatingChatOpen, setFloatingChatOpen] = useState(false);
    const [overlayPresentationId, setOverlayPresentationId] = useState<string | null>(null);
    const [overlayAgentTags, setOverlayAgentTags] = useState<string[]>([]);
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
    const recordedAnalysisContext = useContext(AnalysisContext);
    const liveSession = useContext(LiveSessionContext);
    const liveSessionEnded = sessionMode === 'live'
        && liveSession.recordingState === RecordingState.UPLOAD_READY;
    const analysisContext = useMemo(() => ({
        ...recordedAnalysisContext,
        mapSelected: sessionMode === 'recorded' ? recordedAnalysisContext.mapSelected : null,
        sessionSelected: sessionMode === 'recorded' ? recordedAnalysisContext.sessionSelected : null,
        liveData: liveSession.currentTelemetry,
        TelemetryDataLiveStatus: liveSession.telemetryStatus,
        recordingState: liveSession.recordingState,
        recordingMetadata: liveSession.recordingMetadata,
        recordedSessionDataFilePath: liveSession.recordingFileKey,
        recordedTelemetryDataCount: liveSession.recordedSampleCount,
        recordedSessioStaticsData: liveSession.staticData,
        sessionIntelligence: liveSession.sessionIntelligence,
    }), [liveSession, recordedAnalysisContext, sessionMode]);
    const {
        userSummary,
        userSummaryLoading,
        userSummaryError,
    } = useUserSummary();
    const {
        getLabelName,
        getCategoryLabels,
        loading: labelsLoading,
        error: labelsError,
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
        lastToolStatusKey: null,
        lastToolStatusAt: 0,
        lastSpokenAt: 0,
    });
    const trackGuideLastPosRef = useRef<number | undefined>(undefined);
    const trackGuideTriggeredRef = useRef<Set<string>>(new Set());
    const trackGuideRunTokenRef = useRef(0);
    const activeAgentTagsRef = useRef<string[]>([]);
    const activeAgentSessionRef = useRef<AgentSessionInfo | null>(null);
    const agentVoiceStopRef = useRef<() => void>(() => undefined);
    const mainVoiceStopRef = useRef<() => void>(() => undefined);
    const activeVoiceToolResultRef = useRef<(frame: ToolResultFrame) => boolean>(
        () => false,
    );
    const activeVoiceToolStatusRef = useRef<(data: Record<string, unknown>) => boolean>(
        () => false,
    );
    const agentAutoStartSessionIdRef = useRef<string | null>(null);
    const endedAiShutdownAppliedRef = useRef(false);
    const overlayPresentationRef = useRef<OverlayPresentationSession | null>(null);
    const overlayPresentationsByAiSessionRef = useRef<Map<string, string>>(new Map());
    const overlayAiSessionByVoiceSessionRef = useRef<Map<string, string>>(new Map());
    const overlayStartByVoiceSessionRef = useRef<Map<string, Promise<OverlayPresentationSession | null>>>(new Map());
    const voiceSessionSeenActiveRef = useRef(false);
    const procedurePlanRef = useRef<ProcedurePlan | null>(null);
    const procedurePlanOptedOutRef = useRef(false);
    const planToolRunsRef = useRef<Set<string>>(new Set());
    const lastBroadcastedLiveRangeTodoListKeyRef = useRef<string | null>(null);

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

    const upsertOverlayDisplay = useCallback(<T extends OverlayDisplayType,>(
        type: T,
        snapshot: OverlaySnapshotByType[T],
        metadata: OverlayShellMetadata = {},
        options: Omit<OverlayUpsertOptions, 'metadata'> = {},
        presentationId = overlayPresentationRef.current?.presentationId,
    ) => {
        if (!presentationId) return;
        void overlayDisplayClient.forPresentation(presentationId).upsert(type, snapshot, {
            ...options,
            metadata: {
                ...metadata,
                name: metadata.name ?? (
                    activeAgentSessionRef.current
                        ? getAgentDisplayName(activeAgentSessionRef.current.agentMode)
                        : undefined
                ),
                agentTags: metadata.agentTags ?? activeAgentTagsRef.current,
            },
        }).catch(() => undefined);
    }, []);

    const presentAssistantOverlayMessage = useCallback((
        text: string,
        options: { emotion?: Emotion | null; tags?: string[]; name?: string } = {},
        presentationId?: string,
    ) => {
        try {
            const pillText = text
                .replace(/\*\*(.*?)\*\*/g, '$1')
                .replace(/\*(.*?)\*/g, '$1')
                .replace(/`(.*?)`/g, '$1')
                .replace(/\s+/g, ' ')
                .trim()
                .slice(0, 280);
            if (pillText || options.tags !== undefined) {
                if (pillText) {
                    upsertOverlayDisplay('ai_message', { text: pillText }, {
                        name: options.name,
                        emotion: options.emotion ?? undefined,
                        agentTags: options.tags ?? activeAgentTagsRef.current,
                    }, {}, presentationId);
                }
            }
        } catch { /* ignore unavailable overlay failures */ }
    }, [upsertOverlayDisplay]);

    const displayMapInChat = useCallback((display: AiMapDisplayPayload) => {
        const fallbackText = display.status === 'unavailable'
            ? 'Map is not available'
            : display.note || display.title || display.map?.circuit_name || 'Map';
        upsertOverlayDisplay('map', display);
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
    }, [generateUniqueId, setMessages, upsertOverlayDisplay]);

    const setAgentTag = useCallback((tag: string, active: boolean) => {
        const current = activeAgentTagsRef.current;
        const next = active
            ? Array.from(new Set([...current, tag]))
            : current.filter((item) => item !== tag);
        if (next.length === current.length && next.every((item, index) => item === current[index])) {
            return;
        }
        activeAgentTagsRef.current = next;
        setOverlayAgentTags(next);
    }, []);

    useEffect(() => {
        activeAgentTagsRef.current = [];
        setOverlayAgentTags([]);
    }, []);

    const beginOverlaySession = useCallback(async (
        aiSessionId: string,
        target: 'main' | 'agent',
        agentMode?: AgentSessionMode,
    ): Promise<OverlayPresentationSession | null> => {
        if (!overlaySessionClient.available()) return null;
        const mode = target === 'agent' ? 'agent' : sessionMode;
        const modeTag = mode === 'front_desk'
            ? 'Front Desk'
            : mode === 'user_summary'
                ? 'User Summary'
                : mode.charAt(0).toUpperCase() + mode.slice(1);
        const agentName = target === 'agent' ? getAgentDisplayName(agentMode) : undefined;
        const presentation = await overlaySessionClient.create({
            aiSessionId,
            mode,
            displayIdentity: {
                name: agentName || 'Kestrel',
                emotion: 'idle',
                agentTags: Array.from(new Set([
                    modeTag,
                    ...(agentName ? [agentName] : []),
                    ...activeAgentTagsRef.current,
                ])),
            },
        });
        overlayPresentationsByAiSessionRef.current.set(aiSessionId, presentation.presentationId);
        if (overlaySessionClient.current()?.presentationId === presentation.presentationId) {
            overlayPresentationRef.current = presentation;
            setOverlayPresentationId(presentation.presentationId);
        }
        return presentation;
    }, [sessionMode]);

    const endOverlaySession = useCallback(async (aiSessionId?: string | null) => {
        const presentationId = aiSessionId
            ? overlayPresentationsByAiSessionRef.current.get(aiSessionId)
            : overlayPresentationRef.current?.presentationId;
        if (!presentationId) return;
        await overlaySessionClient.destroy(presentationId);
        if (aiSessionId) overlayPresentationsByAiSessionRef.current.delete(aiSessionId);
        if (overlayPresentationRef.current?.presentationId === presentationId) {
            overlayPresentationRef.current = null;
            setOverlayPresentationId(null);
        }
    }, []);

    useEffect(() => () => {
        const presentationId = overlayPresentationRef.current?.presentationId;
        overlayPresentationRef.current = null;
        if (presentationId) void overlaySessionClient.destroy(presentationId).catch(() => undefined);
    }, []);

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
            return;
        }
    }, []);

    useEffect(() => {
        if (!overlayPresentationId) return;
        if (procedurePlan) {
            upsertOverlayDisplay('procedure_plan', procedurePlan, {}, {}, overlayPresentationId);
            return;
        }
        void overlayDisplayClient.forPresentation(overlayPresentationId).exit(
            { type: 'procedure_plan' },
            'producer_exit',
        ).catch(() => undefined);
    }, [overlayAgentTags, overlayPresentationId, procedurePlan, upsertOverlayDisplay]);

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
        const eventAiSessionId = event.clientSessionId ?? (
            target === 'agent'
                ? activeAgentSessionRef.current?.clientSessionId
                : mainClientSessionIdRef.current
        );
        const presentationId = eventAiSessionId
            ? overlayPresentationsByAiSessionRef.current.get(eventAiSessionId)
            : undefined;
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
            // Send a complete presentation snapshot to the Electron overlay.
            presentAssistantOverlayMessage(cleanText, {
                emotion,
                name: target === 'agent' ? getAgentDisplayName(activeAgentSessionRef.current?.agentMode) : undefined,
            }, presentationId);
            return;
        }
        if (event.kind === 'tool_status') {
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
        if (event.kind === 'tool_call') {
            if (event.runId) {
                upsertOverlayDisplay('tool_status', {
                    runId: event.runId,
                    name: event.name,
                    title: event.title,
                    status: event.status,
                    ok: event.ok,
                    error: event.error ?? null,
                    result: event.result,
                }, {}, { key: event.runId }, presentationId);
            }
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
        presentAssistantOverlayMessage,
        clearProcedurePlan,
        generateUniqueId,
        optOutProcedurePlan,
        setProcedurePlan,
        upsertOverlayDisplay,
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
            kind: 'tool_call',
            runId: envelope.run_id,
            name: envelope.tool_name,
            title: envelope.message || 'Collect live baseline',
            status: envelope.final ? 'completed' : 'started',
            result: getBaselineToolEventResult(envelope),
            ok: !envelopeError,
            error: envelopeError,
        }, activeAgentSessionRef.current ? 'agent' : 'main');

        activeVoiceToolResultRef.current({
            id: envelope.run_id,
            name: envelope.tool_name,
            result: envelope.output,
        });

        const plan = procedurePlanRef.current;
        const request = plan?.requests[plan.currentStep];
        if (
            envelope.final
            && !envelopeError
            && request?.type === 'tool_call'
            && request.name === envelope.tool_name
            && request.status !== 'complete'
        ) {
            advanceProcedurePlanStep(envelope.message || `tool ${envelope.tool_name} completed`);
        }
    }, [advanceProcedurePlanStep, handleSessionVoiceEvent]);

    useEffect(() => {
        const baseline = componentRefs
            .findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)
            ?.current;
        if (!baseline) {
            setBaselineCollectionTag(null);
            return undefined;
        }

        setBaselineCollectionTag(baseline.getTag());
        return baseline.subscribeToolOutput((envelope) => {
            setBaselineCollectionTag(baseline.getTag());
            handleBaselineToolOutput(envelope);
        });
    }, [componentRefRevision, componentRefs, handleBaselineToolOutput]);

    useEffect(() => {
        const baseline = componentRefs
            .findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)
            ?.current;
        setBaselineCollectionTag(baseline?.getTag() ?? null);
    }, [analysisContext?.liveData, componentRefRevision, componentRefs]);

    useEffect(() => {
        if (!overlayPresentationId) return;
        if (!baselineCollectionTag) {
            void overlayDisplayClient.forPresentation(overlayPresentationId).exit(
                { type: 'baseline_progress' },
                'producer_exit',
            ).catch(() => undefined);
            return;
        }
        upsertOverlayDisplay('baseline_progress', baselineCollectionTag, {}, {}, overlayPresentationId);
    }, [baselineCollectionTag, overlayAgentTags, overlayPresentationId, upsertOverlayDisplay]);

    useEffect(() => {
        if (!overlayPresentationId) return;
        const todoList = liveSession.liveRangeTodoListSnapshot;
        if (!todoList || todoList.events.length === 0) {
            lastBroadcastedLiveRangeTodoListKeyRef.current = null;
            void overlayDisplayClient.forPresentation(overlayPresentationId).exit(
                { type: 'live_range_todo' },
                'producer_exit',
            ).catch(() => undefined);
            return;
        }
        const lifecycleKey = JSON.stringify({ overlayPresentationId, agentTags: overlayAgentTags, todoList });
        if (lifecycleKey === lastBroadcastedLiveRangeTodoListKeyRef.current) return;
        lastBroadcastedLiveRangeTodoListKeyRef.current = lifecycleKey;
        upsertOverlayDisplay('live_range_todo', todoList, {}, {}, overlayPresentationId);
    }, [liveSession.liveRangeTodoListSnapshot, overlayAgentTags, overlayPresentationId, upsertOverlayDisplay]);

    const startTrackGuide = useCallback(() => {
        trackGuideRunTokenRef.current += 1;
        setTrackGuideEnabled(true);
    }, []);

    const resolvedSessionId = resolveAssistantRecordedSessionId(
        sessionMode,
        sessionId
            || (analysisContext?.sessionSelected as Record<string, any> | null)?.SessionId,
    );

    const activeScreenContext = useMemo<Record<string, any>>(() => {
        if (sessionMode === 'live') {
            const rawSnapshot = typeof liveSession.sessionIntelligence?.getLiveSessionSnapshot === 'function'
                ? liveSession.sessionIntelligence.getLiveSessionSnapshot()
                : {};
            const snapshot = isRecord(rawSnapshot) ? rawSnapshot : {};
            const track = snapshot.track
                || liveSession.recordingMetadata?.mapName
                || liveSession.staticData.track
                || null;
            const car = snapshot.car
                || liveSession.recordingMetadata?.carName
                || liveSession.staticData.car_model
                || null;

            return toAiChatJsonRecord({
                screen_kind: 'live_session',
                simulator: liveSession.sessionGame,
                recording_state: liveSession.recordingState,
                recording_name: liveSession.recordingMetadata?.sessionName || null,
                track,
                car,
                current_lap: snapshot.current_lap || null,
                completed_laps: snapshot.completed_laps || 0,
                normalized_position: snapshot.normalized_position || 0,
                sample_count: snapshot.sample_count || liveSession.recordedSampleCount,
                latest_telemetry_present: Object.keys(liveSession.currentTelemetry).length > 0,
                latest_telemetry_key_count: Object.keys(liveSession.currentTelemetry).length,
                telemetry_status: liveSession.telemetryStatus,
                session_intelligence: snapshot,
                live_todo: liveSession.liveRangeTodoListSnapshot,
                controls: {
                    live_todo_available: Boolean(liveSession.liveRangeTodoListHandle),
                    recorder_available: Boolean(liveSession.recorderControl),
                },
                visualization_capabilities: {
                    telemetry: true,
                    events: true,
                    sections: true,
                    live_todo: true,
                },
            });
        }

        if (sessionMode === 'recorded') {
            const selectedSession = recordedAnalysisContext.sessionSelected;

            return toAiChatJsonRecord({
                screen_kind: 'recorded_session',
                active_analysis_area: recordedAnalysisContext.activeTab,
                selected_map_id: recordedAnalysisContext.mapSelected || selectedSession?.map || null,
                selected_session: {
                    id: selectedSession?.SessionId || null,
                    name: selectedSession?.session_name || null,
                    map: selectedSession?.map || recordedAnalysisContext.mapSelected || null,
                    car: selectedSession?.car || null,
                },
                recorded_session: {
                    ai_analysis: {
                        status: recordedAnalysisContext.recordedAiAnalysis.status,
                        message: recordedAnalysisContext.recordedAiAnalysis.message || null,
                        session_id: recordedAnalysisContext.recordedAiAnalysis.sessionId,
                        samples_analyzed: recordedAnalysisContext.recordedAiAnalysis.result?.samples_analyzed || 0,
                        result_ready: Boolean(recordedAnalysisContext.recordedAiAnalysis.result),
                    },
                    playback: recordedAnalysisContext.recordedPlaybackSummary,
                },
                analysis_actions: {
                    run_ai_analysis: true,
                    read_ai_analysis: true,
                    read_recorded_context: true,
                },
                visualization_controls: {
                    active: recordedAnalysisContext.activeVisualizations.map(({ id, type }) => ({ id, type })),
                },
            });
        }

        if (sessionMode === 'user_summary') {
            const trackSummaries = buildPracticeTrackSummaryViews(
                asRecord(userSummary),
                getLabelName,
                getCategoryLabels,
            );
            const summaryState = userSummaryLoading || labelsLoading
                ? 'loading'
                : userSummaryError || labelsError
                    ? 'error'
                    : trackSummaries.length > 0
                        ? 'ready'
                        : 'empty';

            return toAiChatJsonRecord({
                screen_kind: 'user_summary',
                summary_scope: 'Most recent 10 practice sessions by track section.',
                summary_state: summaryState,
                loading: userSummaryLoading || labelsLoading,
                error: userSummaryError || labelsError || null,
                track_count: trackSummaries.length,
                normalized_summary: trackSummaries,
                query_capabilities: {
                    map_lookup: true,
                    available_maps: true,
                    search: true,
                },
            });
        }

        const isAnalysisScreen = activeScreen.componentName === AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS;
        return toAiChatJsonRecord({
            screen_kind: 'front_desk',
            active_analysis_area: isAnalysisScreen ? recordedAnalysisContext.activeTab : 'mapLists',
            selected_map_id: isAnalysisScreen ? recordedAnalysisContext.mapSelected : null,
            assistance_scope: 'General navigation, onboarding, map selection, and session selection.',
            capabilities: {
                screen_tools: false,
                general_assistance: true,
            },
        });
    }, [
        activeScreen.componentName,
        getCategoryLabels,
        getLabelName,
        labelsError,
        labelsLoading,
        liveSession,
        recordedAnalysisContext,
        sessionMode,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const aiSessionContext = useMemo(() => {
        const effectiveMode = sessionMode;
        const selectedSession = isRecord(activeScreenContext.selected_session)
            ? activeScreenContext.selected_session
            : null;
        const liveSnapshot = isRecord(activeScreenContext.session_intelligence)
            ? activeScreenContext.session_intelligence
            : null;
        const recordedSession = isRecord(activeScreenContext.recorded_session)
            ? activeScreenContext.recorded_session
            : {};
        const recordedAiAnalysis = isRecord(recordedSession.ai_analysis)
            ? recordedSession.ai_analysis
            : {};
        const recordedPlaybackSummary = isRecord(recordedSession.playback)
            ? recordedSession.playback
            : {};
        const summaryTrackCount = Number(activeScreenContext.track_count) || 0;
        const summaryState = String(activeScreenContext.summary_state || 'empty');
        const liveRecordingActive = activeScreenContext.recording_state === RecordingState.RECORDING;
        const latestTelemetryPresent = Boolean(activeScreenContext.latest_telemetry_present);
        const latestTelemetryKeyCount = Number(activeScreenContext.latest_telemetry_key_count) || 0;
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
            context_kind: effectiveMode,
            context_description: getContextDescription(effectiveMode),
            session_mode: effectiveMode,
            session_id: resolvedSessionId || null,
            recording_state: activeScreenContext.recording_state || null,
            live_recording_active: liveRecordingActive,
            active_tab: activeScreenContext.active_analysis_area || null,
            selected_map_id: activeScreenContext.selected_map_id || selectedSession?.map || null,
            active_screen: {
                screen_id: activeScreen.componentName || 'front-desk',
                label: activeScreen.label,
                assistant_mode: effectiveMode,
                recorded_session_id: resolvedSessionId || null,
                context: activeScreenContext,
            },
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
            track: activeScreenContext.track || selectedSession?.map || null,
            car: activeScreenContext.car || selectedSession?.car || null,
            current_lap: activeScreenContext.current_lap ?? liveSnapshot?.current_lap ?? null,
            normalized_position: activeScreenContext.normalized_position ?? liveSnapshot?.normalized_position ?? null,
            completed_laps: activeScreenContext.completed_laps ?? liveSnapshot?.completed_laps ?? null,
            sample_count: activeScreenContext.sample_count ?? liveSnapshot?.sample_count ?? 0,
            capabilities: {
                live_session: liveRecordingActive,
                recorded_session: effectiveMode === 'recorded',
                front_desk: effectiveMode === 'front_desk',
                ...(effectiveMode === 'user_summary' ? { user_summary: summaryState === 'ready' } : {}),
                active_screen_tools: [],
            },
            selected_session: selectedSession
                ? {
                    id: selectedSession.id || null,
                    name: selectedSession.name || null,
                    map: selectedSession.map || null,
                    car: selectedSession.car || null,
                }
                : null,
            telemetry: {
                live_available: liveRecordingActive,
                latest_sample_present: latestTelemetryPresent,
                latest_sample_key_count: latestTelemetryKeyCount,
                live_status: activeScreenContext.telemetry_status ?? null,
                recorded_file_loaded: Number(recordedPlaybackSummary.sampleCount) > 0,
                recorded_sample_count: recordedPlaybackSummary.sampleCount ?? 0,
            },
            recorded_session: {
                ai_analysis: {
                    status: recordedAiAnalysis.status || 'idle',
                    message: recordedAiAnalysis.message || null,
                    session_id: recordedAiAnalysis.session_id || null,
                    samples_analyzed: recordedAiAnalysis.samples_analyzed ?? 0,
                    result_ready: Boolean(recordedAiAnalysis.result_ready),
                },
                playback: {
                    session_id: recordedPlaybackSummary.sessionId || null,
                    sample_count: recordedPlaybackSummary.sampleCount ?? 0,
                    duration_seconds: recordedPlaybackSummary.durationSeconds ?? 0,
                    playback_index: recordedPlaybackSummary.playbackIndex ?? 0,
                    playback_time_seconds: recordedPlaybackSummary.playbackTimeSeconds ?? 0,
                    active_segment: recordedPlaybackSummary.activeSegment ?? null,
                },
            },
        };

        if (effectiveMode !== 'user_summary') {
            return context;
        }

        return {
            ...context,
            user_summary: {
                loaded: summaryState === 'ready',
                loading: summaryState === 'loading',
                error: activeScreenContext.error || null,
                track_count: summaryTrackCount,
            },
        };
    }, [
        activeScreen.componentName,
        activeScreen.label,
        activeScreenContext,
        activeAgentSession,
        livePerformanceAnalystEnabled,
        procedurePlan,
        resolvedSessionId,
        sessionMode,
        TrackGuideEnabled,
    ]);

    const inactiveAgentToolHandlers = useMemo(() => ({}), []);
    const getProcedurePlan = useCallback(() => procedurePlanRef.current, []);
    const getOpportunityTelemetryRows = useCallback(() => opportunityForecastRowsRef.current, []);
    const notifyAiForLiveRangeEvent = useCallback(async ({
        event,
        data,
        lap,
        eta_seconds: etaSeconds,
        sessionIntelligence,
        signal,
    }: LiveRangeTodoEventCallbackContext) => {
        if (signal.aborted) throw new Error('Live range to-do notification was aborted.');
        const notificationOptions = isRecord(data) ? data : {};
        const rangeRequest = isRecord(notificationOptions.telemetry_range_summary)
            ? notificationOptions.telemetry_range_summary
            : isRecord(notificationOptions.telemetry_range)
                ? notificationOptions.telemetry_range
                : null;
        const includeRangeSummary = Boolean(rangeRequest)
            || notificationOptions.include_telemetry_range_summary === true;
        let telemetryRangeSummary: Record<string, JsonValue> | undefined;

        if (includeRangeSummary) {
            const startPosition = Number(
                rangeRequest?.start_position ?? notificationOptions.range_start_position,
            );
            const endPosition = Number(
                rangeRequest?.end_position ?? notificationOptions.range_end_position,
            );
            if (
                !Number.isFinite(startPosition)
                || startPosition < 0
                || startPosition > 1
                || !Number.isFinite(endPosition)
                || endPosition < 0
                || endPosition > 1
            ) {
                throw new Error('AI notification telemetry range summaries require start_position and end_position from 0 through 1.');
            }
            const requestedLap = rangeRequest?.lap;
            const rangeWindow = sessionIntelligence.getTelemetryWindowForNormalizedRange({
                start_position: startPosition,
                end_position: endPosition,
                lap: requestedLap === 'current' || requestedLap === 'last' || typeof requestedLap === 'number'
                    ? requestedLap
                    : lap,
            });
            telemetryRangeSummary = {
                status: rangeWindow.status,
                start_position: rangeWindow.startPosition,
                end_position: rangeWindow.endPosition,
                lap: rangeWindow.lap,
                start_sample_idx: rangeWindow.startSampleIdx ?? null,
                end_sample_idx: rangeWindow.endSampleIdx ?? null,
                telemetry_row_count: rangeWindow.rows.length,
            };
        }

        const payload = {
            source: 'live_range_todo_list',
            event: typeof notificationOptions.event === 'string' && notificationOptions.event.trim()
                ? notificationOptions.event.trim()
                : 'live_range_todo_event_due',
            event_id: event.id,
            content: event.content,
            normalized_position: event.normalized_position,
            lead_time_seconds: event.lead_time_seconds,
            eta_seconds: etaSeconds,
            lap: lap ?? null,
            ...(telemetryRangeSummary ? { telemetry_range_summary: telemetryRangeSummary } : {}),
        };
        const sent = activeVoiceToolStatusRef.current(payload);
        if (!sent) throw new Error('AI session is not connected; the due notification could not be sent.');
    }, []);
    const liveRangeTodoAiAdapter = useMemo(() => createLiveRangeTodoAiAdapter(
        liveSession.liveRangeTodoListHandle,
        notifyAiForLiveRangeEvent,
    ), [liveSession.liveRangeTodoListHandle, notifyAiForLiveRangeEvent]);
    const setLiveRangeTodoList = useCallback((args: Record<string, unknown>) => (
        liveRangeTodoAiAdapter.set(args)
    ), [liveRangeTodoAiAdapter]);
    const updateLiveRangeTodoList = useCallback((args: Record<string, unknown>) => (
        liveRangeTodoAiAdapter.update(args)
    ), [liveRangeTodoAiAdapter]);
    const getLiveRangeTodoList = useCallback(() => (
        liveRangeTodoAiAdapter.get()
    ), [liveRangeTodoAiAdapter]);

    const resetLivePerformanceAnalystRuntime = useCallback(() => {
        const analystAgent = livePerformanceAnalystStateRef.current;
        if (analystAgent.intervalId) {
            clearInterval(analystAgent.intervalId);
        }
        analystAgent.intervalId = null;
        analystAgent.inFlight = false;
        analystAgent.enabled = false;
        analystAgent.lastToolStatusKey = null;
        analystAgent.lastToolStatusAt = 0;
        analystAgent.lastSpokenAt = 0;
        analystAgent.analysisSessionId = null;
        analysisContext?.sessionIntelligence?.clearFocusSection?.();
        setLivePerformanceAnalystAgentEnabled(false);
        procedurePlanOptedOutRef.current = false;
        clearProcedurePlan();
    }, [analysisContext?.sessionIntelligence, clearProcedurePlan, setLivePerformanceAnalystAgentEnabled]);

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
        if (sessionMode !== 'live' || !isLiveSessionAiAvailable(analysisContext?.recordingState)) {
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
                const overlayAiSessionId = createClientSessionId(`overlay-agent-${agentMode}`);
                overlayAiSessionByVoiceSessionRef.current.set(existing.clientSessionId, overlayAiSessionId);
                const overlayStart = beginOverlaySession(overlayAiSessionId, 'agent', agentMode);
                overlayStartByVoiceSessionRef.current.set(existing.clientSessionId, overlayStart);
                void overlayStart.catch(() => undefined);
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
        const overlayAiSessionId = createClientSessionId(`overlay-agent-${agentMode}`);
        overlayAiSessionByVoiceSessionRef.current.set(clientSessionId, overlayAiSessionId);
        const overlayStart = beginOverlaySession(overlayAiSessionId, 'agent', agentMode);
        overlayStartByVoiceSessionRef.current.set(clientSessionId, overlayStart);
        void overlayStart.catch(() => undefined);
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

        return {
            status: 'started',
            conversation_role: 'agent',
            agent_mode: agentMode,
            agent_session_id: clientSessionId,
            parent_client_session_id: mainClientSessionIdRef.current,
        };
    }, [
        resetAgentRuntimes,
        analysisContext?.recordingState,
        beginOverlaySession,
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
        const stopAgentVoice = agentVoiceStopRef.current;
        window.setTimeout(() => stopAgentVoice?.(), 0);
        resetAgentRuntimes();
        setActiveAgentSession(null);
        activeAgentSessionRef.current = null;
        agentAutoStartSessionIdRef.current = null;
        setAgentTag(getAgentDisplayName(current.agentMode), false);
        const overlayAiSessionId = overlayAiSessionByVoiceSessionRef.current.get(current.clientSessionId);
        overlayAiSessionByVoiceSessionRef.current.delete(current.clientSessionId);
        overlayStartByVoiceSessionRef.current.delete(current.clientSessionId);
        void endOverlaySession(overlayAiSessionId).catch(() => undefined);

        return {
            status: 'stopped',
            conversation_role: 'agent',
            agent_mode: current.agentMode,
            agent_session_id: current.clientSessionId,
        };
    }, [endOverlaySession, resetAgentRuntimes, setAgentTag]);

    const aiChatHandle = useMemo<AiChatHandle>(() => ({
        getComponentName: () => name,
        getSessionMode: () => sessionMode,
        getRecordingState: () => analysisContext?.recordingState ?? null,
        startAgentSession,
        stopAgentSession,
        startTrackGuide,
        setTrackGuideEnabled: setTrackGuideAgentEnabled,
        setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
        advanceProcedurePlanStep,
        getProcedurePlan,
        clearProcedurePlan,
        setProcedurePlan,
        setAgentTagActive: setAgentTag,
        getOpportunityTelemetryRows,
        getOpportunityAgentState: () => opportunityAgentStateRef.current,
        getLivePerformanceAnalystState: () => livePerformanceAnalystStateRef.current,
        getLabelName,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        displayMap: displayMapInChat,
    }), [
        advanceProcedurePlanStep,
        analysisContext?.recordingState,
        clearProcedurePlan,
        displayMapInChat,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLabelName,
        getOpportunityTelemetryRows,
        getProcedurePlan,
        name,
        sessionMode,
        setAgentTag,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startAgentSession,
        startTrackGuide,
        stopAgentSession,
    ]);
    useRegisterAiToolComponentRef(name, aiChatHandle);

    const toolHandlers = useMemo(() => createAiCommandRegistry({
        componentRefs,
        sessionId: resolvedSessionId,
        sessionMode,
    }), [componentRefs, resolvedSessionId, sessionMode]);

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
        componentRefs,
        sessionId: resolvedSessionId,
        sessionMode,
    }), [componentRefs, resolvedSessionId, sessionMode]);

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
    const sendAgentVoiceToolStatus = agentVoiceConversation.sendToolStatus;
    const stopMainVoiceConversation = voiceConversation.stop;
    const stopAgentVoiceConversation = agentVoiceConversation.stop;

    useEffect(() => {
        mainVoiceStopRef.current = voiceConversation.stop;
    }, [voiceConversation.stop]);

    useEffect(() => {
        agentVoiceStopRef.current = agentVoiceConversation.stop;
    }, [agentVoiceConversation.stop]);

    useEffect(() => {
        if (!liveSessionEnded) {
            endedAiShutdownAppliedRef.current = false;
            return;
        }
        if (endedAiShutdownAppliedRef.current) return;
        endedAiShutdownAppliedRef.current = true;

        stopMainVoiceConversation();
        stopAgentVoiceConversation();
        const activeVoiceSessionId = activeAgentSessionRef.current?.clientSessionId ?? mainClientSessionIdRef.current;
        const overlayAiSessionId = overlayAiSessionByVoiceSessionRef.current.get(activeVoiceSessionId);
        overlayAiSessionByVoiceSessionRef.current.delete(activeVoiceSessionId);
        overlayStartByVoiceSessionRef.current.delete(activeVoiceSessionId);
        void endOverlaySession(overlayAiSessionId).catch(() => undefined);
        resetAgentRuntimes();
        agentAutoStartSessionIdRef.current = null;
        setActiveAgentSession((current) => {
            if (!current) return current;
            const stopped = { ...current, status: 'stopped' as const };
            activeAgentSessionRef.current = stopped;
            return stopped;
        });
    }, [endOverlaySession, liveSessionEnded, resetAgentRuntimes, stopAgentVoiceConversation, stopMainVoiceConversation]);

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
        const overlayStart = overlayStartByVoiceSessionRef.current.get(activeAgentSessionId)
            ?? Promise.resolve(null);
        overlayStart.catch(() => null).then(() => {
            if (activeAgentSessionRef.current?.clientSessionId !== activeAgentSessionId) return;
            return startAgentVoiceConversation(
                overlayAiSessionByVoiceSessionRef.current.get(activeAgentSessionId),
            );
        }).catch((err) => {
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
                componentRefs,
                sessionId: resolvedSessionId,
                sessionMode,
                recordingState: analysisContext?.recordingState,
                conversationRole: 'agent',
                activeAgentSession: current,
                analysisContext,
                sessionIntelligence: analysisContext?.sessionIntelligence,
                opportunityAgentState: opportunityAgentStateRef.current,
                livePerformanceAnalystState: livePerformanceAnalystStateRef.current,
                startTrackGuide,
                setTrackGuideEnabled: setTrackGuideAgentEnabled,
                setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
                advanceProcedurePlanStep,
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
                setLiveRangeTodoList,
                updateLiveRangeTodoList,
                getLiveRangeTodoList,
                displayMap: displayMapInChat,
            }, {}, {
                sendToolStatus: agentVoiceConversation.sendToolStatus,
            });
        }, 0);
    }, [
        activeAgentSession,
        TrackGuideEnabled,
        advanceProcedurePlanStep,
        analysisContext,
        analysisContext?.recordingState,
        clearProcedurePlan,
        componentRefs,
        displayMapInChat,
        agentVoiceConversation.state,
        agentVoiceConversation.sendToolStatus,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLiveRangeTodoList,
        getLabelName,
        getOpportunityTelemetryRows,
        getProcedurePlan,
        resolvedSessionId,
        sessionMode,
        setAgentTag,
        setLiveRangeTodoList,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startTrackGuide,
        stopAgentSession,
        updateLiveRangeTodoList,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    useEffect(() => {
        const sessionIntelligence = analysisContext?.sessionIntelligence;
        if (
            sessionMode !== 'live'
            || !isLiveSessionAiAvailable(analysisContext?.recordingState)
            || !sessionIntelligence
            || !activeAgentSession
        ) return;

        return sessionIntelligence.onLiveAnalystToolStatus((toolStatus) => {
            if (!livePerformanceAnalystStateRef.current.enabled) return;
            sendAgentVoiceToolStatus(toolStatus);
        });
    }, [
        activeAgentSession,
        analysisContext?.recordingState,
        analysisContext?.sessionIntelligence,
        sendAgentVoiceToolStatus,
        sessionMode,
    ]);

    const activeVoiceConversation = activeAgentSession ? agentVoiceConversation : voiceConversation;
    const vState = activeVoiceConversation.state;
    const voiceActive = vState === 'listening' || vState === 'speaking';
    const modelPickerDisabled = voiceActive || vState === 'connecting';
    const micDisabled = activeVoiceConversation.micDisabled;
    const sendActiveVoiceToolStatus = activeVoiceConversation.sendToolStatus;
    const sendActiveVoiceToolResult = activeVoiceConversation.sendToolResult;
    useEffect(() => {
        if (vState === 'connecting' || vState === 'listening' || vState === 'speaking') {
            voiceSessionSeenActiveRef.current = true;
            return;
        }
        if (vState !== 'idle' || !voiceSessionSeenActiveRef.current) return;
        if (activeAgentSession?.status === 'starting') return;
        voiceSessionSeenActiveRef.current = false;
        const activeVoiceSessionId = activeAgentSessionRef.current?.clientSessionId ?? mainClientSessionIdRef.current;
        const overlayAiSessionId = overlayAiSessionByVoiceSessionRef.current.get(activeVoiceSessionId);
        overlayAiSessionByVoiceSessionRef.current.delete(activeVoiceSessionId);
        overlayStartByVoiceSessionRef.current.delete(activeVoiceSessionId);
        void endOverlaySession(overlayAiSessionId).catch(() => undefined);
    }, [activeAgentSession?.status, endOverlaySession, vState]);
    useEffect(() => {
        activeVoiceToolResultRef.current = sendActiveVoiceToolResult;
        activeVoiceToolStatusRef.current = sendActiveVoiceToolStatus;
        return () => {
            if (activeVoiceToolResultRef.current === sendActiveVoiceToolResult) {
                activeVoiceToolResultRef.current = () => false;
            }
            if (activeVoiceToolStatusRef.current === sendActiveVoiceToolStatus) {
                activeVoiceToolStatusRef.current = () => false;
            }
        };
    }, [sendActiveVoiceToolResult, sendActiveVoiceToolStatus]);

    const canOpenFloatingChat = overlaySessionClient.available();

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
                    'running',
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
        analystAgent.lastToolStatusKey = null;
        analystAgent.lastToolStatusAt = 0;
        analystAgent.lastSpokenAt = 0;
        setLivePerformanceAnalystAgentEnabled(false);
        procedurePlanOptedOutRef.current = false;
        clearProcedurePlan();
        setAgentTag('Track Guide', false);
        setAgentTag('Overtake', false);
        setAgentTag('Live Analyst', false);
    }, [clearProcedurePlan, sessionMode, setAgentTag, setLivePerformanceAnalystAgentEnabled, setTrackGuideAgentEnabled, stopAgentSession]);

    const toggleFloatingChat = useCallback(async () => {
        try {
            const enabled = !floatingChatOpen;
            await overlaySessionClient.setEnabled(enabled);
            setFloatingChatOpen(enabled);
        } catch (err) {
            console.warn('Failed to toggle floating chat:', err);
        }
    }, [floatingChatOpen]);

    useEffect(() => {
        const api = (window as any).electronAPI;
        if (!api?.onFloatingChatClosed) return;
        const unsubscribe = api.onFloatingChatClosed(() => {
            overlayPresentationRef.current = null;
            setOverlayPresentationId(null);
            setFloatingChatOpen(false);
        });
        if (api.isOverlayEnabled) {
            api.isOverlayEnabled()
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
            const managerNames = [
                AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
                AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
            ];
            managerNames.forEach((managerName) => {
                const manager = componentRefs.findComponentRef<VisualizationManagerHandle>(managerName)?.current;
                const existingCharts = manager?.getCurrentVisualizations() ?? [];
                existingCharts.forEach(chart => {
                if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                        manager?.closeVisualization({ name: chart.name });
                    }
                });
            });
        }
    }, [TrackGuideEnabled, componentRefs, sessionId]);

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

            [
                AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
                AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
            ].forEach((managerName) => {
                const manager = componentRefs.findComponentRef<VisualizationManagerHandle>(managerName)?.current;
                manager?.getCurrentVisualizations().forEach((chart) => {
                    if (chart.type === 'imitation-guidance-chart' && chart.data?.autoManaged) {
                        manager.closeVisualization({ name: chart.name });
                    }
                });
            });
        };
    }, [componentRefs]);

    useEffect(() => {
        setEnvironment(detectEnvironment());
    }, []);

    const handleSendMessage = async (override?: string) => {
        const text = (override ?? inputValue).trim();
        if (!text || isLoading || liveSessionEnded) return;
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
                    : sessionMode === 'front_desk'
                    ? { content: 'Start the assistant connection first. Front desk context will be sent with the request.' }
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
    const screenMode = sessionMode;
    const transcriptLabel = activeAgentSession
        ? `${getAgentDisplayName(activeAgentSession.agentMode).toUpperCase()} TRANSCRIPT`
        : screenMode === 'front_desk'
        ? 'FRONT DESK TRANSCRIPT'
        : screenMode === 'recorded'
        ? 'RECORDED TRANSCRIPT'
        : screenMode === 'user_summary'
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
        vState === 'speaking' ? 'Kestrel' :
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
        if (liveSessionEnded) return;
        const agentSession = activeAgentSessionRef.current;
        const voiceSessionId = agentSession?.clientSessionId ?? mainClientSessionIdRef.current;
        const overlayTarget = agentSession ? 'agent' : 'main';
        if (vState === 'idle' || vState === 'error') {
            const overlayAiSessionId = createClientSessionId(`overlay-${overlayTarget}`);
            overlayAiSessionByVoiceSessionRef.current.set(voiceSessionId, overlayAiSessionId);
            const overlayStart = beginOverlaySession(overlayAiSessionId, overlayTarget, agentSession?.agentMode);
            overlayStartByVoiceSessionRef.current.set(voiceSessionId, overlayStart);
            void overlayStart
                .catch((overlayError) => {
                    console.warn(
                        'AI overlay failed to initialize; continuing voice conversation without it:',
                        overlayError,
                    );
                })
                .then(() => activeVoiceConversation.start(overlayAiSessionId))
                .catch((err) => {
                    console.error('Voice conversation failed to start:', err);
                    void endOverlaySession(overlayAiSessionId).catch(() => undefined);
                });
        } else {
            const overlayAiSessionId = overlayAiSessionByVoiceSessionRef.current.get(voiceSessionId);
            overlayAiSessionByVoiceSessionRef.current.delete(voiceSessionId);
            overlayStartByVoiceSessionRef.current.delete(voiceSessionId);
            activeVoiceConversation.stop();
            void endOverlaySession(overlayAiSessionId).catch(() => undefined);
        }
    };

    const toggleMicDisabled = () => {
        if (liveSessionEnded) return;
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
            <div className="ai-chat__grid-bg" aria-hidden="true" />

            {/* Header */}
            <div className="ai-chat__header">
                <span className="ai-chat__eyebrow">
                    <span className="ai-chat__eyebrow-dot" />
                    {title}
                </span>
                <div className="ai-chat__header-meta">
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
                        disabled={isLoading || liveSessionEnded}
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
                                ? 'Hide the always-on-top AI overlay'
                                : 'Show the always-on-top AI overlay'}
                        >
                            <OverlayIcon size={14} />
                            <span>{floatingChatOpen ? 'Overlay On' : 'Overlay Off'}</span>
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
                            disabled={vState === 'connecting' || liveSessionEnded}
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
                        Push <kbd>PTT</kbd> or say <kbd>&ldquo;Hey Kestrel&rdquo;</kbd><br />
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

                    <LiveRangeTodoListDisplay
                        snapshot={liveSession.liveRangeTodoListSnapshot}
                        surface="chat"
                    />

                    {procedurePlan && (
                        <ProcedurePlanDisplay plan={procedurePlan} onClear={clearProcedurePlan} />
                    )}

                    <div className="ai-chat__msgs" ref={messagesScrollRef} onScroll={handleMessagesScroll}>
                        {liveSessionEnded && (
                            <div className="ai-chat__ended-notice" role="status">
                                This session has already ended. AI chat is unavailable because live recording is off.
                            </div>
                        )}
                        {messages.map((message) => (
                            <AiMessageDisplay
                                key={message.id}
                                message={message}
                                debugMode={debugMode}
                                assistantAvatarLabel={activeAgentSession ? 'LA' : 'AI'}
                                assistantWhoLabel={activeAgentSession ? getAgentDisplayName(activeAgentSession.agentMode).toUpperCase() : 'Kestrel'}
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
                            : sessionMode === 'front_desk'
                                ? 'Ask the front desk.'
                            : sessionMode === 'recorded'
                                ? 'Ask about this recording.'
                                : sessionMode === 'user_summary'
                                    ? 'Ask about your summary.'
                                    : 'Ask about the live session.'
                    }
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading || liveSessionEnded}
                />
                <button
                    type="button"
                    className="ai-chat__btn ai-chat__btn--primary"
                    onClick={() => handleSendMessage()}
                    disabled={!inputValue.trim() || isLoading || liveSessionEnded}
                    title="Send"
                >
                    SEND
                </button>
            </div>
        </div>
    );
};

export default AiChat;
