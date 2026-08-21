import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { flushSync } from 'react-dom';
import './ai-chat.css';
import type { AnalysisContextType } from 'views/lap-analysis/analysis-context';
import type { LiveSessionRuntime } from 'views/live-session/live-session-types';
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
    AiCommandRegistry,
    LivePerformanceAnalystState,
    OpportunityAgentState,
} from './ai-command-registry';
import {
    useVoiceConversation,
    type FrontendToolHandler,
    type VoiceEvent,
} from './use-voice-conversation';
import { AiMapDisplayPayload } from './AiMapToolDisplay';
import AiMessageDisplay, { type AiChatDisplayMessage } from './AiMessageDisplay';
import {
    Goal,
    LiveRangeTodoList,
    LiveRangeTodoListRunner,
    ProcedurePlanWorkflow,
    buildProcedurePlan,
    isProcedurePlanClearEvent,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
    type AiToolDispatcher,
    type GoalHandle,
    type AiToolOperation,
    type LiveRangeTodoListHandle,
    type ProcedurePlanHandle,
    type ProcedurePlanState,
    type GoalSnapshot,
    createAiToolOperationFrom,
} from 'components/ai-engineering-tools';
import { isLiveSessionAiAvailable, RecordingState } from 'views/lap-analysis/recording-state';
import {
    resolveAssistantRecordedSessionId,
    resolveRegisteredAssistantIdentity,
} from 'views/lap-analysis/assistant-session-mode';
import type { AssistantActiveScreen } from 'views/lap-analysis/assistant-session-mode';
import {
    AI_TOOL_COMPONENT_NAMES,
    NamedAiToolComponentHandle,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
    useAiToolComponentRefs,
    useOptionalAiToolComponentSnapshot,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    NoProcedurePlanError,
    NonLiveContextLiveToolsUnavailableError,
    ProcedurePlanAdvanceFailedError,
} from 'contexts/AiToolComponentError';
import {
    CircuitMapLookupFailedError,
    InvalidProcedurePlanRequestsError,
    ToolExecutionError,
    UnsupportedAgentModeError,
} from './ai-tool-base';
import { getAccTelemetryTrackKey } from 'views/lap-analysis/visualization/charts/circuitTrackLayout';
import {
    overlaySessionClient,
} from 'views/floating-chat/overlay-display-client';
import type {
    AiOverlayPresentationSession,
    AiOverlayShellMetadata,
} from 'views/floating-chat/ai-overlay-types';
import type { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { createAiMessageOverlayComponent } from './AiMessageDisplay.overlay-source';
import { createAiMapOverlayComponent } from './AiMapToolDisplay.overlay-source';
import { createToolStatusOverlayComponent } from './ToolMessageDisplay.overlay-source';
import { createDriverExpertComparisonOverlayComponent } from 'components/driver-expert-comparison/DriverExpertComparisonGraph.overlay-source';
import type { DriverExpertComparisonSnapshot } from 'components/driver-expert-comparison';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';

const asFrontendToolHandlers = (
    registry: AiCommandRegistry,
): Record<string, FrontendToolHandler> => registry as unknown as Record<string, FrontendToolHandler>;

type AiChatSessionMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';

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

interface AiChatConversationProps extends AiChatProps {
    selectedChatLlmModel: string;
    setSelectedChatLlmModel: React.Dispatch<React.SetStateAction<string>>;
    debugMode: boolean;
    setDebugMode: React.Dispatch<React.SetStateAction<boolean>>;
    emotionGifs: Partial<Record<Emotion, string>>;
    setEmotionGifs: React.Dispatch<React.SetStateAction<Partial<Record<Emotion, string>>>>;
    floatingChatOpen: boolean;
    toggleFloatingChat: () => Promise<void>;
    overlayClosedGeneration: number;
}

export interface AiChatHandle extends NamedAiToolComponentHandle {
    getSessionMode(): AiChatSessionMode;
    getRecordingState(): RecordingState | null;
    startAgentSession(agentMode: AgentSessionMode, args?: Record<string, any>): AiToolOperation<AgentSessionStartResult>;
    stopAgentSession(agentSessionId?: string | null): AiToolOperation<AgentSessionStopResult>;
    startTrackGuide(): void;
    setTrackGuideEnabled(enabled: boolean): void;
    setLivePerformanceAnalystEnabled(enabled: boolean): void;
    createGoal(args: Record<string, unknown>, dispatchTool: AiToolDispatcher): ReturnType<GoalHandle['createGoal']>;
    createProcedurePlan(args: Record<string, unknown>, dispatchTool: AiToolDispatcher): ReturnType<ProcedurePlanHandle['createProcedurePlan']>;
    initializeLiveRangeTodoList(): LiveRangeTodoListHandle;
    setAgentTagActive(tag: string, active: boolean): void;
    getOpportunityTelemetryRows(): Record<string, any>[];
    getOpportunityAgentState(): OpportunityAgentState;
    getLivePerformanceAnalystState(): LivePerformanceAnalystState;
    getLabelName(labelId: string): string | undefined;
    getCategoryLabels(category: string): string[];
    getCircuitMapById(id: string): ReturnType<ReturnType<typeof useCircuitMaps>['getCircuitMapById']>;
    getCircuitMapByTrack: ReturnType<typeof useCircuitMaps>['getCircuitMapByTrack'];
    displayMap(display: AiMapDisplayPayload): void;
    displayDriverExpertComparison(
        snapshot: DriverExpertComparisonSnapshot,
    ): void;
    showMap(args: Record<string, unknown>): AiToolOperation<ShowMapAiResult>;
}

export type ShowMapAiResult = { status: string; [key: string]: unknown };

type ActiveWorkflow =
    | { kind: 'goal'; key: number; dispatchTool: AiToolDispatcher }
    | { kind: 'procedure_plan'; key: number; dispatchTool: AiToolDispatcher }
    | { kind: 'live_range_todo'; key: number; runner: LiveRangeTodoListRunner };

type PendingWorkflow =
    | Omit<Extract<ActiveWorkflow, { kind: 'goal' }>, 'key'>
    | Omit<Extract<ActiveWorkflow, { kind: 'procedure_plan' }>, 'key'>
    | Omit<Extract<ActiveWorkflow, { kind: 'live_range_todo' }>, 'key'>;

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

const createClientSessionId = (prefix: string): string =>
    `${prefix}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

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

const AiChatConversation: React.FC<AiChatConversationProps> = ({
    name,
    activeScreen,
    selectedChatLlmModel,
    setSelectedChatLlmModel,
    debugMode,
    setDebugMode,
    emotionGifs,
    setEmotionGifs,
    floatingChatOpen,
    toggleFloatingChat,
    overlayClosedGeneration,
}) => {
    const {
        sessionId,
        sessionMode,
        title,
    } = resolveRegisteredAssistantIdentity(activeScreen);
    const { directory: componentRefs } = useAiToolComponentRefs();
    const [mainMessages, setMainMessages] = useState<Message[]>([]);
    const [agentMessages, setAgentMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const mainClientSessionIdRef = useRef<string>(createClientSessionId('main'));
    const [activeAgentSession, setActiveAgentSession] = useState<AgentSessionInfo | null>(null);

    // Loading and mode states
    const [isLoading] = useState(false);
    const [TrackGuideEnabled, setTrackGuideEnabled] = useState(false);
    const [, setProcedurePlanState] = useState<ProcedurePlanState | null>(null);
    const [, setGoalSnapshot] = useState<GoalSnapshot | null>(null);
    const [activeWorkflow, setActiveWorkflow] = useState<ActiveWorkflow | null>(null);
    const workflowKeyRef = useRef(0);

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');

    // Emotion GIF settings — keyed by Emotion, values are data URLs.
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
    const recordedAnalysisContext = useOptionalAiToolComponentSnapshot<AnalysisContextType>(
        activeScreen.componentName === AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS
            ? AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS
            : null,
    );
    const liveSession = useOptionalAiToolComponentSnapshot<LiveSessionRuntime>(
        sessionMode === 'live' ? AI_TOOL_COMPONENT_NAMES.LIVE_SESSION : null,
    );
    const liveSessionEnded = sessionMode === 'live'
        && liveSession?.recordingState === RecordingState.UPLOAD_READY;
    const analysisContext = useMemo(() => ({
        ...(recordedAnalysisContext ?? {}),
        mapSelected: sessionMode === 'recorded' ? recordedAnalysisContext?.mapSelected ?? null : null,
        sessionSelected: sessionMode === 'recorded' ? recordedAnalysisContext?.sessionSelected ?? null : null,
        liveData: liveSession?.currentTelemetry ?? {},
        TelemetryDataLiveStatus: liveSession?.telemetryStatus ?? null,
        recordingState: liveSession?.recordingState ?? null,
        recordingMetadata: liveSession?.recordingMetadata ?? null,
        recordedSessionDataFilePath: liveSession?.recordingFileKey ?? null,
        recordedTelemetryDataCount: liveSession?.recordedSampleCount ?? 0,
        recordedSessioStaticsData: liveSession?.staticData ?? {},
        getLiveSessionSnapshot: liveSession?.getLiveSessionSnapshot,
    }), [liveSession, recordedAnalysisContext, sessionMode]);
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
        enabled: false,
    });
    const trackGuideLastPosRef = useRef<number | undefined>(undefined);
    const trackGuideTriggeredRef = useRef<Set<string>>(new Set());
    const trackGuideRunTokenRef = useRef(0);
    const activeAgentTagsRef = useRef<string[]>([]);
    const activeAgentSessionRef = useRef<AgentSessionInfo | null>(null);
    const agentVoiceStopRef = useRef<() => void>(() => undefined);
    const mainVoiceStopRef = useRef<() => void>(() => undefined);
    const agentAutoStartSessionIdRef = useRef<string | null>(null);
    const endedAiShutdownAppliedRef = useRef(false);
    const overlayPresentationRef = useRef<AiOverlayPresentationSession | null>(null);
    const ownedOverlayPresentationIdsRef = useRef<Set<string>>(new Set());
    const overlayInvalidationTokenRef = useRef(0);
    const overlayPresentationsByAiSessionRef = useRef<Map<string, string>>(new Map());
    const overlayAiSessionByVoiceSessionRef = useRef<Map<string, string>>(new Map());
    const overlayStartByVoiceSessionRef = useRef<Map<string, Promise<AiOverlayPresentationSession | null>>>(new Map());
    const overlayComponentRefsRef = useRef<Map<string, {
        presentationId: string;
        ref: React.MutableRefObject<MutableAiOverlayComponent<any> | null>;
    }>>(new Map());
    const overlayComponentSequenceRef = useRef(0);
    const voiceSessionSeenActiveRef = useRef(false);
    const procedurePlanRef = useRef<ProcedurePlanState | null>(null);
    const procedurePlanOptedOutRef = useRef(false);
    const activeToolHandlersRef = useRef<Record<string, FrontendToolHandler>>({});
    const conversationDisposedRef = useRef(false);
    const pendingTimersRef = useRef<Set<number>>(new Set());
    const liveRangeTodoListRunnerRef = useRef<LiveRangeTodoListRunner | null>(null);
    const liveRangeTodoListSessionGameRef = useRef(liveSession?.sessionGame ?? null);

    useEffect(() => {
        liveRangeTodoListRunnerRef.current?.acceptTelemetry(liveSession?.currentTelemetry ?? {});
    }, [liveSession?.currentTelemetry]);

    useEffect(() => {
        const previousSessionGame = liveRangeTodoListSessionGameRef.current;
        const sessionGame = liveSession?.sessionGame ?? null;
        liveRangeTodoListSessionGameRef.current = sessionGame;
        if (previousSessionGame === sessionGame) return;
        if (sessionGame === null) return;
        liveRangeTodoListRunnerRef.current?.reset();
    }, [liveSession?.sessionGame]);

    const scheduleConversationTimeout = useCallback((callback: () => void, delay = 0) => {
        const timeoutId = window.setTimeout(() => {
            pendingTimersRef.current.delete(timeoutId);
            if (!conversationDisposedRef.current) callback();
        }, delay);
        pendingTimersRef.current.add(timeoutId);
        return timeoutId;
    }, []);

    useEffect(() => {
        const pendingTimers = pendingTimersRef.current;
        conversationDisposedRef.current = false;
        return () => {
            conversationDisposedRef.current = true;
            trackGuideRunTokenRef.current += 1;
            overlayInvalidationTokenRef.current += 1;
            activeAgentSessionRef.current = null;
            activeToolHandlersRef.current = {};
            const liveRangeTodoListRunner = liveRangeTodoListRunnerRef.current;
            liveRangeTodoListRunnerRef.current = null;
            liveRangeTodoListRunner?.dispose();
            pendingTimers.forEach((timeoutId) => window.clearTimeout(timeoutId));
            pendingTimers.clear();
        };
    }, []);

    const mountWorkflow = useCallback((workflow: PendingWorkflow) => {
        const next = { ...workflow, key: ++workflowKeyRef.current } as ActiveWorkflow;
        flushSync(() => {
            setGoalSnapshot(null);
            procedurePlanRef.current = null;
            setProcedurePlanState(null);
            setActiveWorkflow(next);
        });
    }, []);

    const dispatchActiveVoiceTool = useCallback<AiToolDispatcher>((
        toolName,
        args = {},
    ) => {
        const handler = activeToolHandlersRef.current[toolName];
        if (!handler) {
            return createAiToolOperationFrom(() => {
            throw new ToolExecutionError(
                `The active AI session could not execute '${toolName}'.`,
            );
            });
        }
        return handler(args) as ReturnType<AiToolDispatcher>;
    }, []);


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

    const publishOverlayComponent = useCallback(<TSnapshot,>(
        componentName: string,
        presentationId: string | undefined,
        createComponent: (name: string) => MutableAiOverlayComponent<TSnapshot>,
        snapshot: TSnapshot,
        metadata: AiOverlayShellMetadata = {},
    ) => {
        if (conversationDisposedRef.current || !presentationId) return;
        let registered = overlayComponentRefsRef.current.get(componentName);
        if (!registered) {
            const ref: React.MutableRefObject<MutableAiOverlayComponent<any> | null> = {
                current: createComponent(componentName),
            };
            registered = { presentationId, ref };
            overlayComponentRefsRef.current.set(componentName, registered);
            componentRefs.registerComponentRef(ref);
        }
        registered.ref.current?.publish(snapshot, {
            presentationId,
            metadata: {
                ...metadata,
                name: metadata.name ?? (
                    activeAgentSessionRef.current
                        ? getAgentDisplayName(activeAgentSessionRef.current.agentMode)
                        : undefined
                ),
                agentTags: metadata.agentTags ?? activeAgentTagsRef.current,
            },
        });
    }, [componentRefs]);

    const releaseOverlayComponents = useCallback((presentationId?: string) => {
        overlayComponentRefsRef.current.forEach((registered, componentName) => {
            if (presentationId && registered.presentationId !== presentationId) return;
            componentRefs.unregisterComponentRef(registered.ref);
            overlayComponentRefsRef.current.delete(componentName);
        });
    }, [componentRefs]);

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
                    const targetPresentationId = presentationId
                        ?? overlayPresentationRef.current?.presentationId;
                    publishOverlayComponent(
                        `ai-message:${targetPresentationId}`,
                        targetPresentationId,
                        createAiMessageOverlayComponent,
                        { text: pillText },
                        {
                        name: options.name,
                        emotion: options.emotion ?? undefined,
                        agentTags: options.tags ?? activeAgentTagsRef.current,
                        },
                    );
                }
            }
        } catch { /* ignore unavailable overlay failures */ }
    }, [publishOverlayComponent]);

    const displayMapInChat = useCallback((display: AiMapDisplayPayload) => {
        if (conversationDisposedRef.current) return;
        const fallbackText = display.status === 'unavailable'
            ? 'Map is not available'
            : display.note || display.title || display.map?.circuit_name || 'Map';
        const presentationId = overlayPresentationRef.current?.presentationId;
        publishOverlayComponent(
            `ai-map:${presentationId}`,
            presentationId,
            createAiMapOverlayComponent,
            display,
        );
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
    }, [generateUniqueId, publishOverlayComponent, setMessages]);

    const displayDriverExpertComparison = useCallback((
        snapshot: DriverExpertComparisonSnapshot,
    ) => {
        const presentationId = overlayPresentationRef.current?.presentationId;
        publishOverlayComponent(
            `driver-expert-comparison:${presentationId}:${++overlayComponentSequenceRef.current}`,
            presentationId,
            createDriverExpertComparisonOverlayComponent,
            snapshot,
        );
    }, [publishOverlayComponent]);

    const showMap = useCallback(async (
        args: Record<string, unknown>,
    ): Promise<Record<string, unknown>> => {
        const normalize = (value: unknown) => (
            typeof value === 'string' && value.trim() ? value.trim() : undefined
        );
        const selectedMap = recordedAnalysisContext?.sessionSelected?.map
            || recordedAnalysisContext?.mapSelected
            || liveSession?.getLiveSessionSnapshot().track;
        const candidates = [
            args.map_id,
            args.source_track_key,
            args.map_name,
            selectedMap,
        ].map(normalize).filter((value): value is string => Boolean(value));
        try {
            let map = null;
            let resolvedBy: 'id' | 'track' | null = null;
            for (const candidate of candidates) {
                map = await getCircuitMapById(candidate);
                if (map) {
                    resolvedBy = 'id';
                    break;
                }
            }
            if (!map) {
                for (const candidate of candidates) {
                    map = await getCircuitMapByTrack(
                        'acc',
                        getAccTelemetryTrackKey(candidate) || candidate,
                    );
                    if (map) {
                        resolvedBy = 'track';
                        break;
                    }
                }
            }
            const clamp = (value: unknown) => {
                const parsed = Number(value);
                return Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : undefined;
            };
            const start = clamp(args.section_start ?? args.start);
            const end = clamp(args.section_end ?? args.end);
            const label = normalize(args.section_label ?? args.label);
            const section = start === undefined && end === undefined && !label
                ? undefined
                : { start, end, label };
            const title = normalize(args.title) || 'Map';
            const note = normalize(args.message ?? args.note);
            if (!map) {
                const requestedMap = candidates[0];
                const reason = requestedMap
                    ? `No circuit map is available for "${requestedMap}".`
                    : 'No circuit map is available for the current session.';
                displayMapInChat({
                    status: 'unavailable',
                    requestedMap,
                    title,
                    note,
                    reason,
                    section,
                });
                return {
                    status: 'unavailable',
                    message: 'Map is not available',
                    requested_map: requestedMap ?? null,
                    resolved_by: null,
                    reason,
                    section: section ?? null,
                };
            }
            displayMapInChat({
                status: 'ready',
                map,
                requestedMap: candidates[0],
                title,
                note,
                section,
            });
            return {
                status: 'displayed',
                map_id: map.id,
                circuit_name: map.circuit_name,
                requested_map: candidates[0] ?? null,
                source_track_key: map.source_track_key ?? null,
                resolved_by: resolvedBy,
                reason: null,
                section: section ?? null,
            };
        } catch (error) {
            throw new CircuitMapLookupFailedError(
                error instanceof Error && error.message
                    ? error.message
                    : 'Failed to look up the requested circuit map.',
                { cause: error },
            );
        }
    }, [
        displayMapInChat,
        getCircuitMapById,
        getCircuitMapByTrack,
        liveSession?.getLiveSessionSnapshot,
        recordedAnalysisContext?.mapSelected,
        recordedAnalysisContext?.sessionSelected?.map,
    ]);

    const setAgentTag = useCallback((tag: string, active: boolean) => {
        const current = activeAgentTagsRef.current;
        const next = active
            ? Array.from(new Set([...current, tag]))
            : current.filter((item) => item !== tag);
        if (next.length === current.length && next.every((item, index) => item === current[index])) {
            return;
        }
        activeAgentTagsRef.current = next;
    }, []);

    useEffect(() => {
        activeAgentTagsRef.current = [];
    }, []);

    const beginOverlaySession = useCallback(async (
        aiSessionId: string,
        target: 'main' | 'agent',
        agentMode?: AgentSessionMode,
    ): Promise<AiOverlayPresentationSession | null> => {
        if (!overlaySessionClient.available()) return null;
        const previousPresentationId = overlayPresentationRef.current?.presentationId;
        const overlayInvalidationToken = overlayInvalidationTokenRef.current;
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
        if (
            conversationDisposedRef.current
            || overlayInvalidationToken !== overlayInvalidationTokenRef.current
        ) {
            void overlaySessionClient.destroy(presentation.presentationId).catch(() => undefined);
            return null;
        }
        if (previousPresentationId && previousPresentationId !== presentation.presentationId) {
            releaseOverlayComponents(previousPresentationId);
        }
        ownedOverlayPresentationIdsRef.current.add(presentation.presentationId);
        overlayPresentationsByAiSessionRef.current.set(aiSessionId, presentation.presentationId);
        if (overlaySessionClient.current()?.presentationId === presentation.presentationId) {
            overlayPresentationRef.current = presentation;
        }
        return presentation;
    }, [releaseOverlayComponents, sessionMode]);

    const endOverlaySession = useCallback(async (aiSessionId?: string | null) => {
        const presentationId = aiSessionId
            ? overlayPresentationsByAiSessionRef.current.get(aiSessionId)
            : overlayPresentationRef.current?.presentationId;
        if (!presentationId) return;
        releaseOverlayComponents(presentationId);
        ownedOverlayPresentationIdsRef.current.delete(presentationId);
        if (aiSessionId) overlayPresentationsByAiSessionRef.current.delete(aiSessionId);
        await overlaySessionClient.destroy(presentationId);
        if (conversationDisposedRef.current) return;
        if (overlayPresentationRef.current?.presentationId === presentationId) {
            overlayPresentationRef.current = null;
        }
    }, [releaseOverlayComponents]);

    useEffect(() => () => {
        overlayPresentationRef.current = null;
        overlayComponentRefsRef.current.forEach(({ ref }) => {
            componentRefs.unregisterComponentRef(ref);
        });
        overlayComponentRefsRef.current.clear();
        const presentationIds = Array.from(ownedOverlayPresentationIdsRef.current);
        ownedOverlayPresentationIdsRef.current.clear();
        overlayPresentationsByAiSessionRef.current.clear();
        overlayAiSessionByVoiceSessionRef.current.clear();
        overlayStartByVoiceSessionRef.current.clear();
        presentationIds.forEach((presentationId) => {
            void overlaySessionClient.destroy(presentationId).catch(() => undefined);
        });
    }, [componentRefs]);

    useEffect(() => {
        if (overlayClosedGeneration === 0) return;
        overlayInvalidationTokenRef.current += 1;
        overlayPresentationRef.current = null;
        ownedOverlayPresentationIdsRef.current.clear();
        overlayPresentationsByAiSessionRef.current.clear();
        overlayAiSessionByVoiceSessionRef.current.clear();
        overlayStartByVoiceSessionRef.current.clear();
        releaseOverlayComponents();
    }, [overlayClosedGeneration, releaseOverlayComponents]);

    const setTrackGuideAgentEnabled = useCallback((enabled: boolean) => {
        if (!enabled) {
            trackGuideRunTokenRef.current += 1;
        }
        setTrackGuideEnabled(enabled);
    }, []);

    const setLivePerformanceAnalystAgentEnabled = useCallback((enabled: boolean) => {
        livePerformanceAnalystStateRef.current.enabled = enabled;
    }, []);

    const observeBackgroundWorkflow = useCallback((
        operation: AiToolOperation<unknown, object>,
    ) => {
        void operation.result.catch((error) => {
            console.error('Background workflow failed.', error);
        });
    }, []);

    const setProcedurePlan = useCallback((plan: ProcedurePlanState | null) => {
        const existing = componentRefs
            .findComponentRef<ProcedurePlanHandle>(AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN)
            ?.current ?? null;
        if (!plan) {
            existing?.clearProcedurePlan();
            procedurePlanRef.current = null;
            setProcedurePlanState(null);
            return;
        }
        if (existing) {
            observeBackgroundWorkflow(existing.createProcedurePlan(plan));
            return;
        }
        mountWorkflow({ kind: 'procedure_plan', dispatchTool: dispatchActiveVoiceTool });
        void awaitNamedComponentHandle<ProcedurePlanHandle>(
            componentRefs,
            AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
        ).then((handle) => {
            if (conversationDisposedRef.current) return;
            observeBackgroundWorkflow(handle.createProcedurePlan(plan));
        });
    }, [componentRefs, dispatchActiveVoiceTool, mountWorkflow, observeBackgroundWorkflow]);

    const advanceProcedurePlanStep = useCallback(async (reason?: string) => {
        const runner = componentRefs
            .findComponentRef<ProcedurePlanHandle>(AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN)
            ?.current;
        if (!runner) {
            throw new NoProcedurePlanError(
                name,
                'The procedure plan could not be advanced.',
            );
        }

        try {
            const operation = runner.advancePlanStep(reason);
            const result = await operation.result;
            if (result instanceof Error) throw result;
            return result;
        } catch (error) {
            throw new ProcedurePlanAdvanceFailedError(
                name,
                error instanceof Error && error.message
                    ? error.message
                    : 'The procedure plan could not be advanced.',
                { cause: error },
            );
        }
    }, [componentRefs, name]);

    const clearProcedurePlan = useCallback(() => {
        const runner = componentRefs
            .findComponentRef<ProcedurePlanHandle>(AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN)
            ?.current;
        runner?.clearProcedurePlan();
        procedurePlanRef.current = null;
        setProcedurePlanState(null);
    }, [componentRefs]);

    const optOutProcedurePlan = useCallback(() => {
        procedurePlanOptedOutRef.current = true;
        setProcedurePlan(null);
    }, [setProcedurePlan]);

    // Racing engineer voice conversation. The hook owns mic, WS, and
    // audio playback; it ALSO multiplexes the tool-relay text channel on
    // the same WS — frontend tools listed below are reachable from the
    // backend LLM via JSON text frames.
    const handleSessionVoiceEvent = useCallback((event: VoiceEvent, target: 'main' | 'agent') => {
        if (conversationDisposedRef.current) return;
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
            setTargetMessages(prev => {
                const messagesWithoutLoading = prev.filter(m => !m.isLoading);
                const transcript = event.text.trim();
                const lastMessage = messagesWithoutLoading[messagesWithoutLoading.length - 1];

                if (
                    event.source !== 'typed'
                    && lastMessage?.isUser
                    && (lastMessage.kind ?? 'chat') === 'chat'
                ) {
                    return messagesWithoutLoading.slice(0, -1).concat({
                        ...lastMessage,
                        content: `${lastMessage.content} ${transcript}`,
                    });
                }

                return messagesWithoutLoading.concat({
                    id: generateUniqueId('user-voice'),
                    content: transcript,
                    isUser: true,
                    timestamp: new Date(),
                    kind: 'chat',
                });
            });
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
                publishOverlayComponent(
                    `tool-status:${presentationId}:${encodeURIComponent(event.runId)}`,
                    presentationId,
                    createToolStatusOverlayComponent,
                    {
                        runId: event.runId,
                        name: event.name,
                        title: event.title,
                        status: event.status,
                        ok: event.ok,
                        error: event.message ?? null,
                        result: event.result,
                    },
                );
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
                                error: event.message ?? null,
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
                        error: event.message ?? null,
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
        publishOverlayComponent,
        setProcedurePlan,
    ]);

    const handleMainVoiceEvent = useCallback((event: VoiceEvent) => {
        handleSessionVoiceEvent(event, 'main');
    }, [handleSessionVoiceEvent]);

    const handleAgentVoiceEvent = useCallback((event: VoiceEvent) => {
        handleSessionVoiceEvent(event, 'agent');
    }, [handleSessionVoiceEvent]);

    const startTrackGuide = useCallback(() => {
        trackGuideRunTokenRef.current += 1;
        setTrackGuideEnabled(true);
    }, []);

    const resolvedSessionId = resolveAssistantRecordedSessionId(
        sessionMode,
        sessionId
            || (analysisContext?.sessionSelected as Record<string, any> | null)?.SessionId,
    );

    const aiSessionContext = useMemo(() => ({
        session_mode: sessionMode,
    }), [sessionMode]);

    const inactiveAgentToolHandlers = useMemo(() => ({}), []);
    const getProcedurePlan = useCallback(() => procedurePlanRef.current, []);
    const getOpportunityTelemetryRows = useCallback(() => opportunityForecastRowsRef.current, []);
    const createGoal = useCallback((
        args: Record<string, unknown>,
        dispatchTool: AiToolDispatcher,
    ): ReturnType<GoalHandle['createGoal']> => {
        try {
            mountWorkflow({ kind: 'goal', dispatchTool });
            return resolveNamedComponentHandle<GoalHandle>(
                componentRefs,
                AI_TOOL_COMPONENT_NAMES.GOAL,
            ).createGoal(args as any);
        } catch (error) {
            return createAiToolOperationFrom(() => { throw error; });
        }
    }, [componentRefs, mountWorkflow]);

    const createProcedurePlan = useCallback((
        args: Record<string, unknown>,
        dispatchTool: AiToolDispatcher,
    ): ReturnType<ProcedurePlanHandle['createProcedurePlan']> => {
        try {
            const plan = buildProcedurePlan({
                ...args,
                event: typeof args.event === 'string' && args.event.trim()
                    ? args.event
                    : 'procedure_plan_started',
            });
            if (!plan) {
                throw new InvalidProcedurePlanRequestsError(
                    'Provide a goal and at least one request with a title.',
                );
            }
            mountWorkflow({ kind: 'procedure_plan', dispatchTool });
            return resolveNamedComponentHandle<ProcedurePlanHandle>(
                componentRefs,
                AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
            ).createProcedurePlan(plan);
        } catch (error) {
            return createAiToolOperationFrom(() => { throw error; });
        }
    }, [componentRefs, mountWorkflow]);

    const initializeLiveRangeTodoList = useCallback((): LiveRangeTodoListHandle => {
        let runner = liveRangeTodoListRunnerRef.current;
        if (!runner) {
            runner = new LiveRangeTodoListRunner(
                AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
                (snapshot) => {
                    if (snapshot !== null || liveRangeTodoListRunnerRef.current !== runner) return;
                    liveRangeTodoListRunnerRef.current = null;
                    setActiveWorkflow((current) => (
                        current?.kind === 'live_range_todo' && current.runner === runner
                            ? null
                            : current
                    ));
                },
            );
            liveRangeTodoListRunnerRef.current = runner;
            runner.addComponentRef(componentRefs);
        }
        if (activeWorkflow?.kind !== 'live_range_todo' || activeWorkflow.runner !== runner) {
            mountWorkflow({ kind: 'live_range_todo', runner });
        }
        return runner;
    }, [activeWorkflow, componentRefs, mountWorkflow]);

    const resetLivePerformanceAnalystRuntime = useCallback(() => {
        const analystAgent = livePerformanceAnalystStateRef.current;
        analystAgent.enabled = false;
        setLivePerformanceAnalystAgentEnabled(false);
        procedurePlanOptedOutRef.current = false;
        setActiveWorkflow(null);
        setGoalSnapshot(null);
        const liveRangeTodoListRunner = liveRangeTodoListRunnerRef.current;
        liveRangeTodoListRunnerRef.current = null;
        liveRangeTodoListRunner?.dispose();
        procedurePlanRef.current = null;
        setProcedurePlanState(null);
    }, [setLivePerformanceAnalystAgentEnabled]);

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
        if (!['track_guide', 'overtake', 'live_performance_analyst'].includes(agentMode)) {
            throw new UnsupportedAgentModeError(
                'Supported agent modes are track_guide, overtake, and live_performance_analyst.',
            );
        }
        if (sessionMode !== 'live' || !isLiveSessionAiAvailable(analysisContext?.recordingState)) {
            throw new NonLiveContextLiveToolsUnavailableError(
                name,
                'Agent sessions are only available in live session mode.',
            );
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
        name,
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
        scheduleConversationTimeout(() => stopAgentVoice?.());
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
    }, [endOverlaySession, resetAgentRuntimes, scheduleConversationTimeout, setAgentTag]);

    const aiChatHandle = useMemo<AiChatHandle>(() => ({
        getComponentName: () => name,
        getSessionMode: () => sessionMode,
        getRecordingState: () => analysisContext?.recordingState ?? null,
        startAgentSession: (agentMode, args) => createAiToolOperationFrom(
            () => startAgentSession(agentMode, args),
        ),
        stopAgentSession: (agentSessionId) => createAiToolOperationFrom(
            () => stopAgentSession(agentSessionId),
        ),
        startTrackGuide,
        setTrackGuideEnabled: setTrackGuideAgentEnabled,
        setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
        createGoal,
        createProcedurePlan,
        initializeLiveRangeTodoList,
        setAgentTagActive: setAgentTag,
        getOpportunityTelemetryRows,
        getOpportunityAgentState: () => opportunityAgentStateRef.current,
        getLivePerformanceAnalystState: () => livePerformanceAnalystStateRef.current,
        getLabelName,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        displayMap: displayMapInChat,
        displayDriverExpertComparison,
        showMap: (args) => createAiToolOperationFrom(
            async () => await showMap(args) as ShowMapAiResult,
        ),
    }), [
        analysisContext?.recordingState,
        createGoal,
        createProcedurePlan,
        displayDriverExpertComparison,
        displayMapInChat,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLabelName,
        getOpportunityTelemetryRows,
        initializeLiveRangeTodoList,
        name,
        sessionMode,
        setAgentTag,
        setLivePerformanceAnalystAgentEnabled,
        setTrackGuideAgentEnabled,
        startAgentSession,
        startTrackGuide,
        stopAgentSession,
        showMap,
    ]);
    const aiChatRef = useRef<AiChatHandle | null>(aiChatHandle);
    aiChatRef.current = aiChatHandle;
    useRegisterAiToolComponentRef(aiChatRef);

    const toolHandlers = useMemo(() => createAiCommandRegistry({
        componentRefs,
        sessionId: resolvedSessionId,
        sessionMode,
        conversationRole: 'main',
        sessionGame: liveSession?.sessionGame ?? null,
    }), [componentRefs, liveSession?.sessionGame, resolvedSessionId, sessionMode]);

    const selectedChatLlmModelOption = getChatLlmModelOption(selectedChatLlmModel);
    const voiceConversation = useVoiceConversation({
        sessionId: resolvedSessionId,
        conversationRole: 'main',
        clientSessionId: mainClientSessionIdRef.current,
        chatLlmModel: selectedChatLlmModelOption.value,
        sessionContext: aiSessionContext,
        onEvent: handleMainVoiceEvent,
        toolHandlers: asFrontendToolHandlers(toolHandlers),
    });
    const agentSessionContext = useMemo(() => (
        activeAgentSession
            ? {
                session_mode: sessionMode,
                agent_mode: activeAgentSession.agentMode,
            }
            : null
    ), [activeAgentSession, sessionMode]);

    const agentToolHandlers = useMemo(() => createAiCommandRegistry({
        componentRefs,
        sessionId: resolvedSessionId,
        sessionMode,
        conversationRole: 'agent',
        agentMode: activeAgentSession?.agentMode,
        sessionGame: liveSession?.sessionGame ?? null,
    }), [
        activeAgentSession?.agentMode,
        componentRefs,
        liveSession?.sessionGame,
        resolvedSessionId,
        sessionMode,
    ]);
    activeToolHandlersRef.current = asFrontendToolHandlers(
        activeAgentSession ? agentToolHandlers : toolHandlers,
    );

    const agentVoiceConversation = useVoiceConversation({
        sessionId: resolvedSessionId,
        conversationRole: 'agent',
        clientSessionId: activeAgentSession?.clientSessionId,
        parentClientSessionId: activeAgentSession?.parentClientSessionId,
        chatLlmModel: selectedChatLlmModelOption.value,
        sessionContext: agentSessionContext || undefined,
        onEvent: handleAgentVoiceEvent,
        toolHandlers: activeAgentSession
            ? asFrontendToolHandlers(agentToolHandlers)
            : inactiveAgentToolHandlers,
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
            if (conversationDisposedRef.current) return;
            if (activeAgentSessionRef.current?.clientSessionId !== activeAgentSessionId) return;
            return startAgentVoiceConversation(
                overlayAiSessionByVoiceSessionRef.current.get(activeAgentSessionId),
            );
        }).catch((err) => {
            if (conversationDisposedRef.current) return;
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

        scheduleConversationTimeout(() => {
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
                getLiveSessionSnapshot: analysisContext?.getLiveSessionSnapshot,
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
                displayMap: displayMapInChat,
            }, {}, (status) => {
                sendAgentVoiceToolStatus(status);
            }).catch((error) => {
                console.error(`Failed to start ${current.agentMode} runtime.`, error);
                if (conversationDisposedRef.current) return;
                if (activeAgentSessionRef.current?.clientSessionId !== current.clientSessionId) return;
                setActiveAgentSession((session) => session
                    ? { ...session, status: 'error' }
                    : session);
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
        sendAgentVoiceToolStatus,
        getCategoryLabels,
        getCircuitMapById,
        getCircuitMapByTrack,
        getLabelName,
        getOpportunityTelemetryRows,
        getProcedurePlan,
        resolvedSessionId,
        scheduleConversationTimeout,
        sessionMode,
        setAgentTag,
        setLivePerformanceAnalystAgentEnabled,
        setProcedurePlan,
        setTrackGuideAgentEnabled,
        startTrackGuide,
        stopAgentSession,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const activeVoiceConversation = activeAgentSession ? agentVoiceConversation : voiceConversation;
    const vState = activeVoiceConversation.state;
    const voiceActive = vState === 'listening' || vState === 'speaking';
    const modelPickerDisabled = voiceActive || vState === 'connecting';
    const micDisabled = activeVoiceConversation.micDisabled;
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
    const canOpenFloatingChat = overlaySessionClient.available();

    const addGuidanceMessage = useCallback((content: string) => {
        if (conversationDisposedRef.current) return;
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
            if (conversationDisposedRef.current) return;
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
            opportunityAgent.inFlight = false;
            analystAgent.enabled = false;

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
                    if (conversationDisposedRef.current) return;
                    console.warn(
                        'AI overlay failed to initialize; continuing voice conversation without it:',
                        overlayError,
                    );
                })
                .then(() => {
                    if (conversationDisposedRef.current) return;
                    return activeVoiceConversation.start(overlayAiSessionId);
                })
                .catch((err) => {
                    if (conversationDisposedRef.current) return;
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

                    {activeWorkflow?.kind === 'goal' && (
                        <Goal
                            key={activeWorkflow.key}
                            name={AI_TOOL_COMPONENT_NAMES.GOAL}
                            dispatchTool={activeWorkflow.dispatchTool}
                            onSnapshotChange={setGoalSnapshot}
                            surface="chat"
                        />
                    )}
                    {activeWorkflow?.kind === 'procedure_plan' && (
                        <ProcedurePlanWorkflow
                            key={activeWorkflow.key}
                            name={AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN}
                            dispatchTool={activeWorkflow.dispatchTool}
                            onSnapshotChange={(next) => {
                                procedurePlanRef.current = next;
                                setProcedurePlanState(next);
                            }}
                            surface="chat"
                        />
                    )}
                    {activeWorkflow?.kind === 'live_range_todo' && (
                        <LiveRangeTodoList
                            key={activeWorkflow.key}
                            runner={activeWorkflow.runner}
                            surface="chat"
                        />
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

const AiChat: React.FC<AiChatProps> = ({ name, activeScreen }) => {
    const { conversationKey } = resolveRegisteredAssistantIdentity(activeScreen);
    const [selectedChatLlmModel, setSelectedChatLlmModel] = useState(
        DEFAULT_CHAT_LLM_MODEL_OPTION.value,
    );
    const [debugMode, setDebugMode] = useState(false);
    const [emotionGifs, setEmotionGifs] = useState<Partial<Record<Emotion, string>>>(() => {
        try { return JSON.parse(localStorage.getItem(EMOTION_GIFS_KEY) || '{}'); }
        catch { return {}; }
    });
    const [floatingChatOpen, setFloatingChatOpen] = useState(false);
    const [overlayClosedGeneration, setOverlayClosedGeneration] = useState(0);

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
        if (!api) return;
        let disposed = false;
        const unsubscribe = api.onFloatingChatClosed?.(() => {
            setFloatingChatOpen(false);
            setOverlayClosedGeneration((generation) => generation + 1);
        });
        if (api.isOverlayEnabled) {
            api.isOverlayEnabled()
                .then((open: boolean) => {
                    if (!disposed) setFloatingChatOpen(Boolean(open));
                })
                .catch(() => undefined);
        }
        return () => {
            disposed = true;
            try { unsubscribe?.(); } catch { /* ignore */ }
        };
    }, []);

    return (
        <AiChatConversation
            key={conversationKey}
            name={name}
            activeScreen={activeScreen}
            selectedChatLlmModel={selectedChatLlmModel}
            setSelectedChatLlmModel={setSelectedChatLlmModel}
            debugMode={debugMode}
            setDebugMode={setDebugMode}
            emotionGifs={emotionGifs}
            setEmotionGifs={setEmotionGifs}
            floatingChatOpen={floatingChatOpen}
            toggleFloatingChat={toggleFloatingChat}
            overlayClosedGeneration={overlayClosedGeneration}
        />
    );
};

export default AiChat;
