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
    getFrontendToolSchemasForSessionMode,
    QUERY_SCOPE_SCHEMA,
} from './ai-command-registry';
import { getCornersForTrack } from 'views/lap-analysis/session-intelligence/track-corners';
import type { CornerDefinition } from 'views/lap-analysis/session-intelligence/types';
import type { LivePerformanceAnalystState, OpportunityAgentState } from './ai-command-registry';
import { useVoiceConversation, VoiceEvent } from './use-voice-conversation';
import AiMapToolDisplay, { AiMapDisplayPayload } from './AiMapToolDisplay';
import {
    buildLiveProcedurePlan,
    getProcedurePlanAdvanceBlock,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
    type ProcedurePlan,
} from './ai-chat-plan';

type AiChatSessionMode = 'live' | 'recorded' | 'user_summary';

const EMOTIONS = ['idle', 'sad', 'vibing', 'scared', 'waiting', 'hearing'] as const;
type Emotion = typeof EMOTIONS[number];
const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const EMOTION_TAG_RE = /^\[([a-z]+)\]\s*/;
const MAX_OVERTAKE_AGENT_ROWS = 300;
const BASELINE_PROGRESS_MESSAGE_ID = 'live-baseline-progress';
const TRANSCRIPT_BOTTOM_THRESHOLD_PX = 48;

function extractEmotion(text: string): { emotion: Emotion | null; cleanText: string } {
    const m = text.match(EMOTION_TAG_RE);
    if (m && (EMOTIONS as readonly string[]).includes(m[1])) {
        return { emotion: m[1] as Emotion, cleanText: text.slice(m[0].length) };
    }
    return { emotion: null, cleanText: text };
}

type MessageKind = 'chat' | 'tool' | 'progress';

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
        name: string;
        title: string;
        status: 'started' | 'completed';
        ok?: boolean;
        error?: string | null;
    };
    mapDisplay?: AiMapDisplayPayload;
    progress?: {
        label: string;
        value: number;
        detail?: string;
        startMarkerLabel?: string;
        startMarkerValue?: number;
    };
}


interface AiChatProps {
    sessionId?: string;
    sessionMode?: AiChatSessionMode;
    title?: string;
}

const formatClock = (d: Date) =>
    `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;

const getProcedurePlanRequestMeta = (request: ProcedurePlan['requests'][number]): string => {
    const parts = [
        request.type,
        request.method,
        request.name,
        request.url,
    ].filter((part): part is string => Boolean(part));
    return parts.join(' · ');
};

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
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');

    // Loading and mode states
    const [isLoading] = useState(false);
    const [debugMode, setDebugMode] = useState(false);
    const [TrackGuideEnabled, setTrackGuideEnabled] = useState(false);
    const [livePerformanceAnalystEnabled, setLivePerformanceAnalystEnabled] = useState(false);
    const [procedurePlan, setProcedurePlanState] = useState<ProcedurePlan | null>(null);

    const [environment, setEnvironment] = useState<'electron' | 'web'>('web');
    const [floatingChatOpen, setFloatingChatOpen] = useState(false);

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
    const procedurePlanRef = useRef<ProcedurePlan | null>(null);
    const procedurePlanOptedOutRef = useRef(false);

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

    const displayMapInChat = useCallback((display: AiMapDisplayPayload) => {
        const fallbackText = display.status === 'unavailable'
            ? 'Map is not available'
            : display.note || display.title || display.map?.circuit_name || 'Map';
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
    }, [generateUniqueId]);

    const broadcastPillMessage = useCallback((text: string, options: { emotion?: Emotion | null; tags?: string[] } = {}) => {
        try {
            const pillText = text
                .replace(/\*\*(.*?)\*\*/g, '$1')
                .replace(/\*(.*?)\*/g, '$1')
                .replace(/`(.*?)`/g, '$1')
                .replace(/\s+/g, ' ')
                .trim()
                .slice(0, 280);
            if (pillText || options.tags !== undefined) {
                localStorage.setItem('acla-pill-msg', JSON.stringify({
                    text: pillText,
                    ts: Date.now(),
                    emotion: options.emotion ?? undefined,
                    tags: options.tags ?? activeAgentTagsRef.current,
                }));
            }
        } catch { /* ignore storage write failures */ }
    }, []);

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

    useEffect(() => {
        if (sessionMode !== 'live' || !livePerformanceAnalystEnabled) {
            setMessages(prev => prev.filter((message) => message.id !== BASELINE_PROGRESS_MESSAGE_ID));
            return;
        }

        const snapshot = analysisContext?.sessionIntelligence?.getLiveSessionSnapshot?.();
        if (!snapshot || snapshot.status === 'empty') {
            return;
        }

        const value = Math.max(0, Math.min(100, Number(snapshot.baseline_progress_percent ?? 0)));
        const detail = snapshot.baseline_ready
            ? 'Baseline complete. Classifier analysis is starting.'
            : snapshot.baseline_collection_started
                ? `Lap ${snapshot.current_lap + 1} baseline`
                : 'Start at normalized position 0';

        setMessages(prev => {
            const progressMessage: Message = {
                id: BASELINE_PROGRESS_MESSAGE_ID,
                content: 'Collecting baseline',
                isUser: false,
                timestamp: new Date(),
                kind: 'progress',
                progress: {
                    label: 'Collecting baseline',
                    value,
                    detail,
                    startMarkerLabel: 'Start',
                    startMarkerValue: 0,
                },
            };

            const existingIndex = prev.findIndex((message) => message.id === BASELINE_PROGRESS_MESSAGE_ID);
            if (existingIndex === -1) {
                return prev.concat(progressMessage);
            }

            const next = prev.slice();
            next[existingIndex] = {
                ...next[existingIndex],
                progress: progressMessage.progress,
            };
            return next;
        });
    }, [
        analysisContext?.liveData,
        analysisContext?.sessionIntelligence,
        livePerformanceAnalystEnabled,
        sessionMode,
    ]);

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
        procedurePlanRef.current = plan;
        setProcedurePlanState(plan);
    }, []);

    const advanceProcedurePlanStep = useCallback((reason?: string) => {
        const current = procedurePlanRef.current;
        if (!current) {
            return { status: 'unavailable', error: 'no_procedure_plan' };
        }

        const liveSnapshot = sessionMode === 'live'
            ? analysisContext?.sessionIntelligence?.getLiveSessionSnapshot?.()
            : null;
        const hasLiveFocus = sessionMode === 'live'
            ? Boolean(analysisContext?.sessionIntelligence?.getFocusSection?.())
            : false;
        const blocked = getProcedurePlanAdvanceBlock(current, liveSnapshot, hasLiveFocus);
        if (blocked) return blocked;

        const nextStep = Math.min(current.currentStep + 1, current.requests.length - 1);
        const nextPlan = { ...current, currentStep: nextStep };
        setProcedurePlan(nextPlan);

        return {
            status: nextStep === current.currentStep ? 'complete' : 'advanced',
            current_request: nextStep,
            current_step: nextStep,
            request: nextPlan.requests[nextStep],
            step: nextPlan.requests[nextStep]?.title,
            reason,
        };
    }, [analysisContext?.sessionIntelligence, sessionMode, setProcedurePlan]);

    const clearProcedurePlan = useCallback(() => {
        setProcedurePlan(null);
    }, [setProcedurePlan]);

    const optOutProcedurePlan = useCallback(() => {
        procedurePlanOptedOutRef.current = true;
        setProcedurePlan(null);
    }, [setProcedurePlan]);

    // Racing engineer voice conversation. The hook owns mic, WS, and
    // audio playback; it ALSO multiplexes the tool-relay text channel on
    // the same WS — frontend tools listed below are reachable from the
    // backend LLM via JSON text frames.
    const handleVoiceEvent = (event: VoiceEvent) => {
        if (event.kind === 'user_transcript') {
            if (isProcedurePlanOptOutRequest(event.text)) {
                optOutProcedurePlan();
            }
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
            // Backend strips the [emotion] tag before sending the transcript,
            // but fall back to frontend parsing for robustness.
            const { emotion, cleanText } = event.emotion
                ? { emotion: event.emotion as Emotion, cleanText: event.text }
                : extractEmotion(event.text);
            setMessages(prev => prev
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
            broadcastPillMessage(cleanText, { emotion });
            return;
        }
        if (event.kind === 'observation') {
            const plan = buildLiveProcedurePlan(event.data);
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
                ok: event.ok,
                error: event.error,
            });
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

    const startTrackGuide = () => {
        trackGuideRunTokenRef.current += 1;
        setTrackGuideEnabled(true);
    };

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

        return {
            assistant_surface: 'lap_analysis_ai_chat',
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
                user_summary: summaryLoaded,
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
                live_snapshot: liveSnapshot,
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
        livePerformanceAnalystEnabled,
        procedurePlan,
        resolvedSessionId,
        sessionMode,
        TrackGuideEnabled,
        userSummary,
        userSummaryError,
        userSummaryLoading,
    ]);

    const frontendTools = useMemo(
        () => getFrontendToolSchemasForSessionMode(sessionMode),
        [sessionMode],
    );

    const voiceConversation = useVoiceConversation({
        sessionId: resolvedSessionId,
        sessionContext: aiSessionContext,
        onEvent: handleVoiceEvent,
        frontendTools,
        querySchemaScope: QUERY_SCOPE_SCHEMA,
        toolHandlers: createAiCommandRegistry({
            sessionId: resolvedSessionId,
            sessionMode,
            analysisContext,
            sessionIntelligence: analysisContext?.sessionIntelligence,
            opportunityAgentState: opportunityAgentStateRef.current,
            livePerformanceAnalystState: livePerformanceAnalystStateRef.current,
            startTrackGuide,
            setTrackGuideEnabled: setTrackGuideAgentEnabled,
            setLivePerformanceAnalystEnabled: setLivePerformanceAnalystAgentEnabled,
            advanceProcedurePlanStep,
            getProcedurePlan: () => procedurePlanRef.current,
            clearProcedurePlan,
            setProcedurePlan,
            setAgentTagActive: setAgentTag,
            getOpportunityTelemetryRows: () => opportunityForecastRowsRef.current,
            userSummary,
            userSummaryLoading,
            userSummaryError,
            getLabelName,
            getCategoryLabels,
            getCircuitMapById,
            getCircuitMapByTrack,
            displayMap: displayMapInChat,
        }),
    });

    const requestNextPlanStep = useCallback(() => {
        const current = procedurePlanRef.current;
        const advanced = advanceProcedurePlanStep('driver_requested_next_step');
        if (!current || (advanced.status !== 'advanced' && advanced.status !== 'complete')) return;

        voiceConversation.sendObservation({
            source: 'procedure_plan_ui',
            event: 'procedure_plan_next_step_requested',
            goal: current.goal,
            requests: current.requests,
            current_request: advanced.current_request ?? current.currentStep,
            request: advanced.request ?? current.requests[current.currentStep],
        });
    }, [advanceProcedurePlanStep, voiceConversation]);

    const vState = voiceConversation.state;
    const voiceActive = vState === 'listening' || vState === 'speaking';
    const micDisabled = voiceConversation.micDisabled;
    const canOpenFloatingChat = typeof window !== 'undefined'
        && Boolean((window as any).electronAPI?.openFloatingChat);

    useEffect(() => {
        if (sessionMode === 'live') {
            return;
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
        procedurePlanOptedOutRef.current = false;
        clearProcedurePlan();
        setAgentTag('Track Guide', false);
        setAgentTag('Overtake', false);
        setAgentTag('Live Analyst', false);
    }, [clearProcedurePlan, sessionMode, setAgentTag, setLivePerformanceAnalystAgentEnabled, setTrackGuideAgentEnabled]);

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
    }, [generateUniqueId]);

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
    }, [analysisContext?.latestGuidanceMessage, generateUniqueId, TrackGuideEnabled]);

    const welcomeContent = useMemo(() => {
        const selectedSessionName = (analysisContext?.sessionSelected as Record<string, any> | null)?.session_name;
        if (sessionMode === 'recorded') {
            return selectedSessionName
                ? `Recorded session assistant ready for ${selectedSessionName}. Questions will use saved playback, AI analysis, and session metadata.`
                : 'Recorded session assistant ready. Questions will use saved playback, AI analysis, and session metadata.';
        }
        if (sessionMode === 'user_summary') {
            return 'User summary assistant ready. Questions will use saved practice summary and aggregate session history.';
        }
        return 'Live session assistant ready. Questions will use streaming telemetry context.';
    }, [analysisContext?.sessionSelected, sessionMode]);

    useEffect(() => {
        setMessages((previous) => {
            const welcomeMessage: Message = {
                id: 'welcome',
                content: welcomeContent,
                isUser: false,
                timestamp: new Date(),
                kind: 'chat',
            };

            if (previous.length === 0) {
                return [welcomeMessage];
            }

            if (previous[0].id === 'welcome' && previous[0].content !== welcomeContent) {
                return [{ ...previous[0], content: welcomeContent, timestamp: new Date() }, ...previous.slice(1)];
            }

            return previous;
        });
    }, [welcomeContent]);

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
        const sent = voiceConversation.sendUserText(text);
        if (!sent) {
            setMessages(prev => prev.concat({
                id: generateUniqueId('ai'),
                ...(sessionMode === 'recorded'
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
    const sessionModeLabel = sessionMode === 'recorded'
        ? 'Recorded Session'
        : sessionMode === 'user_summary'
            ? 'User Summary'
            : 'Live Session';
    const transcriptLabel = sessionMode === 'recorded'
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
            voiceConversation.start().catch((err) => {
                console.error('Voice conversation failed to start:', err);
            });
        } else {
            voiceConversation.stop();
        }
    };

    const toggleMicDisabled = () => {
        voiceConversation.setMicDisabled(!micDisabled);
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
                    <span className="ai-chat__chip ai-chat__chip--blue">{sessionModeLabel}</span>
                    {environment === 'electron' && (
                        <span className="ai-chat__chip ai-chat__chip--green">Desktop</span>
                    )}
                    {voiceConversation.error && (
                        <span className="ai-chat__chip ai-chat__chip--red" title={voiceConversation.error}>
                            Voice Error
                        </span>
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
                        className={`ai-chat__mic-wave ${useLiveBars ? 'ai-chat__mic-wave--live' : (vState === 'idle' || micDisabled) ? 'ai-chat__mic-wave--idle' : ''}`}
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

                </aside>

                <section className="ai-chat__transcript">
                    <div className="ai-chat__transcript-head">
                        <span className="ai-chat__transcript-title">
                            <span className="ai-chat__eyebrow-dot" />
                            {transcriptLabel}
                        </span>
                        <span className="ai-chat__transcript-time">{clock}</span>
                    </div>

                    {procedurePlan && (
                        <div className="ai-chat__plan" aria-label="Procedure plan">
                            <div className="ai-chat__plan-head">
                                <div>
                                    <span className="ai-chat__plan-kicker">PLAN</span>
                                    <div className="ai-chat__plan-goal">{procedurePlan.goal}</div>
                                </div>
                                <button
                                    type="button"
                                    className="ai-chat__btn ai-chat__btn--blue ai-chat__plan-next"
                                    onClick={requestNextPlanStep}
                                    disabled={procedurePlan.currentStep >= procedurePlan.requests.length - 1}
                                    title="Ask the assistant to move the plan to the next request"
                                >
                                    NEXT REQUEST
                                </button>
                            </div>
                            <ul className="ai-chat__plan-list">
                                {procedurePlan.requests.map((request, index) => {
                                    const isActive = index === procedurePlan.currentStep;
                                    const isDone = index < procedurePlan.currentStep;
                                    const meta = getProcedurePlanRequestMeta(request);
                                    return (
                                        <li
                                            key={`${index}-${request.type}-${request.title}`}
                                            className={[
                                                'ai-chat__plan-step',
                                                isActive ? 'ai-chat__plan-step--active' : '',
                                                isDone ? 'ai-chat__plan-step--done' : '',
                                            ].filter(Boolean).join(' ')}
                                        >
                                            <span className="ai-chat__plan-step-dot" aria-hidden="true" />
                                            <span className="ai-chat__plan-step-text">
                                                <span>{request.title}</span>
                                                {meta && (
                                                    <span className="ai-chat__plan-step-meta">{meta}</span>
                                                )}
                                                {request.detail && (
                                                    <span className="ai-chat__plan-step-detail">{request.detail}</span>
                                                )}
                                            </span>
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    )}

                    <div className="ai-chat__msgs" ref={messagesScrollRef} onScroll={handleMessagesScroll}>
                        {messages.map((message) => {
                            if (message.kind === 'progress' && message.progress) {
                                const progress = Math.round(message.progress.value);
                                const startMarkerValue = Math.max(0, Math.min(100, Number(message.progress.startMarkerValue ?? 0)));
                                return (
                                    <div key={message.id} className="ai-chat__progress">
                                        <div className="ai-chat__progress-head">
                                            <span>{message.progress.label}</span>
                                            <span>{progress}%</span>
                                        </div>
                                        <div
                                            className="ai-chat__progress-track"
                                            role="progressbar"
                                            aria-valuenow={progress}
                                            aria-valuemin={0}
                                            aria-valuemax={100}
                                        >
                                            <div className="ai-chat__progress-fill" style={{ width: `${progress}%` }} />
                                            {message.progress.startMarkerLabel && (
                                                <div className="ai-chat__progress-marker" style={{ left: `${startMarkerValue}%` }}>
                                                    <span>{message.progress.startMarkerLabel}</span>
                                                </div>
                                            )}
                                        </div>
                                        {message.progress.detail && (
                                            <div className="ai-chat__progress-detail">{message.progress.detail}</div>
                                        )}
                                    </div>
                                );
                            }

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

                            const role: 'driver' | 'acla' | 'guidance' = message.isUser
                                ? 'driver'
                                : message.id.includes('guidance') ? 'guidance' : 'acla';
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
                                                {message.mapDisplay && (
                                                    <AiMapToolDisplay display={message.mapDisplay} />
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

            {/* Input row */}
            <div className="ai-chat__input-row">
                <input
                    className="ai-chat__input"
                    placeholder={
                        voiceActive
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
