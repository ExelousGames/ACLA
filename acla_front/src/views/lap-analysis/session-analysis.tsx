import './session-analysis.css';

import { Box, Tabs } from '@radix-ui/themes';
import { ChatBubbleIcon, ChevronLeftIcon, ChevronRightIcon } from '@radix-ui/react-icons';
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import {
    AiChatScreenHandle,
    RECORDED_SCREEN_TOOL_NAMES,
    SCREEN_VISUALIZATION_TOOL_NAMES,
    createAiChatScreenToolHandlers,
    toAiChatJsonRecord,
    toAiChatJsonValue,
    useAiChatScreen,
    useAiChatScreenRegistration,
} from 'contexts/AiChatScreenContext';
import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import SessionAnalysisSplit from './sessionAnalysis/session-analysis-split';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { AnalysisContext } from './analysis-context';
import AiChat from './ai-chat/ai-chat';
import {
    RecordedAiAnalysisState,
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
    getRecordedAnalysisStateForResult,
    normalizeSegmentClassificationResult,
} from './recorded-session-analysis';
import {
    resolveRegisteredAssistantIdentity,
} from './assistant-session-mode';

const RECORDED_AI_ANALYSIS_TIMEOUT_MS = 120000;
const RECORDED_SESSION_TOOL_HANDLERS = createAiChatScreenToolHandlers([
    ...RECORDED_SCREEN_TOOL_NAMES,
    ...SCREEN_VISUALIZATION_TOOL_NAMES,
]);

export const SessionAnalysisProvider = ({ children }: { children: React.ReactNode }) => {
    const [mapSelected, setMap] = useState<string | null>(null);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    const [activeVisualizations, setActiveVisualizations] = useState<VisualizationInstance[]>([]);
    const [latestGuidanceMessage, setLatestGuidanceMessage] = useState<string | null>(null);
    const [recordedAiAnalysis, setRecordedAiAnalysis] = useState<RecordedAiAnalysisState>(createIdleRecordedAiAnalysis());
    const [recordedPlaybackSummary, setRecordedPlaybackSummary] = useState(createEmptyRecordedPlaybackSummary());
    const recordedAiAnalysisCacheRef = useRef<Map<string, RecordedAiAnalysisState>>(new Map());

    const runRecordedAiAnalysis = useCallback(async ({ force = false }: { force?: boolean } = {}): Promise<RecordedAiAnalysisState> => {
        const sessionId = sessionSelected?.SessionId;
        if (!sessionId) {
            const nextState: RecordedAiAnalysisState = {
                ...createIdleRecordedAiAnalysis(null),
                status: 'error',
                message: 'No recorded session is selected.',
            };
            setRecordedAiAnalysis(nextState);
            return nextState;
        }

        const cached = recordedAiAnalysisCacheRef.current.get(sessionId);
        if (cached && !force) {
            setRecordedAiAnalysis(cached);
            return cached;
        }

        setRecordedAiAnalysis({
            sessionId,
            status: 'loading',
            message: 'Running AI segment analysis...',
            result: cached?.result ?? null,
        });

        try {
            const response = await apiService.post('/racing-session/segment-classification', {
                session_id: sessionId,
            }, { timeout: RECORDED_AI_ANALYSIS_TIMEOUT_MS });
            const result = normalizeSegmentClassificationResult(response.data as any, sessionId);
            const nextState: RecordedAiAnalysisState = {
                sessionId,
                result,
                ...getRecordedAnalysisStateForResult(result),
            };
            recordedAiAnalysisCacheRef.current.set(sessionId, nextState);
            setRecordedAiAnalysis(nextState);
            return nextState;
        } catch (error: any) {
            const nextState: RecordedAiAnalysisState = {
                sessionId,
                status: 'error',
                message: error?.data?.message || error?.message || 'Failed to run AI segment analysis.',
                result: cached?.result ?? null,
            };
            setRecordedAiAnalysis(nextState);
            return nextState;
        }
    }, [sessionSelected?.SessionId]);

    const sendGuidanceToChat = useCallback((message: string) => {
        setLatestGuidanceMessage((previous) => previous === message ? previous : message);
    }, []);

    useEffect(() => {
        if (mapSelected !== null) setActiveTab('sessionLists');
        if (sessionSelected !== null) setActiveTab('session');
    }, [mapSelected, sessionSelected]);

    useEffect(() => {
        if (activeTab === 'mapLists') {
            setMap(null);
            setSession(null);
        } else if (activeTab === 'sessionLists') {
            setSession(null);
        }
    }, [activeTab]);

    useEffect(() => {
        const sessionId = sessionSelected?.SessionId || null;
        setRecordedPlaybackSummary(createEmptyRecordedPlaybackSummary(sessionId));
        setRecordedAiAnalysis(sessionId
            ? recordedAiAnalysisCacheRef.current.get(sessionId) || createIdleRecordedAiAnalysis(sessionId)
            : createIdleRecordedAiAnalysis());
    }, [sessionSelected?.SessionId]);

    const contextValue = useMemo(() => ({
        activeTab,
        mapSelected,
        sessionSelected,
        activeVisualizations,
        latestGuidanceMessage,
        recordedAiAnalysis,
        recordedPlaybackSummary,
        setMap,
        setSession,
        setRecordedPlaybackSummary,
        runRecordedAiAnalysis,
        setActiveTab,
        setActiveVisualizations,
        sendGuidanceToChat,
    }), [
        activeTab,
        activeVisualizations,
        latestGuidanceMessage,
        mapSelected,
        recordedAiAnalysis,
        recordedPlaybackSummary,
        runRecordedAiAnalysis,
        sendGuidanceToChat,
        sessionSelected,
    ]);

    return <AnalysisContext.Provider value={contextValue}>{children}</AnalysisContext.Provider>;
};

export const SessionAnalysisAssistant = () => {
    const { activeScreen } = useAiChatScreen();
    const [isOpen, setIsOpen] = useState(false);
    const assistantIdentity = resolveRegisteredAssistantIdentity(activeScreen);
    const assistantClassName = `main-dashboard-assistant${isOpen ? ' main-dashboard-assistant--open' : ' main-dashboard-assistant--folded'}`;

    return (
        <aside className={assistantClassName} aria-label="AI Assistant">
            <button
                type="button"
                className="main-dashboard-assistant__toggle"
                onClick={() => setIsOpen((open) => !open)}
                aria-controls="main-dashboard-assistant-body"
                aria-expanded={isOpen}
                aria-label={isOpen ? 'Fold AI Assistant' : 'Open AI Assistant'}
                title={isOpen ? 'Fold AI Assistant' : 'Open AI Assistant'}
            >
                {isOpen ? <ChevronRightIcon /> : <ChevronLeftIcon />}
                <ChatBubbleIcon />
            </button>
            <div id="main-dashboard-assistant-body" className="main-dashboard-assistant__body" aria-hidden={!isOpen}>
                <AiChat
                    key={assistantIdentity.conversationKey}
                    sessionId={assistantIdentity.sessionId}
                    sessionMode={assistantIdentity.sessionMode}
                    title={assistantIdentity.title}
                />
            </div>
        </aside>
    );
};

const SessionAnalysis = () => {
    const analysisContext = useContext(AnalysisContext);
    const analysisContextRef = useRef(analysisContext);
    analysisContextRef.current = analysisContext;
    const componentRef = useRef<AiChatScreenHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getAiContext: () => {
                const current = analysisContextRef.current;
                const selectedSession = current.sessionSelected;
                const recorded = current.activeTab === 'session' && Boolean(selectedSession?.SessionId);

                if (!recorded) {
                    return toAiChatJsonRecord({
                        screen_kind: 'front_desk',
                        active_analysis_area: current.activeTab,
                        selected_map_id: current.mapSelected,
                        assistance_scope: 'General navigation, onboarding, map selection, and session selection.',
                        capabilities: {
                            screen_tools: false,
                            general_assistance: true,
                        },
                    });
                }

                return toAiChatJsonRecord({
                    screen_kind: 'recorded_session',
                    active_analysis_area: current.activeTab,
                    selected_map_id: current.mapSelected || selectedSession?.map || null,
                    selected_session: {
                        id: selectedSession?.SessionId || null,
                        name: selectedSession?.session_name || null,
                        map: selectedSession?.map || current.mapSelected || null,
                        car: selectedSession?.car || null,
                    },
                    recorded_session: {
                        ai_analysis: {
                            status: current.recordedAiAnalysis.status,
                            message: current.recordedAiAnalysis.message || null,
                            session_id: current.recordedAiAnalysis.sessionId,
                            samples_analyzed: current.recordedAiAnalysis.result?.samples_analyzed || 0,
                            result_ready: Boolean(current.recordedAiAnalysis.result),
                        },
                        playback: toAiChatJsonValue(current.recordedPlaybackSummary),
                    },
                    analysis_actions: {
                        run_ai_analysis: true,
                        read_ai_analysis: true,
                        read_recorded_context: true,
                    },
                    visualization_controls: {
                        active: current.activeVisualizations.map(({ id, type }) => ({ id, type })),
                    },
                });
            },
            getToolHandlers: () => {
                const current = analysisContextRef.current;
                return current.activeTab === 'session' && current.sessionSelected?.SessionId
                    ? RECORDED_SESSION_TOOL_HANDLERS
                    : {};
            },
        };
    }

    const { activeTab, mapSelected, sessionSelected, setActiveTab } = analysisContext;
    const isRecordedScreen = activeTab === 'session' && Boolean(sessionSelected?.SessionId);
    const registration = useMemo(() => ({
        screenId: isRecordedScreen ? 'recorded-session' : 'front-desk',
        assistantMode: isRecordedScreen ? 'recorded' as const : 'front_desk' as const,
        pillLabel: isRecordedScreen
            ? sessionSelected?.session_name || 'Recorded Session'
            : 'Front Desk',
        ...(isRecordedScreen && sessionSelected?.SessionId
            ? { recordedSessionId: sessionSelected.SessionId }
            : {}),
        componentRef,
        getPillInfo: () => isRecordedScreen
            ? {
                title: sessionSelected?.session_name || 'Recorded Session',
                description: 'Selected recording, playback, AI analysis, and visualization workspace.',
                status: analysisContext.recordedAiAnalysis.status === 'error'
                    ? { label: 'Analysis error', tone: 'error' as const }
                    : analysisContext.recordedAiAnalysis.status === 'loading'
                        ? { label: 'Analyzing', tone: 'info' as const }
                        : { label: 'Ready', tone: 'success' as const },
                facts: [
                    { label: 'Track', value: sessionSelected?.map || mapSelected || '—' },
                    { label: 'Car', value: sessionSelected?.car || '—' },
                    { label: 'Samples', value: analysisContext.recordedPlaybackSummary.sampleCount.toLocaleString() },
                    { label: 'Playback', value: `${analysisContext.recordedPlaybackSummary.playbackTimeSeconds.toFixed(1)}s` },
                ],
            }
            : {
                title: 'Front Desk',
                description: 'General help for navigation, maps, and choosing a recorded session.',
                status: { label: 'General assistance', tone: 'info' as const },
                facts: [
                    { label: 'Area', value: activeTab === 'sessionLists' ? 'Recorded sessions' : 'Circuit maps' },
                    { label: 'Selected map', value: mapSelected || 'None' },
                ],
            },
    }), [
        activeTab,
        analysisContext.recordedAiAnalysis.status,
        analysisContext.recordedPlaybackSummary.playbackTimeSeconds,
        analysisContext.recordedPlaybackSummary.sampleCount,
        isRecordedScreen,
        mapSelected,
        sessionSelected?.SessionId,
        sessionSelected?.car,
        sessionSelected?.map,
        sessionSelected?.session_name,
    ]);
    useAiChatScreenRegistration(registration);

    return (
        <Tabs.Root className="LiveAnalysisTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
            <Tabs.List className="live-analysis-tablists" justify="start">
                <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                {mapSelected === null ? null : <Tabs.Trigger value="sessionLists">{mapSelected}</Tabs.Trigger>}
                {sessionSelected === null ? null : <Tabs.Trigger value="session">Session {sessionSelected.session_name}</Tabs.Trigger>}
            </Tabs.List>
            <Box className="live-analysis-container">
                <Tabs.Content className="TabContent" value="mapLists"><MapList /></Tabs.Content>
                <Tabs.Content className="TabContent" value="sessionLists"><SessionList /></Tabs.Content>
                <Tabs.Content className="TabContent" value="session"><SessionAnalysisSplit /></Tabs.Content>
            </Box>
        </Tabs.Root>
    );
};

export default SessionAnalysis;
