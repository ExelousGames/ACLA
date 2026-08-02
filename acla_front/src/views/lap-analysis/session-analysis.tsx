import './session-analysis.css';

import { Box, Tabs } from '@radix-ui/themes';
import { ChatBubbleIcon, ChevronLeftIcon, ChevronRightIcon } from '@radix-ui/react-icons';
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
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
    buildAssistantConversationKey,
    resolveAssistantRecordedSessionId,
    resolveAssistantSessionMode,
    type SessionAnalysisAssistantMode,
} from './assistant-session-mode';

const RECORDED_AI_ANALYSIS_TIMEOUT_MS = 120000;

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

type SessionAnalysisAssistantProps = {
    assistantModeOverride?: SessionAnalysisAssistantMode;
};

export const SessionAnalysisAssistant = ({ assistantModeOverride }: SessionAnalysisAssistantProps = {}) => {
    const analysisContext = useContext(AnalysisContext);
    const liveSession = useContext(LiveSessionContext);
    const [isOpen, setIsOpen] = useState(false);
    const assistantSessionId = analysisContext.sessionSelected?.SessionId;
    const assistantSessionMode = resolveAssistantSessionMode({
        assistantModeOverride,
        sessionId: assistantSessionId,
        recordingState: liveSession.recordingState,
    });
    const assistantSessionLabel = assistantSessionMode === 'user_summary'
        ? 'User Summary'
        : assistantSessionMode === 'front_desk'
            ? 'Front Desk'
            : assistantSessionMode === 'recorded'
                ? analysisContext.sessionSelected?.session_name || 'Recorded Session'
                : liveSession.recordingMetadata?.sessionName || 'Live Telemetry';
    const effectiveAssistantSessionId = resolveAssistantRecordedSessionId(
        assistantSessionMode,
        assistantSessionId,
    );
    const assistantConversationKey = buildAssistantConversationKey(assistantSessionMode, effectiveAssistantSessionId);
    const assistantClassName = `main-dashboard-assistant${isOpen ? ' main-dashboard-assistant--open' : ' main-dashboard-assistant--folded'}`;
    const assistantTitleMode = assistantSessionMode === 'user_summary'
        ? 'User Summary'
        : assistantSessionMode === 'recorded'
            ? 'Recorded'
            : assistantSessionMode === 'front_desk'
                ? 'Front Desk'
                : 'Live';

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
                    key={assistantConversationKey}
                    sessionId={effectiveAssistantSessionId}
                    sessionMode={assistantSessionMode}
                    title={assistantSessionMode === 'user_summary'
                        ? 'AI Assistant - User Summary'
                        : assistantSessionMode === 'front_desk'
                            ? 'AI Assistant - Front Desk'
                            : `AI Assistant - ${assistantTitleMode} - ${assistantSessionLabel}`}
                />
            </div>
        </aside>
    );
};

const SessionAnalysis = () => {
    const { activeTab, mapSelected, sessionSelected, setActiveTab } = useContext(AnalysisContext);

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
