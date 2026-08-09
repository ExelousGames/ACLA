import './session-analysis.css';

import { Box, Tabs } from '@radix-ui/themes';
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    NamedAiToolComponentHandle,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import SessionList from './session-list/session-list';
import MapList from './map-list/map-list';
import SessionAnalysisSplit from './sessionAnalysis/session-analysis-split';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { AnalysisContext } from './analysis-context';
import {
    RecordedAiAnalysisState,
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
    getRecordedAnalysisStateForResult,
    normalizeSegmentClassificationResult,
} from './recorded-session-analysis';

const RECORDED_AI_ANALYSIS_TIMEOUT_MS = 120000;
export interface SessionAnalysisHandle extends NamedAiToolComponentHandle {
    getSelectedSession(): RacingSessionDetailedInfoDto | null;
    getMapSelected(): string | null;
    getRecordedAiAnalysis(): RecordedAiAnalysisState;
    getRecordedPlaybackSummary(): ReturnType<typeof createEmptyRecordedPlaybackSummary>;
    runRecordedAiAnalysis(options?: { force?: boolean }): Promise<RecordedAiAnalysisState>;
    requestSessionAnalysis(sessionId?: string): Promise<any>;
    requestPerformanceInsights(sessionId: string | undefined, analysisType?: string): Promise<any>;
    requestLapComparison(sessionIds: string[], metrics?: string[]): Promise<any>;
    requestExpertLineGuidance(sessionId: string | undefined, dataTypes?: string[]): Promise<any>;
    requestTelemetryData(sessionId: string | undefined, dataTypes?: string[]): Promise<any>;
}

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

const SessionAnalysis = ({ name }: { name: string }) => {
    const analysisContext = useContext(AnalysisContext);
    const analysisContextRef = useRef(analysisContext);
    analysisContextRef.current = analysisContext;
    const componentRef = useRef<SessionAnalysisHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getSelectedSession: () => analysisContextRef.current.sessionSelected,
            getMapSelected: () => analysisContextRef.current.mapSelected,
            getRecordedAiAnalysis: () => analysisContextRef.current.recordedAiAnalysis,
            getRecordedPlaybackSummary: () => analysisContextRef.current.recordedPlaybackSummary,
            runRecordedAiAnalysis: (options) => analysisContextRef.current.runRecordedAiAnalysis(options),
            requestSessionAnalysis: (sessionId) => apiService.post('/racing-session/detailed-info', { id: sessionId }),
            requestPerformanceInsights: (sessionId, analysisType = 'comprehensive') => apiService.post('/ai/performance-analysis', {
                session_id: sessionId,
                analysis_type: analysisType,
            }),
            requestLapComparison: (sessionIds, metrics = ['lap_times']) => apiService.post('/racing-session/compare', {
                session_ids: sessionIds,
                metrics,
            }),
            requestExpertLineGuidance: (sessionId, dataTypes = ['speed', 'acceleration', 'braking', 'steering']) => apiService.post('/ai/expert-line-guidance', {
                session_id: sessionId,
                data_types: dataTypes,
            }),
            requestTelemetryData: (sessionId, dataTypes = ['speed', 'acceleration']) => apiService.post('/racing-session/telemetry', {
                session_id: sessionId,
                data_types: dataTypes,
            }),
        };
    }
    useRegisterAiToolComponentRef(name, componentRef.current);

    const { activeTab, mapSelected, sessionSelected, setActiveTab } = analysisContext;

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
