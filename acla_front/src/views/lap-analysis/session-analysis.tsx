import './session-analysis.css';

import { Box, Tabs } from '@radix-ui/themes';
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    NamedAiToolComponentHandle,
    useOptionalAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    AiToolComponentErrorConstructor,
    ExpertLineGuidanceFailedError,
    LapComparisonFailedError,
    NoRecordedSessionError,
    PerformanceInsightsFailedError,
    RecordedAnalysisFailedError,
    SessionAnalysisFailedError,
    SessionAnalysisComponentError,
    TelemetryDataFailedError,
} from 'contexts/AiToolComponentError';
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
import { getSegmentLabelIds } from './visualization/charts/segmentClassificationDisplay';
import {
    openAnalysisResultsVisualization,
    resolveAnalysisLabel,
} from './visualization/open-analysis-results-visualization';
import {
    createAiToolOperationFrom,
    type AiToolOperation,
} from 'components/ai-engineering-tools';

export type RecordedAnalysisAiResult = {
    status: unknown;
    message?: unknown;
    session_id: unknown;
    session_name: unknown;
    map: unknown;
    car: unknown;
    analysis: unknown;
};

export type RecordedSessionContextAiResult = {
    status: 'ready';
    session_id: string;
    track: unknown;
    car: unknown;
};

export type RecordedTelemetryAnalysisAiResult = {
    status: unknown;
    message?: unknown;
    analysis: unknown;
    telemetry_stats: null;
    chart_id: string | null;
    component_name: string | null;
};

const RECORDED_AI_ANALYSIS_TIMEOUT_MS = 120000;

const getRequestFailureMessage = (error: unknown, fallback: string): string => {
    const value = error as any;
    return value?.response?.data?.message
        || value?.data?.message
        || value?.message
        || fallback;
};

const requestSessionAnalysisOperation = async <T,>(
    componentName: string,
    ErrorType: AiToolComponentErrorConstructor<SessionAnalysisComponentError>,
    fallbackMessage: string,
    request: () => Promise<T>,
): Promise<T> => {
    try {
        return await request();
    } catch (error) {
        if (error instanceof SessionAnalysisComponentError) throw error;
        throw new ErrorType(
            componentName,
            getRequestFailureMessage(error, fallbackMessage),
            { cause: error },
        );
    }
};

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
    runRecordedAnalysisForAi(args: Record<string, any>): AiToolOperation<RecordedAnalysisAiResult>;
    getRecordedAnalysisForAi(args: Record<string, any>): AiToolOperation<RecordedAnalysisAiResult>;
    getRecordedSessionContextForAi(args: Record<string, any>): AiToolOperation<RecordedSessionContextAiResult>;
    analyzeTelemetryForAi(args: Record<string, any>): AiToolOperation<RecordedTelemetryAnalysisAiResult>;
}

const getAiAnalysisLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, 50) : 20;
};

const compactRecordedAnalysisForAi = (
    componentName: string,
    selected: RacingSessionDetailedInfoDto | null,
    mapSelected: string | null,
    state: RecordedAiAnalysisState,
    limit: number,
    getLabelName: (labelId: string) => string | undefined,
): RecordedAnalysisAiResult => {
    if (!selected?.SessionId) {
        throw new NoRecordedSessionError(componentName, 'No recorded session is selected.');
    }
    if (state.status === 'error') {
        throw new RecordedAnalysisFailedError(
            componentName,
            state.message || 'Recorded-session analysis failed.',
        );
    }
    const result = state.result;
    return {
        status: state.status,
        ...(state.message ? { message: state.message } : {}),
        session_id: selected.SessionId,
        session_name: selected.session_name || null,
        map: selected.map || mapSelected,
        car: selected.car || null,
        analysis: result ? {
            status: result.status,
            session_id: result.session_id,
            samples_analyzed: result.samples_analyzed,
            segments: result.segments.slice(0, limit).map((segment) => ({
                id: segment.id ?? null,
                start_index: segment.start_index,
                end_index: segment.end_index,
                track_section: segment.track_section
                    ? getLabelName(segment.track_section) || segment.track_section
                    : null,
                labels: getSegmentLabelIds(segment)
                    .map((labelId) => getLabelName(labelId) || labelId),
                ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
            })),
        } : null,
    };
};

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
            throw new NoRecordedSessionError(
                AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
                nextState.message!,
            );
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
            throw new RecordedAnalysisFailedError(
                AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
                nextState.message!,
                { cause: error },
            );
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
    const componentRefs = useOptionalAiToolComponentRefDirectory();
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
            requestSessionAnalysis: (sessionId) => requestSessionAnalysisOperation(
                name,
                SessionAnalysisFailedError,
                'Failed to load the session analysis.',
                () => apiService.post('/racing-session/detailed-info', { id: sessionId }),
            ),
            requestPerformanceInsights: (sessionId, analysisType = 'comprehensive') => requestSessionAnalysisOperation(
                name,
                PerformanceInsightsFailedError,
                'Failed to load performance insights.',
                () => apiService.post('/ai/performance-analysis', {
                    session_id: sessionId,
                    analysis_type: analysisType,
                }),
            ),
            requestLapComparison: (sessionIds, metrics = ['lap_times']) => requestSessionAnalysisOperation(
                name,
                LapComparisonFailedError,
                'Failed to compare lap times.',
                () => apiService.post('/racing-session/compare', {
                    session_ids: sessionIds,
                    metrics,
                }),
            ),
            requestExpertLineGuidance: (sessionId, dataTypes = ['speed', 'acceleration', 'braking', 'steering']) => requestSessionAnalysisOperation(
                name,
                ExpertLineGuidanceFailedError,
                'Failed to load expert-line guidance.',
                () => apiService.post('/ai/expert-line-guidance', {
                    session_id: sessionId,
                    data_types: dataTypes,
                }),
            ),
            requestTelemetryData: (sessionId, dataTypes = ['speed', 'acceleration']) => requestSessionAnalysisOperation(
                name,
                TelemetryDataFailedError,
                'Failed to load telemetry data.',
                () => apiService.post('/racing-session/telemetry', {
                    session_id: sessionId,
                    data_types: dataTypes,
                }),
            ),
            runRecordedAnalysisForAi: (args) => createAiToolOperationFrom(async () => {
                const state = await analysisContextRef.current.runRecordedAiAnalysis({
                    force: args.force === true,
                });
                if (componentRefs && state.result) {
                    await openAnalysisResultsVisualization({
                        directory: componentRefs,
                        managerName: AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
                        result: state.result,
                        records: analysisContextRef.current.sessionSelected?.data ?? [],
                    });
                }
                return compactRecordedAnalysisForAi(
                    name,
                    analysisContextRef.current.sessionSelected,
                    analysisContextRef.current.mapSelected,
                    state,
                    getAiAnalysisLimit(args.limit),
                    (labelId) => resolveAnalysisLabel(componentRefs, labelId),
                );
            }),
            getRecordedAnalysisForAi: (args) => createAiToolOperationFrom(() => compactRecordedAnalysisForAi(
                name,
                analysisContextRef.current.sessionSelected,
                analysisContextRef.current.mapSelected,
                analysisContextRef.current.recordedAiAnalysis,
                getAiAnalysisLimit(args.limit),
                (labelId) => resolveAnalysisLabel(componentRefs, labelId),
            )),
            getRecordedSessionContextForAi: (_args) => createAiToolOperationFrom(() => {
                const selected = analysisContextRef.current.sessionSelected;
                if (!selected?.SessionId) {
                    throw new NoRecordedSessionError(name, 'No recorded session is selected.');
                }
                return {
                    status: 'ready',
                    session_id: selected.SessionId,
                    track: selected.map || analysisContextRef.current.mapSelected,
                    car: selected.car || null,
                };
            }),
            analyzeTelemetryForAi: (args) => createAiToolOperationFrom(async () => {
                const state = await analysisContextRef.current.runRecordedAiAnalysis({
                    force: args.force === true,
                });
                const compact = compactRecordedAnalysisForAi(
                    name,
                    analysisContextRef.current.sessionSelected,
                    analysisContextRef.current.mapSelected,
                    state,
                    getAiAnalysisLimit(args.limit),
                    (labelId) => resolveAnalysisLabel(componentRefs, labelId),
                );
                const chart = componentRefs && state.result
                    ? await openAnalysisResultsVisualization({
                        directory: componentRefs,
                        managerName: AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
                        result: state.result,
                        records: analysisContextRef.current.sessionSelected?.data ?? [],
                    })
                    : { chart_id: null, component_name: null };
                return {
                    status: compact.status,
                    ...(compact.message ? { message: compact.message } : {}),
                    analysis: compact.analysis ?? null,
                    telemetry_stats: null,
                    ...chart,
                };
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
