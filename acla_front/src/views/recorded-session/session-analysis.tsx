import './session-analysis.css';

import { Box, Tabs, Text } from '@radix-ui/themes';
import React, { useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import {
    OPERATION_COMPONENT_NAMES,
    ObservableOperationComponentHandle,
    useOptionalOperationComponentRefDirectory,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import {
    OperationComponentErrorConstructor,
    ExpertLineGuidanceFailedError,
    LapComparisonFailedError,
    NoRecordedSessionError,
    PerformanceInsightsFailedError,
    RecordedAnalysisFailedError,
    SessionAnalysisFailedError,
    SessionAnalysisComponentError,
    TelemetryDataFailedError,
} from 'contexts/OperationComponentError';
import SessionList from './session-list/session-list';
import LocalIRacingTelemetry from './local-iracing/LocalIRacingTelemetry';
import MapList from './map-list/map-list';
import SessionAnalysisSplit from './sessionAnalysis/session-analysis-split';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { AnalysisContext, AnalysisContextType } from './analysis-context';
import { RecordedSessionDataProvider, useRecordedSessionData, RecordedSessionData } from './data/RecordedSessionDataProvider';
import {
    RecordedAiAnalysisState,
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
    getRecordedAnalysisStateForResult,
} from './recorded-session-analysis';
import { analyzeRecordedTelemetry } from './analyze-recorded-telemetry';
import { getSegmentLabelIds } from '../session-shared/visualization/charts/segmentClassificationDisplay';
import {
    openAnalysisResultsVisualization,
    resolveAnalysisLabel,
} from '../session-shared/visualization/open-analysis-results-visualization';
import {
    createOperationFrom,
    type Operation,
} from 'components/ai-operations';

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

const getRequestFailureMessage = (error: unknown, fallback: string): string => {
    const value = error as any;
    return value?.response?.data?.message
        || value?.data?.message
        || value?.message
        || fallback;
};

const requestSessionAnalysisOperation = async <T,>(
    componentName: string,
    ErrorType: OperationComponentErrorConstructor<SessionAnalysisComponentError>,
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

export interface SessionAnalysisHandle extends ObservableOperationComponentHandle<AnalysisContextType> {
    getSelectedSession(): RacingSessionDetailedInfoDto | null;
    getRecordedSessionData(): RecordedSessionData;
    getMapSelected(): string | null;
    getRecordedAiAnalysis(): RecordedAiAnalysisState;
    getRecordedPlaybackSummary(): ReturnType<typeof createEmptyRecordedPlaybackSummary>;
    runRecordedAiAnalysis(options?: { force?: boolean }): Promise<RecordedAiAnalysisState>;
    requestSessionAnalysis(sessionId?: string): Promise<any>;
    requestPerformanceInsights(sessionId: string | undefined, analysisType?: string): Promise<any>;
    requestLapComparison(sessionIds: string[], metrics?: string[]): Promise<any>;
    requestExpertLineGuidance(sessionId: string | undefined, dataTypes?: string[]): Promise<any>;
    requestTelemetryData(sessionId: string | undefined, dataTypes?: string[]): Promise<any>;
    runRecordedAnalysisForAi(args: Record<string, any>): Operation<RecordedAnalysisAiResult>;
    getRecordedAnalysisForAi(args: Record<string, any>): Operation<RecordedAnalysisAiResult>;
    getRecordedSessionContextForAi(args: Record<string, any>): Operation<RecordedSessionContextAiResult>;
    analyzeTelemetryForAi(args: Record<string, any>): Operation<RecordedTelemetryAnalysisAiResult>;
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
    return (
        <RecordedSessionDataProvider session={sessionSelected} map={mapSelected}>
            <SessionAnalysisStateProvider
                mapSelected={mapSelected}
                setMap={setMap}
                sessionSelected={sessionSelected}
                setSession={setSession}
            >
                {children}
            </SessionAnalysisStateProvider>
        </RecordedSessionDataProvider>
    );
};

const SessionAnalysisStateProvider = ({ children, mapSelected, setMap, sessionSelected, setSession }: {
    children: React.ReactNode;
} & Pick<AnalysisContextType, 'mapSelected' | 'setMap' | 'sessionSelected' | 'setSession'>) => {
    const recordedSessionData = useRecordedSessionData();
    const [activeTab, setActiveTab] = useState('mapLists');
    const [activeVisualizations, setActiveVisualizations] = useState<VisualizationInstance[]>([]);
    const [latestGuidanceMessage, setLatestGuidanceMessage] = useState<string | null>(null);
    const [recordedAiAnalysis, setRecordedAiAnalysis] = useState<RecordedAiAnalysisState>(createIdleRecordedAiAnalysis());
    const [recordedPlaybackSummary, setRecordedPlaybackSummary] = useState(createEmptyRecordedPlaybackSummary());
    const recordedAiAnalysisCacheRef = useRef<Map<string, RecordedAiAnalysisState>>(new Map());
    const activeAnalysisRef = useRef<{
        controller: AbortController;
        promise: Promise<RecordedAiAnalysisState>;
    } | null>(null);

    useLayoutEffect(() => () => {
        activeAnalysisRef.current?.controller.abort();
        activeAnalysisRef.current = null;
    }, [sessionSelected, recordedSessionData]);

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
                OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
                nextState.message!,
            );
        }

        if (activeAnalysisRef.current && !force) return activeAnalysisRef.current.promise;
        activeAnalysisRef.current?.controller.abort();
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

        const controller = new AbortController();
        const promise = Promise.resolve().then(async () => {
            try {
                if (recordedSessionData.sessionId !== sessionId || recordedSessionData.status !== 'ready') {
                    throw new Error(recordedSessionData.message || 'Recorded telemetry is not ready for analysis.');
                }
                const result = await analyzeRecordedTelemetry({
                    sessionId,
                    track: sessionSelected?.map || mapSelected,
                    car: sessionSelected?.car,
                    table: recordedSessionData.table,
                    signal: controller.signal,
                    onProgress: (completed, total) => {
                        if (!controller.signal.aborted) setRecordedAiAnalysis({
                            sessionId,
                            status: 'loading',
                            message: `Analyzing telemetry: ${completed.toLocaleString()} / ${total.toLocaleString()} samples...`,
                            result: cached?.result ?? null,
                        });
                    },
                });
                if (controller.signal.aborted) throw new Error('Recorded analysis was cancelled.');
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
                    message: getRequestFailureMessage(error, 'Failed to run AI segment analysis.'),
                    result: cached?.result ?? null,
                };
                if (!controller.signal.aborted) setRecordedAiAnalysis(nextState);
                throw new RecordedAnalysisFailedError(
                    OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
                    nextState.message!,
                    { cause: error },
                );
            } finally {
                if (activeAnalysisRef.current?.controller === controller) activeAnalysisRef.current = null;
            }
        });
        activeAnalysisRef.current = { controller, promise };
        return promise;
    }, [mapSelected, recordedSessionData, sessionSelected]);

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

    return (
        <AnalysisContext.Provider value={contextValue}>
            {children}
        </AnalysisContext.Provider>
    );
};

type SessionAnalysisProps = { name: string; source?: 'cloud' | 'iracing' };

export const SessionAnalysisContent = ({ name, source = 'cloud' }: SessionAnalysisProps) => {
    const analysisContext = useContext(AnalysisContext);
    const recordedSessionData = useRecordedSessionData();
    const recordedSessionDataRef = useRef(recordedSessionData);
    recordedSessionDataRef.current = recordedSessionData;
    const componentRefs = useOptionalOperationComponentRefDirectory();
    const analysisContextRef = useRef(analysisContext);
    analysisContextRef.current = analysisContext;
    const assistantSnapshotListenersRef = useRef(new Set<() => void>());
    const componentRef = useRef<SessionAnalysisHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getAssistantSnapshot: () => analysisContextRef.current,
            subscribeAssistantSnapshot: (listener) => {
                assistantSnapshotListenersRef.current.add(listener);
                return () => assistantSnapshotListenersRef.current.delete(listener);
            },
            getSelectedSession: () => analysisContextRef.current.sessionSelected,
            getRecordedSessionData: () => recordedSessionDataRef.current,
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
            runRecordedAnalysisForAi: (args) => createOperationFrom(async () => {
                const state = await analysisContextRef.current.runRecordedAiAnalysis({
                    force: args.force === true,
                });
                if (componentRefs && state.result) {
                    await openAnalysisResultsVisualization({
                        directory: componentRefs,
                        managerName: OPERATION_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
                        result: state.result,
                        records: [...recordedSessionDataRef.current.table],
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
            }, 'complete'),
            getRecordedAnalysisForAi: (args) => createOperationFrom(() => compactRecordedAnalysisForAi(
                name,
                analysisContextRef.current.sessionSelected,
                analysisContextRef.current.mapSelected,
                analysisContextRef.current.recordedAiAnalysis,
                getAiAnalysisLimit(args.limit),
                (labelId) => resolveAnalysisLabel(componentRefs, labelId),
            ), 'complete'),
            getRecordedSessionContextForAi: (_args) => createOperationFrom(() => {
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
            }, 'ready'),
            analyzeTelemetryForAi: (args) => createOperationFrom(async () => {
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
                        managerName: OPERATION_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER,
                        result: state.result,
                        records: [...recordedSessionDataRef.current.table],
                    })
                    : { chart_id: null, component_name: null };
                return {
                    status: compact.status,
                    ...(compact.message ? { message: compact.message } : {}),
                    analysis: compact.analysis ?? null,
                    telemetry_stats: null,
                    ...chart,
                };
            }, 'complete'),
        };
    }
    useRegisterOperationComponentRef(componentRef);
    useLayoutEffect(() => {
        assistantSnapshotListenersRef.current.forEach((listener) => listener());
    }, [analysisContext]);

    const { activeTab, mapSelected, sessionSelected, setActiveTab } = analysisContext;

    if (source === 'iracing') return <LocalIRacingTelemetry />;

    return (
        <Tabs.Root className="LiveAnalysisTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
            <Box px="4" pt="4"><Text size="3" weight="bold">Cloud saved</Text></Box>
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

const SessionAnalysis = ({ name, source = 'cloud' }: SessionAnalysisProps) => (
    <SessionAnalysisProvider>
        <SessionAnalysisContent name={name} source={source} />
    </SessionAnalysisProvider>
);

export default SessionAnalysis;
