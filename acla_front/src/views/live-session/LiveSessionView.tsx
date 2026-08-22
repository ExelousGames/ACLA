import React, { useContext, useLayoutEffect, useRef } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    ComponentRefUnavailableError,
    ObservableAiToolComponentHandle,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
    useOptionalAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    AiToolError,
    NoCornerDataError,
    NoTelemetryForScopeError,
    TelemetryAnalysisFailedError,
    InvalidToolCallError,
} from 'views/lap-analysis/ai-chat/ai-tool-base';
import { BaselineCollectionNotStartedError } from 'contexts/AiToolComponentError';
import apiService from 'services/api.service';
import type {
    BaselineCollectionHandle,
    BaselineCollectionPayload,
} from './BaselineCollection';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import {
    normalizeSegmentClassificationResult,
    type SegmentClassificationResult,
} from 'views/lap-analysis/recorded-session-analysis';
import { getSegmentLabelIds } from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import { openAnalysisResultsVisualization } from 'views/lap-analysis/visualization/open-analysis-results-visualization';
import { getVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import { LiveSessionContext, LiveSessionProvider } from './LiveSessionContext';
import LiveSessionGameStatus, { LIVE_SESSION_GAME_LABELS } from './LiveSessionGameStatus';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import LiveAnalysisSessionRecording from 'views/lap-analysis/liveAnalysisSessionRecording';
import LiveSessionDetectionManager from 'views/lap-analysis/LiveSessionDetectionManager';
import { RecordingState } from 'views/lap-analysis/recording-state';
import { getTelemetryLap } from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    createTelemetryScopeCollector,
    reduceTelemetrySamples,
} from 'views/lap-analysis/session-intelligence/telemetry-query';
import type {
    CornerLookahead,
    QueryResult,
    QueryScope,
    ReduceOp,
    TelemetryQuery,
} from 'views/lap-analysis/session-intelligence/types';
import type {
    QueryTelemetryMetricArguments,
    QueryTelemetryMetricResult,
    TelemetryMetricReduce,
} from 'views/lap-analysis/ai-chat/ai-command-registry';
import type { LiveSessionAnalysisResultPage } from './live-session-analysis-results';
import type { LiveSessionRuntime, LiveSessionSnapshot } from './live-session-types';
import type { LiveEventLogHandle } from './LiveEventLog';
import type { EventSearchParams } from './event-log/EventLog';
import { liveTelemetryStore } from './live-telemetry-store';
import {
    createAiToolOperation,
    createAiToolOperationFrom,
    type AiToolOperation,
} from 'components/ai-engineering-tools';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';
const LIVE_EVENT_LOG_COMPONENT_NAME = getVisualizationComponentName('event-log');

export type LiveEventLogAiResult = { status: 'complete'; events: unknown[] };
export type LiveNextCornerAiResult = {
    status: 'complete';
    corner: { name: string; track_position: number; distance_ahead: number };
};
export type LiveBaselineRestartAiResult = {
    status: 'complete';
    progress_percent: 0;
    message: string;
};
export type LiveBaselineAnalysisAiResult = {
    status: 'ready' | 'empty';
};
export type LiveTelemetryAnalysisAiResult = {
    status: 'ready' | 'empty';
    message: string;
    analysis: unknown;
    telemetry_stats: { row_count: number; field_count: number };
    chart_id: string | null;
    component_name: string | null;
};
export interface LiveSessionHandle extends ObservableAiToolComponentHandle<LiveSessionRuntime> {
    getRecordingState(): RecordingState;
    getCurrentTelemetry(): Record<string, any>;
    queryTelemetryMetric<TReduce extends ReduceOp>(args: TelemetryQuery<TReduce>): Promise<QueryResult<TReduce>>;
    getTelemetryForScope(scope: QueryScope): Promise<Record<string, any>[]>;
    getEventLog(args: Record<string, any>): any[];
    getNextCorner(): CornerLookahead | null;
    getLiveSessionSnapshot(): LiveSessionSnapshot;
    getLatestAnalysisResultPage(): LiveSessionAnalysisResultPage | null;
    queryTelemetryMetricForAi<TReduce extends TelemetryMetricReduce>(
        args: QueryTelemetryMetricArguments<TReduce>,
    ): AiToolOperation<QueryTelemetryMetricResult<TReduce>>;
    getEventLogForAi(args: Record<string, any>): AiToolOperation<LiveEventLogAiResult>;
    getNextCornerForAi(): AiToolOperation<LiveNextCornerAiResult>;
    collectLiveBaselineForAi(args: Record<string, any>): AiToolOperation<BaselineCollectionPayload>;
    restartLiveBaselineForAi(): AiToolOperation<LiveBaselineRestartAiResult>;
    analyzeLiveRecordedAnalysisForAi(args: Record<string, any>): AiToolOperation<LiveBaselineAnalysisAiResult>;
    analyzeTelemetryForAi(args: Record<string, any>): AiToolOperation<LiveTelemetryAnalysisAiResult>;
}

const hasExactKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean => {
    const actualKeys = Object.keys(value);
    return actualKeys.length === keys.length && keys.every((key) => (
        Object.prototype.hasOwnProperty.call(value, key)
    ));
};

const isFiniteNumber = (value: unknown): value is number => (
    typeof value === 'number' && Number.isFinite(value)
);

const isQueryScope = (value: unknown): value is QueryScope => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
    const scope = value as Record<string, unknown>;
    switch (scope.type) {
        case 'now':
            return hasExactKeys(scope, ['type']);
        case 'last_seconds':
            return hasExactKeys(scope, ['type', 'seconds'])
                && isFiniteNumber(scope.seconds);
        case 'event':
            return hasExactKeys(scope, ['type', 'eventType', 'which'])
                && ['CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'].includes(scope.eventType as string)
                && (scope.which === 'last' || scope.which === 'current');
        case 'lap':
            return hasExactKeys(scope, ['type', 'lap'])
                && (scope.lap === 'current' || scope.lap === 'last'
                    || (isFiniteNumber(scope.lap) && Number.isInteger(scope.lap)));
        case 'range':
            return hasExactKeys(scope, ['type', 'start', 'end'])
                && isFiniteNumber(scope.start) && Number.isInteger(scope.start)
                && isFiniteNumber(scope.end) && Number.isInteger(scope.end);
        default:
            return false;
    }
};

const validateTelemetryMetricArguments = <TReduce extends TelemetryMetricReduce>(
    args: QueryTelemetryMetricArguments<TReduce>,
): QueryTelemetryMetricArguments<TReduce> => {
    const value = args as unknown as Record<string, unknown>;
    const fields = value?.fields;
    const validFields = Array.isArray(fields)
        && fields.length > 0
        && fields.every((field) => (
            typeof field === 'string'
            && field.length > 0
            && field.trim() === field
        ));
    if (!value || typeof value !== 'object' || Array.isArray(value)
        || !hasExactKeys(value, ['fields', 'scope', 'reduce'])
        || !validFields
        || !isQueryScope(value.scope)
        || !['avg', 'min', 'max', 'stats'].includes(value.reduce as string)) {
        throw new InvalidToolCallError(
            'query_telemetry_metric requires nonempty string fields, a valid scope, and reduce set to avg, min, max, or stats.',
        );
    }
    return args;
};

const getAiLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, 50) : 20;
};

const getTelemetryStats = (rows: Record<string, any>[]) => ({
    row_count: rows.length,
    field_count: Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).length,
});

const compactClassification = (result: SegmentClassificationResult, limit: number) => ({
    status: result.status,
    session_id: result.session_id,
    samples_analyzed: result.samples_analyzed,
    segments: result.segments.slice(0, limit).map((segment) => ({
        id: segment.id ?? null,
        start_index: segment.start_index,
        end_index: segment.end_index,
        track_section: segment.track_section ?? null,
        labels: getSegmentLabelIds(segment),
        ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
    })),
});

const getBaselineHandle = async (
    directory: AiToolComponentRefDirectory | null,
): Promise<BaselineCollectionHandle> => {
    if (!directory) {
        throw new ComponentRefUnavailableError(
            AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
            'The active dashboard component-ref directory is unavailable.',
        );
    }
    const existing = directory.findComponentRef<BaselineCollectionHandle>(
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    )?.current;
    if (existing) return existing;
    const manager = resolveNamedComponentHandle<VisualizationManagerHandle>(
        directory,
        AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
    );
    manager.requestVisualization({
        name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
        type: 'baseline-collection',
    });
    return awaitNamedComponentHandle<BaselineCollectionHandle>(
        directory,
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    );
};

const LimitedLiveWorkspace = ({ game }: { game: Exclude<DesktopGame, 'acc'> }) => (
    <div
        className="live-session-limited-workspace"
        data-testid="limited-live-workspace"
        role="region"
        aria-label={`${LIVE_SESSION_GAME_LABELS[game]} limited live workspace`}
    >
        <span className="live-session-limited-workspace__eyebrow">Limited workspace</span>
        <h2>{LIVE_SESSION_GAME_LABELS[game]}</h2>
        <p>
            This session keeps the selected game locked, but ACC telemetry and recording controls
            are not available for this simulator yet.
        </p>
    </div>
);

export const LiveSessionContent = ({ name }: { name: string }) => {
    const liveSession = useContext(LiveSessionContext);
    const componentRefs = useOptionalAiToolComponentRefDirectory();
    const componentRefsRef = useRef(componentRefs);
    componentRefsRef.current = componentRefs;
    const liveSessionRef = useRef(liveSession);
    liveSessionRef.current = liveSession;
    const assistantSnapshotListenersRef = useRef(new Set<() => void>());
    const componentRef = useRef<LiveSessionHandle | null>(null);

    const getMountedEventLog = () => componentRefsRef.current
        ?.findComponentRef<LiveEventLogHandle>(LIVE_EVENT_LOG_COMPONENT_NAME)
        ?.current ?? null;
    const findLiveEvents = (args: Record<string, any>) => {
        const eventLog = getMountedEventLog();
        if (!eventLog) return [];
        return eventLog.findEvents({
            eventType: args.eventType ?? args.event_type,
            scope: args.scope ?? 'last',
            n: args.n,
        } as EventSearchParams);
    };
    const resolveLiveTelemetryScope = (
        scope: QueryScope,
    ): Exclude<QueryScope, { type: 'event' }> | null => {
        if (scope.type !== 'event') return scope;
        const matches = findLiveEvents({
            eventType: scope.eventType,
            scope: scope.which === 'current' ? 'lap_current' : 'last',
        });
        const event = matches[matches.length - 1];
        return event
            ? { type: 'range', start: event.startSampleIdx, end: event.endSampleIdx + 1 }
            : null;
    };
    const getTelemetryForLiveScope = async (scope: QueryScope) => {
        const resolvedScope = resolveLiveTelemetryScope(scope);
        if (!resolvedScope) return [];

        const runtime = liveSessionRef.current;
        const collector = createTelemetryScopeCollector(
            resolvedScope,
            getTelemetryLap(liveTelemetryStore.getSnapshot().currentTelemetry),
        );
        await runtime.streamRecordedTelemetry((rows) => collector.addRows(rows));
        return collector.getRows();
    };
    const queryLiveTelemetry = async <TReduce extends ReduceOp>(args: TelemetryQuery<TReduce>) => {
        const samples = await getTelemetryForLiveScope(args.scope);
        return reduceTelemetrySamples(samples, args.fields, args.reduce);
    };
    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getAssistantSnapshot: () => liveSessionRef.current,
            subscribeAssistantSnapshot: (listener) => {
                assistantSnapshotListenersRef.current.add(listener);
                return () => assistantSnapshotListenersRef.current.delete(listener);
            },
            getRecordingState: () => liveSessionRef.current.recordingState,
            getCurrentTelemetry: () => liveTelemetryStore.getSnapshot().currentTelemetry,
            queryTelemetryMetric: (args) => queryLiveTelemetry(args),
            getTelemetryForScope: (scope) => getTelemetryForLiveScope(scope),
            getEventLog: (args) => findLiveEvents(args),
            getNextCorner: () => liveSessionRef.current.getNextCorner(),
            getLiveSessionSnapshot: () => liveSessionRef.current.getLiveSessionSnapshot(),
            getLatestAnalysisResultPage: () => {
                const pages = liveSessionRef.current.analysisResultPages;
                return pages[pages.length - 1] ?? null;
            },
            queryTelemetryMetricForAi: (args) => createAiToolOperationFrom(async () => {
                const query = validateTelemetryMetricArguments(args);
                return {
                    status: 'ready' as const,
                    data: await queryLiveTelemetry(query),
                };
            }),
            getEventLogForAi: (args) => createAiToolOperationFrom(() => ({
                status: 'complete',
                events: findLiveEvents(args),
            })),
            getNextCornerForAi: () => createAiToolOperationFrom(() => {
                const corner = liveSessionRef.current.getNextCorner();
                if (!corner) throw new NoCornerDataError('No upcoming corner data is available.');
                return {
                    status: 'complete',
                    corner: {
                        name: corner.name,
                        track_position: corner.trackPosition,
                        distance_ahead: corner.distanceAhead,
                    },
                };
            }),
            collectLiveBaselineForAi: (args) => {
                const timeoutSeconds = Number(args.timeout_seconds);
                const options = {
                    timeoutMs: Number.isFinite(timeoutSeconds) && timeoutSeconds > 0
                        ? timeoutSeconds * 1000
                        : 600000,
                };
                const mountedHandle = componentRefs?.findComponentRef<BaselineCollectionHandle>(
                    AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                )?.current;
                if (mountedHandle) {
                    return createAiToolOperation(mountedHandle.startCollection(options).result);
                }

                return createAiToolOperationFrom<BaselineCollectionPayload>(async () => {
                    const handle = await getBaselineHandle(componentRefs);
                    return handle.startCollection(options).result;
                });
            },
            restartLiveBaselineForAi: () => createAiToolOperationFrom(async () => {
                const handle = componentRefs?.findComponentRef<BaselineCollectionHandle>(
                    AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                )?.current;
                if (!handle) {
                    throw new BaselineCollectionNotStartedError(
                        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                        'Baseline collection is not in progress. Start a new collection instead.',
                    );
                }
                const restart = handle.restartCollection();
                const restarted = await restart.result;
                if (restarted instanceof Error) return restarted;
                return {
                    status: 'complete',
                    progress_percent: 0,
                    message: 'Baseline collection restart completed.',
                };
            }),
            analyzeLiveRecordedAnalysisForAi: (args) => createAiToolOperationFrom(async () => {
                const handle = await getBaselineHandle(componentRefs);
                const analysis = await handle.requestAnalysis(args).result;
                if (analysis instanceof Error) return analysis;
                return { status: analysis.status };
            }),
            analyzeTelemetryForAi: (args) => createAiToolOperationFrom(async () => {
                const rows = await getTelemetryForLiveScope(args.scope);
                if (rows.length === 0) {
                    throw new NoTelemetryForScopeError('No telemetry rows matched the requested scope.');
                }
                try {
                    const snapshot = liveSessionRef.current.getLiveSessionSnapshot();
                    const response = await apiService.post('/racing-session/analyze-live-recorded-analysis', {
                        track: snapshot.track,
                        car: snapshot.car,
                        baseline_lap: snapshot.current_lap,
                        records: rows,
                    }, { timeout: 120000 });
                    const result = normalizeSegmentClassificationResult(
                        response.data as any,
                        `live-scope-${Date.now()}`,
                    );
                    const chart = componentRefs
                        ? await openAnalysisResultsVisualization({
                            directory: componentRefs,
                            managerName: AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
                            result,
                            records: rows,
                        })
                        : { chart_id: null, component_name: null };
                    return {
                        status: result.segments.length > 0 ? 'ready' : 'empty',
                        message: result.segments.length > 0
                            ? 'Telemetry analysis is ready.'
                            : 'Telemetry analysis found no classified segments.',
                        analysis: compactClassification(result, getAiLimit(args.limit)),
                        telemetry_stats: getTelemetryStats(rows),
                        ...chart,
                    };
                } catch (error) {
                    if (error instanceof AiToolError) throw error;
                    throw new TelemetryAnalysisFailedError(
                        error instanceof Error && error.message
                            ? error.message
                            : 'Failed to analyze telemetry.',
                        { cause: error },
                    );
                }
            }),
        };
    }
    useRegisterAiToolComponentRef(componentRef);
    useLayoutEffect(() => {
        assistantSnapshotListenersRef.current.forEach((listener) => listener());
    }, [liveSession]);


    const { restorationError, sessionGame } = liveSession;

    return (
        <section className="live-session-view" aria-label="Live Session">
            <LiveSessionDetectionManager />
            <LiveAnalysisSessionRecording recorderHostId={LIVE_SESSION_RECORDER_HOST_ID} />
            {sessionGame === null ? (
                <div className="live-session-waiting" data-testid="live-session-gate">
                    <LiveSessionGameStatus />
                    <div className="live-session-waiting__copy">
                        <span className="live-session-waiting__eyebrow">Live session gate</span>
                        <h2>Choose the simulator for this session</h2>
                        <p>
                            Start a new session when your simulator is detected. The selected game
                            will stay fixed until you upload or discard the session.
                        </p>
                    </div>
                </div>
            ) : (
                <>
                    <LiveSessionGameStatus />
                    {restorationError && (
                        <div className="live-session-recovery-error" role="alert">
                            {restorationError}
                        </div>
                    )}
                    <div className="live-session-view__workspace">
                        {sessionGame === 'acc'
                            ? <LiveTelemetryWorkspace name={AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER} />
                            : <LimitedLiveWorkspace game={sessionGame} />}
                    </div>
                    <div
                        id={LIVE_SESSION_RECORDER_HOST_ID}
                        className="live-session-view__recorder"
                        data-testid="live-session-recorder-host"
                    />
                </>
            )}
        </section>
    );
};

const getAuthenticatedOwnerEmail = (): string | null => {
    if (typeof window === 'undefined') return null;
    return window.localStorage.getItem('username');
};

const LiveSessionView = ({
    name,
    ownerEmail = getAuthenticatedOwnerEmail(),
}: {
    name: string;
    ownerEmail?: string | null;
}) => (
    <LiveSessionProvider ownerEmail={ownerEmail}>
        <LiveSessionContent name={name} />
    </LiveSessionProvider>
);

export default LiveSessionView;
