import React, { useContext, useRef } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    ComponentRefUnavailableError,
    NamedAiToolComponentHandle,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
    useOptionalAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    BaselineCollectionIncompleteError,
} from 'contexts/AiToolComponentError';
import {
    AiToolError,
    LiveSectionClassificationFailedError,
    LiveSectionTelemetryUnavailableError,
    NoCornerDataError,
    NoTelemetryForScopeError,
    SectionNotFoundError,
    TelemetryAnalysisFailedError,
    InvalidToolCallError,
} from 'views/lap-analysis/ai-chat/ai-tool-base';
import apiService from 'services/api.service';
import type {
    BaselineCollectionHandle,
    BaselineCollectionPayload,
    BaselineCollectionStatus,
} from './BaselineCollection';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import {
    normalizeSegmentClassificationResult,
    type SegmentClassificationResult,
} from 'views/lap-analysis/recorded-session-analysis';
import { getSegmentLabelIds } from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import { openAnalysisResultsVisualization } from 'views/lap-analysis/visualization/open-analysis-results-visualization';
import { LiveSessionContext } from './LiveSessionContext';
import LiveSessionGameStatus, { LIVE_SESSION_GAME_LABELS } from './LiveSessionGameStatus';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import { RecordingState } from 'views/lap-analysis/recording-state';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import type {
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
import {
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    type AiToolOperation,
} from 'components/ai-engineering-tools';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';

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
export type LiveSectionClassificationAiResult = {
    status: 'recorded';
    classification: unknown;
    focus: unknown;
    comparison: unknown;
    analysis: unknown;
    telemetry_stats: { row_count: number; field_count: number };
    chart_id: string | null;
    component_name: string | null;
};

export interface LiveSessionHandle extends NamedAiToolComponentHandle {
    getRecordingState(): RecordingState;
    getSessionIntelligence(): SessionIntelligence;
    getCurrentTelemetry(): Record<string, any>;
    queryTelemetryMetric<TReduce extends ReduceOp>(args: TelemetryQuery<TReduce>): QueryResult<TReduce>;
    getTelemetryForScope(scope: QueryScope): Record<string, any>[];
    getEventLog(args: Record<string, any>): any[];
    getNextCorner(): any;
    getLiveSessionSnapshot(): LiveSessionSnapshot;
    getLiveSectionHistory(limit: number): any[];
    getLiveSectionTelemetry(args: Record<string, any>): any;
    recordLiveSectionClassification(args: Record<string, any>): any;
    getLatestAnalysisResultPage(): LiveSessionAnalysisResultPage | null;
    queryTelemetryMetricForAi<TReduce extends TelemetryMetricReduce>(
        args: QueryTelemetryMetricArguments<TReduce>,
    ): AiToolOperation<QueryTelemetryMetricResult<TReduce>>;
    getEventLogForAi(args: Record<string, any>): AiToolOperation<LiveEventLogAiResult>;
    getNextCornerForAi(): AiToolOperation<LiveNextCornerAiResult>;
    collectLiveBaselineForAi(args: Record<string, any>): AiToolOperation<BaselineCollectionPayload, BaselineCollectionStatus>;
    restartLiveBaselineForAi(): AiToolOperation<LiveBaselineRestartAiResult>;
    analyzeLiveRecordedAnalysisForAi(args: Record<string, any>): AiToolOperation<LiveBaselineAnalysisAiResult>;
    analyzeTelemetryForAi(args: Record<string, any>): AiToolOperation<LiveTelemetryAnalysisAiResult>;
    classifyLiveSectionForAi(args: Record<string, any>): AiToolOperation<LiveSectionClassificationAiResult>;
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

type LiveSessionSnapshot = ReturnType<SessionIntelligence['getLiveSessionSnapshot']>;

const EMPTY_LIVE_SNAPSHOT: LiveSessionSnapshot = {
    status: 'empty',
    track: '',
    car: '',
    current_lap: 0,
    completed_laps: 0,
    normalized_position: 0,
    sample_count: 0,
    live_session_type: 'unknown',
    baseline_ready: false,
    baseline_collection_started: false,
    baseline_progress_percent: 0,
    baseline_lap: null,
    completed_lap_count: 0,
    section_count: 0,
};

const getLiveSnapshot = (
    sessionIntelligence: Partial<Pick<SessionIntelligence, 'getLiveSessionSnapshot'>> | null,
): LiveSessionSnapshot => (
    typeof sessionIntelligence?.getLiveSessionSnapshot === 'function'
        ? sessionIntelligence.getLiveSessionSnapshot()
        : EMPTY_LIVE_SNAPSHOT
);

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

const LiveSessionView = ({ name }: { name: string }) => {
    const liveSession = useContext(LiveSessionContext);
    const componentRefs = useOptionalAiToolComponentRefDirectory();
    const liveSessionRef = useRef(liveSession);
    liveSessionRef.current = liveSession;
    const componentRef = useRef<LiveSessionHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getRecordingState: () => liveSessionRef.current.recordingState,
            getSessionIntelligence: () => liveSessionRef.current.sessionIntelligence,
            getCurrentTelemetry: () => liveSessionRef.current.currentTelemetry,
            queryTelemetryMetric: (args) => liveSessionRef.current.sessionIntelligence.query(args),
            getTelemetryForScope: (scope) => liveSessionRef.current.sessionIntelligence.getRowsForScope(scope),
            getEventLog: (args) => liveSessionRef.current.sessionIntelligence.findEvents(args as any),
            getNextCorner: () => liveSessionRef.current.sessionIntelligence.getNextCorner(),
            getLiveSessionSnapshot: () => getLiveSnapshot(liveSessionRef.current.sessionIntelligence),
            getLiveSectionHistory: (limit) => liveSessionRef.current.sessionIntelligence.getSectionHistory(limit),
            getLiveSectionTelemetry: (args) => liveSessionRef.current.sessionIntelligence.getSectionTelemetryWindow({
                section_id: args.section_id || args.sectionId,
                section_name: args.section_name || args.sectionName,
                lap: args.lap,
            }),
            recordLiveSectionClassification: (args) => liveSessionRef.current.sessionIntelligence.recordSectionClassification(args),
            getLatestAnalysisResultPage: () => {
                const pages = liveSessionRef.current.analysisResultPages;
                return pages[pages.length - 1] ?? null;
            },
            queryTelemetryMetricForAi: (args) => createAiToolOperationFrom(() => {
                const query = validateTelemetryMetricArguments(args);
                return {
                    status: 'ready' as const,
                    data: liveSessionRef.current.sessionIntelligence.query(query),
                };
            }),
            getEventLogForAi: (args) => createAiToolOperationFrom(() => ({
                status: 'complete',
                events: liveSessionRef.current.sessionIntelligence.findEvents(args as any),
            })),
            getNextCornerForAi: () => createAiToolOperationFrom(() => {
                const corner = liveSessionRef.current.sessionIntelligence.getNextCorner();
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
                if (mountedHandle) return mountedHandle.startCollection(options);

                const statusDeferred = [0, 1, 25, 50, 75, 100]
                    .map(() => createAiToolDeferred<BaselineCollectionStatus>());
                const result = (async () => {
                    try {
                        const handle = await getBaselineHandle(componentRefs);
                        const operation = handle.startCollection(options);
                        operation.statuses.forEach((status, index) => {
                            void status.then(
                                (value) => statusDeferred[index]?.resolve(value),
                                (error) => statusDeferred[index]?.reject(error),
                            );
                        });
                        operation.statuses.slice(statusDeferred.length).forEach((status) => {
                            void status.catch(() => undefined);
                        });
                        const value = await operation.result;
                        statusDeferred.forEach((deferred, index) => {
                            if (!deferred.settled) deferred.resolve({
                                ...(value instanceof Error
                                    ? { status: 'waiting_for_start' as const, progress_percent: 0, car: null, track: null, message: value.message }
                                    : value),
                                event: index <= 1 ? 'baseline_collecting' : 'baseline_progress',
                                milestone: [0, 1, 25, 50, 75, 100][index],
                                skipped: true,
                            });
                        });
                        return value;
                    } catch (error) {
                        statusDeferred.forEach((deferred) => deferred.reject(error));
                        throw error;
                    }
                })();
                return createAiToolOperation(result, statusDeferred.map((status) => status.promise));
            },
            restartLiveBaselineForAi: () => createAiToolOperationFrom(async () => {
                const handle = await getBaselineHandle(componentRefs);
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
                const rows = liveSessionRef.current.sessionIntelligence.getRowsForScope(args.scope);
                if (rows.length === 0) {
                    throw new NoTelemetryForScopeError('No telemetry rows matched the requested scope.');
                }
                try {
                    const snapshot = getLiveSnapshot(liveSessionRef.current.sessionIntelligence);
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
            classifyLiveSectionForAi: (args) => createAiToolOperationFrom(async () => {
                const baseline = await getBaselineHandle(componentRefs);
                if (!baseline.getLapRecord()?.records?.length) {
                    throw new BaselineCollectionIncompleteError(
                        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                        'Complete baseline collection before classifying a live section.',
                    );
                }
                const intelligence = liveSessionRef.current.sessionIntelligence;
                const window = intelligence.getSectionTelemetryWindow({
                    section_id: args.section_id || args.sectionId,
                    section_name: args.section_name || args.sectionName,
                    lap: args.lap,
                });
                if (window.status !== 'ready' || !window.rows.length) {
                    throw new LiveSectionTelemetryUnavailableError(
                        'No telemetry is available for the selected live section.',
                    );
                }
                try {
                    const snapshot = getLiveSnapshot(intelligence);
                    const response = await apiService.post('/racing-session/analyze-live-recorded-analysis', {
                        track: snapshot.track,
                        car: snapshot.car,
                        baseline_lap: window.lap,
                        records: window.rows,
                    }, { timeout: 120000 });
                    const result = normalizeSegmentClassificationResult(
                        response.data as any,
                        `live-section-${window.section?.id || 'unknown'}-${window.lap}`,
                    );
                    const labels = result.segments.flatMap(getSegmentLabelIds);
                    const mistakeLabels = labels.filter((label) => label === 'MSP' || label === 'MSR');
                    const classification = intelligence.recordSectionClassification({
                        section_id: window.section?.id || args.section_id,
                        section_name: window.section?.name || args.section_name,
                        lap: window.lap,
                        start_sample_idx: window.startSampleIdx,
                        end_sample_idx: window.endSampleIdx,
                        mistake_count: mistakeLabels.length,
                        expert_adherence_count: labels.filter((label) => label.startsWith('EA')).length,
                        severity: mistakeLabels.length,
                        confidence: result.segments.length > 0 ? 1 : 0,
                        parent_label: labels[0] || null,
                        child_labels: labels.slice(1),
                        telemetry_stats: getTelemetryStats(window.rows),
                    });
                    if (!classification) {
                        throw new SectionNotFoundError('The requested live section was not found.');
                    }
                    const chart = componentRefs
                        ? await openAnalysisResultsVisualization({
                            directory: componentRefs,
                            managerName: AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
                            result,
                            records: window.rows,
                        })
                        : { chart_id: null, component_name: null };
                    const focus = intelligence.getFocusSection();
                    return {
                        status: 'recorded',
                        classification,
                        focus: focus ? {
                            section: focus.section,
                            baseline: focus.baseline,
                            selected_at: focus.selectedAt,
                            reason: focus.reason,
                            score: focus.score,
                            timing: intelligence.getSectionTiming(focus.section),
                        } : null,
                        comparison: classification
                            ? intelligence.compareFocusedSection(classification)
                            : null,
                        analysis: compactClassification(result, getAiLimit(args.limit)),
                        telemetry_stats: getTelemetryStats(window.rows),
                        ...chart,
                    };
                } catch (error) {
                    if (error instanceof AiToolError) throw error;
                    throw new LiveSectionClassificationFailedError(
                        error instanceof Error && error.message
                            ? error.message
                            : 'Failed to classify the live section.',
                        { cause: error },
                    );
                }
            }),
        };
    }
    useRegisterAiToolComponentRef(name, componentRef.current!);


    const { restorationError, sessionGame } = liveSession;

    return (
        <section className="live-session-view" aria-label="Live Session">
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

export default LiveSessionView;
