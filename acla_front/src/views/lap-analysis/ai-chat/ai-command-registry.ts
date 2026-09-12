import type { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import { detectOvertakeTacticalState } from './overtake-agent-detector';
import {
    InvalidOperationCallError,
    NoLiveSessionError,
    NoLiveTelemetryError,
    OperationExecutionError,
    OperationNotRegisteredError,
    createControlledOperation,
    createOperationFrom,
    asTool,
    asWorkflow,
    type Tool,
    type Workflow,
    type Operation,
    type OperationExecutionOutput,
    type OperationStatusPayload,
} from './operation-base';
import {
    OPERATION_COMPONENT_NAMES,
    ComponentRefUnavailableError,
    resolveNamedComponentHandle,
    type OperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import {
    NonLiveContextLiveOperationsUnavailableError,
    RecordedSessionLiveOperationsUnavailableError,
} from 'contexts/OperationComponentError';
import { isLiveSessionAiAvailable, type RecordingState } from 'views/lap-analysis/recording-state';
import type {
    WorkflowDispatcher,
    OperationKind,
    WorkflowPanelHandle,
    LiveRangeTodoListHandle,
    ProcedurePlanHandle,
    OperationQueryResult,
} from 'components/ai-operations';
import type { BaselineCollectionHandle } from 'views/live-session/BaselineCollection';
import type { AiChatHandle } from './ai-chat';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import type { LiveSessionSnapshot } from 'views/live-session/live-session-types';
import type { SessionAnalysisHandle } from 'views/lap-analysis/session-analysis';
import type { UserSummaryHandle } from 'views/user-summary/user-summary';
import type { AiMapDisplayPayload } from './AiMapToolDisplay';
import type {
    AnalysisResultOverlayResult,
    AnalysisResultsChartHandle,
    FilteredAnalysisSegmentsSnapshot,
} from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import type {
    ApplyAnalysisResultQueryInput,
    QueryLapAnalysisResultInput,
    QueryLapAnalysisResultOutput,
} from 'views/lap-analysis/visualization/charts/analysisResultsQuery';
import { getSingletonVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import type { QueryResult, QueryScope } from 'views/lap-analysis/session-intelligence/types';
import {
    getDriverExpertReplayDurationMs,
    hasComparableDriverExpertData,
} from 'components/driver-expert-comparison';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import type { AppendProcedurePlanInput } from 'components/ai-operations/ProcedurePlan';
import type { AppendRepeatablePlanInput } from 'components/ai-operations/RepeatablePlan';
import type { CreateLiveRangeTodoListInput } from 'components/ai-operations/live-range-todo-list-types';
import { WorkflowComponentBase } from 'components/ai-operations/WorkflowComponentBase';
import { readToolCall, type ToolCall } from 'components/ai-operations/tool';
import { readWorkflowCall, type WorkflowCall } from 'components/ai-operations/workflow';
import type {
    ProcedurePlanInput,
    RepeatablePlanInput,
    LiveRangeTodoListInput,
} from 'components/ai-operations';

export type {
    ApplyAnalysisResultQueryInput,
    ApplyAnalysisResultQueryOutput,
    QueryLapAnalysisResultInput,
    QueryLapAnalysisResultOutput,
} from 'views/lap-analysis/visualization/charts/analysisResultsQuery';

export type AgentSessionMode = 'track_guide' | 'overtake' | 'live_performance_analyst';
export type AgentSessionStatus = 'starting' | 'active' | 'stopping' | 'stopped' | 'error';
export type AgentSessionRole = 'main' | 'agent';

export interface AgentSessionInfo {
    sessionRole: AgentSessionRole;
    clientSessionId: string;
    parentClientSessionId: string | null;
    agentMode: AgentSessionMode;
    status: AgentSessionStatus;
}

export type AgentSessionStartResult = {
    status: 'started' | 'already_running';
    conversation_role: 'agent';
    agent_mode: AgentSessionMode;
    agent_session_id?: string;
    parent_client_session_id?: string | null;
};

export type AgentSessionStopResult = {
    status: 'stopped' | 'not_running';
    conversation_role: 'agent';
    agent_mode?: AgentSessionMode;
    agent_session_id?: string | null;
};

export interface FrontendAiCommandContext {
    workflowCaller?: WorkflowComponentBase<any>;
    componentRefs?: OperationComponentRefDirectory;
    sessionId?: string;
    sessionMode?: 'front_desk' | 'live' | 'recorded' | 'user_summary';
    conversationRole?: AgentSessionRole;
    agentMode?: AgentSessionMode;
    sessionGame?: DesktopGame | null;
}

export interface OpportunityAgentState {
    intervalId: ReturnType<typeof setInterval> | null;
    inFlight: boolean;
    lastAlertKey: string | null;
    lastAlertAt: number;
}

export interface LivePerformanceAnalystState {
    enabled: boolean;
}

export type FilteredComparisonSkipReason =
    | 'already_queued'
    | 'invalid_start_position'
    | 'comparison_unavailable'
    | 'invalid_replay_duration';

export interface AddFilteredDriverExpertComparisonsResult {
    [key: string]: unknown;
    status: 'ready' | 'empty' | 'busy';
    active_page_id: string | null;
    applied_view: string | null;
    committed_query: string | null;
    matched_count: number;
    queued_count: number;
    skipped_count: number;
    skipped_segments: Array<{
        segment_id: string;
        event_id: string;
        reason_code: FilteredComparisonSkipReason;
    }>;
}

export interface DisplaySpecificResultInOverlayArguments {
    page_id: string;
    result_id: string;
}

export type DisplaySpecificResultInOverlayResult = AnalysisResultOverlayResult;

export interface AiCommandRegistryContext extends FrontendAiCommandContext {
    recordingState?: RecordingState | null;
    activeAgentSession?: AgentSessionInfo | null;
    analysisContext?: any;
    getLiveSessionSnapshot?: (() => LiveSessionSnapshot) | null;
    opportunityAgentState: OpportunityAgentState;
    livePerformanceAnalystState?: LivePerformanceAnalystState;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
    setLivePerformanceAnalystEnabled?: (enabled: boolean) => void;
    setAgentTagActive?: (tag: string, active: boolean) => void;
    startAgentSession?: (
        agentMode: AgentSessionMode,
        args?: Record<string, any>,
    ) => AgentSessionStartResult | Promise<AgentSessionStartResult>;
    stopAgentSession?: (
        agentSessionId?: string | null,
    ) => AgentSessionStopResult | Promise<AgentSessionStopResult>;
    getOpportunityTelemetryRows: () => Record<string, any>[];
    userSummary?: Record<string, any>;
    userSummaryLoading?: boolean;
    userSummaryError?: string;
    getLabelName?: (labelId: string) => string | undefined;
    getCategoryLabels?: (category: string) => string[];
    getCircuitMapById?: (id: string) => Promise<CircuitMapDto | null>;
    getCircuitMapByTrack?: (
        game: CircuitMapGame,
        sourceTrackKey: string | null | undefined,
    ) => Promise<CircuitMapDto | null>;
    displayMap?: (display: AiMapDisplayPayload) => void;
}

const DEFAULT_OVERTAKE_AGENT_INTERVAL_SECONDS = 5;
const OVERTAKE_AGENT_REPEAT_ALERT_MS = 20000;

const clampInterval = (value: unknown, fallback: number, min: number, max: number) => {
    const parsed = Number(value);
    const seconds = Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
    return Math.min(max, Math.max(min, seconds));
};

const isLiveSessionContext = (context: AiCommandRegistryContext): boolean =>
    (!context.sessionMode || context.sessionMode === 'live')
    && isLiveSessionAiAvailable(context.recordingState);

const getLiveUnavailableError = (context: AiCommandRegistryContext) => (
    context.sessionMode === 'recorded'
        ? RecordedSessionLiveOperationsUnavailableError
        : NonLiveContextLiveOperationsUnavailableError
);

const getBaselineReadiness = (context: AiCommandRegistryContext) => {
    const baseline = context.componentRefs?.findComponentRef<BaselineCollectionHandle>(
        OPERATION_COMPONENT_NAMES.BASELINE_COLLECTION,
    )?.current;
    const record = baseline?.getLapRecord() ?? null;
    const tag = baseline?.getTag() ?? null;
    return { record, tag, ready: Boolean(record?.records?.length) };
};

const buildLiveAnalystSnapshot = (context: AiCommandRegistryContext) => {
    const snapshot = context.getLiveSessionSnapshot?.();
    const { record, tag, ready } = getBaselineReadiness(context);
    return {
        ...(snapshot ?? {}),
        baseline_ready: ready,
        baseline_collection_started: ready || tag?.status === 'collecting',
        baseline_progress_percent: ready ? 100 : tag?.progress_percent ?? 0,
        baseline_lap_id: record?.lap_id ?? tag?.baseline_lap_id ?? null,
        baseline_record_sample_count: record?.sample_count ?? 0,
    };
};

export const startAgentRuntime = async (
    agentMode: AgentSessionMode,
    context: AiCommandRegistryContext,
    args: Record<string, unknown>,
    publishStatus: (data: Record<string, unknown>) => void = () => undefined,
): Promise<OperationExecutionOutput> => {
    if (!isLiveSessionContext(context)) {
        const ErrorType = getLiveUnavailableError(context);
        throw new ErrorType(
            OPERATION_COMPONENT_NAMES.LIVE_SESSION,
            'Agent runtime requires an active live recording.',
        );
    }
    if (agentMode === 'track_guide') {
        context.startTrackGuide();
        context.setAgentTagActive?.('Track Guide', true);
        return { status: 'started', agent_mode: agentMode, enabled: true };
    }
    if (agentMode === 'overtake') {
        const rows = context.getOpportunityTelemetryRows();
        if (rows.length === 0) {
            throw new NoLiveTelemetryError('No live telemetry is available for overtake analysis.');
        }
        const state = context.opportunityAgentState;
        if (state.intervalId) return { status: 'already_running', agent_mode: agentMode };
        const intervalSeconds = clampInterval(
            args.interval_seconds,
            DEFAULT_OVERTAKE_AGENT_INTERVAL_SECONDS,
            2,
            15,
        );
        const runCycle = (notify: boolean) => {
            if (state.inFlight) return { status: 'skipped_in_flight' };
            state.inFlight = true;
            try {
                const telemetry = context.getOpportunityTelemetryRows();
                const tactical = detectOvertakeTacticalState(telemetry);
                if (notify && tactical.status === 'actionable') {
                    const key = `${tactical.event}:${tactical.opponent_id ?? tactical.opponent_slot}:${tactical.projected_section ?? tactical.next_corner?.name}`;
                    const now = Date.now();
                    if (state.lastAlertKey !== key || now - state.lastAlertAt > OVERTAKE_AGENT_REPEAT_ALERT_MS) {
                        state.lastAlertKey = key;
                        state.lastAlertAt = now;
                        publishStatus({ ...tactical, source: 'overtake_agent', agent_mode: agentMode });
                    }
                }
                return { status: 'checked', tactical_state: tactical, telemetry_rows: telemetry.length };
            } finally {
                state.inFlight = false;
            }
        };
        const initial = runCycle(false);
        state.intervalId = setInterval(() => runCycle(true), intervalSeconds * 1000);
        context.setAgentTagActive?.('Overtake', true);
        return { status: 'started', agent_mode: agentMode, interval_seconds: intervalSeconds, initial };
    }

    if (!context.getLiveSessionSnapshot) {
        throw new NoLiveSessionError('Live session snapshot is unavailable.');
    }
    const state = context.livePerformanceAnalystState ?? { enabled: false };
    if (state.enabled) {
        return {
            status: 'already_running',
            agent_mode: agentMode,
            snapshot: buildLiveAnalystSnapshot(context),
        };
    }
    state.enabled = true;
    context.setLivePerformanceAnalystEnabled?.(true);
    context.setAgentTagActive?.('Live Analyst', true);
    publishStatus({
        source: 'live_performance_analyst',
        agent_mode: agentMode,
        event: 'live_analysis_started',
        snapshot: buildLiveAnalystSnapshot(context),
    });
    return {
        status: 'started',
        agent_mode: agentMode,
        snapshot: buildLiveAnalystSnapshot(context),
    };
};

export type FrontendOperationName = typeof definitionList[number]['name'];
export type FrontendAiQueryName = Extract<FrontendOperationName, `query_${string}`>;

export type TelemetryMetricReduce = 'avg' | 'min' | 'max' | 'stats';

export type QueryTelemetryMetricArguments<
    TReduce extends TelemetryMetricReduce,
> = {
    fields: string[];
    scope: QueryScope;
    reduce: TReduce;
};

export type QueryTelemetryMetricResult<
    TReduce extends TelemetryMetricReduce,
> = OperationQueryResult<QueryResult<TReduce>>;

export type FrontendAiQueryContractMap = {
    query_lap_analysis_result: (
        args: QueryLapAnalysisResultInput,
    ) => Tool<QueryLapAnalysisResultOutput>;
    query_telemetry_metric: <TReduce extends TelemetryMetricReduce>(
        args: QueryTelemetryMetricArguments<TReduce>,
    ) => Tool<QueryTelemetryMetricResult<TReduce>>;
};

type AssertTrue<TValue extends true> = TValue;
type QueryContractKeysAreExact = (
    [FrontendAiQueryName] extends [keyof FrontendAiQueryContractMap]
        ? [keyof FrontendAiQueryContractMap] extends [FrontendAiQueryName]
            ? true
            : false
        : false
);

export type FrontendAiQueryContractCoverage = AssertTrue<QueryContractKeysAreExact>;

const validateAnalysisResultQueryArguments = (
    args: unknown,
): QueryLapAnalysisResultInput => {
    const validationMessage = 'query_lap_analysis_result requires exactly one non-empty string property named query.';
    if (!args || typeof args !== 'object' || Array.isArray(args)) {
        throw new InvalidOperationCallError(validationMessage);
    }
    const value = args as Record<string, unknown>;
    const keys = Reflect.ownKeys(value);
    const queryProperty = Object.getOwnPropertyDescriptor(value, 'query');
    if (keys.length !== 1
        || keys[0] !== 'query'
        || !queryProperty
        || !('value' in queryProperty)
        || typeof queryProperty.value !== 'string'
        || !queryProperty.value.trim()) {
        throw new InvalidOperationCallError(validationMessage);
    }
    return { query: queryProperty.value };
};

const validateApplyAnalysisResultQueryArguments = (
    args: unknown,
): ApplyAnalysisResultQueryInput => {
    const validationMessage = 'apply_query_to_analysis_result requires a non-empty string property named query and accepts only an optional integer property named page_number.';
    if (!args || typeof args !== 'object' || Array.isArray(args)) {
        throw new InvalidOperationCallError(validationMessage);
    }
    const value = args as Record<string, unknown>;
    const keys = Reflect.ownKeys(value);
    const queryProperty = Object.getOwnPropertyDescriptor(value, 'query');
    const pageNumberProperty = Object.getOwnPropertyDescriptor(value, 'page_number');
    if (
        keys.some((key) => key !== 'query' && key !== 'page_number')
        || !queryProperty
        || !('value' in queryProperty)
        || typeof queryProperty.value !== 'string'
        || !queryProperty.value.trim()
        || (pageNumberProperty && (
            !('value' in pageNumberProperty)
            || typeof pageNumberProperty.value !== 'number'
            || !Number.isInteger(pageNumberProperty.value)
        ))
    ) {
        throw new InvalidOperationCallError(validationMessage);
    }
    return {
        query: queryProperty.value,
        ...(pageNumberProperty ? { page_number: pageNumberProperty.value as number } : {}),
    };
};

const validateDisplaySpecificResultArguments = (
    args: unknown,
): DisplaySpecificResultInOverlayArguments => {
    const validationMessage = 'display_specific_result_in_overlay requires exactly two non-empty string properties named page_id and result_id.';
    if (!args || typeof args !== 'object' || Array.isArray(args)) {
        throw new InvalidOperationCallError(validationMessage);
    }
    const value = args as Record<string, unknown>;
    const keys = Reflect.ownKeys(value);
    const pageIdProperty = Object.getOwnPropertyDescriptor(value, 'page_id');
    const resultIdProperty = Object.getOwnPropertyDescriptor(value, 'result_id');
    if (
        keys.length !== 2
        || keys.some((key) => key !== 'page_id' && key !== 'result_id')
        || !pageIdProperty
        || !('value' in pageIdProperty)
        || typeof pageIdProperty.value !== 'string'
        || !pageIdProperty.value.trim()
        || !resultIdProperty
        || !('value' in resultIdProperty)
        || typeof resultIdProperty.value !== 'string'
        || !resultIdProperty.value.trim()
    ) {
        throw new InvalidOperationCallError(validationMessage);
    }
    return {
        page_id: pageIdProperty.value,
        result_id: resultIdProperty.value,
    };
};

type FrontendOperationDefinition = {
    readonly name: string;
    readonly kind: OperationKind;
    readonly componentName: string;
    readonly execute: (
        context: FrontendAiCommandContext,
        args: Record<string, any>,
        dispatchNested: WorkflowDispatcher,
        signal?: AbortSignal,
    ) => Operation<OperationExecutionOutput, OperationStatusPayload>;
};

const getDirectory = (context: FrontendAiCommandContext): OperationComponentRefDirectory => {
    if (context.componentRefs) return context.componentRefs;
    throw new ComponentRefUnavailableError(
        'dashboard',
        'The active dashboard component-ref directory is unavailable.',
    );
};

const getComponent = <T,>(context: FrontendAiCommandContext, name: string): T => (
    resolveNamedComponentHandle(getDirectory(context), name) as T
);

const hasOwn = (value: Record<string, unknown>, key: string): boolean => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const validateNoArguments = (args: unknown, toolName: string): void => {
    if (!isRecord(args) || Reflect.ownKeys(args).length > 0) {
        throw new InvalidOperationCallError(`${toolName} does not accept arguments.`);
    }
};

const createLiveRangeAbortError = (): Error => {
    const error = new Error('The live range to-do task was aborted.');
    error.name = 'AbortError';
    return error;
};

const displaySpecificResultInOverlay = (
    context: FrontendAiCommandContext,
    args: unknown,
    signal?: AbortSignal,
): ReturnType<AnalysisResultsChartHandle['displaySpecificResultInOverlay']> => {
    const request = validateDisplaySpecificResultArguments(args);
    return getComponent<AnalysisResultsChartHandle>(
        context,
        getSingletonVisualizationComponentName('analysis-results'),
    ).displaySpecificResultInOverlay(request.page_id, request.result_id, signal);
};

type EligibleFilteredComparison = {
    segmentId: string;
    eventId: string;
    normalizedPosition: number;
    replayDurationMs: number;
    leadTimeSeconds: number;
    title: string;
    section?: string;
};

const createFilteredComparisonResult = (
    snapshot: FilteredAnalysisSegmentsSnapshot,
): AddFilteredDriverExpertComparisonsResult => ({
    status: snapshot.status,
    active_page_id: snapshot.activePageId,
    applied_view: snapshot.appliedView,
    committed_query: snapshot.committedQuery,
    matched_count: snapshot.segments.length,
    queued_count: 0,
    skipped_count: 0,
    skipped_segments: [],
});

const queueFilteredDriverExpertComparisons = async (
    context: FrontendAiCommandContext,
    snapshot: FilteredAnalysisSegmentsSnapshot,
    dispatchNested: WorkflowDispatcher,
    signal: AbortSignal,
): Promise<AddFilteredDriverExpertComparisonsResult> => {
    const result = createFilteredComparisonResult(snapshot);
    if (snapshot.status !== 'ready') return result;

    const eligible: EligibleFilteredComparison[] = [];
    snapshot.segments.forEach((segment) => {
        const eventId = `analysis-comparison:${segment.id}`;
        const skip = (reasonCode: FilteredComparisonSkipReason) => {
            result.skipped_segments.push({
                segment_id: segment.id,
                event_id: eventId,
                reason_code: reasonCode,
            });
        };
        const start = segment.normalizedPositionRange?.start;
        if (typeof start !== 'number' || !Number.isFinite(start) || start < 0 || start > 1) {
            skip('invalid_start_position');
            return;
        }
        if (!hasComparableDriverExpertData(segment.comparison, context.sessionGame ?? null)) {
            skip('comparison_unavailable');
            return;
        }
        const replayDurationMs = getDriverExpertReplayDurationMs(segment.comparison);
        if (!Number.isFinite(replayDurationMs) || replayDurationMs <= 0) {
            skip('invalid_replay_duration');
            return;
        }
        eligible.push({
            segmentId: segment.id,
            eventId,
            normalizedPosition: start,
            replayDurationMs,
            leadTimeSeconds: (replayDurationMs / 1000) + 2,
            title: segment.title
                ? `${segment.title}: Driver vs Expert`
                : 'Driver vs Expert',
            section: segment.section,
        });
    });

    if (snapshot.segments.length > 0 && eligible.length === 0) {
        throw new OperationExecutionError(
            'The filtered analysis results contain no showable overlay graphs.',
        );
    }

    if (eligible.length > 0) {
        dispatchNested.validate('display_specific_result_in_overlay');
        if (!snapshot.activePageId) {
            throw new OperationExecutionError(
                'The filtered analysis results do not identify a retained page.',
            );
        }
        const mounted = getDirectory(context).findComponentRef<LiveRangeTodoListHandle>(
            OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        )?.current;
        const existingIds = new Set(
            mounted?.get().todo_list?.events.map((event) => event.id) ?? [],
        );
        const pending = eligible.filter((comparison) => {
            if (existingIds.has(comparison.eventId)) {
                result.skipped_segments.push({
                    segment_id: comparison.segmentId,
                    event_id: comparison.eventId,
                    reason_code: 'already_queued',
                });
                return false;
            }
            existingIds.add(comparison.eventId);
            return true;
        });
        const voiceDurations = pending.length > 0
            ? await getComponent<AnalysisResultsChartHandle>(
                context, getSingletonVisualizationComponentName('analysis-results'),
            ).prepareComparisonVoices(snapshot.activePageId, pending.map(({ segmentId }) => segmentId), signal)
            : {};
        if (signal.aborted) throw createLiveRangeAbortError();
        // Telemetry can drain and dispose the queue while voices are being prepared.
        const current = getDirectory(context).findComponentRef<LiveRangeTodoListHandle>(OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)?.current;
        const queuedIds = new Set(current?.get().todo_list?.events.map((event) => event.id) ?? []);
        const operations: LiveRangeTodoListInput['workflow']['operations'] = [];
        pending.forEach((comparison) => {
            if (queuedIds.has(comparison.eventId)) {
                result.skipped_segments.push({
                    segment_id: comparison.segmentId,
                    event_id: comparison.eventId,
                    reason_code: 'already_queued',
                });
                return;
            }
            comparison.leadTimeSeconds = Math.max(
                comparison.replayDurationMs, voiceDurations[comparison.segmentId],
            ) / 1000 + 2;
            operations.push({ operation: { name: 'display_specific_result_in_overlay', event: {
                id: comparison.eventId,
                normalized_position: comparison.normalizedPosition,
                lead_time_seconds: comparison.leadTimeSeconds,
                content: {
                    title: comparison.title,
                    ...(comparison.section
                        ? { description: `Section: ${comparison.section}` }
                        : {}),
                },
                }, arguments: { page_id: snapshot.activePageId, result_id: comparison.segmentId } } });
            result.queued_count += 1;
        });
        if (operations.length) {
            const appended = getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
                .appendLiveRangeTodoList({ workflow: { name: 'add_event_to_live_range_todo_list', operations } }, dispatchNested);
            const output = await appended.result;
            if (output instanceof Error) throw output;
        }
    }

    result.skipped_count = result.skipped_segments.length;
    return result;
};

const definitionList = Object.freeze([
    {
        name: 'start_agent_session',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .startAgentSession(args.agent_mode ?? args.agentMode, args),
    },
    {
        name: 'stop_agent_session',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .stopAgentSession(args.agent_session_id ?? args.agentSessionId),
    },
    {
        name: 'add_event_to_live_range_todo_list',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
            .appendLiveRangeTodoList(args as LiveRangeTodoListInput, dispatchNested),
    },
    {
        name: 'create_live_range_todo_list',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
            .createLiveRangeTodoList(args as CreateLiveRangeTodoListInput, dispatchNested),
    },
    {
        name: 'add_analysis_result_to_do_list',
        kind: 'tool',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args, dispatchNested) => {
            const controller = createControlledOperation<AddFilteredDriverExpertComparisonsResult>();
            void Promise.resolve().then(async () => {
                if (controller.signal.aborted) return;
                validateNoArguments(
                    args,
                    'add_analysis_result_to_do_list',
                );
                // Queue the currently displayed results, preserving the view's applied filter and order.
                const snapshot = getComponent<AnalysisResultsChartHandle>(
                    context,
                    getSingletonVisualizationComponentName('analysis-results'),
                ).getFilteredSegments();
                const result = await queueFilteredDriverExpertComparisons(
                    context, snapshot, dispatchNested, controller.signal,
                );
                controller.resolve(result.status, result);
            }).catch((error) => {
                controller.reject('failed', error instanceof Error ? error : new Error(String(error)));
            });
            return asTool(controller.operation);
        },
    },
    {
        name: 'display_specific_result_in_overlay',
        kind: 'tool',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args, _dispatchNested, signal) => (
            displaySpecificResultInOverlay(context, args, signal)
        ),
    },
    {
        name: 'get_live_range_todo_list',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        execute: (context) => getComponent<LiveRangeTodoListHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)
            .getForAi(),
    },
    {
        name: 'collect_live_baseline',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .collectLiveBaselineForAi(args),
    },
    {
        name: 'restart_live_baseline',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .restartLiveBaselineForAi(),
    },
    {
        name: 'analyze_live_recorded_analysis',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .analyzeLiveRecordedAnalysisForAi(args),
    },
    {
        name: 'apply_query_to_analysis_result',
        kind: 'tool',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args) => {
            const request = validateApplyAnalysisResultQueryArguments(args);
            return getComponent<AnalysisResultsChartHandle>(
                context,
                getSingletonVisualizationComponentName('analysis-results'),
            ).applyAnalysisResultQuery(request);
        },
    },
    {
        name: 'query_lap_analysis_result',
        kind: 'tool',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args) => {
            const query = validateAnalysisResultQueryArguments(args);
            return getComponent<AnalysisResultsChartHandle>(
                context,
                getSingletonVisualizationComponentName('analysis-results'),
            ).queryLapAnalysisResult(query);
        },
    },
    {
        name: 'create_repeatable_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => {
            return getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
                .createRepeatablePlan(args as RepeatablePlanInput, dispatchNested);
        },
    },
    {
        name: 'append_repeatable_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
            .appendRepeatablePlan(args as AppendRepeatablePlanInput, dispatchNested),
    },
    {
        name: 'append_procedure_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
            .appendProcedurePlan(args as AppendProcedurePlanInput, dispatchNested),
    },
    {
        name: 'advance_plan_step',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN,
        execute: (context, args) => getComponent<ProcedurePlanHandle>(context, OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN)
            .advancePlanStep(typeof args.reason === 'string' ? args.reason : undefined),
    },
    {
        name: 'clear_procedure_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN,
        execute: (context, args) => getComponent<ProcedurePlanHandle>(context, OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN)
            .clearProcedurePlan(typeof args.reason === 'string' ? args.reason : undefined),
    },
    {
        name: 'set_procedure_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => {
            return getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
                .createProcedurePlan(args as ProcedurePlanInput, dispatchNested);
        },
    },
    {
        name: 'get_next_corner',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .getNextCornerForAi(),
    },
    {
        name: 'query_telemetry_metric',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .queryTelemetryMetricForAi(
                args as QueryTelemetryMetricArguments<TelemetryMetricReduce>,
            ),
    },
    {
        name: 'get_event_log',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
            .getEventLogForAi(args),
    },
    {
        name: 'get_user_summary_map_level',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context, args) => getComponent<UserSummaryHandle>(context, OPERATION_COMPONENT_NAMES.USER_SUMMARY)
            .getUserSummaryMapLevel(args),
    },
    {
        name: 'get_available_user_summary_maps',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context) => getComponent<UserSummaryHandle>(context, OPERATION_COMPONENT_NAMES.USER_SUMMARY)
            .getAvailableUserSummaryMaps(),
    },
    {
        name: 'search_user_summary_map_level',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context, args) => getComponent<UserSummaryHandle>(context, OPERATION_COMPONENT_NAMES.USER_SUMMARY)
            .searchUserSummaryMapLevel(args),
    },
    {
        name: 'show_map',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .showMap(args),
    },
    {
        name: 'run_recorded_ai_analysis',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)
            .runRecordedAnalysisForAi(args),
    },
    {
        name: 'get_recorded_session_analysis',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)
            .getRecordedAnalysisForAi(args),
    },
    {
        name: 'get_recorded_session_context',
        kind: 'tool',
        componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)
            .getRecordedSessionContextForAi(args),
    },
    {
        name: 'analyze_telemetry',
        kind: 'tool',
        componentName: 'session-mode-analysis',
        execute: (context, args) => context.sessionMode === 'recorded'
            ? getComponent<SessionAnalysisHandle>(context, OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)
                .analyzeTelemetryForAi(args)
            : getComponent<LiveSessionHandle>(context, OPERATION_COMPONENT_NAMES.LIVE_SESSION)
                .analyzeTelemetryForAi(args),
    },
] as const satisfies readonly FrontendOperationDefinition[]);

type FrontendOperationDefinitionMap = {
    [TDefinition in typeof definitionList[number] as TDefinition['name']]: TDefinition;
};

export type FrontendWorkflowName = {
    [TName in FrontendOperationName]: FrontendOperationDefinitionMap[TName]['kind'] extends 'workflow'
        ? TName : never;
}[FrontendOperationName];

export type FrontendToolName = Exclude<FrontendOperationName, FrontendWorkflowName>;

export type FrontendOperation<TName extends FrontendOperationName> = (
    TName extends FrontendAiQueryName
        ? ReturnType<FrontendAiQueryContractMap[TName]>
        : ReturnType<FrontendOperationDefinitionMap[TName]['execute']> extends Operation<
            infer TResult, infer TStatus, infer TTerminationStatus
        >
            ? FrontendOperationDefinitionMap[TName]['kind'] extends 'workflow'
                ? Workflow<TResult, TStatus, TTerminationStatus>
                : Tool<TResult, TStatus, TTerminationStatus>
            : never
);

type NonQueryAiCommandRegistry = {
    [TName in Exclude<
        FrontendOperationName,
        FrontendAiQueryName | 'display_specific_result_in_overlay'
    >]: (
        args: TName extends keyof WorkflowInputMap ? WorkflowInputMap[TName] : Record<string, unknown>,
    ) => FrontendOperation<TName>;
};

type WorkflowInputMap = {
    set_procedure_plan: ProcedurePlanInput;
    append_procedure_plan: AppendProcedurePlanInput;
    append_repeatable_plan: AppendRepeatablePlanInput;
    create_live_range_todo_list: CreateLiveRangeTodoListInput;
    create_repeatable_plan: RepeatablePlanInput;
    add_event_to_live_range_todo_list: LiveRangeTodoListInput;
};

type RawAiCommandRegistry = NonQueryAiCommandRegistry & FrontendAiQueryContractMap & {
    display_specific_result_in_overlay(
        args: DisplaySpecificResultInOverlayArguments,
    ): FrontendOperation<'display_specific_result_in_overlay'>;
};

export type AiCommandRegistry = {
    [Name in keyof RawAiCommandRegistry]: Name extends FrontendToolName
        ? RawAiCommandRegistry[Name] & ((
            args: ToolCall<{ arguments?: Parameters<RawAiCommandRegistry[Name]>[0] }>,
        ) => ReturnType<RawAiCommandRegistry[Name]>)
        : (args: Name extends keyof WorkflowInputMap ? WorkflowInputMap[Name]
            : WorkflowCall<Name & FrontendWorkflowName, { operations: []; reason?: string }>) => ReturnType<RawAiCommandRegistry[Name]>;
};

const definitions = Object.fromEntries(
    definitionList.map((definition) => [definition.name, definition]),
) as FrontendOperationDefinitionMap;

export const frontendOperationRegistry = definitions;

export const createWorkflowDispatcher = (
    context: FrontendAiCommandContext,
): WorkflowDispatcher => {
    const validate = (name: string): void => {
        if (!Object.prototype.hasOwnProperty.call(definitions, name)) {
            throw new OperationNotRegisteredError(`Operation '${name}' is not registered.`);
        }
    };
    return Object.assign((name: FrontendOperationName, args: Record<string, unknown> = {}, signal?: AbortSignal, caller = context.workflowCaller) => {
        validate(name);
        return dispatchOperation({ ...context, workflowCaller: caller }, name, args, signal);
    }, { validate, workflowCaller: context.workflowCaller });
};

export const createWorkflowToolDispatcher = createWorkflowDispatcher;

const dispatchOperation = (
    context: FrontendAiCommandContext,
    name: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
    nativeCall = false,
): Tool<OperationExecutionOutput, OperationStatusPayload> | Workflow<OperationExecutionOutput, OperationStatusPayload> => {
    const definition = Object.prototype.hasOwnProperty.call(definitions, name)
        ? definitions[name as FrontendOperationName]
        : undefined;
    try {
        if (signal?.aborted) throw createLiveRangeAbortError();
        context.workflowCaller?.assertAvailable();
        if (!definition) throw new OperationNotRegisteredError(`Operation '${name}' is not registered.`);
        if (nativeCall && definition.kind === 'tool' && isRecord(args) && hasOwn(args, 'tool')) {
            const call = readToolCall(args);
            if (!call || call.name !== name
                || Reflect.ownKeys(call).some((key) => key !== 'name' && key !== 'arguments')
                || (call.arguments !== undefined && !isRecord(call.arguments))) {
                throw new InvalidOperationCallError(`Provide tool with name '${name}' and an arguments object.`);
            }
            args = (call.arguments as Record<string, unknown> | undefined) ?? {};
        } else if (definition.kind === 'workflow'
            && name !== 'set_procedure_plan' && name !== 'create_repeatable_plan'
            && name !== 'add_event_to_live_range_todo_list' && name !== 'create_live_range_todo_list'
            && name !== 'append_procedure_plan' && name !== 'append_repeatable_plan') {
            const call = readWorkflowCall(args, definition.name);
            const supportsReason = name === 'advance_plan_step' || name === 'clear_procedure_plan';
            if (!call || !Array.isArray(call.operations) || call.operations.length !== 0
                || Reflect.ownKeys(call).some((key) => key !== 'name' && key !== 'operations' && !(supportsReason && key === 'reason'))
                || (call.reason !== undefined && typeof call.reason !== 'string')) {
                throw new InvalidOperationCallError(`Provide workflow with name '${name}' and an empty operations list.`);
            }
            args = call.reason !== undefined ? { reason: call.reason } : {};
        }
        if (context.workflowCaller && ['advance_plan_step', 'clear_procedure_plan'].includes(name)) {
            const target = getDirectory(context).findComponentRef(definition.componentName)?.current;
            if (target instanceof WorkflowComponentBase) target.assertCanReplace(context.workflowCaller);
        }
        const dispatchNested = createWorkflowDispatcher(context);
        const result: Operation<OperationExecutionOutput, OperationStatusPayload> = (
            definition.execute(context, args, dispatchNested, signal)
        );
        if (definition.kind === 'tool' && 'kind' in result && result.kind !== 'tool') {
            throw new InvalidOperationCallError(`Tool '${name}' returned a workflow.`);
        }
        const operation = definition.kind === 'workflow' ? asWorkflow(result) : asTool(result);
        if (signal) {
            const abortOperation = () => operation.abort();
            if (signal.aborted) {
                abortOperation();
            } else {
                signal.addEventListener('abort', abortOperation, { once: true });
                operation.notifyTerminated(() => {
                    signal.removeEventListener('abort', abortOperation);
                });
            }
        }
        return operation;
    } catch (error) {
        const operation = createOperationFrom(() => { throw error; }, 'failed');
        return definition?.kind === 'workflow'
            ? asWorkflow(operation, { completed_step_count: 0, stopped_at_step: null }) : asTool(operation);
    }
};

export const createAiCommandRegistry = (
    context: FrontendAiCommandContext,
): AiCommandRegistry => Object.fromEntries(
    definitionList.map((definition) => [
        definition.name,
        (args: Record<string, any>, signal?: AbortSignal, caller?: WorkflowComponentBase<any>, nativeCall = true) => dispatchOperation(
            { ...context, workflowCaller: caller ?? context.workflowCaller },
            definition.name,
            args,
            signal,
            nativeCall,
        ),
    ]),
) as unknown as AiCommandRegistry;

export const isAiCommandName = (name: string): name is FrontendOperationName => (
    Object.prototype.hasOwnProperty.call(definitions, name)
);
