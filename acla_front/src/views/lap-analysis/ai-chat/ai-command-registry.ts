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
    InvalidLiveRangeTodoListError,
    NonLiveContextLiveOperationsUnavailableError,
    RecordedSessionLiveOperationsUnavailableError,
} from 'contexts/OperationComponentError';
import { isLiveSessionAiAvailable, type RecordingState } from 'views/lap-analysis/recording-state';
import type {
    ToolDispatcher,
    OperationKind,
    RepeatablePlanHandle,
    WorkflowPanelHandle,
    LiveRangeTodoEventInput,
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
    QueryAnalysisResultInput,
    QueryAnalysisResultOutput,
} from 'views/lap-analysis/visualization/charts/analysisResultsQuery';
import { getSingletonVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import type { QueryResult, QueryScope } from 'views/lap-analysis/session-intelligence/types';
import {
    getDriverExpertReplayDurationMs,
    hasComparableDriverExpertData,
} from 'components/driver-expert-comparison';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { parseProcedurePlanInput } from 'components/ai-operations/ProcedurePlan';
import { validateGoalRequest } from 'components/ai-operations/RepeatablePlan';
import type {
    ProcedurePlanInput,
    RepeatablePlanInput,
    LiveRangeTodoListInput,
} from 'components/ai-operations';

export type {
    ApplyAnalysisResultQueryInput,
    ApplyAnalysisResultQueryOutput,
    QueryAnalysisResultInput,
    QueryAnalysisResultOutput,
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
    query_analysis_result: (
        args: QueryAnalysisResultInput,
    ) => Tool<QueryAnalysisResultOutput>;
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
): QueryAnalysisResultInput => {
    const validationMessage = 'query_analysis_result requires exactly one non-empty string property named query.';
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
        dispatchNested: ToolDispatcher,
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

const invalidLiveRangeTodoList = (message: string): never => {
    throw new InvalidLiveRangeTodoListError(
        OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        message,
    );
};

const assertExactKeys = (
    value: Record<string, unknown>,
    allowed: readonly string[],
    label: string,
): void => {
    const unsupported = Reflect.ownKeys(value).find((key) => (
        typeof key !== 'string' || !allowed.includes(key)
    ));
    if (unsupported !== undefined) {
        invalidLiveRangeTodoList(
            `${label} property '${String(unsupported)}' is not supported.`,
        );
    }
};

const isJsonSafe = (value: unknown, ancestors = new Set<object>()): boolean => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value !== 'object') return false;
    if (ancestors.has(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) return false;
    ancestors.add(value);
    const valid = Array.isArray(value)
        ? value.every((entry) => isJsonSafe(entry, ancestors))
        : Reflect.ownKeys(value).every((key) => (
            typeof key === 'string'
            && isJsonSafe((value as Record<string, unknown>)[key], ancestors)
        ));
    ancestors.delete(value);
    return valid;
};

type PreparedLiveRangeTodoEvent = {
    event: Omit<LiveRangeTodoEventInput, 'taskStart'>;
    tool: {
        name: FrontendToolName;
        arguments: Record<string, unknown>;
    };
};

const validateLiveRangeTodoBatch = (
    args: unknown,
    dispatchNested: ToolDispatcher,
): PreparedLiveRangeTodoEvent[] => {
    const workflowName = 'add_event_to_live_range_todo_list';
    if (!isRecord(args)) invalidLiveRangeTodoList(`Provide a ${workflowName} envelope.`);
    const envelope = args as Record<string, unknown>;
    assertExactKeys(envelope, [workflowName], 'Live range to-do envelope');
    if (!hasOwn(envelope, workflowName) || !isRecord(envelope[workflowName])) {
        invalidLiveRangeTodoList(`Provide a ${workflowName} envelope.`);
    }
    const request = envelope[workflowName] as Record<string, unknown>;
    assertExactKeys(request, ['tools'], 'Live range to-do request');
    if (!Array.isArray(request.tools) || request.tools.length === 0) {
        invalidLiveRangeTodoList('Provide at least one tool to schedule.');
    }
    const rawEvents = request.tools as unknown[];

    const ids = new Set<string>();
    return rawEvents.map((item, index) => {
        const itemLabel = `Live range to-do item ${index + 1}`;
        if (!isRecord(item)) invalidLiveRangeTodoList(`${itemLabel} must be an object.`);
        const keys = Reflect.ownKeys(item as object);
        if (keys.length !== 1 || typeof keys[0] !== 'string') {
            invalidLiveRangeTodoList(`${itemLabel} requires exactly one tool-name key.`);
        }
        const toolName = keys[0] as string;
        try {
            dispatchNested.validate(toolName);
        } catch (error) {
            invalidLiveRangeTodoList(error instanceof Error ? error.message : String(error));
        }
        const toolValue = (item as Record<string, unknown>)[toolName];
        if (!isRecord(toolValue)) invalidLiveRangeTodoList(`${itemLabel} tool must be an object.`);
        const rawItem = toolValue as Record<string, unknown>;
        assertExactKeys(rawItem, ['event', 'arguments'], itemLabel);
        if (!hasOwn(rawItem, 'event') || !hasOwn(rawItem, 'arguments')) {
            invalidLiveRangeTodoList(`${itemLabel} requires event and arguments objects.`);
        }

        const eventValue = rawItem.event;
        if (!isRecord(eventValue)) invalidLiveRangeTodoList(`${itemLabel} event must be an object.`);
        const rawEvent = eventValue as Record<string, unknown>;
        assertExactKeys(
            rawEvent,
            ['id', 'normalized_position', 'lead_time_seconds', 'content'],
            `${itemLabel} event`,
        );
        const id = typeof rawEvent.id === 'string' ? rawEvent.id.trim() : '';
        if (!id) invalidLiveRangeTodoList(`${itemLabel} event requires a non-empty id.`);
        if (ids.has(id)) invalidLiveRangeTodoList(`Duplicate live range to-do event id: ${id}.`);
        ids.add(id);
        if (
            typeof rawEvent.normalized_position !== 'number'
            || !Number.isFinite(rawEvent.normalized_position)
            || rawEvent.normalized_position < 0
            || rawEvent.normalized_position > 1
        ) {
            invalidLiveRangeTodoList(`Event '${id}' normalized_position must be between 0 and 1.`);
        }
        if (hasOwn(rawEvent, 'lead_time_seconds') && (
            typeof rawEvent.lead_time_seconds !== 'number'
            || !Number.isFinite(rawEvent.lead_time_seconds)
            || rawEvent.lead_time_seconds < 0
        )) {
            invalidLiveRangeTodoList(`Event '${id}' lead_time_seconds must be zero or greater.`);
        }
        if (!isRecord(rawEvent.content)) {
            invalidLiveRangeTodoList(`Event '${id}' requires a structured content object.`);
        }
        const rawContent = rawEvent.content as Record<string, unknown>;
        assertExactKeys(rawContent, ['title', 'description'], `Event '${id}' content`);
        const title = typeof rawContent.title === 'string'
            ? rawContent.title.trim()
            : '';
        if (!title) invalidLiveRangeTodoList(`Event '${id}' content requires a non-empty title.`);
        if (hasOwn(rawContent, 'description')
            && typeof rawContent.description !== 'string') {
            invalidLiveRangeTodoList(`Event '${id}' content description must be a string.`);
        }

        const rawTool = rawItem;
        if (!hasOwn(rawTool, 'arguments') || !isRecord(rawTool.arguments)) {
            invalidLiveRangeTodoList(`Scheduled tool '${toolName}' requires an arguments object.`);
        }
        if (!isJsonSafe(rawTool.arguments)) {
            invalidLiveRangeTodoList(`Scheduled tool '${toolName}' arguments must be JSON-safe.`);
        }
        const normalizedPosition = rawEvent.normalized_position as number;
        const leadTimeSeconds = rawEvent.lead_time_seconds as number | undefined;
        const description = rawContent.description as string | undefined;
        const toolArguments = rawTool.arguments as Record<string, unknown>;

        return {
            event: {
                id,
                normalized_position: normalizedPosition,
                ...(leadTimeSeconds !== undefined
                    ? { lead_time_seconds: leadTimeSeconds }
                    : {}),
                content: {
                    title,
                    ...(description !== undefined
                        ? { description }
                        : {}),
                },
            },
            tool: {
                name: toolName as FrontendToolName,
                arguments: JSON.parse(JSON.stringify(toolArguments)),
            },
        };
    });
};

const getOrInitializeLiveRangeTodoList = (
    context: FrontendAiCommandContext,
): LiveRangeTodoListHandle => {
    const directory = getDirectory(context);
    const mounted = directory.findComponentRef<LiveRangeTodoListHandle>(
        OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
    )?.current;
    if (mounted) {
        directory.findComponentRef<WorkflowPanelHandle>(OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
            ?.current
            ?.initializeLiveRangeTodoList?.();
        return mounted;
    }
    return getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
        .initializeLiveRangeTodoList();
};

const createScheduledTaskStart = (
    descriptor: PreparedLiveRangeTodoEvent['tool'],
    dispatchNested: ToolDispatcher,
): LiveRangeTodoEventInput['taskStart'] => (signal) => {
    if (signal.aborted) {
        return asTool(createOperationFrom(() => {
            throw createLiveRangeAbortError();
        }, 'failed'));
    }
    return dispatchNested(descriptor.name, descriptor.arguments, signal);
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
    dispatchNested: ToolDispatcher,
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
        const todoList = getOrInitializeLiveRangeTodoList(context);
        const queuedIds = new Set(todoList.get().todo_list?.events.map((event) => event.id) ?? []);
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
            todoList.addEvent({
                id: comparison.eventId,
                normalized_position: comparison.normalizedPosition,
                lead_time_seconds: comparison.leadTimeSeconds,
                content: {
                    title: comparison.title,
                    ...(comparison.section
                        ? { description: `Section: ${comparison.section}` }
                        : {}),
                },
                taskStart: createScheduledTaskStart({
                    name: 'display_specific_result_in_overlay',
                    arguments: {
                        page_id: snapshot.activePageId,
                        result_id: comparison.segmentId,
                    },
                }, dispatchNested),
            });
            result.queued_count += 1;
        });
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
        execute: (context, args, dispatchNested) => {
            const prepared = validateLiveRangeTodoBatch(args, dispatchNested);
            const mounted = getDirectory(context).findComponentRef<LiveRangeTodoListHandle>(
                OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
            )?.current;
            const existingIds = new Set(
                mounted?.get().todo_list?.events.map((event) => event.id) ?? [],
            );
            const collision = prepared.find(({ event }) => existingIds.has(event.id));
            if (collision) {
                invalidLiveRangeTodoList(
                    `Duplicate live range to-do event id: ${collision.event.id}.`,
                );
            }
            const todoList = getOrInitializeLiveRangeTodoList(context);
            prepared.forEach(({ event, tool }) => {
                todoList.addEvent({
                    ...event,
                    taskStart: createScheduledTaskStart(tool, dispatchNested),
                });
            });
            return todoList.getForAi();
        },
    },
    {
        name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        kind: 'workflow',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args, dispatchNested) => {
            const controller = createControlledOperation<AddFilteredDriverExpertComparisonsResult>();
            void Promise.resolve().then(async () => {
                if (controller.signal.aborted) return;
                validateNoArguments(
                    args,
                    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
                );
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
            return controller.operation;
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
        name: 'query_analysis_result',
        kind: 'tool',
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args) => {
            const query = validateAnalysisResultQueryArguments(args);
            return getComponent<AnalysisResultsChartHandle>(
                context,
                getSingletonVisualizationComponentName('analysis-results'),
            ).queryAnalysisResult(query);
        },
    },
    {
        name: 'create_repeatable_plan',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        execute: (context, args, dispatchNested) => {
            const validation = validateGoalRequest(args);
            if ('error' in validation) throw validation.error;
            validation.request.steps.forEach((step) => dispatchNested.validate(step.name));
            dispatchNested.validate(validation.request.stop_when.tool.name);
            return getComponent<WorkflowPanelHandle>(context, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL)
                .createRepeatablePlan(args as RepeatablePlanInput, dispatchNested);
        },
    },
    {
        name: 'retry_repeatable_plan_task',
        kind: 'workflow',
        componentName: OPERATION_COMPONENT_NAMES.REPEATABLE_PLAN,
        execute: (context) => getComponent<RepeatablePlanHandle>(context, OPERATION_COMPONENT_NAMES.REPEATABLE_PLAN)
            .retryFailedTask(),
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
            const plan = parseProcedurePlanInput(args);
            plan.requests.forEach((request) => dispatchNested.validate(request.name!));
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
    create_repeatable_plan: RepeatablePlanInput;
    add_event_to_live_range_todo_list: LiveRangeTodoListInput;
};

export type AiCommandRegistry = NonQueryAiCommandRegistry & FrontendAiQueryContractMap & {
    display_specific_result_in_overlay(
        args: DisplaySpecificResultInOverlayArguments,
    ): FrontendOperation<'display_specific_result_in_overlay'>;
};

const definitions = Object.fromEntries(
    definitionList.map((definition) => [definition.name, definition]),
) as FrontendOperationDefinitionMap;

export const frontendOperationRegistry = definitions;

export const createWorkflowToolDispatcher = (
    context: FrontendAiCommandContext,
): ToolDispatcher => {
    const validate = (name: string): void => {
        const definition = Object.prototype.hasOwnProperty.call(definitions, name)
            ? definitions[name as FrontendOperationName]
            : undefined;
        if (!definition) throw new OperationNotRegisteredError(`Tool '${name}' is not registered.`);
        if (definition.kind !== 'tool') {
            throw new OperationNotRegisteredError(`Workflow '${name}' cannot be used as a child tool.`);
        }
    };
    return Object.assign((name: FrontendToolName, args: Record<string, unknown> = {}, signal?: AbortSignal) => {
        validate(name);
        const operation = dispatchOperation(context, name, args, signal);
        if (operation.kind !== 'tool') {
            throw new InvalidOperationCallError(`Child '${name}' did not return a Tool.`);
        }
        return operation;
    }, { validate });
};

const dispatchOperation = (
    context: FrontendAiCommandContext,
    name: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
): Tool<OperationExecutionOutput, OperationStatusPayload> | Workflow<OperationExecutionOutput, OperationStatusPayload> => {
    const definition = Object.prototype.hasOwnProperty.call(definitions, name)
        ? definitions[name as FrontendOperationName]
        : undefined;
    try {
        if (!definition) throw new OperationNotRegisteredError(`Operation '${name}' is not registered.`);
        const dispatchNested = createWorkflowToolDispatcher(context);
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
        return definition?.kind === 'workflow' ? asWorkflow(operation) : asTool(operation);
    }
};

export const createAiCommandRegistry = (
    context: FrontendAiCommandContext,
): AiCommandRegistry => Object.fromEntries(
    definitionList.map((definition) => [
        definition.name,
        (args: Record<string, any>, signal?: AbortSignal) => dispatchOperation(
            context,
            definition.name,
            args,
            signal,
        ),
    ]),
) as unknown as AiCommandRegistry;

export const isAiCommandName = (name: string): name is FrontendOperationName => (
    Object.prototype.hasOwnProperty.call(definitions, name)
);
