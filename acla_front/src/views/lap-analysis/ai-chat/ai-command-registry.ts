import type { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { detectOvertakeTacticalState } from './overtake-agent-detector';
import {
    CreateGoalToolUnavailableError,
    InvalidToolCallError,
    NoLiveSessionError,
    NoLiveTelemetryError,
    RetryGoalTaskToolUnavailableError,
    ToolNotRegisteredError,
    createAiToolOperationFrom,
    type AiToolOperation,
    type AiToolExecutionOutput,
    type AiToolStatusPayload,
} from './ai-tool-base';
import {
    AI_TOOL_COMPONENT_NAMES,
    ComponentRefUnavailableError,
    resolveNamedComponentHandle,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    InvalidLiveRangeTodoListError,
    NonLiveContextLiveToolsUnavailableError,
    RecordedSessionLiveToolsUnavailableError,
} from 'contexts/AiToolComponentError';
import { isLiveSessionAiAvailable, type RecordingState } from 'views/lap-analysis/recording-state';
import type {
    AiToolDispatcher,
    GoalHandle,
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
    ProcedurePlanHandle,
    ProcedurePlanRunResult,
    ProcedurePlanState,
    AiToolQueryResult,
} from 'components/ai-engineering-tools';
import {
    calculateForwardCircularDistance,
    getLiveRangeNormalizedPosition,
} from 'components/ai-engineering-tools';
import type { BaselineCollectionHandle } from 'views/live-session/BaselineCollection';
import type { AiChatHandle } from './ai-chat';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import type { SessionAnalysisHandle } from 'views/lap-analysis/session-analysis';
import type { UserSummaryHandle } from 'views/user-summary/user-summary';
import type { AiMapDisplayPayload } from './AiMapToolDisplay';
import type {
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
import type { DriverExpertComparisonSnapshot } from 'components/driver-expert-comparison';
import { OVERLAY_COMPARISON_COMPLETION_PAUSE_MS } from 'views/floating-chat/ai-overlay-types';

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
    componentRefs?: AiToolComponentRefDirectory;
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
    queued_timing: Array<{
        segment_id: string;
        event_id: string;
        normalized_position: number;
        replay_duration_ms: number;
        lead_time_seconds: number;
    }>;
    skipped_segments: Array<{
        segment_id: string;
        event_id: string;
        reason_code: FilteredComparisonSkipReason;
    }>;
}

export interface AiCommandRegistryContext extends FrontendAiCommandContext {
    recordingState?: RecordingState | null;
    activeAgentSession?: AgentSessionInfo | null;
    analysisContext?: any;
    sessionIntelligence?: SessionIntelligence | null;
    opportunityAgentState: OpportunityAgentState;
    livePerformanceAnalystState?: LivePerformanceAnalystState;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
    setLivePerformanceAnalystEnabled?: (enabled: boolean) => void;
    advanceProcedurePlanStep?: (reason?: string) => Promise<ProcedurePlanRunResult>;
    getProcedurePlan?: () => ProcedurePlanState | null;
    clearProcedurePlan?: () => void;
    setProcedurePlan?: (plan: ProcedurePlanState | null) => void;
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
        ? RecordedSessionLiveToolsUnavailableError
        : NonLiveContextLiveToolsUnavailableError
);

const getBaselineReadiness = (context: AiCommandRegistryContext) => {
    const baseline = context.componentRefs?.findComponentRef<BaselineCollectionHandle>(
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    )?.current;
    const record = baseline?.getLapRecord() ?? null;
    const tag = baseline?.getTag() ?? null;
    return { record, tag, ready: Boolean(record?.records?.length) };
};

const buildLiveAnalystSnapshot = (context: AiCommandRegistryContext) => {
    const snapshot = context.sessionIntelligence?.getLiveSessionSnapshot() as Record<string, any> | undefined;
    const { record, tag, ready } = getBaselineReadiness(context);
    return {
        ...(snapshot ?? {}),
        baseline_ready: ready,
        baseline_collection_started: ready || tag?.status === 'collecting',
        baseline_progress_percent: ready ? 100 : tag?.progress_percent ?? 0,
        baseline_lap: record?.lap ?? tag?.baseline_lap ?? null,
        baseline_record_sample_count: record?.sample_count ?? 0,
    };
};

export const startAgentRuntime = async (
    agentMode: AgentSessionMode,
    context: AiCommandRegistryContext,
    args: Record<string, unknown>,
    publishStatus: (data: Record<string, unknown>) => void = () => undefined,
): Promise<AiToolExecutionOutput> => {
    if (!isLiveSessionContext(context)) {
        const ErrorType = getLiveUnavailableError(context);
        throw new ErrorType(
            AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
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

    const intelligence = context.sessionIntelligence;
    if (!intelligence) throw new NoLiveSessionError('Live session intelligence is unavailable.');
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

export const FRONTEND_AI_TOOL_NAMES = Object.freeze([
    'start_agent_session',
    'stop_agent_session',
    'add_event_to_live_range_todo_list',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'apply_analysis_result_query',
    'query_analysis_result',
    'create_goal',
    'retry_goal_task',
    'advance_plan_step',
    'clear_procedure_plan',
    'set_procedure_plan',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
    'show_map',
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
    'analyze_telemetry',
] as const);

export type FrontendAiToolName = typeof FRONTEND_AI_TOOL_NAMES[number];
export type FrontendAiQueryName = Extract<FrontendAiToolName, `query_${string}`>;

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
> = AiToolQueryResult<QueryResult<TReduce>>;

export type FrontendAiQueryContractMap = {
    query_analysis_result: (
        args: QueryAnalysisResultInput,
    ) => AiToolOperation<QueryAnalysisResultOutput>;
    query_telemetry_metric: <TReduce extends TelemetryMetricReduce>(
        args: QueryTelemetryMetricArguments<TReduce>,
    ) => AiToolOperation<QueryTelemetryMetricResult<TReduce>>;
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
        throw new InvalidToolCallError(validationMessage);
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
        throw new InvalidToolCallError(validationMessage);
    }
    return { query: queryProperty.value };
};

const validateApplyAnalysisResultQueryArguments = (
    args: unknown,
): ApplyAnalysisResultQueryInput => {
    const validationMessage = 'apply_analysis_result_query requires a non-empty string property named query and accepts only an optional integer property named page_number.';
    if (!args || typeof args !== 'object' || Array.isArray(args)) {
        throw new InvalidToolCallError(validationMessage);
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
        throw new InvalidToolCallError(validationMessage);
    }
    return {
        query: queryProperty.value,
        ...(pageNumberProperty ? { page_number: pageNumberProperty.value as number } : {}),
    };
};

type WorkflowOwner = 'chat' | 'goal' | 'procedure_plan' | 'live_range_todo';

type FrontendAiToolDefinition = {
    readonly name: FrontendAiToolName;
    readonly componentName: string;
    readonly execute: (
        context: FrontendAiCommandContext,
        args: Record<string, any>,
        dispatchNested: AiToolDispatcher,
    ) => AiToolOperation<AiToolExecutionOutput, AiToolStatusPayload>;
};

const getDirectory = (context: FrontendAiCommandContext): AiToolComponentRefDirectory => {
    if (context.componentRefs) return context.componentRefs;
    throw new ComponentRefUnavailableError(
        'dashboard',
        'The active dashboard component-ref directory is unavailable.',
    );
};

const getComponent = <T,>(context: FrontendAiCommandContext, name: string): T => (
    resolveNamedComponentHandle(getDirectory(context), name) as T
);

const WORKFLOW_CONTROL_TOOLS = new Set<FrontendAiToolName>([
    'create_goal',
    'retry_goal_task',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
]);

const LIVE_RANGE_TODO_TOOLS = new Set<FrontendAiToolName>([
    'add_event_to_live_range_todo_list',
    'get_live_range_todo_list',
]);

const NON_CHILD_LIVE_SESSION_TOOLS = new Set<FrontendAiToolName>([
    'start_agent_session',
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
]);

const LIVE_RANGE_TODO_NESTED_TOOLS = new Set<FrontendAiToolName>(
    FRONTEND_AI_TOOL_NAMES.filter((name) => (
        !WORKFLOW_CONTROL_TOOLS.has(name)
        && name !== 'add_event_to_live_range_todo_list'
        && name !== 'add_filtered_driver_expert_comparisons_to_live_range_todo_list'
        && !NON_CHILD_LIVE_SESSION_TOOLS.has(name)
    )),
);

const GOAL_STEP_TOOLS = new Set<FrontendAiToolName>([
    'stop_agent_session',
    'add_event_to_live_range_todo_list',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'apply_analysis_result_query',
    'query_analysis_result',
    'advance_plan_step',
    'clear_procedure_plan',
    'set_procedure_plan',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
    'show_map',
    'analyze_telemetry',
]);

export const isGoalStepAvailableForContext = (
    context: Pick<FrontendAiCommandContext, 'sessionMode' | 'conversationRole' | 'agentMode'>,
    name: string,
): boolean => (
    context.sessionMode === 'live'
    && context.conversationRole === 'agent'
    && context.agentMode === 'live_performance_analyst'
    && GOAL_STEP_TOOLS.has(name as FrontendAiToolName)
);

const assertAvailable = (
    context: FrontendAiCommandContext,
    name: FrontendAiToolName,
    owner: WorkflowOwner,
) => {
    const isChildLiveSession = context.sessionMode === 'live'
        && context.conversationRole === 'agent';
    if (owner === 'chat' && LIVE_RANGE_TODO_TOOLS.has(name) && !isChildLiveSession) {
        throw new ToolNotRegisteredError(
            `Tool '${name}' is available only in child live-agent sessions.`,
        );
    }
    if (owner === 'live_range_todo' && (
        !isChildLiveSession
        || !LIVE_RANGE_TODO_NESTED_TOOLS.has(name)
    )) {
        throw new ToolNotRegisteredError(
            `Tool '${name}' cannot be scheduled by the live range to-do list.`,
        );
    }
    if (owner === 'goal' && !isGoalStepAvailableForContext(context, name)) {
        throw new ToolNotRegisteredError(`Tool '${name}' is unavailable inside a goal.`);
    }
    if (owner === 'procedure_plan' && WORKFLOW_CONTROL_TOOLS.has(name)) {
        throw new ToolNotRegisteredError(`Tool '${name}' is unavailable inside a procedure plan.`);
    }
    if (name === 'create_goal' && (
        context.sessionMode !== 'live'
        || context.conversationRole !== 'agent'
        || context.agentMode !== 'live_performance_analyst'
    )) {
        throw new CreateGoalToolUnavailableError(
            'Goal creation is available only to the live performance analyst.',
        );
    }
    if (name === 'retry_goal_task' && (
        context.sessionMode !== 'live'
        || context.conversationRole !== 'agent'
        || context.agentMode !== 'live_performance_analyst'
    )) {
        throw new RetryGoalTaskToolUnavailableError(
            'Goal task retry is available only to the live performance analyst.',
        );
    }
    if (name === 'add_filtered_driver_expert_comparisons_to_live_range_todo_list' && (
        context.sessionMode !== 'live'
        || context.conversationRole !== 'agent'
        || context.agentMode !== 'live_performance_analyst'
    )) {
        throw new ToolNotRegisteredError(
            `Tool '${name}' is available only to the live performance analyst.`,
        );
    }
};

const hasOwn = (value: Record<string, unknown>, key: string): boolean => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const validateNoArguments = (args: unknown, toolName: string): void => {
    if (!isRecord(args) || Reflect.ownKeys(args).length > 0) {
        throw new InvalidToolCallError(`${toolName} does not accept arguments.`);
    }
};

const invalidLiveRangeTodoList = (message: string): never => {
    throw new InvalidLiveRangeTodoListError(
        AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
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
        name: FrontendAiToolName;
        arguments: Record<string, unknown>;
    };
};

const validateLiveRangeTodoBatch = (
    args: unknown,
): PreparedLiveRangeTodoEvent[] => {
    if (!isRecord(args)) invalidLiveRangeTodoList('Provide an object containing events.');
    const request = args as Record<string, unknown>;
    assertExactKeys(request, ['events'], 'Live range to-do request');
    if (!Array.isArray(request.events) || request.events.length === 0) {
        invalidLiveRangeTodoList('Provide at least one event to add.');
    }
    const rawEvents = request.events as unknown[];

    const ids = new Set<string>();
    return rawEvents.map((item, index) => {
        const itemLabel = `Live range to-do item ${index + 1}`;
        if (!isRecord(item)) invalidLiveRangeTodoList(`${itemLabel} must be an object.`);
        const rawItem = item as Record<string, unknown>;
        assertExactKeys(rawItem, ['event', 'tool'], itemLabel);
        if (!hasOwn(rawItem, 'event') || !hasOwn(rawItem, 'tool')) {
            invalidLiveRangeTodoList(`${itemLabel} requires event and tool objects.`);
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

        const toolValue = rawItem.tool;
        if (!isRecord(toolValue)) invalidLiveRangeTodoList(`${itemLabel} tool must be an object.`);
        const rawTool = toolValue as Record<string, unknown>;
        assertExactKeys(rawTool, ['name', 'arguments'], `${itemLabel} tool`);
        const toolName = typeof rawTool.name === 'string' ? rawTool.name.trim() : '';
        if (!toolName || !LIVE_RANGE_TODO_NESTED_TOOLS.has(toolName as FrontendAiToolName)) {
            invalidLiveRangeTodoList(
                `Tool '${toolName || '(missing)'}' cannot be scheduled by the live range to-do list.`,
            );
        }
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
                name: toolName as FrontendAiToolName,
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
        AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
    )?.current;
    if (mounted) {
        directory.findComponentRef<AiChatHandle>(AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            ?.current
            ?.initializeLiveRangeTodoList?.();
        return mounted;
    }
    return getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
        .initializeLiveRangeTodoList();
};

const createScheduledTaskStart = (
    descriptor: PreparedLiveRangeTodoEvent['tool'],
    dispatchNested: AiToolDispatcher,
): LiveRangeTodoEventInput['taskStart'] => async (signal) => {
    if (signal.aborted) {
        const error = new Error('The live range to-do task was aborted.');
        error.name = 'AbortError';
        throw error;
    }
    const operation = dispatchNested(descriptor.name, descriptor.arguments);
    const result = await operation.result;
    if (result instanceof Error) throw result;
    return result;
};

const createLiveRangeAbortError = (): Error => {
    const error = new Error('The live range to-do task was aborted.');
    error.name = 'AbortError';
    return error;
};

const waitForLiveRangeReplay = (
    durationMs: number,
    signal: AbortSignal,
): Promise<void> => new Promise((resolve, reject) => {
    if (signal.aborted) {
        reject(createLiveRangeAbortError());
        return;
    }

    let timer: ReturnType<typeof setTimeout>;
    const handleAbort = () => {
        clearTimeout(timer);
        signal.removeEventListener('abort', handleAbort);
        reject(createLiveRangeAbortError());
    };
    timer = setTimeout(() => {
        signal.removeEventListener('abort', handleAbort);
        resolve();
    }, durationMs + OVERLAY_COMPARISON_COMPLETION_PAUSE_MS);
    signal.addEventListener('abort', handleAbort, { once: true });
});

type EligibleFilteredComparison = {
    segmentId: string;
    eventId: string;
    normalizedPosition: number;
    replayDurationMs: number;
    leadTimeSeconds: number;
    estimatedLapTimeMs: number | null;
    overlay: DriverExpertComparisonSnapshot;
    section?: string;
};

const getPositiveFiniteNumber = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

const getLatestLiveArrivalTiming = (
    aiChat: AiChatHandle,
    todoList: LiveRangeTodoListHandle,
): { currentPosition: number | null; estimatedLapTimeMs: number | null } => {
    const rows = aiChat.getOpportunityTelemetryRows?.() ?? [];
    let currentPosition: number | null = null;
    let estimatedLapTimeMs: number | null = null;
    for (let index = rows.length - 1; index >= 0; index -= 1) {
        const row = rows[index];
        if (currentPosition === null) {
            currentPosition = getLiveRangeNormalizedPosition(row) ?? null;
        }
        if (estimatedLapTimeMs === null) {
            estimatedLapTimeMs = [
                row.Graphics_estimated_lap_time,
                row.Graphics_last_time,
                row.Graphics_best_time,
            ].map(getPositiveFiniteNumber).find((value) => value !== null) ?? null;
        }
        if (currentPosition !== null && estimatedLapTimeMs !== null) break;
    }
    if (currentPosition === null) {
        currentPosition = todoList.get().todo_list?.current_position ?? null;
    }
    return { currentPosition, estimatedLapTimeMs };
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
    queued_timing: [],
    skipped_segments: [],
});

const queueFilteredDriverExpertComparisons = (
    context: FrontendAiCommandContext,
    snapshot: FilteredAnalysisSegmentsSnapshot,
): AddFilteredDriverExpertComparisonsResult => {
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
        const title = segment.title
            ? `${segment.title}: Driver vs Expert`
            : 'Driver vs Expert';
        const end = segment.normalizedPositionRange?.end;
        const segmentDistance = typeof end === 'number' && Number.isFinite(end)
            && end >= 0 && end <= 1
            ? calculateForwardCircularDistance(start, end)
            : 0;
        eligible.push({
            segmentId: segment.id,
            eventId,
            normalizedPosition: start,
            replayDurationMs,
            leadTimeSeconds: (replayDurationMs / 1000) + 2,
            estimatedLapTimeMs: segmentDistance > 0
                ? replayDurationMs / segmentDistance
                : null,
            overlay: {
                title,
                comparison: segment.comparison!,
                game: context.sessionGame ?? null,
            },
            section: segment.section,
        });
    });

    if (eligible.length > 0) {
        const todoList = getOrInitializeLiveRangeTodoList(context);
        const aiChat = getComponent<AiChatHandle>(
            context,
            AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        );
        const liveTiming = getLatestLiveArrivalTiming(aiChat, todoList);
        const existingIds = new Set(
            todoList.get().todo_list?.events.map((event) => event.id) ?? [],
        );
        eligible.forEach((comparison) => {
            if (existingIds.has(comparison.eventId)) {
                result.skipped_segments.push({
                    segment_id: comparison.segmentId,
                    event_id: comparison.eventId,
                    reason_code: 'already_queued',
                });
                return;
            }
            existingIds.add(comparison.eventId);
            const lapTimeMs = liveTiming.estimatedLapTimeMs
                ?? comparison.estimatedLapTimeMs;
            const etaSeconds = liveTiming.currentPosition !== null && lapTimeMs !== null
                ? calculateForwardCircularDistance(
                    liveTiming.currentPosition,
                    comparison.normalizedPosition,
                ) * lapTimeMs / 1000
                : null;
            todoList.addEvent({
                id: comparison.eventId,
                normalized_position: comparison.normalizedPosition,
                lead_time_seconds: comparison.leadTimeSeconds,
                ...(etaSeconds !== null ? { eta_seconds: etaSeconds } : {}),
                content: {
                    title: comparison.overlay.title,
                    ...(comparison.section
                        ? { description: `Section: ${comparison.section}` }
                        : {}),
                },
                taskStart: async (signal) => {
                    if (signal.aborted) throw createLiveRangeAbortError();
                    aiChat.displayDriverExpertComparison(comparison.overlay);
                    await waitForLiveRangeReplay(comparison.replayDurationMs, signal);
                },
            });
            result.queued_timing.push({
                segment_id: comparison.segmentId,
                event_id: comparison.eventId,
                normalized_position: comparison.normalizedPosition,
                replay_duration_ms: comparison.replayDurationMs,
                lead_time_seconds: comparison.leadTimeSeconds,
            });
        });
    }

    result.queued_count = result.queued_timing.length;
    result.skipped_count = result.skipped_segments.length;
    return result;
};

const definitionList = Object.freeze([
    {
        name: 'start_agent_session',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .startAgentSession(args.agent_mode ?? args.agentMode, args),
    },
    {
        name: 'stop_agent_session',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .stopAgentSession(args.agent_session_id ?? args.agentSessionId),
    },
    {
        name: 'add_event_to_live_range_todo_list',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args, dispatchNested) => {
            const prepared = validateLiveRangeTodoBatch(args);
            const todoList = getOrInitializeLiveRangeTodoList(context);
            const existingIds = new Set(
                todoList.get().todo_list?.events.map((event) => event.id) ?? [],
            );
            const collision = prepared.find(({ event }) => existingIds.has(event.id));
            if (collision) {
                invalidLiveRangeTodoList(
                    `Duplicate live range to-do event id: ${collision.event.id}.`,
                );
            }
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
        componentName: getSingletonVisualizationComponentName('analysis-results'),
        execute: (context, args) => createAiToolOperationFrom(() => {
            validateNoArguments(
                args,
                'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
            );
            const snapshot = getComponent<AnalysisResultsChartHandle>(
                context,
                getSingletonVisualizationComponentName('analysis-results'),
            ).getFilteredSegments();
            return queueFilteredDriverExpertComparisons(context, snapshot);
        }),
    },
    {
        name: 'get_live_range_todo_list',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        execute: (context) => getComponent<LiveRangeTodoListHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)
            .getForAi(),
    },
    {
        name: 'collect_live_baseline',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .collectLiveBaselineForAi(args),
    },
    {
        name: 'restart_live_baseline',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .restartLiveBaselineForAi(),
    },
    {
        name: 'analyze_live_recorded_analysis',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .analyzeLiveRecordedAnalysisForAi(args),
    },
    {
        name: 'apply_analysis_result_query',
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
        name: 'create_goal',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args, dispatchNested) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .createGoal(args, dispatchNested),
    },
    {
        name: 'retry_goal_task',
        componentName: AI_TOOL_COMPONENT_NAMES.GOAL,
        execute: (context) => getComponent<GoalHandle>(context, AI_TOOL_COMPONENT_NAMES.GOAL)
            .retryFailedTask(),
    },
    {
        name: 'advance_plan_step',
        componentName: AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
        execute: (context, args) => getComponent<ProcedurePlanHandle>(context, AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN)
            .advancePlanStep(typeof args.reason === 'string' ? args.reason : undefined),
    },
    {
        name: 'clear_procedure_plan',
        componentName: AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
        execute: (context, args) => getComponent<ProcedurePlanHandle>(context, AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN)
            .clearProcedurePlan(typeof args.reason === 'string' ? args.reason : undefined),
    },
    {
        name: 'set_procedure_plan',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args, dispatchNested) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .createProcedurePlan(args, dispatchNested),
    },
    {
        name: 'get_next_corner',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .getNextCornerForAi(),
    },
    {
        name: 'query_telemetry_metric',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .queryTelemetryMetricForAi(
                args as QueryTelemetryMetricArguments<TelemetryMetricReduce>,
            ),
    },
    {
        name: 'get_event_log',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .getEventLogForAi(args),
    },
    {
        name: 'get_user_summary_map_level',
        componentName: AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context, args) => getComponent<UserSummaryHandle>(context, AI_TOOL_COMPONENT_NAMES.USER_SUMMARY)
            .getUserSummaryMapLevel(args),
    },
    {
        name: 'get_available_user_summary_maps',
        componentName: AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context) => getComponent<UserSummaryHandle>(context, AI_TOOL_COMPONENT_NAMES.USER_SUMMARY)
            .getAvailableUserSummaryMaps(),
    },
    {
        name: 'search_user_summary_map_level',
        componentName: AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
        execute: (context, args) => getComponent<UserSummaryHandle>(context, AI_TOOL_COMPONENT_NAMES.USER_SUMMARY)
            .searchUserSummaryMapLevel(args),
    },
    {
        name: 'show_map',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .showMap(args),
    },
    {
        name: 'run_recorded_ai_analysis',
        componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)
            .runRecordedAnalysisForAi(args),
    },
    {
        name: 'get_recorded_session_analysis',
        componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)
            .getRecordedAnalysisForAi(args),
    },
    {
        name: 'get_recorded_session_context',
        componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
        execute: (context, args) => getComponent<SessionAnalysisHandle>(context, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)
            .getRecordedSessionContextForAi(args),
    },
    {
        name: 'analyze_telemetry',
        componentName: 'session-mode-analysis',
        execute: (context, args) => context.sessionMode === 'recorded'
            ? getComponent<SessionAnalysisHandle>(context, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)
                .analyzeTelemetryForAi(args)
            : getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
                .analyzeTelemetryForAi(args),
    },
] as const satisfies readonly FrontendAiToolDefinition[]);

type FrontendAiToolDefinitionMap = {
    [TDefinition in typeof definitionList[number] as TDefinition['name']]: TDefinition;
};

export type FrontendAiToolOperation<TName extends FrontendAiToolName> = (
    TName extends FrontendAiQueryName
        ? ReturnType<FrontendAiQueryContractMap[TName]>
        : ReturnType<FrontendAiToolDefinitionMap[TName]['execute']>
);

type NonQueryAiCommandRegistry = {
    [TName in Exclude<FrontendAiToolName, FrontendAiQueryName>]: (
        args: Record<string, unknown>,
    ) => FrontendAiToolOperation<TName>;
};

export type AiCommandRegistry = NonQueryAiCommandRegistry & FrontendAiQueryContractMap;

const definitions = Object.fromEntries(
    definitionList.map((definition) => [definition.name, definition]),
) as FrontendAiToolDefinitionMap;

export const frontendAiToolRegistry = definitions;

const dispatchAiTool = (
    context: FrontendAiCommandContext,
    name: string,
    args: Record<string, unknown>,
    owner: WorkflowOwner,
): AiToolOperation<AiToolExecutionOutput, AiToolStatusPayload> => {
    try {
        const definition = definitions[name as FrontendAiToolName];
        if (!definition) throw new ToolNotRegisteredError(`Tool '${name}' is not registered.`);
        assertAvailable(context, definition.name, owner);
        const nestedOwner: WorkflowOwner = definition.name === 'create_goal'
            ? 'goal'
            : definition.name === 'add_event_to_live_range_todo_list'
                ? 'live_range_todo'
                : 'procedure_plan';
        const dispatchNested: AiToolDispatcher = (nestedName, nestedArgs = {}) => dispatchAiTool(
            context,
            nestedName,
            nestedArgs,
            nestedOwner,
        ) as ReturnType<AiToolDispatcher>;
        return definition.execute(context, args, dispatchNested);
    } catch (error) {
        return createAiToolOperationFrom(() => { throw error; });
    }
};

export const createAiCommandRegistry = (
    context: FrontendAiCommandContext,
): AiCommandRegistry => Object.fromEntries(
    definitionList.map((definition) => [
        definition.name,
        (args: Record<string, any>) => dispatchAiTool(
            context,
            definition.name,
            args,
            'chat',
        ),
    ]),
) as unknown as AiCommandRegistry;

export const isAiCommandName = (name: string): name is FrontendAiToolName => (
    name in definitions
);
