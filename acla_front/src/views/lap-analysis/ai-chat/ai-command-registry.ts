import type { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import {
    DEFAULT_ANALYST_COOLDOWN_MS,
    DEFAULT_ANALYST_MIN_DISTANCE,
    DEFAULT_ANALYST_MIN_LEAD_SECONDS,
    hasEnoughCoachingLead,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import { detectOvertakeTacticalState } from './overtake-agent-detector';
import {
    CreateGoalToolUnavailableError,
    LivePerformanceAnalystToolUnavailableError,
    NoLiveSessionError,
    NoLiveTelemetryError,
    RetryGoalTaskToolUnavailableError,
    ToolNotRegisteredError,
    createAiToolOperationFrom,
    mapAiToolOperation,
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
    NonLiveContextLiveToolsUnavailableError,
    RecordedSessionLiveToolsUnavailableError,
} from 'contexts/AiToolComponentError';
import { isLiveSessionAiAvailable, type RecordingState } from 'views/lap-analysis/recording-state';
import type {
    AiToolDispatcher,
    GoalHandle,
    LiveRangeTodoListHandle,
    ProcedurePlanHandle,
    ProcedurePlanRunResult,
    ProcedurePlanState,
    LiveRangeTodoDueStatus,
} from 'components/ai-engineering-tools';
import type { BaselineCollectionHandle } from 'views/live-session/BaselineCollection';
import type { AiChatHandle } from './ai-chat';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import type { SessionAnalysisHandle } from 'views/lap-analysis/session-analysis';
import type { UserSummaryHandle } from 'views/user-summary/user-summary';
import type { AiMapDisplayPayload } from './AiMapToolDisplay';

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
    enrichLiveRangeTodoStatus?: (
        status: LiveRangeTodoDueStatus,
    ) => AiToolStatusPayload | PromiseLike<AiToolStatusPayload>;
}

export interface OpportunityAgentState {
    intervalId: ReturnType<typeof setInterval> | null;
    inFlight: boolean;
    lastAlertKey: string | null;
    lastAlertAt: number;
}

export interface LivePerformanceAnalystState {
    intervalId: ReturnType<typeof setInterval> | null;
    inFlight: boolean;
    enabled: boolean;
    lastToolStatusKey: string | null;
    lastToolStatusAt: number;
    lastSpokenAt: number;
    analysisSessionId?: string | null;
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
const DEFAULT_LIVE_ANALYST_INTERVAL_SECONDS = 4;
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

const buildLiveFocus = (context: AiCommandRegistryContext) => {
    const intelligence = context.sessionIntelligence;
    if (!intelligence || !getBaselineReadiness(context).ready) return null;
    const focus = intelligence.getFocusSection();
    if (!focus) return null;
    return {
        section: focus.section,
        baseline: focus.baseline,
        selected_at: focus.selectedAt,
        reason: focus.reason,
        score: focus.score,
        timing: intelligence.getSectionTiming(focus.section),
        show_map_arguments: {
            source_track_key: intelligence.getLiveSessionSnapshot().track,
            section_start: focus.section.from,
            section_end: focus.section.to,
            section_label: focus.section.name,
            title: 'Live analyst focus',
        },
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
    const state = context.livePerformanceAnalystState ?? {
        intervalId: null,
        inFlight: false,
        enabled: false,
        lastToolStatusKey: null,
        lastToolStatusAt: 0,
        lastSpokenAt: 0,
    };
    if (state.intervalId) {
        return {
            status: 'already_running',
            agent_mode: agentMode,
            snapshot: buildLiveAnalystSnapshot(context),
            focus: buildLiveFocus(context),
        };
    }
    const intervalSeconds = clampInterval(
        args.interval_seconds,
        DEFAULT_LIVE_ANALYST_INTERVAL_SECONDS,
        2,
        12,
    );
    state.enabled = true;
    context.setLivePerformanceAnalystEnabled?.(true);
    context.setAgentTagActive?.('Live Analyst', true);
    intelligence.emitLiveAnalysisPlanStarted();
    const runCycle = (notify: boolean) => {
        if (state.inFlight) return { status: 'skipped_in_flight' };
        state.inFlight = true;
        try {
            const snapshot = buildLiveAnalystSnapshot(context);
            const focus = buildLiveFocus(context);
            if (notify && focus) {
                const now = Date.now();
                const key = `focus:${focus.section.id}:${focus.baseline.lap}:${focus.baseline.observedAt}`;
                const canSpeak = hasEnoughCoachingLead(
                    focus.timing.distanceAhead,
                    focus.timing.secondsAhead,
                    DEFAULT_ANALYST_MIN_DISTANCE,
                    DEFAULT_ANALYST_MIN_LEAD_SECONDS,
                ) && now - state.lastSpokenAt >= DEFAULT_ANALYST_COOLDOWN_MS;
                if (canSpeak && state.lastToolStatusKey !== key) {
                    state.lastToolStatusKey = key;
                    state.lastSpokenAt = now;
                    intelligence.emitLiveAnalysisWindow(snapshot, focus);
                }
            }
            return {
                status: 'checked',
                snapshot,
                section_count: intelligence.getKnownTrackSections().length,
                focus,
                history_count: intelligence.getSectionHistory(80).length,
            };
        } finally {
            state.inFlight = false;
        }
    };
    const initial = runCycle(true);
    state.intervalId = setInterval(() => runCycle(true), intervalSeconds * 1000);
    return { status: 'started', agent_mode: agentMode, interval_seconds: intervalSeconds, initial };
};

export const FRONTEND_AI_TOOL_NAMES = Object.freeze([
    'start_agent_session',
    'stop_agent_session',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'get_live_analysis_mistake_count',
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
    'classify_live_section',
] as const);

export type FrontendAiToolName = typeof FRONTEND_AI_TOOL_NAMES[number];
type WorkflowOwner = 'chat' | 'goal' | 'procedure_plan';

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

const GOAL_STEP_TOOLS = new Set<FrontendAiToolName>([
    'stop_agent_session',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'get_live_analysis_mistake_count',
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
    'classify_live_section',
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
    if (name === 'get_live_analysis_mistake_count' && (
        context.conversationRole !== 'agent'
        || context.agentMode !== 'live_performance_analyst'
    )) {
        throw new LivePerformanceAnalystToolUnavailableError(
            'This tool is available only to the live performance analyst.',
        );
    }
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
        name: 'set_live_range_todo_list',
        componentName: AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
        execute: (context, args) => getComponent<AiChatHandle>(context, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)
            .createLiveRangeTodoList(args),
    },
    {
        name: 'update_live_range_todo_list',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        execute: (context, args) => {
            const operation = getComponent<LiveRangeTodoListHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)
                .updateForAi(args);
            return context.enrichLiveRangeTodoStatus
                ? mapAiToolOperation(operation, (result) => result, context.enrichLiveRangeTodoStatus)
                : operation;
        },
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
        name: 'get_live_analysis_mistake_count',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .getLiveAnalysisMistakeCountForAi(),
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
            .queryTelemetryMetricForAi(args),
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
    {
        name: 'classify_live_section',
        componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        execute: (context, args) => getComponent<LiveSessionHandle>(context, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)
            .classifyLiveSectionForAi(args),
    },
] as const satisfies readonly FrontendAiToolDefinition[]);

type FrontendAiToolDefinitionMap = {
    [TDefinition in typeof definitionList[number] as TDefinition['name']]: TDefinition;
};

export type FrontendAiToolOperation<TName extends FrontendAiToolName> = ReturnType<
    FrontendAiToolDefinitionMap[TName]['execute']
>;

export type AiCommandRegistry = {
    [TName in FrontendAiToolName]: (
        args: Record<string, unknown>,
    ) => FrontendAiToolOperation<TName>;
};

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
        const dispatchNested: AiToolDispatcher = (nestedName, nestedArgs = {}) => dispatchAiTool(
            context,
            nestedName,
            nestedArgs,
            definition.name === 'create_goal' ? 'goal' : 'procedure_plan',
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
) as AiCommandRegistry;

export const isAiCommandName = (name: string): name is FrontendAiToolName => (
    name in definitions
);
