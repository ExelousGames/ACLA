import { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import { ToolHandlerContext } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { AiMapDisplayPayload } from './AiMapToolDisplay';
import {
    DEFAULT_ANALYST_COOLDOWN_MS,
    DEFAULT_ANALYST_MIN_DISTANCE,
    DEFAULT_ANALYST_MIN_LEAD_SECONDS,
    hasEnoughCoachingLead,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    type ProcedurePlanAdvanceResult,
    type ProcedurePlanState,
} from 'components/ai-engineering-tools';
import { detectOvertakeTacticalState } from './overtake-agent-detector';
import {
    AiToolError,
    AiToolDefinition,
    NoLiveSessionError,
    NoLiveTelemetryError,
    ToolNotRegisteredError,
    type AiToolExecutionOutput,
    type ToolOutputEnvelope,
    executeAiToolDefinition,
} from './ai-tool-base';
import type { LiveRangeTodoListToolResult } from 'components/ai-engineering-tools';
import { isLiveSessionAiAvailable, RecordingState } from 'views/lap-analysis/recording-state';
import {
    AI_TOOL_COMPONENT_NAMES,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    NonLiveContextLiveToolsUnavailableError,
    RecordedSessionLiveToolsUnavailableError,
} from 'contexts/AiToolComponentError';
import type { BaselineCollectionHandle } from 'views/live-session/BaselineCollection';
import {
    createRefBasedAiCommandFunctions,
    type RefAiCommandContext,
} from './ai-command-ref-functions';

type AiCommandHandler = (
    args: Record<string, any>,
    ctx: ToolHandlerContext,
) => Promise<ToolOutputEnvelope>;
export type AiCommandToolDefinition = AiToolDefinition<RefAiCommandContext, ToolHandlerContext>;
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
export interface AiCommandRegistryContext {
    componentRefs?: AiToolComponentRefDirectory;
    sessionId?: string;
    sessionMode?: 'front_desk' | 'live' | 'recorded' | 'user_summary';
    recordingState?: RecordingState | null;
    conversationRole?: AgentSessionRole;
    activeAgentSession?: AgentSessionInfo | null;
    analysisContext?: any;
    // Populated during live recording. Null in post-session analysis view.
    sessionIntelligence?: SessionIntelligence | null;
    opportunityAgentState: OpportunityAgentState;
    livePerformanceAnalystState?: LivePerformanceAnalystState;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
    setLivePerformanceAnalystEnabled?: (enabled: boolean) => void;
    advanceProcedurePlanStep?: (reason?: string) => ProcedurePlanAdvanceResult;
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
    setLiveRangeTodoList?: (args: Record<string, unknown>) => LiveRangeTodoListToolResult;
    updateLiveRangeTodoList?: (args: Record<string, unknown>) => LiveRangeTodoListToolResult;
    getLiveRangeTodoList?: () => LiveRangeTodoListToolResult;
    displayMap?: (display: AiMapDisplayPayload) => void;
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

// Frontend-implemented tool capabilities. This file owns executable browser
// handlers only; LLM-facing tool metadata is injected by the backend voice
// gateway from its frontend application tool registry.
const DEFAULT_OVERTAKE_AGENT_INTERVAL_SECONDS = 5;
const OVERTAKE_AGENT_MIN_INTERVAL_SECONDS = 2;
const OVERTAKE_AGENT_MAX_INTERVAL_SECONDS = 15;
const OVERTAKE_AGENT_REPEAT_ALERT_MS = 20000;
const DEFAULT_LIVE_ANALYST_INTERVAL_SECONDS = 4;
const LIVE_ANALYST_MIN_INTERVAL_SECONDS = 2;
const LIVE_ANALYST_MAX_INTERVAL_SECONDS = 12;
const toPositiveNumber = (value: unknown): number | undefined => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
};

const getAgentIntervalSeconds = (value: unknown): number => {
    const parsed = toPositiveNumber(value) ?? DEFAULT_OVERTAKE_AGENT_INTERVAL_SECONDS;
    return Math.min(
        OVERTAKE_AGENT_MAX_INTERVAL_SECONDS,
        Math.max(OVERTAKE_AGENT_MIN_INTERVAL_SECONDS, parsed),
    );
};

const getTacticalAlertKey = (result: any): string => {
    const section = result.projected_section || result.next_corner?.name || 'unknown-section';
    const opponent = result.opponent_id ?? result.opponent_slot ?? 'unknown-opponent';
    return `${result.event}:${opponent}:${section}`;
};





const getLiveAnalystIntervalSeconds = (value: unknown): number => {
    const parsed = toPositiveNumber(value) ?? DEFAULT_LIVE_ANALYST_INTERVAL_SECONDS;
    return Math.min(
        LIVE_ANALYST_MAX_INTERVAL_SECONDS,
        Math.max(LIVE_ANALYST_MIN_INTERVAL_SECONDS, parsed),
    );
};

const getLiveToolsUnavailableErrorType = (context: AiCommandRegistryContext) => (
    context.sessionMode === 'recorded'
        ? RecordedSessionLiveToolsUnavailableError
        : NonLiveContextLiveToolsUnavailableError
);

const isLiveSessionContext = (context: AiCommandRegistryContext): boolean =>
    (!context.sessionMode || context.sessionMode === 'live')
    && isLiveSessionAiAvailable(context.recordingState);



























const buildLiveAnalystUnavailable = (context: AiCommandRegistryContext): AiToolError | null => (
    !isLiveSessionContext(context)
        ? new (getLiveToolsUnavailableErrorType(context))(
            AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
            'Live performance analysis requires an active live recording.',
        )
        : !context.sessionIntelligence
            ? new NoLiveSessionError(
                'Live session intelligence is unavailable.',
            )
            : null
);

const getLiveAnalystState = (context: AiCommandRegistryContext): LivePerformanceAnalystState => {
    if (context.livePerformanceAnalystState) return context.livePerformanceAnalystState;
    return {
        intervalId: null,
        inFlight: false,
        enabled: false,
        lastToolStatusKey: null,
        lastToolStatusAt: 0,
        lastSpokenAt: 0,
    };
};

const getBaselineRecorderReadiness = (context: AiCommandRegistryContext) => {
    const baseline = context.componentRefs
        ?.findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)
        ?.current;
    const record = baseline?.getLapRecord() ?? null;
    const tag = baseline?.getTag() ?? null;
    const ready = Boolean(record?.records?.length);

    return { ready, record, tag };
};

const buildLiveAnalystSnapshot = (context: AiCommandRegistryContext): Record<string, any> => {
    const snapshot = context.sessionIntelligence?.getLiveSessionSnapshot?.() as Record<string, any> | undefined;
    const { record, tag, ready } = getBaselineRecorderReadiness(context);

    return {
        ...(snapshot ?? {}),
        baseline_ready: ready,
        baseline_collection_started: ready || tag?.status === 'collecting',
        baseline_progress_percent: ready ? 100 : tag?.progress_percent ?? snapshot?.baseline_progress_percent ?? 0,
        baseline_lap: record?.lap ?? tag?.baseline_lap ?? snapshot?.baseline_lap ?? null,
        baseline_record_sample_count: record?.sample_count ?? 0,
    };
};

const buildLiveFocusPayload = (context: AiCommandRegistryContext) => {
    const si = context.sessionIntelligence;
    if (!si) return null;

    if (!getBaselineRecorderReadiness(context).ready) return null;

    const focus = si.getFocusSection();
    if (!focus) return null;

    const timing = si.getSectionTiming(focus.section);
    return {
        section: focus.section,
        baseline: focus.baseline,
        selected_at: focus.selectedAt,
        reason: focus.reason,
        score: focus.score,
        timing,
        show_map_arguments: {
            source_track_key: si.getLiveSessionSnapshot().track,
            section_start: focus.section.from,
            section_end: focus.section.to,
            section_label: focus.section.name,
            title: 'Live analyst focus',
            note: focus.reason === 'repeated_mistake'
                ? 'Repeated mistake section'
                : 'Highest priority mistake section',
        },
    };
};






export const startAgentRuntime = async (
    agentMode: AgentSessionMode,
    context: AiCommandRegistryContext,
    args: Record<string, unknown>,
    ctx: ToolHandlerContext,
): Promise<AiToolExecutionOutput> => {
    if (agentMode === 'track_guide') {
        if (!isLiveSessionContext(context)) {
            const ErrorType = getLiveToolsUnavailableErrorType(context);
            throw new ErrorType(
                AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
                'Track guidance requires an active live recording.',
            );
        }
        context.startTrackGuide();
        context.setAgentTagActive?.('Track Guide', true);
        return { status: 'started', agent_mode: 'track_guide', enabled: true };
    }

    if (agentMode === 'overtake') {
        if (!isLiveSessionContext(context)) {
            const ErrorType = getLiveToolsUnavailableErrorType(context);
            throw new ErrorType(
                AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
                'Overtake analysis requires an active live recording.',
            );
        }
        const telemetryRows = context.getOpportunityTelemetryRows();
        if (telemetryRows.length === 0) {
            throw new NoLiveTelemetryError(
                'No live telemetry is available for overtake analysis.',
            );
        }

        const agent = context.opportunityAgentState;
        if (agent.intervalId) {
            context.setAgentTagActive?.('Overtake', true);
            return { status: 'already_running', agent_mode: 'overtake' };
        }

        const intervalSeconds = getAgentIntervalSeconds(args.interval_seconds);

        const runTacticalCycle = async (notify: boolean): Promise<any> => {
            if (agent.inFlight) {
                return { status: 'skipped_in_flight' };
            }

            const rows = context.getOpportunityTelemetryRows();
            if (rows.length === 0) {
                return { status: 'no_live_telemetry' };
            }

            agent.inFlight = true;
            try {
                const result = detectOvertakeTacticalState(rows);

                if (notify && result.status === 'actionable') {
                    const alertKey = getTacticalAlertKey(result);
                    const now = Date.now();
                    if (agent.lastAlertKey !== alertKey || now - agent.lastAlertAt > OVERTAKE_AGENT_REPEAT_ALERT_MS) {
                        agent.lastAlertKey = alertKey;
                        agent.lastAlertAt = now;
                        ctx.sendToolStatus({
                            ...result,
                            source: 'overtake_agent',
                            agent_mode: 'overtake',
                            telemetry_rows: rows.length,
                        });
                    }
                }

                return {
                    status: 'checked',
                    tactical_state: result,
                    telemetry_rows: rows.length,
                };
            } finally {
                agent.inFlight = false;
            }
        };

        const initial = await runTacticalCycle(false);

        agent.intervalId = setInterval(() => {
            void runTacticalCycle(true);
        }, intervalSeconds * 1000);
        context.setAgentTagActive?.('Overtake', true);

        return {
            status: 'started',
            agent_mode: 'overtake',
            interval_seconds: intervalSeconds,
            initial,
        };
    }

    const unavailable = buildLiveAnalystUnavailable(context);
    if (unavailable) throw unavailable;

    const si = context.sessionIntelligence!;
    const agent = getLiveAnalystState(context);
    if (agent.intervalId) {
        agent.enabled = true;
        context.setLivePerformanceAnalystEnabled?.(true);
        context.setAgentTagActive?.('Live Analyst', true);
        const snapshot = buildLiveAnalystSnapshot(context);
        return {
            status: 'already_running',
            agent_mode: 'live_performance_analyst',
            snapshot,
            focus: getBaselineRecorderReadiness(context).ready ? buildLiveFocusPayload(context) : null,
        };
    }

    const intervalSeconds = getLiveAnalystIntervalSeconds(args.interval_seconds);
    agent.enabled = true;
    context.setLivePerformanceAnalystEnabled?.(true);
    context.setAgentTagActive?.('Live Analyst', true);
    si.emitLiveAnalysisPlanStarted();

    const runAnalystCycle = async (notify: boolean): Promise<any> => {
        if (agent.inFlight) {
            return { status: 'skipped_in_flight' };
        }

        agent.inFlight = true;
        try {
            const snapshot = buildLiveAnalystSnapshot(context);
            const sections = si.getKnownTrackSections();
            const baselineReady = getBaselineRecorderReadiness(context).ready;
            const focus = baselineReady ? buildLiveFocusPayload(context) : null;

            if (notify) {
                const now = Date.now();
                if (!baselineReady) {
                    agent.lastToolStatusKey = `warmup:${snapshot.completed_laps}:${snapshot.sample_count}`;
                } else if (focus) {
                    const timing = focus.timing;
                    const key = `focus:${focus.section.id}:${focus.baseline.lap}:${focus.baseline.observedAt}`;
                    const canSpeak = hasEnoughCoachingLead(
                        timing.distanceAhead,
                        timing.secondsAhead,
                        DEFAULT_ANALYST_MIN_DISTANCE,
                        DEFAULT_ANALYST_MIN_LEAD_SECONDS,
                    ) && now - agent.lastSpokenAt >= DEFAULT_ANALYST_COOLDOWN_MS;

                    if (canSpeak && agent.lastToolStatusKey !== key) {
                        agent.lastToolStatusKey = key;
                        agent.lastToolStatusAt = now;
                        agent.lastSpokenAt = now;
                        si.emitLiveAnalysisWindow(snapshot, focus);
                    }
                }
            }

            return {
                status: 'checked',
                snapshot,
                section_count: sections.length,
                focus,
                history_count: si.getSectionHistory(80).length,
            };
        } finally {
            agent.inFlight = false;
        }
    };

    const initial = await runAnalystCycle(true);

    agent.intervalId = setInterval(() => {
        void runAnalystCycle(true);
    }, intervalSeconds * 1000);

    return {
        status: 'started',
        agent_mode: 'live_performance_analyst',
        interval_seconds: intervalSeconds,
        initial,
    };
};


const ALL_AI_TOOL_NAMES = [
    'start_agent_session',
    'stop_agent_session',
    'get_session_analysis',
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
    'get_performance_insights',
    'compare_lap_times',
    'query_telemetry_metric',
    '_get_telemetry_for_scope',
    'get_event_log',
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
    'get_next_corner',
    'get_live_focus_section',
    'get_live_section_history',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'get_live_analysis_mistake_count',
    'create_goal',
    'retry_goal_task',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    '_get_live_section_telemetry',
    '_record_live_section_classification',
    'analyze_telemetry',
    'classify_live_section',
    'follow_expert_line',
    'get_telemetry_data',
    'get_visualization_capabilities',
    'show_map',
    'open_visualization_chart',
    'close_visualization_chart',
    'invoke_visualization_control',
    'update_guidance_once',
    'add_imitation_guidance_chart',
    'remove_imitation_guidance_chart',
    'disable_ui_component',
] as const;

export const isAiCommandName = (name: string): boolean => (
    (ALL_AI_TOOL_NAMES as readonly string[]).includes(name)
);

const COMMON_SESSION_TOOL_NAMES = [
    'show_map',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'stop_agent_session',
] as const;

const LIVE_AGENT_SESSION_TOOL_NAMES = [
    'analyze_telemetry',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'classify_live_section',
] as const;

const USER_SUMMARY_SESSION_TOOL_NAMES = [
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
] as const;

export const isGoalStepAvailableForContext = (
    context: Pick<RefAiCommandContext, 'sessionMode' | 'conversationRole' | 'agentMode'>,
    name: string,
): boolean => {
    if (
        context.sessionMode !== 'live'
        || context.conversationRole !== 'agent'
        || context.agentMode !== 'live_performance_analyst'
        || name === 'create_goal'
        || name === 'retry_goal_task'
    ) {
        return false;
    }
    return new Set<string>([
        ...COMMON_SESSION_TOOL_NAMES,
        ...LIVE_AGENT_SESSION_TOOL_NAMES,
        ...USER_SUMMARY_SESSION_TOOL_NAMES,
        'get_live_analysis_mistake_count',
    ]).has(name);
};

const getToolUiRecord = (uiOutput: unknown): Record<string, any> => (
    uiOutput && typeof uiOutput === 'object' && !Array.isArray(uiOutput)
        ? uiOutput as Record<string, any>
        : {}
);

const getToolAiStatus = (uiOutput: Record<string, any>): string => (
    typeof uiOutput.status === 'string'
        ? uiOutput.status
        : 'complete'
);

const getToolAiMessage = (uiOutput: Record<string, any>, fallback: string): string => (
    typeof uiOutput.message === 'string' && uiOutput.message.trim()
        ? uiOutput.message
        : fallback
);

const omitOkForAi = (value: Record<string, any>): Record<string, unknown> => {
    const { ok: _ok, ...rest } = value;
    return rest;
};

const summarizeMapForAi = (uiOutput: Record<string, any>) => ({
    map_id: uiOutput.map_id ?? null,
    circuit_name: uiOutput.circuit_name ?? null,
    requested_map: uiOutput.requested_map ?? null,
    resolved_by: uiOutput.resolved_by ?? null,
    reason: uiOutput.reason ?? null,
    section: uiOutput.section ?? null,
});

const summarizeLiveRangeTodoListForAi = (uiOutput: Record<string, any>) => {
    const events = Array.isArray(uiOutput.todo_list?.events) ? uiOutput.todo_list.events : [];
    return {
        event_count: events.length,
        pending_count: events.filter((event: Record<string, any>) => event.status === 'pending').length,
        running_count: events.filter((event: Record<string, any>) => event.status === 'running').length,
    };
};

const summarizeProcedureRequestForAi = (request: unknown) => {
    const record = getToolUiRecord(request);
    return {
        title: record.title ?? null,
        tool: record.name ?? record.tool ?? null,
        status: record.status ?? null,
    };
};

const buildToolAiOutput = (
    name: typeof ALL_AI_TOOL_NAMES[number],
    uiOutputValue: unknown,
): Record<string, unknown> => {
    const uiOutput = getToolUiRecord(uiOutputValue);
    const status = getToolAiStatus(uiOutput);
    const output: Record<string, unknown> = {
        name,
        status,
        message: getToolAiMessage(
            uiOutput,
            `${name} ${status}.`,
        ),
    };

    switch (name) {
        case 'query_telemetry_metric':
            output.values = omitOkForAi(uiOutput);
            break;
        case 'get_next_corner':
            output.corner = {
                name: uiOutput.name ?? uiOutput.corner_name ?? null,
                from: uiOutput.from ?? null,
                to: uiOutput.to ?? null,
            };
            break;
        case 'show_map':
            Object.assign(output, summarizeMapForAi(uiOutput));
            break;
        case 'set_live_range_todo_list':
        case 'update_live_range_todo_list':
        case 'get_live_range_todo_list':
            Object.assign(output, summarizeLiveRangeTodoListForAi(uiOutput));
            break;
        case 'set_procedure_plan':
            output.goal = uiOutput.goal ?? null;
            output.request_count = uiOutput.request_count ?? 0;
            output.current_request = uiOutput.current_request ?? null;
            output.request = summarizeProcedureRequestForAi(uiOutput.request);
            break;
        case 'advance_plan_step':
            output.current_request = uiOutput.current_request ?? null;
            output.request = summarizeProcedureRequestForAi(uiOutput.request);
            break;
        case 'clear_procedure_plan':
            output.reason = uiOutput.reason ?? null;
            break;
        case 'get_available_user_summary_maps':
            output.map_count = uiOutput.map_count ?? 0;
            output.map_options = Array.isArray(uiOutput.map_options) ? uiOutput.map_options : [];
            break;
        case 'search_user_summary_map_level':
            output.query = uiOutput.query ?? null;
            output.match_count = uiOutput.match_count ?? 0;
            output.maps = Array.isArray(uiOutput.maps)
                ? uiOutput.maps.map((map: Record<string, any>) => ({
                    id: map.id,
                    name: map.name,
                    matched_fields: map.matched_fields ?? undefined,
                }))
                : [];
            break;
        case 'get_user_summary_map_level':
            output.map_count = uiOutput.map_count ?? 0;
            output.maps = Array.isArray(uiOutput.maps)
                ? uiOutput.maps.map((map: Record<string, any>) => ({
                    id: map.id,
                    name: map.name,
                    section_count: map.section_count,
                    mistake_percent: map.mistake_percent,
                    expert_adherence_percent: map.expert_adherence_percent,
                }))
                : [];
            break;
        case 'run_recorded_ai_analysis':
        case 'get_recorded_session_analysis':
            output.session_id = uiOutput.session_id ?? uiOutput.analysis?.session_id ?? null;
            output.samples_analyzed = uiOutput.samples_analyzed ?? uiOutput.analysis?.samples_analyzed ?? 0;
            break;
        case 'get_recorded_session_context':
            output.session_id = uiOutput.session_id ?? null;
            output.track = uiOutput.track ?? null;
            output.car = uiOutput.car ?? null;
            break;
        case 'get_live_focus_section':
            output.focus_section = uiOutput.focus?.section?.name ?? null;
            output.show_map_arguments = uiOutput.focus?.show_map_arguments ?? null;
            break;
        case 'get_live_section_history':
            output.history_count = Array.isArray(uiOutput.history) ? uiOutput.history.length : 0;
            break;
        case 'get_live_analysis_mistake_count':
            if (typeof uiOutput.mistake_count === 'number') {
                output.mistake_count = uiOutput.mistake_count;
            }
            if (typeof uiOutput.practice_mistake_count === 'number') {
                output.practice_mistake_count = uiOutput.practice_mistake_count;
            }
            if (typeof uiOutput.racing_mistake_count === 'number') {
                output.racing_mistake_count = uiOutput.racing_mistake_count;
            }
            if (typeof uiOutput.baseline_lap === 'number') {
                output.baseline_lap = uiOutput.baseline_lap;
            }
            if (typeof uiOutput.page_id === 'string') output.page_id = uiOutput.page_id;
            if (typeof uiOutput.track === 'string') output.track = uiOutput.track;
            if (typeof uiOutput.car === 'string') output.car = uiOutput.car;
            break;
        case 'create_goal':
        case 'retry_goal_task':
            output.goal = uiOutput.name ?? null;
            output.target = typeof uiOutput.target === 'number' ? uiOutput.target : null;
            output.actual = typeof uiOutput.actual === 'number' ? uiOutput.actual : null;
            output.completed_steps = Array.isArray(uiOutput.completed_steps)
                ? uiOutput.completed_steps
                : [];
            output.determination = uiOutput.determination ?? null;
            output.determination_result = uiOutput.determination_result ?? null;
            output.task_results = Array.isArray(uiOutput.task_results)
                ? uiOutput.task_results
                : [];
            break;
        case '_get_live_section_telemetry':
            output.section = uiOutput.section
                ? {
                    id: uiOutput.section.id ?? null,
                    name: uiOutput.section.name ?? null,
                }
                : null;
            output.row_count = Array.isArray(uiOutput.rows) ? uiOutput.rows.length : 0;
            break;
        case '_record_live_section_classification':
            output.classification = uiOutput.classification
                ? {
                    section_id: uiOutput.classification.sectionId ?? uiOutput.classification.section_id ?? null,
                    section_name: uiOutput.classification.sectionName ?? uiOutput.classification.section_name ?? null,
                    severity: uiOutput.classification.severity ?? null,
                }
                : null;
            break;
        case 'analyze_telemetry':
            output.analysis = uiOutput.analysis ?? null;
            output.telemetry_stats = uiOutput.telemetry_stats ?? null;
            output.chartId = uiOutput.chartId ?? null;
            output.component_name = uiOutput.component_name ?? null;
            break;
        case 'classify_live_section':
            output.classification = uiOutput.classification ?? null;
            output.focus = uiOutput.focus ?? null;
            output.comparison = uiOutput.comparison ?? null;
            output.analysis = uiOutput.analysis ?? null;
            output.telemetry_stats = uiOutput.telemetry_stats ?? null;
            output.chartId = uiOutput.chartId ?? null;
            output.component_name = uiOutput.component_name ?? null;
            break;
        default:
            break;
    }

    return output;
};

const createAiToolDefinition = (
    name: typeof ALL_AI_TOOL_NAMES[number],
): AiCommandToolDefinition => {
    return {
        name,
        schema: { properties: {}, required: [] },
        required: [],
        execute: async (args, context, output, handlerContext) => {
            const handler = createRefBasedAiCommandFunctions(context)[name];
            if (!handler) {
                throw new ToolNotRegisteredError(
                    `Tool ${name} is not registered.`,
                );
            }
            return handler(args, handlerContext, output);
        },
        formatAiOutput: (uiOutput) => buildToolAiOutput(name, uiOutput),
    };
};

export const frontendToolDefinitions: AiCommandToolDefinition[] = ALL_AI_TOOL_NAMES
    .map(createAiToolDefinition);

export const createAiCommandRegistry = (
    context: RefAiCommandContext,
): Record<string, AiCommandHandler> => {
    const registry: Record<string, AiCommandHandler> = {};

    frontendToolDefinitions.forEach((definition) => {
        registry[definition.name] = (args, ctx) => executeAiToolDefinition(
            definition,
            args,
            context,
            ctx,
        );
    });

    return registry;
};
