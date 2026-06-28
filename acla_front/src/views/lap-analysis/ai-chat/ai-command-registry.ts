import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/lap-analysis/visualization/charts/circuitTrackLayout';
import { ToolHandlerContext, FrontendToolSchema } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { AiMapDisplayPayload, AiMapSectionSelection } from './AiMapToolDisplay';
import {
    SegmentClassificationResult,
    RecordedAiAnalysisState,
} from 'views/lap-analysis/recorded-session-analysis';
import {
    getSegmentMainLabelText,
    resolveSegmentChildLabelTexts,
    SegmentClassificationSegment,
} from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import {
    DEFAULT_ANALYST_COOLDOWN_MS,
    DEFAULT_ANALYST_MIN_DISTANCE,
    DEFAULT_ANALYST_MIN_LEAD_SECONDS,
    hasEnoughCoachingLead,
    type LiveAnalystRecordedAnalysisError,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    buildProcedurePlan,
    isProcedurePlanRequestDone,
    type ProcedurePlan,
    type ProcedurePlanRequest,
    type ProcedurePlanStepStatus,
} from './ai-chat-plan';
import {
    PracticeParentSegmentView,
    PracticeSectionSummaryView,
    asRecord,
    buildPracticeTrackSummaryViews,
} from 'views/user-summary/user-summary-model';
import { detectOvertakeTacticalState } from './overtake-agent-detector';

type AiCommandHandler = (args: Record<string, any>, ctx: ToolHandlerContext) => Promise<any>;
export type AgentSessionMode = 'live_performance_analyst';
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
    status: 'started' | 'already_running' | 'error';
    conversation_role: 'agent';
    agent_mode: AgentSessionMode;
    agent_session_id?: string;
    parent_client_session_id?: string | null;
    error?: string;
    message?: string;
};

export type AgentSessionStopResult = {
    status: 'stopped' | 'not_running' | 'error';
    conversation_role: 'agent';
    agent_mode?: AgentSessionMode;
    agent_session_id?: string | null;
    error?: string;
    message?: string;
};
export type ProcedurePlanSubscriberResult = {
    status: Extract<ProcedurePlanStepStatus, 'complete' | 'blocked' | 'failed' | 'skipped'>;
    error?: string;
    message?: string;
};
export type ProcedurePlanSubscriber = (
    request: ProcedurePlanRequest,
    ctx: ToolHandlerContext,
    snapshot?: Record<string, any> | null,
) => Promise<ProcedurePlanSubscriberResult>;

export interface AiCommandRegistryContext {
    sessionId?: string;
    sessionMode?: 'live' | 'recorded' | 'user_summary';
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
    advanceProcedurePlanStep?: (reason?: string) => {
        status: string;
        current_request?: number;
        request?: ProcedurePlanRequest;
        current_step?: number;
        step?: string;
        error?: string;
    };
    getProcedurePlan?: () => ProcedurePlan | null;
    clearProcedurePlan?: () => void;
    setProcedurePlan?: (plan: ProcedurePlan | null) => void;
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
    procedurePlanSubscribers?: Record<string, ProcedurePlanSubscriber>;
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
    lastObservationKey: string | null;
    lastObservationAt: number;
    lastSpokenAt: number;
    analysisSessionId?: string | null;
}

// Frontend-implemented tool capabilities. This file owns executable browser
// handlers and JSON parameter shapes only; LLM-facing tool instructions live
// in the AI service external knowledge base.
// JSON-Schema for QueryScope (see session-intelligence/types.ts). Shared
// shape between `query_telemetry_metric` (frontend) and `analyze_telemetry`
// (server).
//
// Flat shape with a `type` enum discriminator. The per-type field coupling
// (e.g. `type='lap'` requires `lap`) is enforced by `_validate_scope` in
// app/pipelines/chat/__init__.py before tool dispatch, not in JSON Schema.
// Reason: Groq llama-3.3-70b's tool-call validator rejects oneOf+const
// discriminated unions when the model picks an invalid type — the whole
// turn fails server-side. A single flat object with an enum on `type` is
// the shape Groq and similar providers handle reliably.
export const QUERY_SCOPE_SCHEMA = {
    type: 'object',
    properties: {
        type: {
            type: 'string',
            enum: ['now', 'last_seconds', 'event', 'lap', 'range'],
        },
        seconds: { type: 'number' },
        eventType: { type: 'string', enum: ['CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'] },
        which: { type: 'string', enum: ['last', 'current'] },
        lap: {
            oneOf: [
                { type: 'string', enum: ['current', 'last'] },
                { type: 'integer' },
            ],
        },
        start: { type: 'integer' },
        end: { type: 'integer' },
    },
    required: ['type'],
    additionalProperties: false,
} as const;

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

const hasSummaryContent = (summary: Record<string, any>): boolean => Object.keys(summary).length > 0;

const normalizeSummarySearchText = (value: unknown): string => String(value ?? '')
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const getSearchLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed <= 0) return 5;
    return Math.min(parsed, 10);
};

const getRecordedSegmentLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed <= 0) return 20;
    return Math.min(parsed, 50);
};

const getLiveAnalystIntervalSeconds = (value: unknown): number => {
    const parsed = toPositiveNumber(value) ?? DEFAULT_LIVE_ANALYST_INTERVAL_SECONDS;
    return Math.min(
        LIVE_ANALYST_MAX_INTERVAL_SECONDS,
        Math.max(LIVE_ANALYST_MIN_INTERVAL_SECONDS, parsed),
    );
};

const getLiveToolsUnavailableError = (context: AiCommandRegistryContext) => (
    context.sessionMode === 'recorded'
        ? 'recorded_session_live_tools_unavailable'
        : 'non_live_context_live_tools_unavailable'
);

const isLiveSessionContext = (context: AiCommandRegistryContext): boolean =>
    !context.sessionMode || context.sessionMode === 'live';

const isRecordedSessionContext = (context: AiCommandRegistryContext): boolean =>
    context.sessionMode === 'recorded';

const normalizeOptionalString = (value: unknown): string | undefined => (
    typeof value === 'string' && value.trim() ? value.trim() : undefined
);

const clampNormalizedSectionValue = (value: unknown): number | undefined => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return undefined;
    return Math.max(0, Math.min(1, parsed));
};

const buildMapSectionSelection = (args: Record<string, any>): AiMapSectionSelection | undefined => {
    const start = clampNormalizedSectionValue(args.section_start ?? args.start);
    const end = clampNormalizedSectionValue(args.section_end ?? args.end);
    const label = normalizeOptionalString(args.section_label ?? args.label);

    if (start === undefined && end === undefined && !label) return undefined;

    return { start, end, label };
};

const getMapRequestCandidates = (
    args: Record<string, any>,
    context: AiCommandRegistryContext,
): string[] => {
    const selectedSession = context.analysisContext?.sessionSelected as Record<string, any> | null | undefined;
    const liveData = context.analysisContext?.liveData as Record<string, any> | null | undefined;
    return [
        args.map_id,
        args.source_track_key,
        args.map_name,
        context.analysisContext?.mapSelected,
        selectedSession?.map,
        liveData?.Static_track,
        liveData?.Static?.track,
        context.analysisContext?.recordedSessioStaticsData?.track,
    ]
        .map(normalizeOptionalString)
        .filter((value): value is string => Boolean(value));
};

const buildUnavailableMapDisplay = (
    args: Record<string, any>,
    reason: string,
    requestedMap?: string,
): AiMapDisplayPayload => ({
    status: 'unavailable',
    requestedMap,
    title: normalizeOptionalString(args.title) || 'Map',
    note: normalizeOptionalString(args.message ?? args.note),
    reason,
    section: buildMapSectionSelection(args),
});

const summarizePracticeSegments = (segments: PracticeParentSegmentView[]) => segments
    .filter((segment) => segment.count > 0)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 5)
    .map((segment) => ({
        id: segment.id,
        name: segment.name,
        count: segment.count,
        child_segments: segment.childSegments
            .filter((child) => child.count > 0)
            .slice(0, 5)
            .map((child) => ({
                id: child.id,
                name: child.name,
                count: child.count,
                ...(child.startIndex !== undefined ? { start_index: child.startIndex } : {}),
                ...(child.endIndex !== undefined ? { end_index: child.endIndex } : {}),
            })),
    }));

const summarizePracticeSection = (section: PracticeSectionSummaryView) => ({
    id: section.id,
    name: section.name,
    analyzed_time_count: section.analyzedTimeCount,
    mistake_count: section.mistakeCount,
    mistake_percent: section.mistakePercent,
    expert_adherence_count: section.expertAdherenceCount,
    expert_adherence_percent: section.expertAdherencePercent,
    mistake_segments: summarizePracticeSegments(section.mistakeSegments),
    expert_adherence_segments: summarizePracticeSegments(section.expertAdherenceSegments),
    recovery_merge_segments: summarizePracticeSegments(section.recoveryMergeSegments),
});

const buildUserSummaryMapLevel = (
    summary: Record<string, any>,
    args: Record<string, any>,
    context: AiCommandRegistryContext,
) => {
    const tracks = buildPracticeTrackSummaryViews(
        summary,
        context.getLabelName,
        context.getCategoryLabels,
    );
    const requestedMapId = typeof args.map_id === 'string' && args.map_id.trim()
        ? args.map_id.trim()
        : undefined;
    const filteredTracks = requestedMapId
        ? tracks.filter((track) => (
            track.id === requestedMapId
            || track.name.toLowerCase() === requestedMapId.toLowerCase()
        ))
        : tracks;

    return {
        status: 'ready',
        map_count: filteredTracks.length,
        maps: filteredTracks.map((track) => {
            const mistakeCount = track.sections.reduce((sum, section) => sum + section.mistakeCount, 0);
            const expertAdherenceCount = track.sections.reduce((sum, section) => sum + section.expertAdherenceCount, 0);
            const totalAnalyzedTimeCount = track.totalAnalyzedTimeCount
                || track.sections.reduce((sum, section) => sum + section.analyzedTimeCount, 0);

            return {
                id: track.id,
                name: track.name,
                analyzed_session_count: track.analyzedSessionCount,
                skipped_session_count: track.skippedSessionCount,
                failed_session_count: track.failedSessionCount,
                total_analyzed_time_count: totalAnalyzedTimeCount,
                section_count: track.sections.length,
                mistake_count: mistakeCount,
                expert_adherence_count: expertAdherenceCount,
                mistake_percent: totalAnalyzedTimeCount > 0
                    ? (mistakeCount / totalAnalyzedTimeCount) * 100
                    : 0,
                expert_adherence_percent: totalAnalyzedTimeCount > 0
                    ? (expertAdherenceCount / totalAnalyzedTimeCount) * 100
                    : 0,
                top_mistake_sections: track.sections
                    .filter((section) => section.mistakeCount > 0)
                    .sort((a, b) => b.mistakeCount - a.mistakeCount || a.name.localeCompare(b.name))
                    .slice(0, 3)
                    .map(summarizePracticeSection),
                top_expert_adherence_sections: track.sections
                    .filter((section) => section.expertAdherenceCount > 0)
                    .sort((a, b) => b.expertAdherenceCount - a.expertAdherenceCount || a.name.localeCompare(b.name))
                    .slice(0, 3)
                    .map(summarizePracticeSection),
                sections: requestedMapId
                    ? track.sections.map(summarizePracticeSection)
                    : undefined,
            };
        }),
    };
};

type UserSummaryMapLevelResult = ReturnType<typeof buildUserSummaryMapLevel>;
type UserSummaryMapRow = UserSummaryMapLevelResult['maps'][number];

const buildAvailableUserSummaryMaps = (mapLevel: UserSummaryMapLevelResult) => {
    const maps = mapLevel.maps.map((map) => ({
        id: map.id,
        name: map.name,
        analyzed_session_count: map.analyzed_session_count,
        total_analyzed_time_count: map.total_analyzed_time_count,
        section_count: map.section_count,
    }));
    const mapOptions = maps.map((map) => (
        `${map.name} (${map.id}) - ${map.analyzed_session_count} analyzed session${map.analyzed_session_count === 1 ? '' : 's'}`
    ));

    return {
        status: 'ready',
        map_count: mapLevel.map_count,
        maps,
        map_options: mapOptions,
        response_text: mapOptions.length > 0
            ? `Available maps in your summary:\n${mapOptions.map((option) => `- ${option}`).join('\n')}\nWhich map should I inspect?`
            : 'I do not see any maps in your user summary yet.',
    };
};

const getSelectedRecordedSession = (context: AiCommandRegistryContext): Record<string, any> | null => {
    const selectedSession = context.analysisContext?.sessionSelected;
    if (!selectedSession?.SessionId) return null;
    return selectedSession;
};

const summarizeRecordedSegment = (
    segment: SegmentClassificationSegment,
    context: AiCommandRegistryContext,
) => ({
    id: segment.id ?? null,
    start_index: segment.start_index,
    end_index: segment.end_index,
    parent_label: getSegmentMainLabelText(segment, context.getLabelName),
    child_labels: resolveSegmentChildLabelTexts(segment, context.getLabelName),
    label_ids: segment.labels ?? [],
    child_segments: (segment.child_segments || segment.sub_segments || [])
        .slice(0, 8)
        .map((child) => ({
            start_index: child.start_index,
            end_index: child.end_index,
            labels: child.labels,
            label_names: child.labels.map((labelId) => context.getLabelName?.(labelId) || labelId),
        })),
});

const buildRecordedAnalysisToolResult = (
    state: RecordedAiAnalysisState | null | undefined,
    context: AiCommandRegistryContext,
    args: Record<string, any> = {},
) => {
    const selectedSession = getSelectedRecordedSession(context);
    if (!selectedSession) {
        return {
            status: 'error',
            error: 'no_recorded_session',
            message: 'No recorded session is selected.',
        };
    }

    const analysisState = state || context.analysisContext?.recordedAiAnalysis;
    const result = analysisState?.result as SegmentClassificationResult | null | undefined;
    const limit = getRecordedSegmentLimit(args.limit);
    const segments = result && Array.isArray(result.segments) ? result.segments : [];

    return {
        status: analysisState?.status || 'idle',
        message: analysisState?.message || null,
        session_id: selectedSession.SessionId,
        session_name: selectedSession.session_name || null,
        map: selectedSession.map || context.analysisContext?.mapSelected || null,
        car: selectedSession.car || null,
        analysis: result
            ? {
                status: result.status,
                session_id: result.session_id,
                samples_analyzed: result.samples_analyzed,
                segment_count: result.segment_count,
                returned_segment_count: Math.min(segments.length, limit),
                segments: segments.slice(0, limit).map((segment) => summarizeRecordedSegment(segment, context)),
            }
            : null,
    };
};

const buildRecordedSessionContext = (
    context: AiCommandRegistryContext,
    args: Record<string, any> = {},
) => {
    const selectedSession = getSelectedRecordedSession(context);
    if (!selectedSession) {
        return {
            status: 'error',
            error: 'no_recorded_session',
            message: 'No recorded session is selected.',
        };
    }

    const playback = context.analysisContext?.recordedPlaybackSummary;
    return {
        status: 'ready',
        selected_session: {
            id: selectedSession.SessionId,
            name: selectedSession.session_name || null,
            map: selectedSession.map || context.analysisContext?.mapSelected || null,
            car: selectedSession.car || null,
        },
        recorded_telemetry: {
            sample_count: playback?.sampleCount ?? context.analysisContext?.recordedTelemetryDataCount ?? 0,
            duration_seconds: playback?.durationSeconds ?? 0,
            playback_index: playback?.playbackIndex ?? 0,
            playback_time_seconds: playback?.playbackTimeSeconds ?? 0,
            active_segment: playback?.activeSegment ?? null,
        },
        ai_analysis: buildRecordedAnalysisToolResult(
            context.analysisContext?.recordedAiAnalysis,
            context,
            args,
        ),
    };
};

const getSelectedSessionId = (context: AiCommandRegistryContext): string | null => {
    const selectedSession = getSelectedRecordedSession(context);
    return selectedSession?.SessionId || context.sessionId || null;
};

const toRequestPayloadRecord = (request: ProcedurePlanRequest): Record<string, any> => (
    request.payload && typeof request.payload === 'object' && !Array.isArray(request.payload)
        ? request.payload as Record<string, any>
        : {}
);

const getProcedurePlanToolName = (request: ProcedurePlanRequest): string | undefined => {
    const payload = toRequestPayloadRecord(request);
    return normalizeOptionalString(
        request.name
        ?? payload.name
        ?? payload.tool
        ?? payload.tool_name,
    );
};

const normalizeAgentSessionMode = (value: unknown): AgentSessionMode | null => (
    value === 'live_performance_analyst' ? 'live_performance_analyst' : null
);

const getProcedurePlanToolArguments = (request: ProcedurePlanRequest): Record<string, any> => {
    const payload = toRequestPayloadRecord(request);
    const explicitArgs = payload.arguments ?? payload.args ?? payload.parameters ?? payload.params;
    if (explicitArgs && typeof explicitArgs === 'object' && !Array.isArray(explicitArgs)) {
        return explicitArgs as Record<string, any>;
    }

    const {
        name: _name,
        tool: _tool,
        tool_name: _toolName,
        arguments: _arguments,
        args: _args,
        parameters: _parameters,
        params: _params,
        output: _output,
        result_visibility: _resultVisibility,
        resultVisibility: _camelResultVisibility,
        ...rest
    } = payload;
    return rest;
};

const getProcedurePlanToolResultVisibility = (request: ProcedurePlanRequest): 'ai' | 'tag' => {
    const payload = toRequestPayloadRecord(request);
    const visibility = normalizeOptionalString(
        request.result_visibility
        ?? request.output
        ?? payload.result_visibility
        ?? payload.resultVisibility
        ?? payload.output,
    )?.toLowerCase();

    return visibility && ['tag', 'tags', 'ui', 'hidden', 'none'].includes(visibility)
        ? 'tag'
        : 'ai';
};

const runRecordedAnalysisForLiveRequest = async (
    context: AiCommandRegistryContext,
    args: Record<string, any> = {},
): Promise<
    | { status: 'ready'; analysis: ReturnType<typeof buildRecordedAnalysisToolResult> }
    | { status: 'error'; error: LiveAnalystRecordedAnalysisError; message: string }
> => {
    const sessionId = getSelectedSessionId(context);
    if (!sessionId) {
        return {
            status: 'error',
            error: 'recorded_session_required',
            message: 'Live performance analysis needs an uploaded or selected recorded session before it can request classifier analysis.',
        };
    }

    const runAnalysis = context.analysisContext?.runRecordedAiAnalysis;
    if (typeof runAnalysis !== 'function') {
        return {
            status: 'error',
            error: 'recorded_analysis_unavailable',
            message: 'Recorded AI analysis is not available in this view.',
        };
    }

    const state = await runAnalysis({ force: args.force === true });
    if (state.status === 'error') {
        return {
            status: 'error',
            error: 'recorded_analysis_failed',
            message: state.message || 'Failed to run recorded AI analysis.',
        };
    }

    return {
        status: 'ready',
        analysis: buildRecordedAnalysisToolResult(state, context, { limit: 8 }),
    };
};

const buildProcedurePlanSubscribers = (
    context: AiCommandRegistryContext,
): Record<string, ProcedurePlanSubscriber> => ({
    async driver() {
        return { status: 'complete' };
    },

    async live_recorded_analysis(request, _ctx, snapshot) {
        if (snapshot?.baseline_ready !== true) {
            return {
                status: 'blocked',
                error: 'baseline_collection_incomplete',
                message: 'Complete one clean baseline lap before advancing the plan.',
            };
        }

        const analysisStatus = await runRecordedAnalysisForLiveRequest(
            context,
            toRequestPayloadRecord(request),
        );
        if (analysisStatus.status !== 'ready') {
            context.sessionIntelligence?.emitRecordedAnalysisError(
                analysisStatus.error,
                analysisStatus.message,
                snapshot,
            );
            return {
                status: 'failed',
                error: analysisStatus.error,
                message: analysisStatus.message,
            };
        }

        const sessionId = getSelectedSessionId(context);
        const agent = getLiveAnalystState(context);
        agent.analysisSessionId = sessionId;
        context.sessionIntelligence?.emitRecordedAnalysisReady(analysisStatus.analysis, snapshot);
        return { status: 'complete' };
    },
    ...context.procedurePlanSubscribers,
});

const executeProcedurePlanRequest = async (
    request: ProcedurePlanRequest,
    context: AiCommandRegistryContext,
    ctx: ToolHandlerContext,
    snapshot?: Record<string, any> | null,
): Promise<ProcedurePlanSubscriberResult> => {
    if (!request.subscriber) {
        return {
            status: 'blocked',
            error: 'procedure_plan_subscriber_missing',
            message: 'This plan request does not name a frontend subscriber.',
        };
    }
    const subscriber = buildProcedurePlanSubscribers(context)[request.subscriber];
    if (!subscriber) {
        return {
            status: 'blocked',
            error: 'procedure_plan_subscriber_missing',
            message: `No frontend subscriber is registered for "${request.subscriber}".`,
        };
    }
    return subscriber(request, ctx, snapshot);
};

const executeProcedurePlanToolCall = async (
    request: ProcedurePlanRequest,
    registry: Record<string, AiCommandHandler>,
    ctx: ToolHandlerContext,
): Promise<
    | {
        status: 'complete';
        name: string;
        arguments: Record<string, any>;
        resultVisibility: 'ai' | 'tag';
        result: any;
    }
    | {
        status: 'failed';
        error: string;
        message: string;
    }
> => {
    const name = getProcedurePlanToolName(request);
    if (!name) {
        return {
            status: 'failed',
            error: 'procedure_plan_tool_missing',
            message: 'This plan request is a tool_call but does not name a tool.',
        };
    }
    if (name === 'advance_plan_step') {
        return {
            status: 'failed',
            error: 'procedure_plan_tool_recursive',
            message: 'A plan request cannot execute advance_plan_step from inside advance_plan_step.',
        };
    }

    const handler = registry[name];
    if (!handler) {
        return {
            status: 'failed',
            error: 'procedure_plan_tool_unavailable',
            message: `No frontend tool handler is registered for "${name}". Call that tool directly if it is server-side, then advance the plan.`,
        };
    }

    const toolArguments = getProcedurePlanToolArguments(request);
    let result: any;
    try {
        result = await handler(toolArguments, ctx);
    } catch (err) {
        return {
            status: 'failed',
            error: 'procedure_plan_tool_failed',
            message: (err as Error)?.message || String(err),
        };
    }
    if (result && typeof result === 'object' && 'error' in result) {
        return {
            status: 'failed',
            error: String((result as Record<string, any>).error || 'procedure_plan_tool_failed'),
            message: normalizeOptionalString((result as Record<string, any>).message)
                || `The plan tool "${name}" could not complete.`,
        };
    }

    return {
        status: 'complete',
        name,
        arguments: toolArguments,
        resultVisibility: getProcedurePlanToolResultVisibility(request),
        result,
    };
};

const shouldExecuteProcedurePlanRequest = (
    request: ProcedurePlanRequest | undefined,
): request is ProcedurePlanRequest => (
    Boolean(request?.subscriber)
    && request?.subscriber !== 'driver'
    && !isProcedurePlanRequestDone(request)
);

const searchUserSummaryMapLevel = (
    mapLevel: UserSummaryMapLevelResult,
    args: Record<string, any>,
) => {
    const query = normalizeSummarySearchText(args.query);
    const terms = Array.from(new Set(query.split(' ').filter(Boolean)));
    const limit = getSearchLimit(args.limit);

    if (terms.length === 0) {
        return {
            status: 'invalid_query',
            error: 'query_required',
            query,
            match_count: 0,
            map_count: mapLevel.map_count,
            maps: [],
        };
    }

    const matches = mapLevel.maps
        .map((map) => {
            const searchFields: Array<{ name: string; value: unknown; weight: number }> = [
                { name: 'map_name', value: map.name, weight: 8 },
                { name: 'map_id', value: map.id, weight: 6 },
            ];

            map.top_mistake_sections.forEach((section) => {
                searchFields.push({ name: 'top_mistake_section', value: section.name, weight: 5 });
                searchFields.push({ name: 'top_mistake_section_id', value: section.id, weight: 3 });
            });
            map.top_expert_adherence_sections.forEach((section) => {
                searchFields.push({ name: 'top_expert_adherence_section', value: section.name, weight: 4 });
                searchFields.push({ name: 'top_expert_adherence_section_id', value: section.id, weight: 3 });
            });
            if (map.mistake_count > 0) {
                searchFields.push({ name: 'aggregate_kind', value: 'mistake mistakes weakness weak section', weight: 2 });
            }
            if (map.expert_adherence_count > 0) {
                searchFields.push({ name: 'aggregate_kind', value: 'expert adherence strength strong section', weight: 2 });
            }

            const matchedTerms = new Set<string>();
            const matchedFields = new Set<string>();
            let score = 0;

            searchFields.forEach((field) => {
                const value = normalizeSummarySearchText(field.value);
                if (!value) return;

                if (value.includes(query)) {
                    score += field.weight * 2;
                    matchedFields.add(field.name);
                }

                terms.forEach((term) => {
                    if (value.includes(term)) {
                        matchedTerms.add(term);
                        matchedFields.add(field.name);
                        score += field.weight;
                    }
                });
            });

            if (matchedTerms.size !== terms.length) return null;

            return {
                map,
                search_score: score,
                matched_fields: Array.from(matchedFields),
            };
        })
        .filter((match): match is { map: UserSummaryMapRow; search_score: number; matched_fields: string[] } => Boolean(match))
        .sort((a, b) => (
            b.search_score - a.search_score
            || b.map.analyzed_session_count - a.map.analyzed_session_count
            || a.map.name.localeCompare(b.map.name)
        ));

    return {
        status: 'ready',
        query,
        match_count: matches.length,
        map_count: mapLevel.map_count,
        maps: matches.slice(0, limit).map((match) => ({
            ...match.map,
            search_score: match.search_score,
            matched_fields: match.matched_fields,
        })),
    };
};

export const frontendToolSchemas: FrontendToolSchema[] = [
    {
        name: 'start_agent_session',
        description: 'Start a separate child AI agent session. The user should interact with that child session while it is active.',
        properties: {
            agent_mode: {
                type: 'string',
                enum: ['live_performance_analyst'],
                description: 'Agent profile to start.',
            },
        },
        required: ['agent_mode'],
    },
    {
        name: 'stop_agent_session',
        description: 'Stop the active child AI agent session and return focus to the main assistant.',
        properties: {
            agent_session_id: {
                type: 'string',
                description: 'Optional frontend child session id. Defaults to the active agent session.',
            },
        },
        required: [],
    },
    {
        name: 'start_per_turn_coaching',
        properties: {},
        required: [],
    },
    {
        name: 'stop_per_turn_coaching',
        properties: {},
        required: [],
    },
    {
        name: 'start_overtake_agent',
        properties: {
            interval_seconds: {
                type: 'number',
            },
        },
        required: [],
    },
    {
        name: 'stop_overtake_agent',
        properties: {},
        required: [],
    },
    {
        name: 'start_live_performance_analysis',
        description: 'Start the live performance analyst agent. The agent creates a procedure plan, waits for a completed baseline lap, and uses recorded-session analysis to build one focus goal.',
        properties: {
            interval_seconds: {
                type: 'number',
                description: 'How often the frontend should check live section and plan state.',
            },
        },
        required: [],
    },
    {
        name: 'stop_live_performance_analysis',
        description: 'Stop the live performance analyst agent.',
        properties: {},
        required: [],
    },
    {
        name: 'get_live_session_snapshot',
        description: 'Return compact live session state, including lap readiness and detected session type.',
        properties: {},
        required: [],
    },
    {
        name: 'get_live_focus_section',
        description: 'Return the current live analyst focus section, timing, and map-display arguments when available.',
        properties: {},
        required: [],
    },
    {
        name: 'get_live_section_history',
        description: 'Return compact live section classifications already recorded by the AI service.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classifications to return.',
            },
        },
        required: [],
    },
    {
        name: 'advance_plan_step',
        description: 'Report that the current visible procedure plan request is complete so the UI can move to the next request.',
        properties: {
            reason: {
                type: 'string',
                description: 'Short reason the current plan request is complete.',
            },
        },
        required: [],
    },
    {
        name: 'clear_procedure_plan',
        description: 'Clear or terminate the visible procedure plan UI when the plan is no longer useful.',
        properties: {
            reason: {
                type: 'string',
                description: 'Optional short reason the visible plan should be cleared.',
            },
        },
        required: [],
    },
    {
        name: 'set_procedure_plan',
        description: 'Create or replace the visible procedure plan UI with an AI-authored list of requests.',
        properties: {
            goal: {
                type: 'string',
                description: 'Short goal shown above the request list.',
            },
            current_request: {
                type: 'integer',
                description: 'Zero-based index of the active request.',
            },
            requests: {
                type: 'array',
                description: 'Ordered list of requests the assistant plans to perform or ask the UI/backend to perform.',
                items: {
                    type: 'object',
                    properties: {
                        type: { type: 'string' },
                        title: { type: 'string' },
                        name: {
                            type: 'string',
                            description: 'Tool name for tool_call requests.',
                        },
                        subscriber: {
                            type: 'string',
                            description: 'Frontend subscriber that can complete this request, such as driver, live_recorded_analysis, or another registered component.',
                        },
                        status: {
                            type: 'string',
                            enum: ['pending', 'running', 'complete', 'blocked', 'failed', 'skipped'],
                        },
                        detail: { type: 'string' },
                        result_visibility: {
                            type: 'string',
                            enum: ['ai', 'tag'],
                            description: 'Use ai when the assistant needs the tool result. Use tag for UI-only/side-effect tools.',
                        },
                        output: {
                            type: 'string',
                            enum: ['ai', 'tag'],
                            description: 'Alias for result_visibility.',
                        },
                        payload: {
                            type: 'object',
                            description: 'Tool arguments for tool_call requests, or an object containing arguments/args/parameters.',
                        },
                    },
                    required: ['type', 'title'],
                },
            },
        },
        required: ['goal', 'requests'],
    },
    {
        name: 'get_next_corner',
        properties: {},
        required: [],
    },
    {
        name: 'query_telemetry_metric',
        properties: {
            fields: {
                type: 'array',
                items: { type: 'string' },
            },
            scope: QUERY_SCOPE_SCHEMA,
            reduce: {
                type: 'string',
                enum: ['avg', 'min', 'max', 'stats'],
            },
        },
        required: ['fields', 'scope', 'reduce'],
    },
    {
        name: 'get_event_log',
        properties: {
            eventType: {
                type: 'string',
                enum: ['CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'],
            },
            scope: {
                type: 'string',
                enum: ['last', 'last_n', 'lap_current', 'lap_last', 'all'],
            },
            n: {
                type: 'integer',
            },
        },
        required: ['eventType', 'scope'],
    },
    {
        name: 'get_user_summary_map_level',
        properties: {
            map_id: {
                type: 'string',
            },
        },
        required: [],
    },
    {
        name: 'get_available_user_summary_maps',
        properties: {},
        required: [],
    },
    {
        name: 'search_user_summary_map_level',
        properties: {
            query: {
                type: 'string',
            },
            limit: {
                type: 'integer',
            },
        },
        required: ['query'],
    },
    {
        name: 'show_map',
        description: 'Display a circuit map in the chat transcript, optionally highlighting a normalized lap section.',
        properties: {
            map_id: {
                type: 'string',
                description: 'Circuit map id to display. Prefer this when a map id is known.',
            },
            source_track_key: {
                type: 'string',
                description: 'ACC source track key such as brands_hatch, monza, or spa.',
            },
            map_name: {
                type: 'string',
                description: 'Human-readable map or circuit name when no id/key is known.',
            },
            section_start: {
                type: 'number',
                description: 'Start of the highlighted section as normalized lap position from 0 to 1.',
            },
            section_end: {
                type: 'number',
                description: 'End of the highlighted section as normalized lap position from 0 to 1. Values wrapping across start/finish are allowed.',
            },
            section_label: {
                type: 'string',
                description: 'Short label for the highlighted section.',
            },
            title: {
                type: 'string',
                description: 'Short title shown above the map.',
            },
            note: {
                type: 'string',
                description: 'Brief note shown below the map.',
            },
        },
        required: [],
    },
    {
        name: 'run_recorded_ai_analysis',
        description: 'Run or retrieve the AI segment analysis for the currently selected recorded session.',
        properties: {
            force: {
                type: 'boolean',
                description: 'When true, rerun analysis even if a cached result is available.',
            },
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'get_recorded_session_analysis',
        description: 'Return the shared AI segment analysis for the currently selected recorded session.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'get_recorded_session_context',
        description: 'Return compact selected recorded-session, playback, and AI-analysis context.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to include.',
            },
        },
        required: [],
    },
];

const COMMON_TOOL_NAMES = [
    'show_map',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'stop_agent_session',
] as const;

const LIVE_TOOL_NAMES = [
    'start_agent_session',
    'start_per_turn_coaching',
    'stop_per_turn_coaching',
    'start_overtake_agent',
    'stop_overtake_agent',
    'start_live_performance_analysis',
    'stop_live_performance_analysis',
    'get_live_session_snapshot',
    'get_live_focus_section',
    'get_live_section_history',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
] as const;

const USER_SUMMARY_TOOL_NAMES = [
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
] as const;

const RECORDED_TOOL_NAMES = [
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
] as const;

export const getFrontendToolSchemasForSessionMode = (
    sessionMode: AiCommandRegistryContext['sessionMode'] = 'live',
    options: {
        conversationRole?: AgentSessionRole;
        agentMode?: AgentSessionMode | null;
    } = {},
): FrontendToolSchema[] => {
    if (options.conversationRole === 'agent') {
        const agentAllowedNames = new Set<string>([
            ...COMMON_TOOL_NAMES,
            ...LIVE_TOOL_NAMES.filter((name) => name !== 'start_agent_session'),
            ...USER_SUMMARY_TOOL_NAMES,
        ]);
        return frontendToolSchemas.filter((tool) => agentAllowedNames.has(tool.name));
    }

    const allowedNames: Set<string> = sessionMode === 'recorded'
        ? new Set<string>([...COMMON_TOOL_NAMES, ...USER_SUMMARY_TOOL_NAMES, ...RECORDED_TOOL_NAMES])
        : sessionMode === 'user_summary'
            ? new Set<string>([...COMMON_TOOL_NAMES, ...USER_SUMMARY_TOOL_NAMES])
            : new Set<string>([...COMMON_TOOL_NAMES, ...LIVE_TOOL_NAMES, ...USER_SUMMARY_TOOL_NAMES]);

    return frontendToolSchemas.filter((tool) => allowedNames.has(tool.name));
};

const buildLiveAnalystUnavailable = (context: AiCommandRegistryContext) => (
    !isLiveSessionContext(context)
        ? { status: 'error', error: getLiveToolsUnavailableError(context) }
        : !context.sessionIntelligence
            ? { status: 'error', error: 'no_live_session' }
            : null
);

const getLiveAnalystState = (context: AiCommandRegistryContext): LivePerformanceAnalystState => {
    if (context.livePerformanceAnalystState) return context.livePerformanceAnalystState;
    return {
        intervalId: null,
        inFlight: false,
        enabled: false,
        lastObservationKey: null,
        lastObservationAt: 0,
        lastSpokenAt: 0,
    };
};

const buildLiveFocusPayload = (context: AiCommandRegistryContext) => {
    const si = context.sessionIntelligence;
    if (!si) return null;

    const snapshot = si.getLiveSessionSnapshot();
    if (!snapshot.baseline_ready) return null;

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

const buildLiveAnalystPlanError = (
    error: string,
    message: string,
    snapshot?: Record<string, any>,
) => ({
    status: 'error',
    error,
    agent_mode: 'live_performance_analyst',
    ...(snapshot ? { snapshot } : {}),
    message,
});

const buildProcedurePlanStepError = (
    error: string,
    message: string,
    snapshot?: Record<string, any> | null,
) => ({
    status: 'error',
    error,
    ...(snapshot ? { snapshot } : {}),
    message,
});

const getSessionId = (args: Record<string, any>, context: AiCommandRegistryContext): string | undefined =>
    args.session_id ||
    context.sessionId ||
    context.analysisContext?.sessionSelected?.SessionId;

export const createAiCommandRegistry = (context: AiCommandRegistryContext): Record<string, AiCommandHandler> => {
    const registry: Record<string, AiCommandHandler> = {

    // ── Session ───────────────────────────────────────────────────────────────

    async start_agent_session(args) {
        const agentMode = normalizeAgentSessionMode(args.agent_mode || args.agentMode);
        if (!agentMode) {
            return {
                status: 'error',
                error: 'unsupported_agent_mode',
                message: 'Only live_performance_analyst is supported right now.',
            };
        }
        if (!isLiveSessionContext(context)) {
            return {
                status: 'error',
                error: getLiveToolsUnavailableError(context),
                message: 'Agent sessions are only available from live session context.',
            };
        }
        if (!context.startAgentSession) {
            return {
                status: 'error',
                error: 'agent_session_unavailable',
                message: 'This UI cannot start child AI agent sessions.',
            };
        }
        return context.startAgentSession(agentMode, args);
    },

    async stop_agent_session(args) {
        if (!context.stopAgentSession) {
            return {
                status: 'error',
                error: 'agent_session_unavailable',
                message: 'This UI cannot stop child AI agent sessions.',
            };
        }
        return context.stopAgentSession(normalizeOptionalString(args.agent_session_id ?? args.agentSessionId));
    },

    async get_session_analysis(args) {
        return await apiService.post('/racing-session/detailed-info', { id: getSessionId(args, context) });
    },

    async run_recorded_ai_analysis(args) {
        if (!isRecordedSessionContext(context)) {
            return { status: 'error', error: 'not_recorded_mode' };
        }
        if (!getSelectedRecordedSession(context)) {
            return {
                status: 'error',
                error: 'no_recorded_session',
                message: 'No recorded session is selected.',
            };
        }
        const runAnalysis = context.analysisContext?.runRecordedAiAnalysis;
        if (typeof runAnalysis !== 'function') {
            return {
                status: 'error',
                error: 'recorded_analysis_unavailable',
                message: 'Recorded AI analysis is not available in this view.',
            };
        }

        const state = await runAnalysis({ force: args.force === true });
        return buildRecordedAnalysisToolResult(state, context, args);
    },

    async get_recorded_session_analysis(args) {
        if (!isRecordedSessionContext(context)) {
            return { status: 'error', error: 'not_recorded_mode' };
        }
        return buildRecordedAnalysisToolResult(
            context.analysisContext?.recordedAiAnalysis,
            context,
            args,
        );
    },

    async get_recorded_session_context(args) {
        if (!isRecordedSessionContext(context)) {
            return { status: 'error', error: 'not_recorded_mode' };
        }
        return buildRecordedSessionContext(context, args);
    },

    async get_performance_insights(args) {
        return await apiService.post('/ai/performance-analysis', {
            session_id:    getSessionId(args, context),
            analysis_type: args.analysis_type || 'comprehensive',
        });
    },

    async compare_lap_times(args) {
        return await apiService.post('/racing-session/compare', {
            session_ids: args.session_ids,
            metrics:     args.metrics || ['lap_times'],
        });
    },

    // ── Telemetry ─────────────────────────────────────────────────────────────

    // Constrained-reduce variant exposed to the LLM. The schema enforces
    // reduce ∈ {avg,min,max,stats}; we defensively swap any other value
    // for 'stats' so an invalid prompt can't leak rows.
    async query_telemetry_metric(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        const allowed = new Set(['avg', 'min', 'max', 'stats']);
        const reduce = allowed.has(args.reduce) ? args.reduce : 'stats';
        return si.query({ fields: args.fields, scope: args.scope, reduce } as any);
    },

    // Server-internal: backs analyze_telemetry. Returns raw rows over the
    // WS relay so the server-side classifier can consume them. NOT exposed
    // to the LLM (absent from the voice tool schema) — rows must never
    // enter the LLM context.
    async _get_telemetry_for_scope(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return { rows: si.getRowsForScope(args.scope) };
    },

    // ── Event log ─────────────────────────────────────────────────────────────

    async get_event_log(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return { events: si.findEvents(args as any) };
    },

    async get_user_summary_map_level(args) {
        if (context.userSummaryLoading) {
            return { status: 'loading', maps: [] };
        }
        if (context.userSummaryError) {
            return { status: 'error', error: context.userSummaryError, maps: [] };
        }

        const summary = asRecord(context.userSummary);
        if (!hasSummaryContent(summary)) {
            return { status: 'empty', maps: [] };
        }

        return buildUserSummaryMapLevel(summary, args, context);
    },

    async get_available_user_summary_maps() {
        if (context.userSummaryLoading) {
            return { status: 'loading', maps: [] };
        }
        if (context.userSummaryError) {
            return { status: 'error', error: context.userSummaryError, maps: [] };
        }

        const summary = asRecord(context.userSummary);
        if (!hasSummaryContent(summary)) {
            return { status: 'empty', maps: [] };
        }

        return buildAvailableUserSummaryMaps(
            buildUserSummaryMapLevel(summary, {}, context),
        );
    },

    async search_user_summary_map_level(args) {
        if (context.userSummaryLoading) {
            return { status: 'loading', maps: [] };
        }
        if (context.userSummaryError) {
            return { status: 'error', error: context.userSummaryError, maps: [] };
        }

        const summary = asRecord(context.userSummary);
        if (!hasSummaryContent(summary)) {
            return { status: 'empty', maps: [] };
        }

        return searchUserSummaryMapLevel(
            buildUserSummaryMapLevel(summary, {}, context),
            args,
        );
    },

    async get_next_corner() {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return si.getNextCorner() ?? { error: 'no_corner_data' };
    },

    // ── Coaching ──────────────────────────────────────────────────────────────

    async start_per_turn_coaching() {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        context.startTrackGuide();
        context.setAgentTagActive?.('Track Guide', true);
        return { status: 'started', agent_mode: 'track_guide', enabled: true };
    },

    async stop_per_turn_coaching() {
        context.setTrackGuideEnabled(false);
        context.setAgentTagActive?.('Track Guide', false);
        return { status: 'stopped', agent_mode: 'track_guide', enabled: false };
    },

    async start_overtake_agent(args, ctx) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const telemetryRows = context.getOpportunityTelemetryRows();
        if (telemetryRows.length === 0) {
            return { error: 'no_live_telemetry' };
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
                        ctx.sendObservation({
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
    },

    async stop_overtake_agent() {
        const agent = context.opportunityAgentState;
        if (agent.intervalId) {
            clearInterval(agent.intervalId);
        }
        agent.intervalId = null;
        agent.inFlight = false;
        agent.lastAlertKey = null;
        agent.lastAlertAt = 0;
        context.setAgentTagActive?.('Overtake', false);
        return { status: 'stopped', agent_mode: 'overtake' };
    },

    async start_live_performance_analysis(args, ctx) {
        if (context.conversationRole !== 'agent' && context.startAgentSession) {
            return context.startAgentSession('live_performance_analyst', args);
        }

        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const si = context.sessionIntelligence!;
        const agent = getLiveAnalystState(context);
        if (agent.intervalId) {
            agent.enabled = true;
            context.setLivePerformanceAnalystEnabled?.(true);
            context.setAgentTagActive?.('Live Analyst', true);
            const snapshot = si.getLiveSessionSnapshot();
            return {
                status: 'already_running',
                agent_mode: 'live_performance_analyst',
                snapshot,
                focus: snapshot.baseline_ready ? buildLiveFocusPayload(context) : null,
            };
        }

        const intervalSeconds = getLiveAnalystIntervalSeconds(args.interval_seconds);
        agent.enabled = true;
        if (!si.hasCompletedBaselineLap()) {
            si.startBaselineCollectionAtLapStart();
        }
        context.setLivePerformanceAnalystEnabled?.(true);
        context.setAgentTagActive?.('Live Analyst', true);
        si.emitLiveAnalysisPlanStarted();

        const runAnalystCycle = async (notify: boolean): Promise<any> => {
            if (agent.inFlight) {
                return { status: 'skipped_in_flight' };
            }

            agent.inFlight = true;
            try {
                const snapshot = si.getLiveSessionSnapshot();
                const sections = si.getKnownTrackSections();
                let focus = snapshot.baseline_ready ? buildLiveFocusPayload(context) : null;

                if (notify) {
                    const now = Date.now();
                    if (!snapshot.baseline_ready) {
                        agent.lastObservationKey = `warmup:${snapshot.completed_laps}:${snapshot.sample_count}`;
                    } else if (focus) {
                        const timing = focus.timing;
                        const key = `focus:${focus.section.id}:${focus.baseline.lap}:${focus.baseline.observedAt}`;
                        const canSpeak = hasEnoughCoachingLead(
                            timing.distanceAhead,
                            timing.secondsAhead,
                            DEFAULT_ANALYST_MIN_DISTANCE,
                            DEFAULT_ANALYST_MIN_LEAD_SECONDS,
                        ) && now - agent.lastSpokenAt >= DEFAULT_ANALYST_COOLDOWN_MS;

                        if (canSpeak && agent.lastObservationKey !== key) {
                            agent.lastObservationKey = key;
                            agent.lastObservationAt = now;
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
    },

    async stop_live_performance_analysis() {
        const agent = getLiveAnalystState(context);
        if (agent.intervalId) {
            clearInterval(agent.intervalId);
        }
        agent.intervalId = null;
        agent.inFlight = false;
        agent.enabled = false;
        agent.lastObservationKey = null;
        agent.lastObservationAt = 0;
        agent.lastSpokenAt = 0;
        agent.analysisSessionId = null;
        context.sessionIntelligence?.clearFocusSection();
        context.setLivePerformanceAnalystEnabled?.(false);
        context.clearProcedurePlan?.();
        context.setAgentTagActive?.('Live Analyst', false);
        if (context.conversationRole === 'agent' && context.stopAgentSession) {
            await context.stopAgentSession(agent.analysisSessionId);
        }
        return { status: 'stopped', agent_mode: 'live_performance_analyst', enabled: false };
    },

    async get_live_session_snapshot() {
        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const agent = getLiveAnalystState(context);
        return {
            status: 'ready',
            agent_mode: 'live_performance_analyst',
            enabled: agent.enabled,
            snapshot: context.sessionIntelligence!.getLiveSessionSnapshot(),
        };
    },

    async get_live_focus_section() {
        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const snapshot = context.sessionIntelligence!.getLiveSessionSnapshot();
        if (!snapshot.baseline_ready) {
            return buildLiveAnalystPlanError(
                'baseline_collection_incomplete',
                'Complete one clean baseline lap before reading a focus section.',
                snapshot,
            );
        }

        const focus = buildLiveFocusPayload(context);
        if (!focus) {
            return buildLiveAnalystPlanError(
                'focus_section_not_ready',
                'Analyze the completed baseline and select a focus section before reading it.',
                snapshot,
            );
        }

        return {
            status: 'ready',
            agent_mode: 'live_performance_analyst',
            focus,
        };
    },

    async get_live_section_history(args) {
        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const limit = getRecordedSegmentLimit(args.limit);
        return {
            status: 'ready',
            agent_mode: 'live_performance_analyst',
            history: context.sessionIntelligence!.getSectionHistory(limit),
        };
    },

    async set_procedure_plan(args) {
        const plan = buildProcedurePlan({
            ...args,
            event: normalizeOptionalString(args.event) || 'procedure_plan_started',
        });
        if (!plan) {
            return {
                status: 'error',
                error: 'invalid_procedure_plan_requests',
                message: 'Provide a goal and at least one request with a title.',
            };
        }

        context.setProcedurePlan?.(plan);
        return {
            status: 'ready',
            goal: plan.goal,
            request_count: plan.requests.length,
            current_request: plan.currentStep,
            request: plan.requests[plan.currentStep],
        };
    },

    async advance_plan_step(args, ctx) {
        const snapshot = context.sessionIntelligence?.getLiveSessionSnapshot?.() || null;
        const plan = context.getProcedurePlan?.() || null;
        let toolCallResult: any = null;
        if (plan) {
            const activeRequest = plan.requests[plan.currentStep];
            const executableRequest = activeRequest?.type === 'tool_call' && !isProcedurePlanRequestDone(activeRequest)
                ? activeRequest
                : shouldExecuteProcedurePlanRequest(activeRequest)
                    ? activeRequest
                    : undefined;

            if (executableRequest) {
                if (executableRequest.type === 'tool_call') {
                    toolCallResult = await executeProcedurePlanToolCall(executableRequest, registry, ctx);
                } else {
                    toolCallResult = await executeProcedurePlanRequest(
                        executableRequest,
                        context,
                        ctx,
                        snapshot,
                    );
                }
                if (toolCallResult.status === 'blocked' || toolCallResult.status === 'failed') {
                    return buildProcedurePlanStepError(
                        toolCallResult.error || toolCallResult.status,
                        toolCallResult.message || 'The procedure plan request could not be completed.',
                        snapshot,
                    );
                }
            }
        }

        const advanceResult = context.advanceProcedurePlanStep?.(normalizeOptionalString(args.reason)) || {
            status: 'unavailable',
            error: 'no_procedure_plan_ui',
        };
        if (!toolCallResult || !('name' in toolCallResult)) {
            return advanceResult;
        }

        const executedTool = {
            name: toolCallResult.name,
            arguments: toolCallResult.arguments,
            result_visibility: toolCallResult.resultVisibility,
        };
        if (toolCallResult.resultVisibility === 'tag') {
            return {
                ...advanceResult,
                executed_tool: executedTool,
                tool_result: {
                    status: 'completed',
                    result_visibility: 'tag',
                },
            };
        }

        return {
            ...advanceResult,
            executed_tool: executedTool,
            tool_result: toolCallResult.result,
        };
    },

    async clear_procedure_plan(args) {
        context.clearProcedurePlan?.();
        return {
            status: 'cleared',
            reason: normalizeOptionalString(args.reason),
        };
    },

    async _get_live_section_telemetry(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        const snapshot = si.getLiveSessionSnapshot();
        if (!snapshot.baseline_ready) {
            return buildLiveAnalystPlanError(
                'baseline_collection_incomplete',
                'Complete one clean baseline lap before classifying live sections.',
                snapshot,
            );
        }
        return si.getSectionTelemetryWindow({
            section_id: args.section_id || args.sectionId,
            section_name: args.section_name || args.sectionName,
            lap: args.lap,
        });
    },

    async _record_live_section_classification(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        const snapshot = si.getLiveSessionSnapshot();
        if (!snapshot.baseline_ready) {
            return buildLiveAnalystPlanError(
                'baseline_collection_incomplete',
                'Complete one clean baseline lap before recording a live section classification.',
                snapshot,
            );
        }

        const classification = si.recordSectionClassification(args);
        if (!classification) {
            return { status: 'error', error: 'section_not_found' };
        }

        const focus = buildLiveFocusPayload(context);
        const comparison = focus?.section.id === classification.sectionId
            ? si.compareFocusedSection(classification)
            : null;

        return {
            status: 'recorded',
            agent_mode: 'live_performance_analyst',
            classification,
            focus,
            comparison,
        };
    },

    // ── Expert line ───────────────────────────────────────────────────────────

    async follow_expert_line(args) {
        return await apiService.post('/ai/expert-line-guidance', {
            session_id: getSessionId(args, context),
            data_types: args.data_types || ['speed', 'acceleration', 'braking', 'steering'],
        });
    },

    async get_telemetry_data(args) {
        return await apiService.post('/racing-session/telemetry', {
            session_id: getSessionId(args, context),
            data_types: args.data_types || ['speed', 'acceleration'],
        });
    },

    // ── Visualizations ────────────────────────────────────────────────────────

    async track_detail_for_guide() {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        context.startTrackGuide();
        context.setAgentTagActive?.('Track Guide', true);
        return { status: 'guidance_enabled', enabled: true };
    },

    async disable_guide_user_racing() {
        context.setTrackGuideEnabled(false);
        context.setAgentTagActive?.('Track Guide', false);
        return { status: 'guidance_disabled', enabled: false };
    },

    async get_visualization_capabilities() {
        return visualizationController.getVisualizationAssistantContext();
    },

    async show_map(args) {
        const candidates = getMapRequestCandidates(args, context);
        const requestedMap = candidates[0];
        const section = buildMapSectionSelection(args);
        const title = normalizeOptionalString(args.title) || 'Map';
        const note = normalizeOptionalString(args.message ?? args.note);

        let map: CircuitMapDto | null = null;
        let resolvedBy: 'id' | 'track' | null = null;

        if (context.getCircuitMapById) {
            for (const candidate of candidates) {
                map = await context.getCircuitMapById(candidate);
                if (map) {
                    resolvedBy = 'id';
                    break;
                }
            }
        }

        if (!map && context.getCircuitMapByTrack) {
            for (const candidate of candidates) {
                const sourceTrackKey = getAccTelemetryTrackKey(candidate) || candidate;
                map = await context.getCircuitMapByTrack('acc', sourceTrackKey);
                if (map) {
                    resolvedBy = 'track';
                    break;
                }
            }
        }

        if (!map) {
            const reason = requestedMap
                ? `No circuit map is available for "${requestedMap}".`
                : 'No circuit map is available for the current session.';
            context.displayMap?.(buildUnavailableMapDisplay(args, reason, requestedMap));
            return {
                status: 'unavailable',
                message: 'Map is not available',
                requested_map: requestedMap ?? null,
                reason,
            };
        }

        const display: AiMapDisplayPayload = {
            status: 'ready',
            map,
            requestedMap,
            title,
            note,
            section,
        };

        context.displayMap?.(display);

        return {
            status: 'displayed',
            map_id: map.id,
            circuit_name: map.circuit_name,
            source_track_key: map.source_track_key ?? null,
            resolved_by: resolvedBy,
            section: section ?? null,
        };
    },

    async open_visualization_chart(args) {
        return visualizationController.openVisualization(args.type, args.data, args.config);
    },

    async close_visualization_chart(args) {
        return visualizationController.closeVisualization({ id: args.chartId, type: args.type, all: args.all === true });
    },

    async invoke_visualization_control(args) {
        return await visualizationController.invokeVisualizationControl({
            control: args.control,
            id:      args.chartId,
            type:    args.type,
            args:    args.args,
        });
    },

    async update_guidance_once(args) {
        return await visualizationController.invokeVisualizationControl({
            control: 'refresh_once',
            id:      args.chartId,
            type:    args.type || 'imitation-guidance-chart',
            args:    args.args,
        });
    },

    async add_imitation_guidance_chart(args) {
        const result = visualizationController.openVisualization(
            'imitation-guidance-chart',
            { sessionId: getSessionId(args, context), manuallyAdded: true },
            { title: args.title || 'AI Driving Guidance', autoUpdate: args.autoUpdate !== false },
        );
        return { ...result, chartType: 'imitation-guidance-chart' };
    },

    async remove_imitation_guidance_chart(args) {
        const charts = visualizationController.getCurrentInstances()
            .filter(c => c.type === 'imitation-guidance-chart');
        let removed = 0;
        if (args.chartId) {
            if (visualizationController.closeVisualization({ id: args.chartId }).success) removed = 1;
        } else {
            charts.forEach(c => { if (visualizationController.closeVisualization({ id: c.id }).success) removed++; });
        }
        return { success: removed > 0, removedCount: removed };
    },

    async disable_ui_component(args) {
        if (args.component === 'chart' && context.analysisContext) return { success: true };
        return { success: false };
    },
    };

    return registry;
};
