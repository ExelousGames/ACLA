import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/lap-analysis/visualization/charts/circuitTrackLayout';
import { ToolHandlerContext } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { AiMapDisplayPayload, AiMapSectionSelection } from './AiMapToolDisplay';
import {
    SegmentClassificationResult,
    RecordedAiAnalysisState,
    normalizeSegmentClassificationResult,
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
    type ProcedurePlan,
    type ProcedurePlanRequest,
} from './ai-chat-plan';
import {
    PracticeParentSegmentView,
    PracticeSectionSummaryView,
    asRecord,
    buildPracticeTrackSummaryViews,
} from 'views/user-summary/user-summary-model';
import { detectOvertakeTacticalState } from './overtake-agent-detector';
import {
    buildBaselineCollectionToolPayload,
    type BaselineLapRecord,
    type BaselineCollectionTag,
} from './BaselineCollectionTracker';
import {
    AiToolDefinition,
    ToolOutputController,
    type ToolOutputEmitter,
    type ToolOutputEnvelope,
    executeAiToolDefinition,
    getToolEnvelopeError,
} from './ai-tool-base';
import type { LiveRangeTrackerToolResult } from './LiveRangeTracker';

type AiCommandHandler = (args: Record<string, any>, ctx: ToolHandlerContext) => Promise<any>;
export type AiCommandToolDefinition = AiToolDefinition<AiCommandRegistryContext, ToolHandlerContext>;
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
    setBaselineCollectionEnabled?: (enabled: boolean) => void;
    restartBaselineCollection?: () => void;
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
    getBaselineCollectionTag?: () => BaselineCollectionTag | null;
    getBaselineLapRecord?: () => BaselineLapRecord | null;
    getBaselineToolOutput?: () => ToolOutputEnvelope | null;
    subscribeBaselineToolOutput?: (listener: ToolOutputEmitter) => () => void;
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
    setLiveRangeTracker?: (args: Record<string, unknown>) => LiveRangeTrackerToolResult;
    updateLiveRangeTracker?: (args: Record<string, unknown>) => LiveRangeTrackerToolResult;
    getLiveRangeTracker?: () => LiveRangeTrackerToolResult;
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
    lastObservationKey: string | null;
    lastObservationAt: number;
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
const LIVE_RECORDED_ANALYSIS_TIMEOUT_MS = 120000;
const LIVE_RECORDED_ANALYSIS_ENDPOINT = '/racing-session/analyze-live-recorded-analysis';
const DEFAULT_BASELINE_COLLECTION_TIMEOUT_SECONDS = 600;
const MIN_BASELINE_COLLECTION_TIMEOUT_SECONDS = 30;
const MAX_BASELINE_COLLECTION_TIMEOUT_SECONDS = 900;

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

function stripTelemetryFieldToken(value: string): string {
    let token = value.trim();
    if (
        (token.startsWith("'") && token.endsWith("'"))
        || (token.startsWith('"') && token.endsWith('"'))
    ) {
        token = token.slice(1, -1).trim();
    }
    return token;
}

function parseTelemetryFieldString(value: string): string[] {
    const trimmed = value.trim();
    if (!trimmed) return [];

    try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) {
            return normalizeTelemetryFields(parsed);
        }
    } catch {
        // Accept Python-style array strings emitted by some models/tools.
    }

    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        const inner = trimmed.slice(1, -1).trim();
        if (!inner) return [];
        return inner
            .split(',')
            .map(stripTelemetryFieldToken)
            .filter(Boolean);
    }

    if (trimmed.includes(',')) {
        return trimmed
            .split(',')
            .map(stripTelemetryFieldToken)
            .filter(Boolean);
    }

    return [stripTelemetryFieldToken(trimmed)];
}

function normalizeTelemetryFields(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value.flatMap((field) => (
            typeof field === 'string'
                ? parseTelemetryFieldString(field)
                : []
        ));
    }
    if (typeof value === 'string') {
        return parseTelemetryFieldString(value);
    }
    return [];
}

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
    ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
    child_segments: (segment.child_segments || segment.sub_segments || [])
        .slice(0, 8)
        .map((child) => ({
            start_index: child.start_index,
            end_index: child.end_index,
            labels: child.labels,
            label_names: child.labels.map((labelId) => context.getLabelName?.(labelId) || labelId),
            ...(child.time_gap ? { time_gap: child.time_gap } : {}),
        })),
});

const getBaselineRecordPosition = (
    records: Record<string, any>[],
    index: number,
): number | null => {
    if (records.length === 0 || !Number.isFinite(index)) return null;

    const boundedIndex = Math.min(
        records.length - 1,
        Math.max(0, Math.trunc(index)),
    );
    const row = records[boundedIndex];
    if (!row) return null;

    const value = row.Graphics_normalized_car_position
        ?? row.normalized_position
        ?? row.normalizedPosition;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const summarizeLiveRecordedSegment = (
    segment: SegmentClassificationSegment,
    context: AiCommandRegistryContext,
    records: Record<string, any>[],
) => ({
    id: segment.id ?? null,
    start_position: getBaselineRecordPosition(records, segment.start_index),
    end_position: getBaselineRecordPosition(records, segment.end_index),
    parent_label: getSegmentMainLabelText(segment, context.getLabelName),
    child_labels: resolveSegmentChildLabelTexts(segment, context.getLabelName),
    label_ids: segment.labels ?? [],
    ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
    child_segments: (segment.child_segments || segment.sub_segments || [])
        .slice(0, 8)
        .map((child) => ({
            start_position: getBaselineRecordPosition(records, child.start_index),
            end_position: getBaselineRecordPosition(records, child.end_index),
            labels: child.labels,
            label_names: child.labels.map((labelId) => context.getLabelName?.(labelId) || labelId),
            ...(child.time_gap ? { time_gap: child.time_gap } : {}),
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

const buildLiveRecordedAnalysisToolResult = (
    result: SegmentClassificationResult,
    baselineRecord: BaselineLapRecord,
    context: AiCommandRegistryContext,
    args: Record<string, any> = {},
) => {
    const limit = getRecordedSegmentLimit(args.limit);
    const segments = Array.isArray(result.segments) ? result.segments : [];

    return {
        status: result.segment_count > 0 ? 'ready' : 'empty',
        message: result.segment_count > 0
            ? 'Live baseline lap analysis is ready.'
            : 'Live baseline lap analysis found no classified segments.',
        source: 'baseline_lap_record',
        baseline: {
            id: baselineRecord.id,
            lap: baselineRecord.lap,
            track: baselineRecord.track || context.sessionIntelligence?.getLiveSessionSnapshot?.().track || null,
            car: baselineRecord.car || context.sessionIntelligence?.getLiveSessionSnapshot?.().car || null,
            sample_count: baselineRecord.sample_count,
            captured_at: baselineRecord.captured_at,
        },
        analysis: {
            status: result.status,
            session_id: result.session_id,
            samples_analyzed: result.samples_analyzed,
            segment_count: result.segment_count,
            returned_segment_count: Math.min(segments.length, limit),
            ...(typeof result.expert_time_available === 'boolean'
                ? { expert_time_available: result.expert_time_available }
                : {}),
            segments: segments
                .slice(0, limit)
                .map((segment) => summarizeLiveRecordedSegment(segment, context, baselineRecord.records)),
        },
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

const normalizeAgentSessionMode = (value: unknown): AgentSessionMode | null => (
    value === 'track_guide'
    || value === 'overtake'
    || value === 'live_performance_analyst'
        ? value
        : null
);

const runRecordedAnalysisForLiveRequest = async (
    context: AiCommandRegistryContext,
    args: Record<string, any> = {},
    baselineRecordOverride?: BaselineLapRecord | null,
): Promise<
    | { status: 'ready'; analysis: ReturnType<typeof buildLiveRecordedAnalysisToolResult> }
    | { status: 'error'; error: LiveAnalystRecordedAnalysisError; message: string }
> => {
    const baselineRecord = baselineRecordOverride ?? context.getBaselineLapRecord?.() ?? null;
    if (!baselineRecord || baselineRecord.records.length === 0) {
        return {
            status: 'error',
            error: 'baseline_lap_record_required',
            message: 'Live performance analysis requires a recorded baseline lap before it can request classifier analysis.',
        };
    }

    try {
        const response = await apiService.post(LIVE_RECORDED_ANALYSIS_ENDPOINT, {
            track: baselineRecord.track,
            car: baselineRecord.car,
            baseline_lap: baselineRecord.lap,
            records: baselineRecord.records,
        }, { timeout: LIVE_RECORDED_ANALYSIS_TIMEOUT_MS });
        const result = normalizeSegmentClassificationResult(response.data as any, baselineRecord.id);
        return {
            status: 'ready',
            analysis: buildLiveRecordedAnalysisToolResult(result, baselineRecord, context, { limit: 8, ...args }),
        };
    } catch (error: any) {
        return {
            status: 'error',
            error: 'recorded_analysis_failed',
            message: error?.data?.message || error?.message || 'Failed to run live baseline analysis.',
        };
    }
};

const getCachedBaselineLapRecord = (
    context: AiCommandRegistryContext,
): BaselineLapRecord | null => {
    const record = context.getBaselineLapRecord?.() ?? null;
    return record?.records?.length ? record : null;
};

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

const getBaselineRecorderReadiness = (context: AiCommandRegistryContext) => {
    const record = context.getBaselineLapRecord?.() ?? null;
    const tag = context.getBaselineCollectionTag?.() ?? null;
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

const getBaselineCollectionTimeoutMs = (args: Record<string, any>): number => {
    const seconds = toPositiveNumber(args.timeout_seconds) ?? DEFAULT_BASELINE_COLLECTION_TIMEOUT_SECONDS;
    return Math.min(
        MAX_BASELINE_COLLECTION_TIMEOUT_SECONDS,
        Math.max(MIN_BASELINE_COLLECTION_TIMEOUT_SECONDS, seconds),
    ) * 1000;
};

const getCurrentBaselineCollectionPayload = (context: AiCommandRegistryContext) => (
    buildBaselineCollectionToolPayload(
        context.getBaselineCollectionTag?.() ?? null,
        getCachedBaselineLapRecord(context),
    )
);

const finishBaselineCollectionFromEnvelope = (
    envelope: ToolOutputEnvelope,
    output: ToolOutputController,
) => {
    const envelopeError = getToolEnvelopeError(envelope);
    if (envelopeError) {
        return output.error(envelopeError, envelope.ui_output, { message: envelope.message });
    }
    return output.final(envelope.ui_output, { message: envelope.message });
};

const collectBaselineLapFromTrackerOutput = async (
    context: AiCommandRegistryContext,
    args: Record<string, any>,
    output: ToolOutputController,
): Promise<ToolOutputEnvelope> => {
    const unavailable = buildLiveAnalystUnavailable(context);
    if (unavailable) return output.error(unavailable.error || 'baseline_collection_unavailable', unavailable);

    context.setLivePerformanceAnalystEnabled?.(true);
    context.setBaselineCollectionEnabled?.(true);
    context.setAgentTagActive?.('Live Analyst', true);

    const initial = context.getBaselineToolOutput?.() ?? null;
    if (initial?.tool_name === 'collect_live_baseline') {
        if (initial.final) {
            return finishBaselineCollectionFromEnvelope(initial, output);
        }
    }

    const initialPayload = getCurrentBaselineCollectionPayload(context);
    if (initialPayload.status === 'complete') {
        return output.final(initialPayload);
    }

    const timeoutMs = getBaselineCollectionTimeoutMs(args);
    const subscribe = context.subscribeBaselineToolOutput;
    return new Promise((resolve) => {
        let settled = false;
        let unsubscribe: () => void = () => undefined;

        const settle = (result: ToolOutputEnvelope) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeoutId);
            unsubscribe();
            resolve(result);
        };

        const handleEnvelope: ToolOutputEmitter = (envelope) => {
            if (envelope.tool_name !== 'collect_live_baseline') {
                return;
            }

            if (envelope.final) {
                settle(finishBaselineCollectionFromEnvelope(envelope, output));
                return;
            }
        };

        const timeoutId = setTimeout(() => {
            const progress = context.getBaselineToolOutput?.()?.ui_output
                ?? getCurrentBaselineCollectionPayload(context);
            const progressRecord = progress && typeof progress === 'object' && !Array.isArray(progress)
                ? progress as Record<string, any>
                : {};
            const timeoutPayload = {
                status: 'error',
                error: 'baseline_collection_timeout',
                progress_percent: Number(progressRecord.progress_percent ?? 0),
                car: typeof progressRecord.car === 'string' ? progressRecord.car : null,
                track: typeof progressRecord.track === 'string' ? progressRecord.track : null,
                message: 'Baseline collection did not complete before the tool timeout.',
            };
            settle(output.error(
                'baseline_collection_timeout',
                timeoutPayload,
                { message: timeoutPayload.message },
            ));
        }, timeoutMs);

        if (subscribe) {
            unsubscribe = subscribe(handleEnvelope);
            const current = context.getBaselineToolOutput?.();
            if (current) {
                handleEnvelope(current, { final: current.final });
            }
            return;
        }
    });
};

const restartLiveBaselineCollection = (
    context: AiCommandRegistryContext,
    output: ToolOutputController,
): ToolOutputEnvelope => {
    const unavailable = buildLiveAnalystUnavailable(context);
    if (unavailable) {
        return output.error(unavailable.error || 'baseline_restart_unavailable', unavailable);
    }

    if (!context.restartBaselineCollection) {
        return output.error('baseline_restart_unavailable', {
            status: 'error',
            error: 'baseline_restart_unavailable',
            message: 'Baseline restart is not available in this view.',
        });
    }

    context.restartBaselineCollection();
    context.setBaselineCollectionEnabled?.(true);

    const payload = {
        status: 'restarted',
        progress_percent: 0,
        message: 'Baseline collection restarted.',
    };
    return output.final(payload, { message: payload.message });
};

const getSessionId = (args: Record<string, any>, context: AiCommandRegistryContext): string | undefined =>
    args.session_id ||
    context.sessionId ||
    context.analysisContext?.sessionSelected?.SessionId;

export const startAgentRuntime = async (
    agentMode: AgentSessionMode,
    context: AiCommandRegistryContext,
    args: Record<string, any>,
    ctx: ToolHandlerContext,
): Promise<any> => {
    if (agentMode === 'track_guide') {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        context.startTrackGuide();
        context.setAgentTagActive?.('Track Guide', true);
        return { status: 'started', agent_mode: 'track_guide', enabled: true };
    }

    if (agentMode === 'overtake') {
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
    }

    const unavailable = buildLiveAnalystUnavailable(context);
    if (unavailable) return unavailable;

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
};

const createRawAiCommandRegistry = (context: AiCommandRegistryContext): Record<string, AiCommandHandler> => {
    const registry: Record<string, AiCommandHandler> = {

    // ── Session ───────────────────────────────────────────────────────────────

    async start_agent_session(args) {
        const agentMode = normalizeAgentSessionMode(args.agent_mode || args.agentMode);
        if (!agentMode) {
            return {
                status: 'error',
                error: 'unsupported_agent_mode',
                message: 'Supported agent modes are track_guide, overtake, and live_performance_analyst.',
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

    // Constrained-reduce variant. Clamp invalid reduce values to 'stats'
    // so a bad tool call can't leak rows.
    async query_telemetry_metric(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        const allowed = new Set(['avg', 'min', 'max', 'stats']);
        const reduce = allowed.has(args.reduce) ? args.reduce : 'stats';
        const fields = normalizeTelemetryFields(args.fields);
        if (fields.length === 0) {
            return { error: 'telemetry_fields_required' };
        }
        return si.query({ fields, scope: args.scope, reduce } as any);
    },

    // Server-internal: backs analyze_telemetry. Returns raw rows over the
    // WS relay so the server-side classifier can consume them. NOT exposed
    // to the LLM (absent from the backend tool registry) - rows must never
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

    async get_live_focus_section() {
        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const snapshot = buildLiveAnalystSnapshot(context);
        if (!getBaselineRecorderReadiness(context).ready) {
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

    async set_live_range_tracker(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        if (!context.setLiveRangeTracker) {
            return {
                status: 'error',
                error: 'live_range_tracker_unavailable',
                message: 'Live range tracker UI is not mounted.',
            };
        }
        return context.setLiveRangeTracker(args);
    },

    async update_live_range_tracker(args) {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        if (!context.updateLiveRangeTracker) {
            return {
                status: 'error',
                error: 'live_range_tracker_unavailable',
                message: 'Live range tracker UI is not mounted.',
            };
        }
        return context.updateLiveRangeTracker(args);
    },

    async get_live_range_tracker() {
        if (!isLiveSessionContext(context)) return { error: getLiveToolsUnavailableError(context) };
        if (!context.getLiveRangeTracker) {
            return {
                status: 'error',
                error: 'live_range_tracker_unavailable',
                message: 'Live range tracker UI is not mounted.',
            };
        }
        return context.getLiveRangeTracker();
    },

    async analyze_live_recorded_analysis(args) {
        const unavailable = buildLiveAnalystUnavailable(context);
        if (unavailable) return unavailable;

        const baselineRecord = getCachedBaselineLapRecord(context);
        if (!baselineRecord) {
            const snapshot = buildLiveAnalystSnapshot(context);
            const message = 'Live recorded analysis requires a recorded baseline lap before it can run.';
            context.sessionIntelligence?.emitRecordedAnalysisError(
                'baseline_lap_record_required',
                message,
                snapshot,
            );
            return buildLiveAnalystPlanError(
                'baseline_lap_record_required',
                message,
                snapshot,
            );
        }

        const analysisStatus = await runRecordedAnalysisForLiveRequest(
            context,
            args,
            baselineRecord,
        );
        if (analysisStatus.status !== 'ready') {
            context.sessionIntelligence?.emitRecordedAnalysisError(
                analysisStatus.error,
                analysisStatus.message,
                buildLiveAnalystSnapshot(context),
            );
            return buildLiveAnalystPlanError(
                analysisStatus.error,
                analysisStatus.message,
                buildLiveAnalystSnapshot(context),
            );
        }

        const agent = getLiveAnalystState(context);
        agent.analysisSessionId = analysisStatus.analysis.baseline.id;
        context.sessionIntelligence?.emitRecordedAnalysisReady(
            analysisStatus.analysis,
            buildLiveAnalystSnapshot(context),
        );
        return analysisStatus.analysis;
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

    async advance_plan_step(args) {
        return context.advanceProcedurePlanStep?.(normalizeOptionalString(args.reason)) || {
            status: 'unavailable',
            error: 'no_procedure_plan_ui',
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
        const snapshot = buildLiveAnalystSnapshot(context);
        if (!getBaselineRecorderReadiness(context).ready) {
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
        const snapshot = buildLiveAnalystSnapshot(context);
        if (!getBaselineRecorderReadiness(context).ready) {
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
    'set_live_range_tracker',
    'update_live_range_tracker',
    'get_live_range_tracker',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    '_get_live_section_telemetry',
    '_record_live_section_classification',
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

const getToolUiRecord = (uiOutput: unknown): Record<string, any> => (
    uiOutput && typeof uiOutput === 'object' && !Array.isArray(uiOutput)
        ? uiOutput as Record<string, any>
        : {}
);

const getToolAiStatus = (uiOutput: Record<string, any>): string => (
    typeof uiOutput.status === 'string'
        ? uiOutput.status
        : uiOutput.error
            ? 'error'
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

const summarizeLiveRangeForAi = (uiOutput: Record<string, any>) => {
    const ranges = Array.isArray(uiOutput.tracker?.ranges) ? uiOutput.tracker.ranges : [];
    return {
        tracker_status: uiOutput.tracker?.status ?? null,
        range_count: ranges.length,
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

const summarizeLiveRecordedSegmentsForAi = (segments: unknown): Record<string, unknown>[] => (
    Array.isArray(segments)
        ? segments.slice(0, 5).map((segment) => {
            const record = getToolUiRecord(segment);
            return {
                id: record.id ?? null,
                parent_label: record.parent_label ?? null,
                child_labels: Array.isArray(record.child_labels) ? record.child_labels : [],
                start_position: record.start_position ?? null,
                end_position: record.end_position ?? null,
            };
        })
        : []
);

const buildToolAiOutput = (
    name: typeof ALL_AI_TOOL_NAMES[number],
    uiOutputValue: unknown,
): Record<string, unknown> => {
    const uiOutput = getToolUiRecord(uiOutputValue);
    const status = getToolAiStatus(uiOutput);
    const error = typeof uiOutput.error === 'string' ? uiOutput.error : undefined;
    const output: Record<string, unknown> = {
        name,
        status,
        message: getToolAiMessage(
            uiOutput,
            error || `${name} ${status}.`,
        ),
    };
    if (error) {
        output.error = error;
    }

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
        case 'set_live_range_tracker':
        case 'update_live_range_tracker':
        case 'get_live_range_tracker':
            Object.assign(output, summarizeLiveRangeForAi(uiOutput));
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
            output.segment_count = uiOutput.segment_count ?? uiOutput.analysis?.segment_count ?? 0;
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
        case 'analyze_live_recorded_analysis':
            output.source = uiOutput.source ?? null;
            output.segment_count = uiOutput.analysis?.segment_count ?? 0;
            output.samples_analyzed = uiOutput.analysis?.samples_analyzed ?? 0;
            output.expert_time_available = uiOutput.analysis?.expert_time_available ?? null;
            output.segments = summarizeLiveRecordedSegmentsForAi(uiOutput.analysis?.segments);
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
            if (name === 'collect_live_baseline') {
                return collectBaselineLapFromTrackerOutput(context, args, output);
            }
            if (name === 'restart_live_baseline') {
                return restartLiveBaselineCollection(context, output);
            }

            const rawRegistry = createRawAiCommandRegistry(context);
            const handler = rawRegistry[name];
            if (!handler) {
                return output.error('tool_not_registered', {
                    status: 'error',
                    error: 'tool_not_registered',
                    message: `Tool ${name} is not registered.`,
                });
            }
            return handler(args, handlerContext);
        },
        formatAiOutput: (uiOutput) => buildToolAiOutput(name, uiOutput),
    };
};

export const frontendToolDefinitions: AiCommandToolDefinition[] = ALL_AI_TOOL_NAMES
    .map(createAiToolDefinition);

export const createAiCommandRegistry = (
    context: AiCommandRegistryContext,
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

