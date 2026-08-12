import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
} from 'contexts/AiToolComponentRefContext';
import {
    BaselineCollectionIncompleteError,
    BaselineCollectionVisualizationRequiredError,
    ComponentDisableFailedError,
    ComponentRefUnavailableError,
    NoRecordedSessionError,
    NonLiveContextLiveToolsUnavailableError,
    RecordedAnalysisFailedError,
    RecordedSessionLiveToolsUnavailableError,
} from 'contexts/AiToolComponentError';
import type { ToolHandlerContext } from './use-voice-conversation';
import {
    AiToolError,
    AmbiguousComponentTargetError,
    CircuitMapLookupFailedError,
    CreateGoalToolUnavailableError,
    FocusSectionNotReadyError,
    InvalidProcedurePlanRequestsError,
    LivePerformanceAnalystToolUnavailableError,
    LiveSectionClassificationFailedError,
    LiveSectionTelemetryUnavailableError,
    NoCornerDataError,
    NoTelemetryForScopeError,
    NotRecordedModeError,
    RetryGoalTaskToolUnavailableError,
    SectionNotFoundError,
    TelemetryAnalysisFailedError,
    TelemetryFieldsRequiredError,
    UnsupportedAgentModeError,
    VisualizationControlUnavailableError,
    type AiToolExecutionOutput,
    type ToolOutputController,
} from './ai-tool-base';
import type { AiChatHandle } from './ai-chat';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import type { SessionAnalysisHandle } from 'views/lap-analysis/session-analysis';
import type { UserSummaryHandle } from 'views/user-summary/user-summary';
import {
    buildGoalRequest,
    buildProcedurePlan,
    GoalRunner,
    LiveRangeTodoListRunner,
    ProcedurePlanRunner,
} from 'components/ai-engineering-tools';
import {
    createLiveRangeTodoAiAdapter,
    type LiveRangeTodoTaskStartFunctionFactory,
} from './live-range-todo-ai-adapter';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import type { TelemetryOverviewHandle } from 'views/lap-analysis/visualization/charts/TelemetryOverview';
import type { LiveTelemetryOverviewHandle } from 'views/live-session/LiveTelemetryOverview';
import type { EventLogChartHandle } from 'views/lap-analysis/visualization/charts/EventLogChart';
import type { LiveEventLogHandle } from 'views/live-session/LiveEventLog';
import type { MapVisualizationHandle } from 'views/lap-analysis/visualization/charts/MapVisualization';
import type { AnalysisResultsChartHandle } from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import type { ImitationGuidanceChartHandle } from 'views/lap-analysis/visualization/charts/ImitationGuidanceChart';
import {
    deriveTelemetryMetricFamilies,
    getSingletonVisualizationComponentName,
    getTelemetryComponentName,
    getVisualizationComponentName,
    isTelemetryComponentName,
} from 'views/lap-analysis/visualization/visualization-component-names';
import { isLiveSessionAiAvailable } from 'views/lap-analysis/recording-state';
import {
    normalizeSegmentClassificationResult,
    RecordedAiAnalysisState,
    SegmentClassificationResult,
} from 'views/lap-analysis/recorded-session-analysis';
import { getSegmentLabelIds } from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import type { AnalysisResultElement } from 'views/lap-analysis/visualization/charts/analysisResultsModel';
import { adaptAnalysisResultsComparison } from 'views/lap-analysis/visualization/charts/analysisResultsComparisonAdapter';
import {
    type BaselineCollectionHandle,
} from 'views/live-session/BaselineCollection';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/lap-analysis/visualization/charts/circuitTrackLayout';
import type { AiMapDisplayPayload, AiMapSectionSelection } from './AiMapToolDisplay';
import { getLiveAnalysisMistakeCount } from 'views/live-session/live-session-analysis-results';

export interface RefAiCommandContext {
    componentRefs?: AiToolComponentRefDirectory;
    sessionMode?: 'front_desk' | 'live' | 'recorded' | 'user_summary';
    sessionId?: string;
    conversationRole?: 'main' | 'agent';
    agentMode?: 'track_guide' | 'overtake' | 'live_performance_analyst';
}

type RefAiCommandHandler = (
    args: Record<string, any>,
    handlerContext: ToolHandlerContext,
    output: ToolOutputController,
) => Promise<AiToolExecutionOutput> | AiToolExecutionOutput;

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getFailureMessage = (error: unknown, fallback: string): string => {
    if (!isRecord(error)) return error instanceof Error && error.message.trim()
        ? error.message
        : fallback;
    const response = isRecord(error.response) ? error.response : null;
    const responseData = response && isRecord(response.data) ? response.data : null;
    const data = isRecord(error.data) ? error.data : null;
    const message = responseData?.message ?? data?.message ?? error.message;
    return typeof message === 'string' && message.trim() ? message : fallback;
};

const getDirectory = (context: RefAiCommandContext): AiToolComponentRefDirectory => {
    if (context.componentRefs) return context.componentRefs;
    throw new ComponentRefUnavailableError(
        'dashboard',
        'The active dashboard component-ref directory is unavailable.',
    );
};

const getChat = (context: RefAiCommandContext) => resolveNamedComponentHandle<AiChatHandle>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
);

const getLive = (context: RefAiCommandContext) => resolveNamedComponentHandle<LiveSessionHandle>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
);

const getRecorded = (context: RefAiCommandContext) => resolveNamedComponentHandle<SessionAnalysisHandle>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
);

const getSummary = (context: RefAiCommandContext) => resolveNamedComponentHandle<UserSummaryHandle>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
);

const getManagerName = (context: RefAiCommandContext): string => (
    context.sessionMode === 'live'
        ? AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER
        : AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER
);

const getManager = (context: RefAiCommandContext) => resolveNamedComponentHandle<VisualizationManagerHandle>(
    getDirectory(context),
    getManagerName(context),
);

const getBaseline = (
    context: RefAiCommandContext,
): BaselineCollectionHandle => {
    const directory = getDirectory(context);
    if (!directory.findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)?.current) {
        throw new BaselineCollectionVisualizationRequiredError(
            AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            'Baseline collection visualization required. Open Baseline Collection in Live Session and keep it open.',
        );
    }
    return resolveNamedComponentHandle<BaselineCollectionHandle>(
        directory,
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    );
};

const openBaselineVisualization = async (context: RefAiCommandContext) => {
    const manager = getManager(context);
    const requested = manager.requestVisualization({
        name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
        type: 'baseline-collection',
    });

    const handle = await awaitNamedComponentHandle<BaselineCollectionHandle>(
        getDirectory(context),
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    );
    return { requested, handle };
};

const requireLive = (context: RefAiCommandContext): LiveSessionHandle => {
    const ErrorType = context.sessionMode === 'recorded'
        ? RecordedSessionLiveToolsUnavailableError
        : NonLiveContextLiveToolsUnavailableError;
    if (context.sessionMode && context.sessionMode !== 'live') {
        throw new ErrorType(
            AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
            'Live-session AI tools are unavailable in the current session mode.',
        );
    }
    const live = getLive(context);
    if (!isLiveSessionAiAvailable(live.getRecordingState())) {
        throw new ErrorType(
            AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
            'Live-session AI tools require an active live recording.',
        );
    }
    return live;
};

const normalizeOptionalString = (value: unknown): string | undefined => (
    typeof value === 'string' && value.trim() ? value.trim() : undefined
);

const getGoal = (context: RefAiCommandContext) => resolveNamedComponentHandle<GoalRunner>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.GOAL,
);

const getLiveRangeTodoListRunner = (
    context: RefAiCommandContext,
): LiveRangeTodoListRunner => resolveNamedComponentHandle<LiveRangeTodoListRunner>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
);

const getProcedurePlanRunner = (
    context: RefAiCommandContext,
): ProcedurePlanRunner => resolveNamedComponentHandle<ProcedurePlanRunner>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
);

const createLiveRangeNotificationTaskStartFunctionFactory = (
    handlerContext: ToolHandlerContext,
): LiveRangeTodoTaskStartFunctionFactory => (event) => (signal) => {
    if (signal.aborted) return;
    const data = event.data && typeof event.data === 'object' && !Array.isArray(event.data)
        ? event.data as Record<string, unknown>
        : {};
    handlerContext.sendToolStatus({
        source: 'live_range_todo_list',
        event: normalizeOptionalString(data.event) || 'live_range_todo_event_due',
        event_id: event.id,
        content: event.content,
        normalized_position: event.normalized_position,
        lead_time_seconds: event.lead_time_seconds ?? 2,
        data: event.data,
    });
};

const normalizeFields = (value: unknown): string[] => {
    if (Array.isArray(value)) return value.flatMap(normalizeFields);
    if (typeof value !== 'string') return [];
    const trimmed = value.trim();
    if (!trimmed) return [];
    try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return normalizeFields(parsed);
    } catch {
        // Accept Python-style and comma-delimited arrays from model calls.
    }
    return trimmed.replace(/^\[|\]$/g, '').split(',')
        .map((field) => field.replace(/^['"]|['"]$/g, '').trim())
        .filter(Boolean);
};

const getLimit = (value: unknown, fallback = 20, maximum = 50): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, maximum) : fallback;
};

const getLabelText = (chat: AiChatHandle, labelId?: string | null): string | null => (
    labelId ? chat.getLabelName(labelId) || labelId : null
);

const compactSegment = (segment: any, chat: AiChatHandle) => ({
    id: segment.id ?? null,
    start_index: segment.start_index,
    end_index: segment.end_index,
    track_section: getLabelText(chat, segment.track_section),
    labels: getSegmentLabelIds(segment)
        .map((labelId) => getLabelText(chat, labelId))
        .filter(Boolean),
    ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
});

const compactRecordedAnalysis = (
    recorded: SessionAnalysisHandle,
    chat: AiChatHandle,
    state: RecordedAiAnalysisState = recorded.getRecordedAiAnalysis(),
    limit = 20,
) => {
    const selected = recorded.getSelectedSession();
    if (!selected?.SessionId) {
        throw new NoRecordedSessionError(
            AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            'No recorded session is selected.',
        );
    }
    if (state.status === 'error') {
        throw new RecordedAnalysisFailedError(
            AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            state.message || 'Recorded-session analysis failed.',
        );
    }
    const result = state.result;
    return {
        status: state.status,
        message: state.message || null,
        session_id: selected.SessionId,
        session_name: selected.session_name || null,
        map: selected.map || recorded.getMapSelected(),
        car: selected.car || null,
        analysis: result ? {
            status: result.status,
            session_id: result.session_id,
            samples_analyzed: result.samples_analyzed,
            segments: result.segments.slice(0, limit).map((segment) => compactSegment(segment, chat)),
        } : null,
    };
};

const getBaselinePosition = (records: Record<string, any>[], index: number): number | null => {
    const row = records[Math.max(0, Math.min(records.length - 1, Math.trunc(index)))];
    const parsed = Number(row?.Graphics_normalized_car_position ?? row?.normalized_position ?? row?.normalizedPosition);
    return Number.isFinite(parsed) ? parsed : null;
};

const buildAnalysisElements = (
    result: SegmentClassificationResult,
    chat: AiChatHandle,
    records: Record<string, any>[] = [],
): AnalysisResultElement[] => result.segments.map((segment, index) => {
    const start = records.length ? getBaselinePosition(records, segment.start_index) : null;
    const end = records.length ? getBaselinePosition(records, segment.end_index) : null;
    const comparison = records.length && segment.expert_reference_data.length
        ? adaptAnalysisResultsComparison({
            baselineRecords: records,
            expertReferenceData: segment.expert_reference_data,
        })
        : undefined;
    return {
        id: segment.id || `${result.session_id}:segment:${index}`,
        labels: getSegmentLabelIds(segment).map((labelId) => chat.getLabelName(labelId) || labelId),
        ...(segment.track_section ? { section: chat.getLabelName(segment.track_section) || segment.track_section } : {}),
        ...(start !== null && end !== null ? { normalizedPositionRange: { start, end } } : {}),
        ...(comparison?.samples.length ? { comparison } : {}),
        metadata: {
            source: 'ai_classifier',
            start_index: segment.start_index,
            end_index: segment.end_index,
        },
    };
});

const compactTelemetryAnalysis = (
    result: SegmentClassificationResult,
    chat: AiChatHandle,
    limit = 20,
) => ({
    status: result.segments.length > 0 ? 'ready' : 'empty',
    message: result.segments.length > 0
        ? 'Telemetry analysis is ready.'
        : 'Telemetry analysis found no classified segments.',
    analysis: {
        status: result.status,
        session_id: result.session_id,
        samples_analyzed: result.samples_analyzed,
        segments: result.segments.slice(0, limit).map((segment) => compactSegment(segment, chat)),
    },
});

const ensureAnalysisResultsChart = async (
    context: RefAiCommandContext,
    elements: AnalysisResultElement[],
) => {
    const directory = getDirectory(context);
    const name = getSingletonVisualizationComponentName('analysis-results');
    if (directory.findComponentRef<AnalysisResultsChartHandle>(name)?.current) {
        const chart = resolveNamedComponentHandle<AnalysisResultsChartHandle>(directory, name);
        chart.replaceAnalysisResults({ elements });
        const instance = getManager(context).getCurrentVisualizations()
            .find((visualization) => visualization.name === name);
        return {
            success: true,
            message: `Reused chart '${name}'.`,
            componentName: name,
            chartId: instance?.id,
            chartType: instance?.type || 'analysis-results',
            reused: true,
        };
    }

    const manager = getManager(context);
    const requested = manager.requestVisualization({
        name,
        type: 'analysis-results',
        data: { elements },
    });
    const mountedName = requested.componentName || name;
    await directory.awaitComponentRef<AnalysisResultsChartHandle>(mountedName);
    const chart = resolveNamedComponentHandle<AnalysisResultsChartHandle>(directory, mountedName);
    chart.replaceAnalysisResults({ elements });
    return { ...requested, componentName: mountedName };
};

const getMapSection = (args: Record<string, any>): AiMapSectionSelection | undefined => {
    const clamp = (value: unknown) => {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : undefined;
    };
    const start = clamp(args.section_start ?? args.start);
    const end = clamp(args.section_end ?? args.end);
    const label = normalizeOptionalString(args.section_label ?? args.label);
    return start === undefined && end === undefined && !label ? undefined : { start, end, label };
};

const resolveMap = async (
    chat: AiChatHandle,
    args: Record<string, any>,
    context: RefAiCommandContext,
) => {
    let selectedMap: string | null = null;
    try {
        selectedMap = getRecorded(context).getSelectedSession()?.map || getRecorded(context).getMapSelected();
    } catch {
        try {
            selectedMap = getLive(context).getLiveSessionSnapshot().track;
        } catch {
            selectedMap = null;
        }
    }
    const candidates = [args.map_id, args.source_track_key, args.map_name, selectedMap]
        .map(normalizeOptionalString)
        .filter((value): value is string => Boolean(value));
    let map: CircuitMapDto | null = null;
    let resolvedBy: 'id' | 'track' | null = null;
    for (const candidate of candidates) {
        map = await chat.getCircuitMapById(candidate);
        if (map) {
            resolvedBy = 'id';
            break;
        }
    }
    if (!map) {
        for (const candidate of candidates) {
            map = await chat.getCircuitMapByTrack('acc', getAccTelemetryTrackKey(candidate) || candidate);
            if (map) {
                resolvedBy = 'track';
                break;
            }
        }
    }
    return { map, resolvedBy, requestedMap: candidates[0] };
};

const getFocusPayload = (live: LiveSessionHandle) => {
    const intelligence = live.getSessionIntelligence();
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
            source_track_key: live.getLiveSessionSnapshot().track,
            section_start: focus.section.from,
            section_end: focus.section.to,
            section_label: focus.section.name,
            title: 'Live analyst focus',
        },
    };
};

const getVisualizationTarget = (
    manager: VisualizationManagerHandle,
    args: Record<string, unknown>,
): { name: string } => {
    const instances = manager.getCurrentVisualizations();
    const byId = normalizeOptionalString(args.chartId ?? args.chart_id);
    if (byId) {
        const match = instances.find((instance) => instance.id === byId);
        if (!match) {
            throw new ComponentRefUnavailableError(
                byId,
                `Chart '${byId}' is not open.`,
            );
        }
        return { name: match.name };
    }
    const explicit = normalizeOptionalString(args.component_name ?? args.componentName);
    if (explicit) return { name: explicit };
    const type = normalizeOptionalString(args.type) || 'telemetry-overview';
    if (type !== 'telemetry-overview') return { name: getVisualizationComponentName(type, args) };
    const families = deriveTelemetryMetricFamilies(args);
    if (families.length > 0) return { name: getTelemetryComponentName(families) };
    const candidates = instances.map((instance) => instance.name).filter(isTelemetryComponentName).sort();
    if (candidates.length > 1) {
        throw new AmbiguousComponentTargetError(
            'Choose a telemetry metric family and retry the same tool.',
        );
    }
    return { name: candidates[0] || getTelemetryComponentName([]) };
};

const updateMountedVisualization = (
    type: string,
    handle: any,
    data: any,
    config?: any,
) => {
    switch (type) {
        case 'telemetry-overview':
            if ('updateLiveTelemetry' in handle) return (handle as LiveTelemetryOverviewHandle).updateLiveTelemetry(data);
            return (handle as TelemetryOverviewHandle).updateTelemetry(data, config);
        case 'event-log':
            if ('updateLiveEvents' in handle) return (handle as LiveEventLogHandle).updateLiveEvents(data);
            return (handle as EventLogChartHandle).updateEvents(data);
        case 'map-visualization':
            return (handle as MapVisualizationHandle).updateMap(data, config);
        case 'analysis-results':
            return (handle as AnalysisResultsChartHandle).replaceAnalysisResults(data);
        case 'imitation-guidance-chart':
            return (handle as ImitationGuidanceChartHandle).updateGuidanceData(data, config);
        default:
            return true;
    }
};

const openVisualization = async (context: RefAiCommandContext, args: Record<string, unknown>) => {
    const manager = getManager(context);
    const target = getVisualizationTarget(manager, args);
    const type = normalizeOptionalString(args.type) || 'telemetry-overview';
    const directory = getDirectory(context);
    const mountedRef = directory.findComponentRef<any>(target.name!);
    if (mountedRef?.current) {
        const child = resolveNamedComponentHandle<any>(directory, target.name!);
        const instance = manager.getCurrentVisualizations().find(({ name }) => name === target.name);
        updateMountedVisualization(type, child, args.data, args.config);
        return {
            success: true,
            message: `Reused chart '${target.name}'.`,
            componentName: target.name!,
            chartId: instance?.id,
            chartType: instance?.type || type,
            reused: true,
        };
    }
    const requested = manager.requestVisualization({
        name: target.name!,
        type,
        data: args.data,
        config: args.config,
    });
    const mountedName = requested.componentName || target.name!;
    const child = await awaitNamedComponentHandle<any>(directory, mountedName);
    updateMountedVisualization(type, child, args.data, args.config);
    return { ...requested, componentName: mountedName };
};

const invokeVisualizationControl = async (context: RefAiCommandContext, args: Record<string, unknown>) => {
    const manager = getManager(context);
    const target = getVisualizationTarget(manager, args);
    const name = target.name!;
    const child = resolveNamedComponentHandle<any>(getDirectory(context), name);
    const type = manager.getCurrentVisualizations().find((instance) => instance.name === name)?.type || args.type;
    const control = args.control;
    const controlArgs: any = args.args || {};
    let result: any;
    if (type === 'analysis-results') {
        const chart = child as AnalysisResultsChartHandle;
        if (control === 'append_element') result = chart.appendAnalysisResult(controlArgs.element);
        else if (control === 'update_element') result = chart.updateAnalysisResult(controlArgs.id, controlArgs.changes);
        else if (control === 'remove_element') result = chart.removeAnalysisResult(controlArgs.id);
    } else if (type === 'imitation-guidance-chart' && control === 'refresh_once') {
        result = await (child as ImitationGuidanceChartHandle).refreshGuidanceOnce();
    }
    if (!result) {
        throw new VisualizationControlUnavailableError(
            `Control '${String(control)}' is not available for chart '${name}'.`,
        );
    }
    return {
        success: true,
        message: result.message || `Executed '${control}' on chart '${name}'.`,
        componentName: name,
        chartType: type,
        control,
        data: result.data ?? result,
    };
};

const getTelemetryStats = (rows: Record<string, any>[]) => ({
    row_count: rows.length,
    field_count: Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).length,
});

const createHandlers = (context: RefAiCommandContext): Record<string, RefAiCommandHandler> => ({
    async start_agent_session(args) {
        const chat = getChat(context);
        const mode = args.agent_mode ?? args.agentMode;
        if (!['track_guide', 'overtake', 'live_performance_analyst'].includes(mode)) {
            throw new UnsupportedAgentModeError(
                'Supported agent modes are track_guide, overtake, and live_performance_analyst.',
            );
        }
        if (!isLiveSessionAiAvailable(chat.getRecordingState())) {
            const ErrorType = context.sessionMode === 'recorded'
                ? RecordedSessionLiveToolsUnavailableError
                : NonLiveContextLiveToolsUnavailableError;
            throw new ErrorType(
                AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
                'Agent sessions require an active live recording.',
            );
        }
        const result = await chat.startAgentSession(mode, args);
        return result;
    },
    async stop_agent_session(args) {
        const result = await getChat(context).stopAgentSession(
            normalizeOptionalString(args.agent_session_id ?? args.agentSessionId),
        );
        return result;
    },
    async get_session_analysis(args) {
        const recorded = getRecorded(context);
        return recorded.requestSessionAnalysis(
            args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
        );
    },
    async run_recorded_ai_analysis(args) {
        if (context.sessionMode !== 'recorded') {
            throw new NotRecordedModeError('This tool requires recorded-session mode.');
        }
        const recorded = getRecorded(context);
        const chat = getChat(context);
        const state = await recorded.runRecordedAiAnalysis({ force: args.force === true });
        const compact = compactRecordedAnalysis(recorded, chat, state, getLimit(args.limit));
        if (state.result) {
            const records = recorded.getSelectedSession()?.data ?? [];
            await ensureAnalysisResultsChart(context, buildAnalysisElements(state.result, chat, records));
        }
        return compact;
    },
    async get_recorded_session_analysis(args) {
        if (context.sessionMode !== 'recorded') {
            throw new NotRecordedModeError('This tool requires recorded-session mode.');
        }
        return compactRecordedAnalysis(getRecorded(context), getChat(context), undefined, getLimit(args.limit));
    },
    async get_recorded_session_context(args) {
        if (context.sessionMode !== 'recorded') {
            throw new NotRecordedModeError('This tool requires recorded-session mode.');
        }
        const recorded = getRecorded(context);
        const selected = recorded.getSelectedSession();
        if (!selected?.SessionId) {
            throw new NoRecordedSessionError(
                AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
                'No recorded session is selected.',
            );
        }
        const playback = recorded.getRecordedPlaybackSummary();
        return {
            status: 'ready',
            selected_session: { id: selected.SessionId, name: selected.session_name || null, map: selected.map || recorded.getMapSelected(), car: selected.car || null },
            recorded_telemetry: {
                sample_count: playback.sampleCount,
                duration_seconds: playback.durationSeconds,
                playback_index: playback.playbackIndex,
                playback_time_seconds: playback.playbackTimeSeconds,
                active_segment: playback.activeSegment,
            },
            ai_analysis: compactRecordedAnalysis(recorded, getChat(context), undefined, getLimit(args.limit)),
        };
    },
    async get_performance_insights(args) {
        const recorded = getRecorded(context);
        return recorded.requestPerformanceInsights(
            args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
            args.analysis_type,
        );
    },
    async compare_lap_times(args) {
        return getRecorded(context).requestLapComparison(args.session_ids, args.metrics);
    },
    async query_telemetry_metric(args) {
        const live = requireLive(context);
        const fields = normalizeFields(args.fields);
        if (fields.length === 0) {
            throw new TelemetryFieldsRequiredError('Provide at least one telemetry field.');
        }
        const reduce = ['avg', 'min', 'max', 'stats'].includes(args.reduce) ? args.reduce : 'stats';
        return live.queryTelemetryMetric({ fields, scope: args.scope, reduce });
    },
    async _get_telemetry_for_scope(args) {
        const live = requireLive(context);
        return { rows: live.getTelemetryForScope(args.scope) };
    },
    async get_event_log(args) {
        const live = requireLive(context);
        return { events: live.getEventLog(args) };
    },
    async get_user_summary_map_level(args) {
        return getSummary(context).getUserSummaryMapLevel(args);
    },
    async get_available_user_summary_maps() {
        return getSummary(context).getAvailableUserSummaryMaps();
    },
    async search_user_summary_map_level(args) {
        return getSummary(context).searchUserSummaryMapLevel(args);
    },
    async get_next_corner() {
        const live = requireLive(context);
        const corner = live.getNextCorner();
        if (!corner) throw new NoCornerDataError('No upcoming corner data is available.');
        return corner;
    },
    async get_live_focus_section() {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throw new BaselineCollectionIncompleteError(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                'Complete one clean baseline lap before reading a focus section.',
            );
        }
        const focus = getFocusPayload(live);
        if (!focus) {
            throw new FocusSectionNotReadyError(
                'Analyze the completed baseline and select a focus section before reading it.',
            );
        }
        return { status: 'ready', agent_mode: 'live_performance_analyst', focus };
    },
    async get_live_section_history(args) {
        const live = requireLive(context);
        return { status: 'ready', agent_mode: 'live_performance_analyst', history: live.getLiveSectionHistory(getLimit(args.limit, 20, 80)) };
    },
    async set_live_range_todo_list(args, handlerContext) {
        requireLive(context);
        const todo = new LiveRangeTodoListRunner(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        try {
            todo.addComponentRef(getDirectory(context));
            return createLiveRangeTodoAiAdapter(
                todo,
                createLiveRangeNotificationTaskStartFunctionFactory(handlerContext),
            ).set(args);
        } catch (error) {
            todo.dispose();
            throw error;
        }
    },
    async update_live_range_todo_list(args, handlerContext) {
        requireLive(context);
        const todo = getLiveRangeTodoListRunner(context);
        return createLiveRangeTodoAiAdapter(
            todo,
            createLiveRangeNotificationTaskStartFunctionFactory(handlerContext),
        ).update(args);
    },
    async get_live_range_todo_list() {
        requireLive(context);
        return getLiveRangeTodoListRunner(context).get();
    },
    async collect_live_baseline(_args, handlerContext, output) {
        requireLive(context);
        const { handle } = await openBaselineVisualization(context);
        const chat = getChat(context);
        chat.setLivePerformanceAnalystEnabled(true);
        chat.setAgentTagActive('Live Analyst', true);
        const payload = handle.startCollection(handlerContext.toolRunId);
        return payload.status === 'complete'
            ? output.final(payload)
            : output.progress({ ...payload, status: 'started', message: 'Baseline collection started.' });
    },
    async restart_live_baseline(_args, _handlerContext, output) {
        requireLive(context);
        const { handle } = await openBaselineVisualization(context);
        handle.restartCollection();
        return output.final({ status: 'complete', progress_percent: 0, message: 'Baseline collection restart completed.' });
    },
    async analyze_live_recorded_analysis(args) {
        requireLive(context);
        const baseline = getBaseline(context);
        return baseline.requestAnalysis(args);
    },
    async get_live_analysis_mistake_count() {
        if (
            context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throw new LivePerformanceAnalystToolUnavailableError(
                'This tool is available only to the live performance analyst.',
            );
        }
        const live = requireLive(context);
        return getLiveAnalysisMistakeCount(
            live.getLatestAnalysisResultPage(),
            getChat(context).getLabelName,
        );
    },
    async create_goal(args) {
        if (
            context.sessionMode !== 'live'
            || context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throw new CreateGoalToolUnavailableError(
                'Goal creation is available only to the live performance analyst.',
            );
        }
        const chat = getChat(context);
        const built = buildGoalRequest(
            args,
            (step) => chat.selectGoalTaskStartFunction(step),
        );
        if ('error' in built) {
            throw built.error;
        }
        const goal = new GoalRunner(AI_TOOL_COMPONENT_NAMES.GOAL);
        try {
            goal.addComponentRef(getDirectory(context));
            return await goal.create(built.request);
        } catch (error) {
            if (!goal.getSnapshot()) goal.dispose();
            throw error;
        }
    },
    async retry_goal_task() {
        if (
            context.sessionMode !== 'live'
            || context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throw new RetryGoalTaskToolUnavailableError(
                'Goal task retry is available only to the live performance analyst.',
            );
        }
        return getGoal(context).retryFailedTask();
    },
    async set_procedure_plan(args, handlerContext) {
        const chat = getChat(context);
        const plan = buildProcedurePlan(
            { ...args, event: normalizeOptionalString(args.event) || 'procedure_plan_started' },
            (request) => chat.selectTaskStartFunction(request),
        );
        if (!plan) {
            throw new InvalidProcedurePlanRequestsError(
                'Provide a goal and at least one request with a title.',
            );
        }
        const runner = new ProcedurePlanRunner(
            AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN,
            undefined,
            (request, error) => {
                handlerContext.sendToolStatus({
                    source: 'procedure_plan',
                    event: 'procedure_plan_task_rejected',
                    title: request.title,
                    error: error instanceof Error ? error.message : String(error),
                });
            },
        );
        try {
            runner.addComponentRef(getDirectory(context));
            runner.replace(plan);
        } catch (error) {
            runner.dispose();
            throw error;
        }
        return { status: 'ready', goal: plan.goal, request_count: plan.requests.length, current_request: plan.currentStep, request: plan.requests[plan.currentStep] };
    },
    async advance_plan_step(args) {
        return getProcedurePlanRunner(context).advance(normalizeOptionalString(args.reason));
    },
    async clear_procedure_plan(args) {
        getProcedurePlanRunner(context).clear();
        return { status: 'cleared', reason: normalizeOptionalString(args.reason) };
    },
    async _get_live_section_telemetry(args) {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throw new BaselineCollectionIncompleteError(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                'Complete baseline collection before reading live-section telemetry.',
            );
        }
        return live.getLiveSectionTelemetry(args);
    },
    async _record_live_section_classification(args) {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throw new BaselineCollectionIncompleteError(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                'Complete baseline collection before recording a live-section classification.',
            );
        }
        const classification = live.recordLiveSectionClassification(args);
        if (!classification) {
            throw new SectionNotFoundError('The requested live section was not found.');
        }
        const focus = getFocusPayload(live);
        return { status: 'recorded', agent_mode: 'live_performance_analyst', classification, focus, comparison: live.getSessionIntelligence().compareFocusedSection(classification) };
    },
    async follow_expert_line(args) {
        const recorded = getRecorded(context);
        return recorded.requestExpertLineGuidance(
            args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
            args.data_types,
        );
    },
    async get_telemetry_data(args) {
        const recorded = getRecorded(context);
        return recorded.requestTelemetryData(
            args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
            args.data_types,
        );
    },
    async get_visualization_capabilities() {
        return getManager(context).getVisualizationCapabilities();
    },
    async show_map(args) {
        const chat = getChat(context);
        let resolved: Awaited<ReturnType<typeof resolveMap>>;
        try {
            resolved = await resolveMap(chat, args, context);
        } catch (error) {
            throw new CircuitMapLookupFailedError(
                getFailureMessage(error, 'Failed to look up the requested circuit map.'),
                { cause: error },
            );
        }
        const { map, resolvedBy, requestedMap } = resolved;
        const section = getMapSection(args);
        const title = normalizeOptionalString(args.title) || 'Map';
        const note = normalizeOptionalString(args.message ?? args.note);
        if (!map) {
            const reason = requestedMap ? `No circuit map is available for "${requestedMap}".` : 'No circuit map is available for the current session.';
            const display: AiMapDisplayPayload = { status: 'unavailable', requestedMap, title, note, reason, section };
            chat.displayMap(display);
            return { status: 'unavailable', message: 'Map is not available', requested_map: requestedMap ?? null, reason };
        }
        chat.displayMap({ status: 'ready', map, requestedMap, title, note, section });
        return { status: 'displayed', map_id: map.id, circuit_name: map.circuit_name, source_track_key: map.source_track_key ?? null, resolved_by: resolvedBy, section: section ?? null };
    },
    async open_visualization_chart(args) {
        return openVisualization(context, args);
    },
    async close_visualization_chart(args) {
        const manager = getManager(context);
        const target = getVisualizationTarget(manager, args);
        const result = manager.closeVisualization({
            name: target.name,
            type: args.type,
            all: args.all === true,
        });
        return result;
    },
    async invoke_visualization_control(args) {
        return invokeVisualizationControl(context, args);
    },
    async update_guidance_once(args) {
        return invokeVisualizationControl(context, { ...args, type: args.type || 'imitation-guidance-chart', control: 'refresh_once' });
    },
    async add_imitation_guidance_chart(args) {
        return openVisualization(context, {
            ...args,
            type: 'imitation-guidance-chart',
            data: { sessionId: args.session_id || context.sessionId, manuallyAdded: true },
            config: { title: args.title || 'AI Driving Guidance', autoUpdate: args.autoUpdate !== false },
        });
    },
    async remove_imitation_guidance_chart(args) {
        const result = getManager(context).closeVisualization({
            name: args.component_name,
            id: args.chartId,
            type: 'imitation-guidance-chart',
            all: !args.chartId && !args.component_name,
        });
        return result;
    },
    async disable_ui_component(args) {
        const manager = getManager(context);
        const target = getVisualizationTarget(manager, args);
        const child = resolveNamedComponentHandle<any>(getDirectory(context), target.name!);
        const type = manager.getCurrentVisualizations().find((instance) => instance.name === target.name)?.type;
        let success = false;
        if (type === 'telemetry-overview') success = 'disableLiveTelemetry' in child ? child.disableLiveTelemetry() : child.disableTelemetry();
        else if (type === 'event-log') success = 'disableLiveEventLog' in child ? child.disableLiveEventLog() : child.disableEventLog();
        else if (type === 'analysis-results') success = child.disableAnalysisResults();
        else if (type === 'imitation-guidance-chart') success = child.disableGuidance();
        else if (type === 'map-visualization') success = child.disableMap();
        if (!success) {
            throw new ComponentDisableFailedError(
                target.name,
                `Component '${target.name}' could not be disabled.`,
            );
        }
        return { success, component_name: target.name };
    },
    async analyze_telemetry(args) {
        const chat = getChat(context);
        if (context.sessionMode === 'recorded') {
            const recorded = getRecorded(context);
            const state = await recorded.runRecordedAiAnalysis({ force: args.force === true });
            if (!state.result) return compactRecordedAnalysis(recorded, chat, state, getLimit(args.limit));
            const records = recorded.getSelectedSession()?.data ?? [];
            const chart = await ensureAnalysisResultsChart(context, buildAnalysisElements(state.result, chat, records));
            return { ...compactTelemetryAnalysis(state.result, chat, getLimit(args.limit)), chartId: chart.chartId ?? null, component_name: chart.componentName };
        }
        const live = requireLive(context);
        const rows = live.getTelemetryForScope(args.scope);
        if (rows.length === 0) {
            throw new NoTelemetryForScopeError(
                'No telemetry rows matched the requested scope.',
            );
        }
        try {
            const snapshot = live.getLiveSessionSnapshot();
            const response = await apiService.post('/racing-session/analyze-live-recorded-analysis', {
                track: snapshot.track,
                car: snapshot.car,
                baseline_lap: snapshot.current_lap,
                records: rows,
            }, { timeout: 120000 });
            const result = normalizeSegmentClassificationResult(response.data as any, `live-scope-${Date.now()}`);
            const chart = await ensureAnalysisResultsChart(context, buildAnalysisElements(result, chat, rows));
            return { ...compactTelemetryAnalysis(result, chat, getLimit(args.limit)), telemetry_stats: getTelemetryStats(rows), chartId: chart.chartId ?? null, component_name: chart.componentName };
        } catch (error: any) {
            if (error instanceof AiToolError) throw error;
            throw new TelemetryAnalysisFailedError(
                getFailureMessage(error, 'Failed to analyze telemetry.'),
                { cause: error },
            );
        }
    },
    async classify_live_section(args) {
        const live = requireLive(context);
        const chat = getChat(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throw new BaselineCollectionIncompleteError(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                'Complete baseline collection before classifying a live section.',
            );
        }
        const window = live.getLiveSectionTelemetry(args);
        if (window.status !== 'ready' || !Array.isArray(window.rows) || window.rows.length === 0) {
            throw new LiveSectionTelemetryUnavailableError(
                'No telemetry is available for the selected live section.',
            );
        }
        try {
            const snapshot = live.getLiveSessionSnapshot();
            const response = await apiService.post('/racing-session/analyze-live-recorded-analysis', {
                track: snapshot.track,
                car: snapshot.car,
                baseline_lap: window.lap,
                records: window.rows,
            }, { timeout: 120000 });
            const result = normalizeSegmentClassificationResult(response.data as any, `live-section-${window.section?.id || 'unknown'}-${window.lap}`);
            const labels = result.segments.flatMap(getSegmentLabelIds);
            const mistakeLabels = labels.filter((label) => label === 'MSP' || label === 'MSR');
            const classification = live.recordLiveSectionClassification({
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
            const elements = buildAnalysisElements(result, chat, window.rows);
            const chart = await ensureAnalysisResultsChart(context, elements);
            return {
                status: 'recorded',
                classification,
                focus: getFocusPayload(live),
                comparison: live.getSessionIntelligence().compareFocusedSection(classification),
                analysis: compactTelemetryAnalysis(result, chat, getLimit(args.limit)).analysis,
                telemetry_stats: getTelemetryStats(window.rows),
                chartId: chart.chartId ?? null,
                component_name: chart.componentName,
            };
        } catch (error: any) {
            if (error instanceof AiToolError) throw error;
            throw new LiveSectionClassificationFailedError(
                getFailureMessage(error, 'Failed to classify the live section.'),
                { cause: error },
            );
        }
    },
});

export const createRefBasedAiCommandFunctions = (
    context: RefAiCommandContext,
): Record<string, RefAiCommandHandler> => {
    const handlers = createHandlers(context);
    return { ...handlers };
};
