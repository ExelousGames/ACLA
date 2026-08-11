import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    AiToolComponentRefError,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
} from 'contexts/AiToolComponentRefContext';
import type { ToolHandlerContext } from './use-voice-conversation';
import {
    AiToolError,
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
    type GoalHandle,
    type LiveRangeTodoListHandle,
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

function componentError(error: unknown): never {
    if (error instanceof AiToolComponentRefError) {
        throw new AiToolError(error.code, error.message, {
            details: { component_name: error.componentName },
            cause: error,
        });
    }
    throw error;
}

function throwToolError(
    code: string,
    message: string,
    details?: Record<string, unknown>,
    cause?: unknown,
): never {
    throw new AiToolError(code, message, { details, cause });
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getResultMessage = (value: unknown, fallback: string): string => (
    isRecord(value) && typeof value.message === 'string' && value.message.trim()
        ? value.message
        : fallback
);

const getResultCode = (value: unknown, fallback: string): string => (
    isRecord(value) && typeof value.error === 'string' && value.error.trim()
        ? value.error
        : fallback
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

const executeApiRequest = async <T,>(
    request: () => Promise<T>,
    code: string,
    fallbackMessage: string,
): Promise<T> => {
    try {
        return await request();
    } catch (error) {
        if (error instanceof AiToolError) throw error;
        throwToolError(code, getFailureMessage(error, fallbackMessage), undefined, error);
    }
};

const throwErrorResult = (
    value: unknown,
    fallbackCode: string,
    fallbackMessage: string,
): never => throwToolError(
    getResultCode(value, fallbackCode),
    getResultMessage(value, fallbackMessage),
);

const returnOrThrowErrorResult = <T,>(
    value: T,
    fallbackCode: string,
    fallbackMessage: string,
): T => {
    if (isRecord(value) && (
        value.status === 'error'
        || value.status === 'invalid_query'
        || typeof value.error === 'string'
    )) {
        throwErrorResult(value, fallbackCode, fallbackMessage);
    }
    return value;
};

const getDirectory = (context: RefAiCommandContext): AiToolComponentRefDirectory => {
    if (context.componentRefs) return context.componentRefs;
    throw new AiToolComponentRefError(
        'component_ref_unavailable',
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
        throwToolError(
            'baseline_collection_visualization_required',
            'Baseline collection visualization required. Open Baseline Collection in Live Session and keep it open.',
            { component_name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION },
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
    if (!requested.success) return { requested, handle: null };

    const handle = await awaitNamedComponentHandle<BaselineCollectionHandle>(
        getDirectory(context),
        AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
    );
    return { requested, handle };
};

const requireLive = (context: RefAiCommandContext): LiveSessionHandle => {
    const code = context.sessionMode === 'recorded'
        ? 'recorded_session_live_tools_unavailable'
        : 'non_live_context_live_tools_unavailable';
    if (context.sessionMode && context.sessionMode !== 'live') {
        throwToolError(code, 'Live-session AI tools are unavailable in the current session mode.');
    }
    const live = getLive(context);
    if (!isLiveSessionAiAvailable(live.getRecordingState())) {
        throwToolError(code, 'Live-session AI tools require an active live recording.');
    }
    return live;
};

const normalizeOptionalString = (value: unknown): string | undefined => (
    typeof value === 'string' && value.trim() ? value.trim() : undefined
);

const getGoal = (context: RefAiCommandContext) => resolveNamedComponentHandle<GoalHandle>(
    getDirectory(context),
    AI_TOOL_COMPONENT_NAMES.GOAL,
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
        throwToolError('no_recorded_session', 'No recorded session is selected.');
    }
    if (state.status === 'error') {
        throwToolError(
            'recorded_analysis_failed',
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
    if (!requested.success) {
        throwErrorResult(
            requested,
            'analysis_results_visualization_unavailable',
            'The analysis-results visualization could not be opened.',
        );
    }
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
            throwToolError(
                'component_ref_unavailable',
                `Chart '${byId}' is not open.`,
                { component_name: byId },
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
        throwToolError(
            'ambiguous_component_target',
            'Choose a telemetry metric family and retry the same tool.',
            { semantic_candidates: candidates },
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
    if (!requested.success) {
        throwErrorResult(
            requested,
            'visualization_request_failed',
            `Could not open chart '${target.name}'.`,
        );
    }
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
        throwToolError(
            'visualization_control_unavailable',
            `Control '${String(control)}' is not available for chart '${name}'.`,
            { component_name: name, chart_type: type, control },
        );
    }
    if (result.success === false) {
        throwErrorResult(
            result,
            'visualization_control_failed',
            `Control '${String(control)}' failed for chart '${name}'.`,
        );
    }
    return {
        success: result.success !== false,
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
            throwToolError(
                'unsupported_agent_mode',
                'Supported agent modes are track_guide, overtake, and live_performance_analyst.',
            );
        }
        if (!isLiveSessionAiAvailable(chat.getRecordingState())) {
            throwToolError(
                context.sessionMode === 'recorded'
                    ? 'recorded_session_live_tools_unavailable'
                    : 'non_live_context_live_tools_unavailable',
                'Agent sessions require an active live recording.',
            );
        }
        const result = await chat.startAgentSession(mode, args);
        return returnOrThrowErrorResult(
            result,
            'agent_session_start_failed',
            'The agent session could not be started.',
        );
    },
    async stop_agent_session(args) {
        const result = await getChat(context).stopAgentSession(
            normalizeOptionalString(args.agent_session_id ?? args.agentSessionId),
        );
        return returnOrThrowErrorResult(
            result,
            'agent_session_stop_failed',
            'The agent session could not be stopped.',
        );
    },
    async get_session_analysis(args) {
        const recorded = getRecorded(context);
        return executeApiRequest(
            () => recorded.requestSessionAnalysis(
                args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
            ),
            'session_analysis_failed',
            'Failed to load the session analysis.',
        );
    },
    async run_recorded_ai_analysis(args) {
        if (context.sessionMode !== 'recorded') {
            throwToolError('not_recorded_mode', 'This tool requires recorded-session mode.');
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
            throwToolError('not_recorded_mode', 'This tool requires recorded-session mode.');
        }
        return compactRecordedAnalysis(getRecorded(context), getChat(context), undefined, getLimit(args.limit));
    },
    async get_recorded_session_context(args) {
        if (context.sessionMode !== 'recorded') {
            throwToolError('not_recorded_mode', 'This tool requires recorded-session mode.');
        }
        const recorded = getRecorded(context);
        const selected = recorded.getSelectedSession();
        if (!selected?.SessionId) {
            throwToolError('no_recorded_session', 'No recorded session is selected.');
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
        return executeApiRequest(
            () => recorded.requestPerformanceInsights(
                args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
                args.analysis_type,
            ),
            'performance_insights_failed',
            'Failed to load performance insights.',
        );
    },
    async compare_lap_times(args) {
        return executeApiRequest(
            () => getRecorded(context).requestLapComparison(args.session_ids, args.metrics),
            'lap_comparison_failed',
            'Failed to compare lap times.',
        );
    },
    async query_telemetry_metric(args) {
        const live = requireLive(context);
        const fields = normalizeFields(args.fields);
        if (fields.length === 0) {
            throwToolError('telemetry_fields_required', 'Provide at least one telemetry field.');
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
        return returnOrThrowErrorResult(
            getSummary(context).getUserSummaryMapLevel(args),
            'user_summary_unavailable',
            'The user summary is unavailable.',
        );
    },
    async get_available_user_summary_maps() {
        return returnOrThrowErrorResult(
            getSummary(context).getAvailableUserSummaryMaps(),
            'user_summary_unavailable',
            'The user summary is unavailable.',
        );
    },
    async search_user_summary_map_level(args) {
        return returnOrThrowErrorResult(
            getSummary(context).searchUserSummaryMapLevel(args),
            'user_summary_search_failed',
            'The user-summary search could not be completed.',
        );
    },
    async get_next_corner() {
        const live = requireLive(context);
        const corner = live.getNextCorner();
        if (!corner) throwToolError('no_corner_data', 'No upcoming corner data is available.');
        return corner;
    },
    async get_live_focus_section() {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throwToolError(
                'baseline_collection_incomplete',
                'Complete one clean baseline lap before reading a focus section.',
            );
        }
        const focus = getFocusPayload(live);
        if (!focus) {
            throwToolError(
                'focus_section_not_ready',
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
        const todo = resolveNamedComponentHandle<LiveRangeTodoListHandle>(getDirectory(context), AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        return createLiveRangeTodoAiAdapter(
            todo,
            createLiveRangeNotificationTaskStartFunctionFactory(handlerContext),
        ).set(args);
    },
    async update_live_range_todo_list(args, handlerContext) {
        requireLive(context);
        const todo = resolveNamedComponentHandle<LiveRangeTodoListHandle>(getDirectory(context), AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        return createLiveRangeTodoAiAdapter(
            todo,
            createLiveRangeNotificationTaskStartFunctionFactory(handlerContext),
        ).update(args);
    },
    async get_live_range_todo_list() {
        requireLive(context);
        const result = resolveNamedComponentHandle<LiveRangeTodoListHandle>(
            getDirectory(context),
            AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        ).get();
        return returnOrThrowErrorResult(
            result,
            'live_range_todo_list_unavailable',
            'The live range to-do list is unavailable.',
        );
    },
    async collect_live_baseline(_args, handlerContext, output) {
        requireLive(context);
        const { requested, handle } = await openBaselineVisualization(context);
        if (!handle) {
            throwToolError(
                'baseline_collection_visualization_unavailable',
                requested.message || 'The baseline collection visualization is unavailable.',
            );
        }
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
        const { requested, handle } = await openBaselineVisualization(context);
        if (!handle) {
            throwToolError(
                'baseline_collection_visualization_unavailable',
                requested.message || 'The baseline collection visualization is unavailable.',
            );
        }
        handle.restartCollection();
        return output.final({ status: 'complete', progress_percent: 0, message: 'Baseline collection restart completed.' });
    },
    async analyze_live_recorded_analysis(args) {
        requireLive(context);
        const baseline = getBaseline(context);
        const result = await baseline.requestAnalysis(args);
        return returnOrThrowErrorResult(
            result,
            'recorded_analysis_failed',
            'The live baseline analysis could not be completed.',
        );
    },
    async get_live_analysis_mistake_count() {
        if (
            context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throwToolError(
                'live_performance_analyst_tool_unavailable',
                'This tool is available only to the live performance analyst.',
            );
        }
        const live = requireLive(context);
        const result = getLiveAnalysisMistakeCount(
            live.getLatestAnalysisResultPage(),
            getChat(context).getLabelName,
        );
        return returnOrThrowErrorResult(
            result,
            'live_analysis_result_unavailable',
            'No completed live analysis result is available.',
        );
    },
    async create_goal(args) {
        if (
            context.sessionMode !== 'live'
            || context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throwToolError(
                'create_goal_tool_unavailable',
                'Goal creation is available only to the live performance analyst.',
            );
        }
        const chat = getChat(context);
        const built = buildGoalRequest(
            args,
            (step) => chat.selectGoalTaskStartFunction(step),
        );
        if ('error' in built) {
            throwToolError(built.error, 'The goal request is invalid.');
        }
        const result = await getGoal(context).createGoal(built.request);
        return returnOrThrowErrorResult(
            result,
            'goal_execution_failed',
            'The goal could not be completed.',
        );
    },
    async retry_goal_task() {
        if (
            context.sessionMode !== 'live'
            || context.conversationRole !== 'agent'
            || context.agentMode !== 'live_performance_analyst'
        ) {
            throwToolError(
                'retry_goal_task_tool_unavailable',
                'Goal task retry is available only to the live performance analyst.',
            );
        }
        const result = await getGoal(context).retryFailedTask();
        return returnOrThrowErrorResult(
            result,
            'goal_task_retry_failed',
            'The failed goal task could not be retried.',
        );
    },
    async set_procedure_plan(args) {
        const chat = getChat(context);
        const plan = buildProcedurePlan(
            { ...args, event: normalizeOptionalString(args.event) || 'procedure_plan_started' },
            (request) => chat.selectTaskStartFunction(request),
        );
        if (!plan) {
            throwToolError(
                'invalid_procedure_plan_requests',
                'Provide a goal and at least one request with a title.',
            );
        }
        chat.setProcedurePlan(plan);
        return { status: 'ready', goal: plan.goal, request_count: plan.requests.length, current_request: plan.currentStep, request: plan.requests[plan.currentStep] };
    },
    async advance_plan_step(args) {
        const result = getChat(context).advanceProcedurePlanStep(
            normalizeOptionalString(args.reason),
        );
        return returnOrThrowErrorResult(
            result,
            'procedure_plan_advance_failed',
            'The procedure plan could not be advanced.',
        );
    },
    async clear_procedure_plan(args) {
        getChat(context).clearProcedurePlan();
        return { status: 'cleared', reason: normalizeOptionalString(args.reason) };
    },
    async _get_live_section_telemetry(args) {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throwToolError(
                'baseline_collection_incomplete',
                'Complete baseline collection before reading live-section telemetry.',
            );
        }
        return live.getLiveSectionTelemetry(args);
    },
    async _record_live_section_classification(args) {
        const live = requireLive(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throwToolError(
                'baseline_collection_incomplete',
                'Complete baseline collection before recording a live-section classification.',
            );
        }
        const classification = live.recordLiveSectionClassification(args);
        if (!classification) {
            throwToolError('section_not_found', 'The requested live section was not found.');
        }
        const focus = getFocusPayload(live);
        return { status: 'recorded', agent_mode: 'live_performance_analyst', classification, focus, comparison: live.getSessionIntelligence().compareFocusedSection(classification) };
    },
    async follow_expert_line(args) {
        const recorded = getRecorded(context);
        return executeApiRequest(
            () => recorded.requestExpertLineGuidance(
                args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
                args.data_types,
            ),
            'expert_line_guidance_failed',
            'Failed to load expert-line guidance.',
        );
    },
    async get_telemetry_data(args) {
        const recorded = getRecorded(context);
        return executeApiRequest(
            () => recorded.requestTelemetryData(
                args.session_id || context.sessionId || recorded.getSelectedSession()?.SessionId,
                args.data_types,
            ),
            'telemetry_data_failed',
            'Failed to load telemetry data.',
        );
    },
    async get_visualization_capabilities() {
        return getManager(context).getVisualizationCapabilities();
    },
    async show_map(args) {
        const chat = getChat(context);
        const { map, resolvedBy, requestedMap } = await executeApiRequest(
            () => resolveMap(chat, args, context),
            'circuit_map_lookup_failed',
            'Failed to look up the requested circuit map.',
        );
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
        if (isRecord(result) && result.success === false) {
            throwErrorResult(
                result,
                'visualization_close_failed',
                `Could not close chart '${target.name}'.`,
            );
        }
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
        if (isRecord(result) && result.success === false) {
            throwErrorResult(
                result,
                'visualization_close_failed',
                'Could not close the imitation-guidance chart.',
            );
        }
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
            throwToolError(
                'component_disable_failed',
                `Component '${target.name}' could not be disabled.`,
                { component_name: target.name },
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
            throwToolError(
                'no_telemetry_for_scope',
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
            throwToolError(
                'telemetry_analysis_failed',
                getFailureMessage(error, 'Failed to analyze telemetry.'),
                undefined,
                error,
            );
        }
    },
    async classify_live_section(args) {
        const live = requireLive(context);
        const chat = getChat(context);
        const baseline = getBaseline(context);
        if (!baseline.getLapRecord()?.records?.length) {
            throwToolError(
                'baseline_collection_incomplete',
                'Complete baseline collection before classifying a live section.',
            );
        }
        const window = live.getLiveSectionTelemetry(args);
        if (window.status !== 'ready' || !Array.isArray(window.rows) || window.rows.length === 0) {
            throwToolError(
                window.status || 'live_section_telemetry_unavailable',
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
                throwToolError('section_not_found', 'The requested live section was not found.');
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
            throwToolError(
                'live_section_classification_failed',
                getFailureMessage(error, 'Failed to classify the live section.'),
                undefined,
                error,
            );
        }
    },
});

export const createRefBasedAiCommandFunctions = (
    context: RefAiCommandContext,
): Record<string, RefAiCommandHandler> => {
    const handlers = createHandlers(context);
    return Object.fromEntries(Object.entries(handlers).map(([name, handler]) => [
        name,
        async (args: Record<string, any>, handlerContext: ToolHandlerContext, output: ToolOutputController) => {
            try {
                return await handler(args, handlerContext, output);
            } catch (error) {
                return componentError(error);
            }
        },
    ]));
};
