import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { ToolHandlerContext, FrontendToolSchema } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { FIELD_GROUPS } from 'views/lap-analysis/session-intelligence/telemetry-query';
import { getOpportunityForecast } from 'services/opportunityForecastService';

export interface AiCommandRegistryContext {
    sessionId?: string;
    analysisContext?: any;
    // Populated during live recording. Null in post-session analysis view.
    sessionIntelligence?: SessionIntelligence | null;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
    getOpportunityTelemetryRows: () => Record<string, any>[];
}

type AiCommandHandler = (args: Record<string, any>, ctx: ToolHandlerContext) => Promise<any>;

// Single source of truth for the frontend-implemented tool surface exposed
// to the voice LLM. Sent to the AI service over the WS on session start
// (see use-voice-conversation.ts) so the backend doesn't carry a duplicate
// copy. Server-implemented tools (analyze_telemetry, explain_label) stay
// in Python.
//
// `title` is the human-readable label the chat UI renders in the "tool box"
// while a call is in flight.
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

const _FIELD_GROUP_NAMES = Object.keys(FIELD_GROUPS).join(', ');
const DEFAULT_OPPORTUNITY_HORIZON_SECONDS = 10;
const DEFAULT_OPPORTUNITY_WHEN_HORIZONS_SECONDS = [6, 10, 15, 20, 30];
const DEFAULT_OPPORTUNITY_TOP_K = 3;

const toPositiveNumber = (value: unknown): number | undefined => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
};

const getOpportunityHorizons = (args: Record<string, any>): number[] => {
    const explicit = Array.isArray(args.horizon_seconds_options)
        ? args.horizon_seconds_options
            .map(toPositiveNumber)
            .filter((value): value is number => value !== undefined)
        : [];

    if (explicit.length > 0) {
        return Array.from(new Set(explicit)).slice(0, 5);
    }

    const single = toPositiveNumber(args.horizon_seconds);
    if (single !== undefined) {
        return [single];
    }

    return args.question_type === 'when'
        ? DEFAULT_OPPORTUNITY_WHEN_HORIZONS_SECONDS
        : [DEFAULT_OPPORTUNITY_HORIZON_SECONDS];
};

const getOpportunityTopK = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0
        ? Math.min(parsed, 5)
        : DEFAULT_OPPORTUNITY_TOP_K;
};

const summarizeOpportunityForecast = (forecast: any): string => {
    if (forecast.model_status === 'not_trained') {
        return `next ${forecast.horizon_seconds}s: model not trained`;
    }

    const opportunities = Array.isArray(forecast.opportunities) ? forecast.opportunities : [];
    if (opportunities.length === 0) {
        return `next ${forecast.horizon_seconds}s: no strong opportunity`;
    }

    return `next ${forecast.horizon_seconds}s: ${opportunities.map((item: any) => {
        const percent = Math.round(Number(item.probability || 0) * 100);
        const section = item.circuit_section_name ? ` at ${item.circuit_section_name}` : '';
        return `${item.label_name || item.label_id} ${percent}%${section}`;
    }).join(' | ')}`;
};

export const frontendToolSchemas: FrontendToolSchema[] = [
    {
        name: 'start_per_turn_coaching',
        title: 'Starting per-turn coaching',
        description:
            "Activate background per-corner coaching. Observations arrive as " +
            "'[OBSERVATION]' user turns. Use when driver asks to be coached every corner.",
        properties: {},
        required: [],
    },
    {
        name: 'stop_per_turn_coaching',
        title: 'Stopping per-turn coaching',
        description: 'Stop per-corner coaching. Use when driver asks to be left alone.',
        properties: {},
        required: [],
    },
    {
        name: 'opportunity_forecast',
        title: 'Checking opportunity',
        description:
            'Run exactly one live opportunity forecast and return upcoming overtake or defense opportunities. ' +
            'Use when the driver asks if they can overtake, defend, pass, attack, or find an opportunity. ' +
            'For "can I overtake in the next 6 seconds", pass horizon_seconds=6 and question_type=next_seconds. ' +
            'For "any opportunity in the next corner", pass question_type=next_corner. ' +
            'For "when can I overtake", pass question_type=when so several horizons are checked in this one call. ' +
            'Do not use this to start background monitoring.',
        properties: {
            question_type: {
                type: 'string',
                enum: ['next_seconds', 'next_corner', 'when', 'general'],
                description:
                    'Use next_seconds for a specified time, next_corner for the upcoming corner, and when for timing questions.',
            },
            horizon_seconds: {
                type: 'number',
                description:
                    'Single forecast horizon in seconds. Use the driver-specified time, e.g. 6 for "next 6 seconds". Default 10.',
            },
            horizon_seconds_options: {
                type: 'array',
                items: { type: 'number' },
                description:
                    'Multiple horizons to check in one tool call. Use for "when can I overtake"; default is 6, 10, 15, 20, 30.',
            },
            top_k: {
                type: 'integer',
                description: 'Maximum number of opportunity labels to return. Default 3.',
            },
        },
        required: [],
    },
    {
        name: 'get_next_corner',
        title: 'Looking up next corner',
        description:
            'Name and normalized distance of the next corner ahead. Use with opportunity_forecast when the driver asks about the next corner.',
        properties: {},
        required: [],
    },
    {
        name: 'query_telemetry_metric',
        title: 'Querying telemetry',
        description: 'Read a telemetry metric over a scope.',
        properties: {
            fields: {
                type: 'array',
                items: { type: 'string' },
                description:
                    'Field group names (preferred) or raw Physics_* names. ' +
                    `Available groups: ${_FIELD_GROUP_NAMES}.`,
            },
            scope: QUERY_SCOPE_SCHEMA,
            reduce: {
                type: 'string',
                enum: ['avg', 'min', 'max', 'stats'],
                description: 'stats = {avg,min,max,stddev}.',
            },
        },
        required: ['fields', 'scope', 'reduce'],
    },
    {
        name: 'get_event_log',
        title: 'Searching event log',
        description:
            'List racing events with their sample-index ranges. Use to find when ' +
            'something happened before querying telemetry around it.',
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
                description: 'For last_n: how many events.',
            },
        },
        required: ['eventType', 'scope'],
    },
];

const getSessionId = (args: Record<string, any>, context: AiCommandRegistryContext): string | undefined =>
    args.session_id ||
    context.sessionId ||
    context.analysisContext?.sessionSelected?.SessionId;

export const createAiCommandRegistry = (context: AiCommandRegistryContext): Record<string, AiCommandHandler> => ({

    // ── Session ───────────────────────────────────────────────────────────────

    async get_session_analysis(args) {
        return await apiService.post('/racing-session/detailed-info', { id: getSessionId(args, context) });
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

    async query_telemetry(args) {
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return si.query(args as any);
    },

    // Constrained-reduce variant exposed to the LLM. The schema enforces
    // reduce ∈ {avg,min,max,stats}; we defensively swap any other value
    // (incl. legacy 'raw') for 'stats' so a stale prompt can't leak rows.
    async query_telemetry_metric(args) {
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
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return { rows: si.getRowsForScope(args.scope) };
    },

    // ── Event log ─────────────────────────────────────────────────────────────

    async get_event_log(args) {
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return { events: si.findEvents(args as any) };
    },

    async get_next_corner() {
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return si.getNextCorner() ?? { error: 'no_corner_data' };
    },

    // ── Coaching ──────────────────────────────────────────────────────────────

    async start_per_turn_coaching() {
        return { status: 'not_yet_implemented' };
    },

    async stop_per_turn_coaching() {
        return { status: 'stopped' };
    },

    async opportunity_forecast(args) {
        const telemetryRows = context.getOpportunityTelemetryRows();
        if (telemetryRows.length === 0) {
            return { error: 'no_live_telemetry' };
        }

        const questionType = ['next_seconds', 'next_corner', 'when'].includes(args.question_type)
            ? args.question_type
            : 'general';
        const horizonsSeconds = getOpportunityHorizons(args);
        const topK = getOpportunityTopK(args.top_k);
        const nextCorner = questionType === 'next_corner'
            ? context.sessionIntelligence?.getNextCorner() ?? null
            : null;
        const forecasts = await Promise.all(horizonsSeconds.map((horizonSeconds) =>
            getOpportunityForecast({
                telemetry_data: telemetryRows,
                horizon_seconds: horizonSeconds,
                top_k: topK,
            })
        ));
        const firstForecast = forecasts[0];

        return {
            ...firstForecast,
            request: {
                question_type: questionType,
                horizons_seconds: horizonsSeconds,
                top_k: topK,
            },
            forecasts,
            summaries: forecasts.map(summarizeOpportunityForecast),
            next_corner: nextCorner,
            telemetry_rows: telemetryRows.length,
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
        context.startTrackGuide();
        return { status: 'guidance_enabled', enabled: true };
    },

    async disable_guide_user_racing() {
        context.setTrackGuideEnabled(false);
        return { status: 'guidance_disabled', enabled: false };
    },

    async get_visualization_capabilities() {
        return visualizationController.getVisualizationAssistantContext();
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
});
