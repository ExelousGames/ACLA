import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { ToolHandlerContext, FrontendToolSchema } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { FIELD_GROUPS } from 'views/lap-analysis/session-intelligence/telemetry-query';
import { detectOvertakeTacticalState } from './overtake-agent-detector';

export interface AiCommandRegistryContext {
    sessionId?: string;
    analysisContext?: any;
    // Populated during live recording. Null in post-session analysis view.
    sessionIntelligence?: SessionIntelligence | null;
    opportunityAgentState: OpportunityAgentState;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
    setAgentTagActive?: (tag: string, active: boolean) => void;
    getOpportunityTelemetryRows: () => Record<string, any>[];
}

export interface OpportunityAgentState {
    intervalId: ReturnType<typeof setInterval> | null;
    inFlight: boolean;
    lastAlertKey: string | null;
    lastAlertAt: number;
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
const DEFAULT_OVERTAKE_AGENT_INTERVAL_SECONDS = 5;
const OVERTAKE_AGENT_MIN_INTERVAL_SECONDS = 2;
const OVERTAKE_AGENT_MAX_INTERVAL_SECONDS = 15;
const OVERTAKE_AGENT_REPEAT_ALERT_MS = 20000;

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

export const frontendToolSchemas: FrontendToolSchema[] = [
    {
        name: 'start_per_turn_coaching',
        title: 'Starting track guide agent',
        description:
            'Activate the background track guide agent. Use when the driver asks for AI guiding, ' +
            'track guidance, or corner-by-corner coaching during a practice session.',
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
        name: 'start_overtake_agent',
        title: 'Starting overtake agent',
        description:
            'Open continuous overtake agent mode. Use only when the driver explicitly asks to open, enable, watch, ' +
            'monitor, or plan attack/defense overtake agent mode. Do not use for one-off questions like "when can I overtake". ' +
            'The agent uses live car coordinates to detect attack windows and defense threats until stopped.',
        properties: {
            interval_seconds: {
                type: 'number',
                description: 'How often to check while agent mode is active. Default 5; clamped to 2-15.',
            },
        },
        required: [],
    },
    {
        name: 'stop_overtake_agent',
        title: 'Stopping overtake agent',
        description:
            'Stop background overtake planning. Use when the driver asks you to stop watching for passes or leave them alone.',
        properties: {},
        required: [],
    },
    {
        name: 'get_next_corner',
        title: 'Looking up next corner',
        description:
            'Name and normalized distance of the next corner ahead.',
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
