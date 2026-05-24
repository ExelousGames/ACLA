import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';
import { ToolHandlerContext } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

export interface AiCommandRegistryContext {
    sessionId?: string;
    analysisContext?: any;
    // Populated during live recording. Null in post-session analysis view.
    sessionIntelligence?: SessionIntelligence | null;
    startTrackGuide: () => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
}

type AiCommandHandler = (args: Record<string, any>, ctx: ToolHandlerContext) => Promise<any>;

const getSessionId = (args: Record<string, any>, context: AiCommandRegistryContext): string | undefined =>
    args.session_id ||
    context.sessionId ||
    context.analysisContext?.sessionSelected?.SessionId;

export const createAiCommandRegistry = (context: AiCommandRegistryContext): Record<string, AiCommandHandler> => ({

    // ── Session ───────────────────────────────────────────────────────────────

    async get_session_info() {
        return {
            track:   context.analysisContext?.recordedSessioStaticsData?.track   || '',
            car:     context.analysisContext?.recordedSessioStaticsData?.car     || '',
            user_id: context.sessionId || '',
        };
    },

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

    async get_telemetry_schema() {
        const si = context.sessionIntelligence;
        if (!si) return { error: 'no_live_session' };
        return si.getSchema();
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
