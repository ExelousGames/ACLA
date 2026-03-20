import apiService from 'services/api.service';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';

export interface AiCommandRegistryContext {
    sessionId?: string;
    analysisContext?: any;
    startTrackGuide: (responseData: any) => void;
    setTrackGuideEnabled: (enabled: boolean) => void;
}

export const VISUALIZATION_COMMAND_FUNCTIONS = [
    'get_visualization_capabilities',
    'open_visualization_chart',
    'close_visualization_chart',
    'invoke_visualization_control',
    'update_guidance_once'
];

type AiCommandHandler = (args: Record<string, any>, responseData: any) => Promise<any>;

const getSessionIdToUse = (args: Record<string, any>, context: AiCommandRegistryContext) => {
    return args.session_id ||
        context.sessionId ||
        context.analysisContext?.sessionSelected?.SessionId;
};

export const createAiCommandRegistry = (context: AiCommandRegistryContext): Record<string, AiCommandHandler> => {
    return {
        async get_session_analysis(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            return await apiService.post('/racing-session/detailed-info', {
                id: sessionIdToUse
            });
        },

        async get_telemetry_data(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            return await apiService.post('/racing-session/telemetry', {
                session_id: sessionIdToUse,
                data_types: args.data_types || ['speed', 'acceleration']
            });
        },

        async compare_lap_times(args) {
            return await apiService.post('/racing-session/compare', {
                session_ids: args.session_ids,
                metrics: args.metrics || ['lap_times']
            });
        },

        async get_performance_insights(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            return await apiService.post('/ai/performance-analysis', {
                session_id: sessionIdToUse,
                analysis_type: args.analysis_type || 'comprehensive'
            });
        },

        async follow_expert_line(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            return await apiService.post('/ai/expert-line-guidance', {
                session_id: sessionIdToUse,
                data_types: args.data_types || ['speed', 'acceleration', 'braking', 'steering']
            });
        },

        async track_detail_for_guide(args, responseData) {
            context.startTrackGuide(responseData);
            return {
                status: 'Imitation learning guidance enabled - now continuously monitoring telemetry data and displaying AI guidance chart',
                enabled: true,
                chartAdded: true
            };
        },

        async disable_guide_user_racing() {
            context.setTrackGuideEnabled(false);
            return {
                status: 'Imitation learning guidance disabled - no longer monitoring telemetry data and guidance chart removed',
                enabled: false,
                chartRemoved: true
            };
        },

        async disable_ui_component(args) {
            if (args.component === 'chart' && context.analysisContext) {
                console.log('Updating UI component:', args);
                return { success: true, message: 'UI updated successfully' };
            }
            return { success: false, message: 'UI component not found or not supported' };
        },

        async add_imitation_guidance_chart(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            const chartAddResult = visualizationController.openVisualization('imitation-guidance-chart', {
                sessionId: sessionIdToUse,
                manuallyAdded: true
            }, {
                title: args.title || 'AI Driving Guidance',
                autoUpdate: args.autoUpdate !== false
            });

            return {
                ...chartAddResult,
                message: chartAddResult.success
                    ? 'Imitation guidance chart added successfully'
                    : 'Failed to add imitation guidance chart',
                chartType: 'imitation-guidance-chart'
            };
        },

        async remove_imitation_guidance_chart(args) {
            const charts = visualizationController.getCurrentInstances();
            const imitationCharts = charts.filter(chart => chart.type === 'imitation-guidance-chart');

            let removedCount = 0;
            if (args.chartId) {
                const removed = visualizationController.closeVisualization({ id: args.chartId });
                if (removed.success) removedCount = 1;
            } else {
                imitationCharts.forEach(chart => {
                    const removed = visualizationController.closeVisualization({ id: chart.id });
                    if (removed.success) removedCount++;
                });
            }

            return {
                success: removedCount > 0,
                message: `Removed ${removedCount} imitation guidance chart(s)`,
                removedCount
            };
        },

        async get_visualization_capabilities() {
            return visualizationController.getVisualizationAssistantContext();
        },

        async open_visualization_chart(args) {
            return visualizationController.openVisualization(
                args.type,
                args.data,
                args.config
            );
        },

        async close_visualization_chart(args) {
            return visualizationController.closeVisualization({
                id: args.chartId,
                type: args.type,
                all: args.all === true
            });
        },

        async invoke_visualization_control(args) {
            return await visualizationController.invokeVisualizationControl({
                control: args.control,
                id: args.chartId,
                type: args.type,
                args: args.args
            });
        },

        async update_guidance_once(args) {
            return await visualizationController.invokeVisualizationControl({
                control: 'refresh_once',
                id: args.chartId,
                type: args.type || 'imitation-guidance-chart',
                args: args.args
            });
        },

        async get_available_functions(args) {
            const sessionIdToUse = getSessionIdToUse(args, context);
            return {
                functions: getAvailableAiFunctionNames(),
                session_context: !!context.sessionId,
                analysis_context: !!context.analysisContext,
                current_session: sessionIdToUse
            };
        }
    };
};

export const getAvailableAiFunctionNames = () => {
    return [
        'get_session_analysis',
        'get_telemetry_data',
        'compare_lap_times',
        'get_performance_insights',
        'follow_expert_line',
        'track_detail_for_guide',
        'disable_guide_user_racing',
        'disable_ui_component',
        'add_imitation_guidance_chart',
        'remove_imitation_guidance_chart',
        'get_visualization_capabilities',
        'open_visualization_chart',
        'close_visualization_chart',
        'invoke_visualization_control',
        'update_guidance_once',
        'get_available_functions'
    ];
};
