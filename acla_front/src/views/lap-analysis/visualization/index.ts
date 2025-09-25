import { visualizationRegistry } from './VisualizationRegistry';
import SpeedChart from './charts/SpeedChart';
import TelemetryOverview from './charts/TelemetryOverview';
import LapTimeChart from './charts/LapTimeChart';
import MapVisualization from './charts/MapVisualization';
import ImitationGuidanceChart from './charts/ImitationGuidanceChart';

// Register all visualization components
export const initializeVisualizations = () => {
    visualizationRegistry.register('speed-chart', {
        component: SpeedChart,
        name: 'Speed Chart',
        description: 'Displays speed data over time',
        defaultConfig: {
            showGrid: true,
            lineColor: '#3b82f6'
        },
        minWidth: 300,
        minHeight: 250,
        preferredAspectRatio: 16 / 9
    });

    visualizationRegistry.register('telemetry-overview', {
        component: TelemetryOverview,
        name: 'Telemetry Overview',
        description: 'Shows basic telemetry statistics',
        defaultConfig: {},
        minWidth: 250,
        minHeight: 150,
        preferredAspectRatio: 4 / 3
    });

    visualizationRegistry.register('lap-time-chart', {
        component: LapTimeChart,
        name: 'Lap Time Chart',
        description: 'Displays lap times as a bar chart',
        defaultConfig: {
            showAverage: true,
            barColor: '#10b981'
        },
        minWidth: 300,
        minHeight: 200,
        preferredAspectRatio: 16 / 9
    });

    visualizationRegistry.register('map-visualization', {
        component: MapVisualization,
        name: 'Track Map',
        description: 'Interactive track map with session data',
        defaultConfig: {},
        minWidth: 300,
        minHeight: 300,
        preferredAspectRatio: 1
    });

    visualizationRegistry.register('imitation-guidance-chart', {
        component: ImitationGuidanceChart,
        name: 'AI Driving Guidance',
        description: 'Real-time AI guidance based on imitation learning',
        defaultConfig: {},
        minWidth: 380,
        minHeight: 500,
        preferredAspectRatio: 3 / 4
    });
};

// Initialize visualizations when this module is imported
initializeVisualizations();

// Export public API
export { visualizationRegistry } from './VisualizationRegistry';
export { visualizationController } from './VisualizationController';
export type { VisualizationInstance, VisualizationProps } from './VisualizationRegistry';
export type { VisualizationCommand } from './VisualizationController';
