import { ComponentType } from 'react';
import TelemetryOverview from './charts/TelemetryOverview';
import MapVisualization from './charts/MapVisualization';
import ImitationGuidanceChart from './charts/ImitationGuidanceChart';
import ExpertActionsChart from './charts/ExpertActionsChart';
import EventLogChart from './charts/EventLogChart';

export interface VisualizationProps {
    id: string;
    data?: any;
    config?: any;
    width?: string | number;
    height?: string | number;
}

export interface VisualizationAssistantControl {
    name: string;
    description: string;
    requiresOpenChart?: boolean;
    params?: Record<string, string>;
}

export interface VisualizationComponent {
    component: ComponentType<VisualizationProps>;
    name: string;
    description: string;
    assistantControls?: VisualizationAssistantControl[];
    defaultConfig?: any;
    minWidth?: number;
    minHeight?: number;
    preferredAspectRatio?: number;
}

export interface VisualizationInstance {
    id: string;
    type: string;
    data?: any;
    config?: any;
    position?: {
        x: number | string;
        y: number | string;
        width: string | number;
        height: string | number;
    };
}

class VisualizationRegistry {
    private components: Map<string, VisualizationComponent> = new Map();

    register(type: string, component: VisualizationComponent) {
        this.components.set(type, component);
    }

    getComponent(type: string): VisualizationComponent | undefined {
        return this.components.get(type);
    }

    getAllTypes(): string[] {
        return Array.from(this.components.keys());
    }

    getAllComponents(): VisualizationComponent[] {
        return Array.from(this.components.values());
    }
}

export const visualizationRegistry = new VisualizationRegistry();

// Register all visualization components
export const initializeVisualizations = () => {
    visualizationRegistry.register('telemetry-overview', {
        component: TelemetryOverview,
        name: 'Telemetry Overview',
        description: 'Shows basic telemetry statistics',
        defaultConfig: {},
        minWidth: 250,
        minHeight: 150,
        preferredAspectRatio: 4 / 3
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

    visualizationRegistry.register('expert-actions-chart', {
        component: ExpertActionsChart,
        name: 'Expert Actions Chart',
        description: 'Shows predicted expert actions across the lap using imitation learning models',
        defaultConfig: {},
        minWidth: 380,
        minHeight: 420,
        preferredAspectRatio: 4 / 5
    });

    visualizationRegistry.register('event-log', {
        component: EventLogChart,
        name: 'Event Log',
        description: 'Live list of all session events (corners, straights, crashes, overtakes) detected by the sensors',
        defaultConfig: {},
        minWidth: 360,
        minHeight: 260,
        preferredAspectRatio: 4 / 3
    });
};

// Initialize visualizations when this module is imported
initializeVisualizations();

export { visualizationController } from './VisualizationController';
export type { VisualizationCommand } from './VisualizationController';

