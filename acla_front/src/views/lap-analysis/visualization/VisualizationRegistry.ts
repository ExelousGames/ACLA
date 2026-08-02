import { ComponentType } from 'react';
import TelemetryOverview from './charts/TelemetryOverview';
import MapVisualization from './charts/MapVisualization';
import ImitationGuidanceChart from './charts/ImitationGuidanceChart';
import EventLogChart from './charts/EventLogChart';
import AnalysisResultsChart from './charts/AnalysisResultsChart';
import {
    appendAnalysisResultElement,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './charts/analysisResultsModel';

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

export interface VisualizationControlFactoryContext {
    getData: () => any;
    replaceData: (data: any) => boolean;
}

export type VisualizationControlHandler = (
    args?: Record<string, any>,
) => unknown | Promise<unknown>;

export interface VisualizationComponent {
    component: ComponentType<VisualizationProps>;
    name: string;
    description: string;
    availableInRecordedWorkspace?: boolean;
    assistantControls?: VisualizationAssistantControl[];
    createAssistantControlHandlers?: (
        context: VisualizationControlFactoryContext,
    ) => Record<string, VisualizationControlHandler>;
    normalizeData?: (data: unknown) => any;
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

    getRecordedWorkspaceTypes(): string[] {
        return Array.from(this.components.entries())
            .filter(([, component]) => component.availableInRecordedWorkspace !== false)
            .map(([type]) => type);
    }
}

export const visualizationRegistry = new VisualizationRegistry();

// Register all visualization components
export const initializeVisualizations = () => {
    visualizationRegistry.register('telemetry-overview', {
        component: TelemetryOverview,
        name: 'Telemetry Overview',
        description: 'Shows telemetry supplied to the recorded analysis workspace',
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
        description: 'Guidance for telemetry supplied to the recorded analysis workspace',
        defaultConfig: {},
        minWidth: 380,
        minHeight: 500,
        preferredAspectRatio: 3 / 4
    });

    visualizationRegistry.register('event-log', {
        component: EventLogChart,
        name: 'Event Log',
        description: 'Shows event data supplied to the recorded analysis workspace',
        defaultConfig: {},
        minWidth: 360,
        minHeight: 260,
        preferredAspectRatio: 4 / 3
    });

    visualizationRegistry.register('analysis-results', {
        component: AnalysisResultsChart,
        name: 'Analysis Results',
        description: 'Scrollable list of generic labeled analysis elements and their contextual fields',
        availableInRecordedWorkspace: false,
        assistantControls: [
            {
                name: 'append_element',
                description: 'Normalize and append one labeled element, generating its ID when omitted.',
                requiresOpenChart: true,
                params: { element: 'Generic labeled element object.' },
            },
            {
                name: 'update_element',
                description: 'Update labels or contextual fields on an element without changing its ID.',
                requiresOpenChart: true,
                params: { id: 'Existing element ID.', changes: 'Fields to update.' },
            },
            {
                name: 'remove_element',
                description: 'Remove an element by ID.',
                requiresOpenChart: true,
                params: { id: 'Existing element ID.' },
            },
        ],
        createAssistantControlHandlers: ({ getData, replaceData }) => ({
            append_element: (args = {}) => {
                const mutation = appendAnalysisResultElement(getData(), args.element);
                if (mutation.result.success) replaceData(mutation.data);
                return mutation.result;
            },
            update_element: (args = {}) => {
                const mutation = updateAnalysisResultElement(getData(), args.id, args.changes);
                if (mutation.result.success) replaceData(mutation.data);
                return mutation.result;
            },
            remove_element: (args = {}) => {
                const mutation = removeAnalysisResultElement(getData(), args.id);
                if (mutation.result.success) replaceData(mutation.data);
                return mutation.result;
            },
        }),
        normalizeData: normalizeAnalysisResultsData,
        defaultConfig: {},
        minWidth: 320,
        minHeight: 260,
        preferredAspectRatio: 4 / 3,
    });
};

// Initialize visualizations when this module is imported
initializeVisualizations();

export { visualizationController } from './VisualizationController';
export type { VisualizationCommand } from './VisualizationController';

