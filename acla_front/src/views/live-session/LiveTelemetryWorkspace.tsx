import React from 'react';
import type { VisualizationInstance } from 'views/lap-analysis/visualization/VisualizationRegistry';
import VisualizationPanelManager from 'views/lap-analysis/visualization/VisualizationPanelManager';
import AnalysisResultsChart from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import { LiveSessionContext } from './LiveSessionContext';
import { LiveVisualizationInstance } from './live-session-types';
import LiveTrajectoryMap from './LiveTrajectoryMap';
import LiveTelemetryOverview from './LiveTelemetryOverview';
import LiveEventLog from './LiveEventLog';
import LiveRangeTodoList from './LiveRangeTodoList';

const OPTIONAL_VISUALIZATIONS = {
    'telemetry-overview': { name: 'Live Telemetry Overview' },
    'event-log': { name: 'Live Event Log' },
    'analysis-results': { name: 'Analysis Results' },
    'live-range-todo-list': { name: 'Live Range To-do List' },
} as const;

class LiveTelemetryWorkspace extends VisualizationPanelManager<{}, LiveVisualizationInstance> {
    static contextType = LiveSessionContext;
    context!: React.ContextType<typeof LiveSessionContext>;

    protected getManagerTitle() {
        return 'Live Data Visualizations';
    }

    protected getStaticMapTitle() {
        return 'Live 2D Telemetry Trajectory';
    }

    protected getManagerClassName() {
        return 'dynamic-visualization-manager live-telemetry-workspace';
    }

    protected getManagerTestId() {
        return 'live-telemetry-workspace';
    }

    protected getPanelContentTestId(instance: LiveVisualizationInstance) {
        return `live-visualization-content-${instance.type}`;
    }

    protected getRemoveButtonAriaLabel(instance: LiveVisualizationInstance) {
        return `Remove ${OPTIONAL_VISUALIZATIONS[instance.type].name}`;
    }

    protected getPanelTitleWeight(): undefined {
        return undefined;
    }

    protected getPanelTypes() {
        return Object.keys(OPTIONAL_VISUALIZATIONS) as LiveVisualizationInstance['type'][];
    }

    protected getPanelName(type: string) {
        if (!Object.prototype.hasOwnProperty.call(OPTIONAL_VISUALIZATIONS, type)) {
            return undefined;
        }
        return OPTIONAL_VISUALIZATIONS[type as LiveVisualizationInstance['type']].name;
    }

    protected createPanelInstance(type: string): LiveVisualizationInstance | undefined {
        if (!Object.prototype.hasOwnProperty.call(OPTIONAL_VISUALIZATIONS, type)) {
            return undefined;
        }

        return {
            id: `${type}-${Date.now()}`,
            type: type as LiveVisualizationInstance['type'],
            height: 280,
        };
    }

    protected getPanelHeight(instance: LiveVisualizationInstance) {
        return instance.height;
    }

    protected setPanelHeight(instance: LiveVisualizationInstance, height: number): LiveVisualizationInstance {
        return { ...instance, height };
    }

    protected deserializeControllerInstances(instances: VisualizationInstance[]) {
        return instances
            .filter((instance): instance is VisualizationInstance & { type: LiveVisualizationInstance['type'] } => (
                Object.prototype.hasOwnProperty.call(OPTIONAL_VISUALIZATIONS, instance.type)
            ))
            .map((instance): LiveVisualizationInstance => ({
                id: instance.id,
                type: instance.type,
                height: typeof instance.position?.height === 'number' ? instance.position.height : 280,
                data: instance.data,
            }));
    }

    protected serializeControllerInstances(instances: LiveVisualizationInstance[]) {
        return instances.map((instance) => ({
            id: instance.id,
            type: instance.type,
            data: instance.data,
            position: { x: 0, y: 0, width: '100%', height: instance.height },
        }));
    }

    protected renderStaticMap() {
        return <LiveTrajectoryMap />;
    }

    protected renderPanelContent(instance: LiveVisualizationInstance) {
        const events = this.context.sessionIntelligence.getAllEvents();
        if (instance.type === 'telemetry-overview') {
            return <LiveTelemetryOverview telemetry={this.context.currentTelemetry} />;
        }
        if (instance.type === 'event-log') {
            return <LiveEventLog events={events} />;
        }
        if (instance.type === 'live-range-todo-list') {
            return <LiveRangeTodoList />;
        }
        return (
            <AnalysisResultsChart
                id={instance.id}
                data={instance.data}
                width="100%"
                height="100%"
                showElementId={false}
            />
        );
    }
}

export default LiveTelemetryWorkspace;
