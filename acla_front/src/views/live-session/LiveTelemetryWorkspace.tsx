import React, { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import { AI_TOOL_COMPONENT_NAMES, useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import VisualizationPanelManager, {
    VisualizationManagerHandle,
    VisualizationManagerResult,
} from 'views/lap-analysis/visualization/VisualizationPanelManager';
import AnalysisResultsChart from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import { getVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
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

export interface LiveTelemetryWorkspaceProps {
    name: string;
}

class LiveTelemetryWorkspaceImpl extends VisualizationPanelManager<LiveTelemetryWorkspaceProps, LiveVisualizationInstance> {
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

    protected createPanelInstance(
        type: string,
        name: string,
        data?: any,
        config?: any,
    ): LiveVisualizationInstance | undefined {
        if (!Object.prototype.hasOwnProperty.call(OPTIONAL_VISUALIZATIONS, type)) {
            return undefined;
        }

        return {
            name,
            id: `${type}-${Date.now()}`,
            type: type as LiveVisualizationInstance['type'],
            height: 280,
            data,
            config,
        };
    }

    protected getDefaultComponentName(type: string) {
        return type === 'live-range-todo-list'
            ? AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST
            : getVisualizationComponentName(type);
    }

    protected getPanelHeight(instance: LiveVisualizationInstance) {
        return instance.height;
    }

    protected setPanelHeight(instance: LiveVisualizationInstance, height: number): LiveVisualizationInstance {
        return { ...instance, height };
    }

    protected renderStaticMap() {
        const name = getVisualizationComponentName('live-trajectory-map');
        return <LiveTrajectoryMap key={name} name={name} />;
    }

    protected renderPanelContent(instance: LiveVisualizationInstance) {
        const events = this.context.sessionIntelligence.getAllEvents();
        if (instance.type === 'telemetry-overview') {
            return (
                <LiveTelemetryOverview
                    key={instance.name}
                    name={instance.name}
                    telemetry={(instance.data as Record<string, any>) ?? this.context.currentTelemetry}
                    onUpdate={(data) => this.updateVisualization(instance.name, data).success}
                    onDisable={() => this.closeVisualization({ name: instance.name }).success}
                />
            );
        }
        if (instance.type === 'event-log') {
            return (
                <LiveEventLog
                    key={instance.name}
                    name={instance.name}
                    events={Array.isArray(instance.data) ? instance.data : events}
                    onUpdate={(data) => this.updateVisualization(instance.name, data).success}
                    onDisable={() => this.closeVisualization({ name: instance.name }).success}
                />
            );
        }
        if (instance.type === 'live-range-todo-list') {
            return <LiveRangeTodoList key={instance.name} name={instance.name} />;
        }
        return (
            <AnalysisResultsChart
                key={instance.name}
                name={instance.name}
                id={instance.id}
                data={instance.data}
                width="100%"
                height="100%"
                showElementId={false}
                sessionGame={this.context.sessionGame}
                onUpdate={(data, config) => this.updateVisualization(instance.name, data, config).success}
                onDisable={() => this.closeVisualization({ name: instance.name }).success}
            />
        );
    }
}

const unavailable = (name: string): VisualizationManagerResult => ({
    success: false,
    message: `Visualization manager '${name}' is not mounted.`,
});

const LiveTelemetryWorkspace = forwardRef<VisualizationManagerHandle, LiveTelemetryWorkspaceProps>((
    { name },
    forwardedRef,
) => {
    const managerRef = useRef<LiveTelemetryWorkspaceImpl | null>(null);
    const handle = useMemo<VisualizationManagerHandle>(() => ({
        getComponentName: () => name,
        getVisualizationCapabilities: () => managerRef.current?.getVisualizationCapabilities() ?? {
            availableCharts: [],
            openInstances: [],
        },
        getCurrentVisualizations: () => managerRef.current?.getCurrentVisualizations() ?? [],
        requestVisualization: (options) => managerRef.current?.requestVisualization(options) ?? unavailable(name),
        updateVisualization: (componentName, data, config) => (
            managerRef.current?.updateVisualization(componentName, data, config) ?? unavailable(name)
        ),
        closeVisualization: (options) => managerRef.current?.closeVisualization(options) ?? unavailable(name),
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    useRegisterAiToolComponentRef(name, handle);

    return <LiveTelemetryWorkspaceImpl ref={managerRef} name={name} />;
});

LiveTelemetryWorkspace.displayName = 'LiveTelemetryWorkspace';

export default LiveTelemetryWorkspace;
