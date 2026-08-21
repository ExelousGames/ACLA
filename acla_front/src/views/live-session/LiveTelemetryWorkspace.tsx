import React, { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import { AI_TOOL_COMPONENT_NAMES, useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import { VisualizationManagerUnavailableError } from 'contexts/AiToolComponentError';
import VisualizationPanelManager, {
    VisualizationManagerHandle,
} from 'views/lap-analysis/visualization/VisualizationPanelManager';
import AnalysisResultsChart from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import { getVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import { LiveSessionContext } from './LiveSessionContext';
import { LiveVisualizationInstance } from './live-session-types';
import LiveTrajectoryMap from './LiveTrajectoryMap';
import LiveTelemetryOverview from './LiveTelemetryOverview';
import LiveEventLog from './LiveEventLog';
import BaselineCollection from './BaselineCollection';

const OPTIONAL_VISUALIZATIONS = {
    'live-trajectory-map': { name: 'Live 2D Telemetry Trajectory' },
    'telemetry-overview': { name: 'Live Telemetry Overview' },
    'event-log': { name: 'Live Event Log' },
    'analysis-results': { name: 'Analysis Results' },
    'baseline-collection': { name: 'Baseline Collection' },
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
            height: type === 'live-trajectory-map' ? 520 : 280,
            data,
            config,
        };
    }

    protected getDefaultComponentName(type: string) {
        if (type === 'baseline-collection') return AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION;
        return getVisualizationComponentName(type);
    }

    protected getPanelHeight(instance: LiveVisualizationInstance) {
        return instance.height;
    }

    protected setPanelHeight(instance: LiveVisualizationInstance, height: number): LiveVisualizationInstance {
        return { ...instance, height };
    }

    protected isPrimaryVisualization(instance: LiveVisualizationInstance) {
        return instance.type === 'live-trajectory-map';
    }

    protected renderStaticMap() {
        return null;
    }

    protected renderPanelContent(instance: LiveVisualizationInstance) {
        if (instance.type === 'live-trajectory-map') {
            return <LiveTrajectoryMap key={instance.name} name={instance.name} />;
        }
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
                    initialEvents={Array.isArray(instance.data) ? instance.data : undefined}
                    onUpdate={(data) => this.updateVisualization(instance.name, data).success}
                    onDisable={() => this.closeVisualization({ name: instance.name }).success}
                />
            );
        }
        if (instance.type === 'baseline-collection') {
            return <BaselineCollection key={instance.name} name={instance.name} />;
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
                pagination={{
                    pages: this.context.analysisResultPages,
                    activePageId: this.context.activeAnalysisResultPageId,
                    onSelectPage: this.context.selectAnalysisResultPage,
                }}
                onUpdate={(data, config) => (
                    this.context.analysisResultPages.length > 0
                        ? this.context.updateActiveAnalysisResultPage(data)
                        : this.updateVisualization(instance.name, data, config).success
                )}
                onDisable={() => this.closeVisualization({ name: instance.name }).success}
            />
        );
    }
}

const unavailable = (name: string): never => {
    throw new VisualizationManagerUnavailableError(
        name,
        `Visualization manager '${name}' is not mounted.`,
    );
};

const LiveTelemetryWorkspace = forwardRef<VisualizationManagerHandle, LiveTelemetryWorkspaceProps>((
    { name },
    forwardedRef,
) => {
    const managerRef = useRef<LiveTelemetryWorkspaceImpl | null>(null);
    const handle = useMemo<VisualizationManagerHandle>(() => ({
        getComponentName: () => name,
        getVisualizationCapabilities: () => managerRef.current?.getVisualizationCapabilities() ?? unavailable(name),
        getCurrentVisualizations: () => managerRef.current?.getCurrentVisualizations() ?? unavailable(name),
        requestVisualization: (options) => managerRef.current?.requestVisualization({
            ...options,
            name: options.type === 'baseline-collection'
                ? AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION
                : options.name,
        }) ?? unavailable(name),
        updateVisualization: (componentName, data, config) => (
            managerRef.current?.updateVisualization(componentName, data, config) ?? unavailable(name)
        ),
        closeVisualization: (options) => managerRef.current?.closeVisualization(options) ?? unavailable(name),
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterAiToolComponentRef(registeredHandleRef);

    return <LiveTelemetryWorkspaceImpl ref={managerRef} name={name} />;
});

LiveTelemetryWorkspace.displayName = 'LiveTelemetryWorkspace';

export default LiveTelemetryWorkspace;
