import React from 'react';
import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';
import MapVisualization from './charts/MapVisualization';
import VisualizationPanelManager from './VisualizationPanelManager';

interface DynamicVisualizationManagerProps {
    onLayoutChange?: (instances: VisualizationInstance[]) => void;
}

const STATIC_MAP_TYPE = 'map-visualization';
const STATIC_MAP_ID = 'static-map-visualization';

class DynamicVisualizationManager extends VisualizationPanelManager<
    DynamicVisualizationManagerProps,
    VisualizationInstance
> {
    protected getManagerTitle() {
        return 'Data Visualizations';
    }

    protected getStaticMapTitle() {
        return '2D Telemetry Trajectory';
    }

    protected getPanelTypes() {
        return visualizationRegistry.getRecordedWorkspaceTypes().filter((type) => type !== STATIC_MAP_TYPE);
    }

    protected getPanelName(type: string) {
        return visualizationRegistry.getComponent(type)?.name;
    }

    protected createPanelInstance(type: string): VisualizationInstance | undefined {
        const component = visualizationRegistry.getComponent(type);
        if (!component) {
            return undefined;
        }

        return {
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
            type,
            config: component.defaultConfig || {},
            position: { x: 0, y: 0, width: '100%', height: 280 },
        };
    }

    protected getPanelHeight(instance: VisualizationInstance) {
        const height = instance.position?.height;
        return typeof height === 'number' ? height : 280;
    }

    protected setPanelHeight(instance: VisualizationInstance, height: number): VisualizationInstance {
        return {
            ...instance,
            position: {
                ...(instance.position || { x: 0, y: 0, width: '100%', height }),
                height,
            },
        };
    }

    protected deserializeControllerInstances(instances: VisualizationInstance[]) {
        return instances.filter((instance) => instance.type !== STATIC_MAP_TYPE);
    }

    protected serializeControllerInstances(instances: VisualizationInstance[]) {
        return [
            {
                id: STATIC_MAP_ID,
                type: STATIC_MAP_TYPE,
                config: {},
                position: { x: 0, y: 0, width: '100%', height: '100%' },
            },
            ...instances,
        ];
    }

    protected renderStaticMap() {
        return <MapVisualization id={STATIC_MAP_ID} width="100%" height="100%" />;
    }

    protected renderPanelContent(instance: VisualizationInstance) {
        const component = visualizationRegistry.getComponent(instance.type);
        if (!component) {
            return null;
        }

        const Component = component.component;
        return (
            <Component
                id={instance.id}
                data={instance.data}
                config={instance.config}
                width="100%"
                height="100%"
            />
        );
    }

    protected notifyLayoutChange(instances: VisualizationInstance[]) {
        this.props.onLayoutChange?.(instances);
    }
}

export default DynamicVisualizationManager;
