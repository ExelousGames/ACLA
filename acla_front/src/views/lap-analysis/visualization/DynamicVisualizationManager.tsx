import React, { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';
import MapVisualization from './charts/MapVisualization';
import VisualizationPanelManager, {
    VisualizationManagerHandle,
    VisualizationManagerResult,
} from './VisualizationPanelManager';
import type { MapVisualizationHandle } from './charts/MapVisualization';
import { getVisualizationComponentName } from './visualization-component-names';

export interface DynamicVisualizationManagerProps {
    name: string;
    onLayoutChange?: (instances: VisualizationInstance[]) => void;
}

interface DynamicVisualizationManagerImplProps extends DynamicVisualizationManagerProps {
    staticMapRef: React.RefObject<MapVisualizationHandle | null>;
}

const STATIC_MAP_TYPE = 'map-visualization';
const STATIC_MAP_ID = 'static-map-visualization';

class DynamicVisualizationManagerImpl extends VisualizationPanelManager<
    DynamicVisualizationManagerImplProps,
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

    protected createPanelInstance(
        type: string,
        name: string,
        data?: any,
        config?: any,
    ): VisualizationInstance | undefined {
        const component = visualizationRegistry.getComponent(type);
        if (!component) {
            return undefined;
        }

        return {
            name,
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
            type,
            data: component.normalizeData?.(data) ?? data,
            config: { ...(component.defaultConfig || {}), ...(config || {}) },
            position: { x: 0, y: 0, width: '100%', height: 280 },
        };
    }

    protected getDefaultComponentName(type: string) {
        return getVisualizationComponentName(type);
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

    protected renderStaticMap() {
        return (
            <MapVisualization
                ref={this.props.staticMapRef}
                key={getVisualizationComponentName(STATIC_MAP_TYPE)}
                name={getVisualizationComponentName(STATIC_MAP_TYPE)}
                id={STATIC_MAP_ID}
                width="100%"
                height="100%"
            />
        );
    }

    protected renderPanelContent(instance: VisualizationInstance) {
        const component = visualizationRegistry.getComponent(instance.type);
        if (!component) {
            return null;
        }

        const Component = component.component;
        return (
            <Component
                key={instance.name}
                name={instance.name}
                id={instance.id}
                data={instance.data}
                config={instance.config}
                width="100%"
                height="100%"
                onUpdate={(data, config) => this.updateVisualization(instance.name, data, config).success}
                onDisable={() => this.closeVisualization({ name: instance.name }).success}
            />
        );
    }

    protected notifyLayoutChange(instances: VisualizationInstance[]) {
        this.props.onLayoutChange?.(instances);
    }
}

const unavailable = (name: string): VisualizationManagerResult => ({
    success: false,
    message: `Visualization manager '${name}' is not mounted.`,
});

const DynamicVisualizationManager = forwardRef<VisualizationManagerHandle, DynamicVisualizationManagerProps>((
    { name, ...props },
    forwardedRef,
) => {
    const managerRef = useRef<DynamicVisualizationManagerImpl | null>(null);
    const staticMapRef = useRef<MapVisualizationHandle | null>(null);
    const staticMapName = getVisualizationComponentName(STATIC_MAP_TYPE);
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
    const staticMapHandle = useMemo<MapVisualizationHandle>(() => ({
        getComponentName: () => staticMapName,
        updateMap: (data, config) => staticMapRef.current?.updateMap(data, config) ?? false,
        disableMap: () => staticMapRef.current?.disableMap() ?? false,
    }), [staticMapName]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    useRegisterAiToolComponentRef(name, handle);
    useRegisterAiToolComponentRef(staticMapName, staticMapHandle);

    return (
        <DynamicVisualizationManagerImpl
            ref={managerRef}
            name={name}
            staticMapRef={staticMapRef}
            {...props}
        />
    );
});

DynamicVisualizationManager.displayName = 'DynamicVisualizationManager';

export default DynamicVisualizationManager;
