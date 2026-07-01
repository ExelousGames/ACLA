import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';

export interface VisualizationCommand {
    action: 'add' | 'remove' | 'update' | 'clear';
    type?: string;
    id?: string;
    data?: any;
    config?: any;
}

export interface VisualizationAssistantControl {
    name: string;
    description: string;
    requiresOpenChart?: boolean;
    params?: Record<string, string>;
}

export interface VisualizationControlResult {
    success: boolean;
    message: string;
    data?: any;
    chartId?: string;
    chartType?: string;
    control?: string;
}

type VisualizationControlHandler = (args?: Record<string, any>) => unknown | Promise<unknown>;

interface RegisteredInstanceControls {
    type: string;
    controls: VisualizationAssistantControl[];
    handlers: Map<string, VisualizationControlHandler>;
}

export class VisualizationController {
    private static instance: VisualizationController;
    private updateCallback?: (instances: VisualizationInstance[]) => void;
    private currentInstances: VisualizationInstance[] = [];
    private instanceControls: Map<string, RegisteredInstanceControls> = new Map();

    private constructor() { }

    static getInstance(): VisualizationController {
        if (!VisualizationController.instance) {
            VisualizationController.instance = new VisualizationController();
        }
        return VisualizationController.instance;
    }

    setUpdateCallback(callback: (instances: VisualizationInstance[]) => void) {
        this.updateCallback = callback;
    }

    setCurrentInstances(instances: VisualizationInstance[]) {
        this.currentInstances = [...instances];
    }

    private notifyUpdate(instances: VisualizationInstance[]) {
        this.currentInstances = instances;
        this.updateCallback?.(instances);
    }

    private createVisualizationId(type: string): string {
        return `${type}_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    }

    hasVisualizationType(type: string): boolean {
        return this.currentInstances.some(instance => instance.type === type);
    }

    executeCommand(command: VisualizationCommand): boolean {
        try {
            switch (command.action) {
                case 'add':
                    if (!command.type) return false;
                    return this.addVisualization(command.type, command.data, command.config);
                case 'remove':
                    if (!command.id) return false;
                    return this.removeVisualization(command.id);
                case 'update':
                    if (!command.id) return false;
                    return this.updateVisualization(command.id, command.data, command.config);
                case 'clear':
                    return this.clearAll();
                default:
                    console.warn('Unknown visualization command:', command.action);
                    return false;
            }
        } catch (error) {
            console.error('Error executing visualization command:', error);
            return false;
        }
    }

    private addVisualization(type: string, data?: any, config?: any): boolean {
        const component = visualizationRegistry.getComponent(type);
        if (!component) {
            console.warn('Unknown visualization type:', type);
            return false;
        }

        if (this.hasVisualizationType(type)) {
            console.warn('Visualization type already added:', type);
            return false;
        }

        const newVisualization: VisualizationInstance = {
            id: this.createVisualizationId(type),
            type,
            data,
            config: { ...component.defaultConfig, ...config },
            position: { x: 0, y: 0, width: '100%', height: 300 }
        };

        const updatedInstances = [...this.currentInstances, newVisualization];
        this.notifyUpdate(updatedInstances);
        return true;
    }

    private removeVisualization(id: string): boolean {
        const initialLength = this.currentInstances.length;
        const updatedInstances = this.currentInstances.filter(v => v.id !== id);

        if (updatedInstances.length === initialLength) {
            console.warn('Visualization not found:', id);
            return false;
        }

        this.notifyUpdate(updatedInstances);
        return true;
    }

    private updateVisualization(id: string, data?: any, config?: any): boolean {
        const updatedInstances = this.currentInstances.map(v => {
            if (v.id === id) {
                return {
                    ...v,
                    data: data !== undefined ? data : v.data,
                    config: config !== undefined ? { ...v.config, ...config } : v.config
                };
            }
            return v;
        });

        const wasUpdated = updatedInstances.some((v, i) => v !== this.currentInstances[i]);
        if (!wasUpdated) {
            console.warn('Visualization not found or no changes:', id);
            return false;
        }

        this.notifyUpdate(updatedInstances);
        return true;
    }

    private clearAll(): boolean {
        this.notifyUpdate([]);
        return true;
    }

    openVisualization(type: string, data?: any, config?: any): VisualizationControlResult {
        const created = this.addVisualization(type, data, config);
        if (!created) {
            return {
                success: false,
                message: `Unable to open chart '${type}'. It may already be open or unsupported.`,
                chartType: type
            };
        }

        const createdChart = this.currentInstances.find(instance => instance.type === type);
        return {
            success: true,
            message: `Opened chart '${type}'.`,
            chartType: type,
            chartId: createdChart?.id
        };
    }

    closeVisualization(options: { id?: string; type?: string; all?: boolean }): VisualizationControlResult {
        if (options.id) {
            const removed = this.removeVisualization(options.id);
            return {
                success: removed,
                message: removed ? `Closed chart '${options.id}'.` : `Chart '${options.id}' was not found.`,
                chartId: options.id
            };
        }

        if (!options.type) {
            return {
                success: false,
                message: 'closeVisualization requires either an id or a type.'
            };
        }

        const matching = this.currentInstances.filter(instance => instance.type === options.type);
        if (matching.length === 0) {
            return {
                success: false,
                message: `No open charts found for type '${options.type}'.`,
                chartType: options.type
            };
        }

        const toClose = options.all ? matching : [matching[0]];
        let removedCount = 0;
        toClose.forEach(instance => {
            if (this.removeVisualization(instance.id)) {
                removedCount += 1;
            }
        });

        return {
            success: removedCount > 0,
            message: `Closed ${removedCount} chart(s) of type '${options.type}'.`,
            chartType: options.type,
            data: { removedCount }
        };
    }

    registerInstanceControls(
        instanceId: string,
        type: string,
        controls: VisualizationAssistantControl[],
        handlers: Record<string, VisualizationControlHandler>
    ) {
        const handlerEntries = Object.entries(handlers || {});
        this.instanceControls.set(instanceId, {
            type,
            controls,
            handlers: new Map(handlerEntries)
        });
    }

    unregisterInstanceControls(instanceId: string) {
        this.instanceControls.delete(instanceId);
    }

    async invokeVisualizationControl(options: {
        control: string;
        id?: string;
        type?: string;
        args?: Record<string, any>;
    }): Promise<VisualizationControlResult> {
        const { control, id, type, args } = options;
        if (!control) {
            return {
                success: false,
                message: 'Missing control name for invokeVisualizationControl.'
            };
        }

        const targetInstance = id
            ? this.currentInstances.find(instance => instance.id === id)
            : this.currentInstances.find(instance => instance.type === type);

        if (!targetInstance) {
            return {
                success: false,
                message: id
                    ? `Chart '${id}' is not open.`
                    : `No open chart found for type '${type ?? 'unknown'}'.`,
                chartId: id,
                chartType: type,
                control
            };
        }

        const registration = this.instanceControls.get(targetInstance.id);
        const handler = registration?.handlers.get(control);
        if (!handler) {
            return {
                success: false,
                message: `Control '${control}' is not available for chart '${targetInstance.id}'.`,
                chartId: targetInstance.id,
                chartType: targetInstance.type,
                control,
                data: {
                    availableControls: registration?.controls.map(item => item.name) ?? []
                }
            };
        }

        const result = await handler(args);
        return {
            success: true,
            message: `Executed '${control}' on chart '${targetInstance.id}'.`,
            chartId: targetInstance.id,
            chartType: targetInstance.type,
            control,
            data: result
        };
    }

    getVisualizationAssistantContext() {
        const chartTypes = visualizationRegistry.getAllTypes();
        const availableCharts = chartTypes.map(type => {
            const component = visualizationRegistry.getComponent(type);
            const openInstances = this.currentInstances.filter(instance => instance.type === type);
            return {
                type,
                name: component?.name ?? type,
                description: component?.description ?? '',
                openCount: openInstances.length,
                canOpen: !this.hasVisualizationType(type),
                controls: component?.assistantControls ?? []
            };
        });

        const openInstances = this.currentInstances.map(instance => {
            const registration = this.instanceControls.get(instance.id);
            return {
                id: instance.id,
                type: instance.type,
                controls: registration?.controls ?? visualizationRegistry.getComponent(instance.type)?.assistantControls ?? []
            };
        });

        return {
            availableCharts,
            openInstances
        };
    }

    // Utility methods for external use
    addTelemetryOverview(data?: any) {
        return this.executeCommand({ action: 'add', type: 'telemetry-overview', data });
    }

    addExpertActionsChart(data?: any) {
        return this.executeCommand({ action: 'add', type: 'expert-actions-chart', data });
    }

    getAvailableTypes(): string[] {
        return visualizationRegistry
            .getAllTypes()
            .filter(type => !this.hasVisualizationType(type));
    }

    getCurrentInstances(): VisualizationInstance[] {
        return [...this.currentInstances];
    }
}

export const visualizationController = VisualizationController.getInstance();
