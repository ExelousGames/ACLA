import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';

export interface VisualizationCommand {
    action: 'add' | 'remove' | 'update' | 'clear';
    type?: string;
    id?: string;
    data?: any;
    config?: any;
}

export class VisualizationController {
    private static instance: VisualizationController;
    private updateCallback?: (instances: VisualizationInstance[]) => void;
    private currentInstances: VisualizationInstance[] = [];

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

    executeCommand(command: VisualizationCommand): boolean {
        try {
            switch (command.action) {
                case 'add':
                    return this.addVisualization(command.type!, command.data, command.config);
                case 'remove':
                    return this.removeVisualization(command.id!);
                case 'update':
                    return this.updateVisualization(command.id!, command.data, command.config);
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

        const newVisualization: VisualizationInstance = {
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type,
            data,
            config: { ...component.defaultConfig, ...config },
            position: { x: 0, y: 0, width: '100%', height: 300 }
        };

        const updatedInstances = [...this.currentInstances, newVisualization];
        this.updateCallback?.(updatedInstances);
        this.currentInstances = updatedInstances;
        return true;
    }

    private removeVisualization(id: string): boolean {
        const initialLength = this.currentInstances.length;
        const updatedInstances = this.currentInstances.filter(v => v.id !== id);

        if (updatedInstances.length === initialLength) {
            console.warn('Visualization not found:', id);
            return false;
        }

        this.updateCallback?.(updatedInstances);
        this.currentInstances = updatedInstances;
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

        this.updateCallback?.(updatedInstances);
        this.currentInstances = updatedInstances;
        return true;
    }

    private clearAll(): boolean {
        this.updateCallback?.([]);
        this.currentInstances = [];
        return true;
    }

    // Utility methods for external use
    addSpeedChart(data?: any) {
        return this.executeCommand({ action: 'add', type: 'speed-chart', data });
    }

    addTelemetryOverview(data?: any) {
        return this.executeCommand({ action: 'add', type: 'telemetry-overview', data });
    }

    addLapTimeChart(data?: any) {
        return this.executeCommand({ action: 'add', type: 'lap-time-chart', data });
    }

    getAvailableTypes(): string[] {
        return visualizationRegistry.getAllTypes();
    }

    getCurrentInstances(): VisualizationInstance[] {
        return [...this.currentInstances];
    }
}

export const visualizationController = VisualizationController.getInstance();
