import { ComponentType } from 'react';

export interface VisualizationProps {
    id: string;
    data?: any;
    config?: any;
    width?: string | number;
    height?: string | number;
}

export interface VisualizationComponent {
    component: ComponentType<VisualizationProps>;
    name: string;
    description: string;
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
