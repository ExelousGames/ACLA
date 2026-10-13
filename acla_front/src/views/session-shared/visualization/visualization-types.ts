export interface VisualizationProps {
    name: string;
    id: string;
    data?: any;
    config?: any;
    width?: string | number;
    height?: string | number;
    onUpdate?: (data?: any, config?: any) => boolean;
    onDisable?: () => boolean;
}
