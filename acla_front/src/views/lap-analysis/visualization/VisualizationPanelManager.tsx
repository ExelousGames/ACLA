import React from 'react';
import { Box, Button, DropdownMenu, Flex, IconButton, Text } from '@radix-ui/themes';
import { Cross2Icon, DragHandleDots2Icon, PlusIcon } from '@radix-ui/react-icons';
import type { NamedAiToolComponentHandle } from 'contexts/AiToolComponentRefContext';
import {
    AiToolComponentErrorConstructor,
    VisualizationCloseFailedError,
    VisualizationComponentError,
    VisualizationRequestFailedError,
    VisualizationUpdateFailedError,
} from 'contexts/AiToolComponentError';
import './DynamicVisualizationManager.css';

export interface ManagedVisualizationInstance {
    name: string;
    id: string;
    type: string;
    data?: any;
    config?: any;
}

export interface VisualizationManagerResult {
    success: true;
    message: string;
    componentName?: string;
    chartId?: string;
    chartType?: string;
    reused?: boolean;
    data?: any;
}

export interface VisualizationManagerHandle extends NamedAiToolComponentHandle {
    getVisualizationCapabilities(): Record<string, any>;
    getCurrentVisualizations(): ManagedVisualizationInstance[];
    requestVisualization(options: {
        name: string;
        type: string;
        data?: any;
        config?: any;
    }): VisualizationManagerResult;
    updateVisualization(name: string, data?: any, config?: any): VisualizationManagerResult;
    closeVisualization(options: {
        name?: string;
        id?: string;
        type?: string;
        all?: boolean;
    }): VisualizationManagerResult;
}

interface VisualizationPanelManagerState<TInstance> {
    visualizations: TInstance[];
    draggingId: string | null;
    dropTargetId: string | null;
    resizingId: string | null;
}

interface ResizeState {
    id: string;
    pointerId: number;
    startY: number;
    startHeight: number;
    currentHeight: number;
}

/**
 * Shared visualization workspace behavior. Subclasses retain ownership of their
 * session-specific panel catalog, instance shape, map, and panel contents.
 */
abstract class VisualizationPanelManager<
    TProps extends { name: string },
    TInstance extends ManagedVisualizationInstance,
> extends React.Component<TProps, VisualizationPanelManagerState<TInstance>> implements VisualizationManagerHandle {
    protected static readonly MIN_PANEL_HEIGHT = 180;
    protected static readonly MAX_PANEL_HEIGHT = 900;

    state: VisualizationPanelManagerState<TInstance> = {
        visualizations: [],
        draggingId: null,
        dropTargetId: null,
        resizingId: null,
    };

    private currentVisualizations: TInstance[] = [];
    private resizeState: ResizeState | null = null;

    private runManagerOperation<T>(
        ErrorType: AiToolComponentErrorConstructor<VisualizationComponentError>,
        fallbackMessage: string,
        operation: () => T,
    ): T {
        try {
            return operation();
        } catch (error) {
            if (
                error instanceof ErrorType
                && error.componentName === this.props.name
            ) {
                throw error;
            }
            throw new ErrorType(
                this.props.name,
                error instanceof Error && error.message ? error.message : fallbackMessage,
                { cause: error },
            );
        }
    }

    protected abstract getManagerTitle(): React.ReactNode;
    protected abstract getStaticMapTitle(): React.ReactNode;
    protected abstract getPanelTypes(): string[];
    protected abstract getPanelName(type: string): string | undefined;
    protected abstract createPanelInstance(
        type: string,
        name: string,
        data?: any,
        config?: any,
    ): TInstance | undefined;
    protected abstract getDefaultComponentName(type: string): string;
    protected abstract getPanelHeight(instance: TInstance): number;
    protected abstract setPanelHeight(instance: TInstance, height: number): TInstance;
    protected abstract renderStaticMap(): React.ReactNode;
    protected abstract renderPanelContent(instance: TInstance): React.ReactNode;

    protected getManagerClassName(): string {
        return 'dynamic-visualization-manager';
    }

    protected getManagerTestId(): string | undefined {
        return undefined;
    }

    protected getPanelContentTestId(_instance: TInstance): string | undefined {
        return undefined;
    }

    protected getRemoveButtonAriaLabel(_instance: TInstance): string | undefined {
        return undefined;
    }

    protected getPanelTitleWeight(): 'medium' | undefined {
        return 'medium';
    }

    protected notifyLayoutChange(_instances: TInstance[]): void { }

    private applyVisualizations(next: TInstance[]) {
        this.currentVisualizations = next;
        this.setState({ visualizations: next });
        this.notifyLayoutChange(next);
    }

    private getAvailableTypes(): string[] {
        return this.getPanelTypes().filter((type) => (
            !this.state.visualizations.some((visualization) => visualization.type === type)
        ));
    }

    private addVisualization = (type: string) => {
        const name = this.getDefaultComponentName(type);
        if (this.currentVisualizations.some((visualization) => visualization.name === name)) {
            return;
        }

        const instance = this.createPanelInstance(type, name);
        if (!instance) {
            return;
        }

        this.applyVisualizations([...this.currentVisualizations, instance]);
    };

    public getComponentName = (): string => this.props.name;

    public getCurrentVisualizations = (): ManagedVisualizationInstance[] => (
        this.currentVisualizations.map((instance) => ({ ...instance }))
    );

    public getVisualizationCapabilities = (): Record<string, any> => ({
        availableCharts: this.getPanelTypes().map((type) => {
            const openInstances = this.currentVisualizations.filter((instance) => instance.type === type);
            return {
                type,
                name: this.getPanelName(type) ?? type,
                openCount: openInstances.length,
                canOpen: type === 'telemetry-overview' || openInstances.length === 0,
            };
        }),
        openInstances: this.currentVisualizations.map(({ id, name, type }) => ({ id, name, type })),
    });

    public requestVisualization = ({
        name,
        type,
        data,
        config,
    }: {
        name: string;
        type: string;
        data?: any;
        config?: any;
    }): VisualizationManagerResult => this.runManagerOperation(
        VisualizationRequestFailedError,
        `Unable to open chart '${type}'.`,
        () => {
            const exact = this.currentVisualizations.find((instance) => instance.name === name);
            if (exact) {
                this.applyVisualizations(this.currentVisualizations.map((instance) => (
                    instance.name === name
                        ? { ...instance, data: data ?? instance.data, config: config === undefined ? instance.config : { ...instance.config, ...config } }
                        : instance
                )));
                return {
                    success: true,
                    message: `Reused chart '${name}'.`,
                    componentName: name,
                    chartId: exact.id,
                    chartType: exact.type,
                    reused: true,
                };
            }

            if (type !== 'telemetry-overview') {
                const singleton = this.currentVisualizations.find((instance) => instance.type === type);
                if (singleton) {
                    this.applyVisualizations(this.currentVisualizations.map((instance) => (
                        instance.name === singleton.name
                            ? { ...instance, data: data ?? instance.data, config: config === undefined ? instance.config : { ...instance.config, ...config } }
                            : instance
                    )));
                    return {
                        success: true,
                        message: `Reused chart '${singleton.name}'.`,
                        componentName: singleton.name,
                        chartId: singleton.id,
                        chartType: singleton.type,
                        reused: true,
                    };
                }
            }

            const instance = this.createPanelInstance(type, name, data, config);
            if (!instance) {
                throw new VisualizationRequestFailedError(
                    this.props.name,
                    `Unable to open chart '${type}'.`,
                );
            }
            this.applyVisualizations([...this.currentVisualizations, instance]);
            return {
                success: true,
                message: `Opened chart '${type}'.`,
                componentName: name,
                chartId: instance.id,
                chartType: type,
                reused: false,
            };
        },
    );

    public updateVisualization = (
        name: string,
        data?: any,
        config?: any,
    ): VisualizationManagerResult => this.runManagerOperation(
        VisualizationUpdateFailedError,
        `Could not update chart '${name}'.`,
        () => {
            const existing = this.currentVisualizations.find((instance) => instance.name === name);
            if (!existing) {
                throw new VisualizationUpdateFailedError(
                    this.props.name,
                    `Chart '${name}' is not open.`,
                );
            }
            this.applyVisualizations(this.currentVisualizations.map((instance) => (
                instance.name === name
                    ? {
                        ...instance,
                        data: data === undefined ? instance.data : data,
                        config: config === undefined ? instance.config : { ...instance.config, ...config },
                    }
                    : instance
            )));
            return {
                success: true,
                message: `Updated chart '${name}'.`,
                componentName: name,
                chartId: existing.id,
                chartType: existing.type,
            };
        },
    );

    public closeVisualization = (options: {
        name?: string;
        id?: string;
        type?: string;
        all?: boolean;
    }): VisualizationManagerResult => this.runManagerOperation(
        VisualizationCloseFailedError,
        'No matching open chart was found.',
        () => {
            const matches = this.currentVisualizations.filter((instance) => (
                options.name ? instance.name === options.name
                    : options.id ? instance.id === options.id
                        : options.type ? instance.type === options.type
                            : false
            ));
            const closing = options.all ? matches : matches.slice(0, 1);
            if (closing.length === 0) {
                throw new VisualizationCloseFailedError(
                    this.props.name,
                    options.name
                        ? `Chart '${options.name}' is not open.`
                        : 'No matching open chart was found.',
                );
            }
            const names = new Set(closing.map((instance) => instance.name));
            this.applyVisualizations(this.currentVisualizations.filter((instance) => !names.has(instance.name)));
            return {
                success: true,
                message: `Closed ${closing.length} chart(s).`,
                componentName: closing[0].name,
                chartId: closing[0].id,
                chartType: closing[0].type,
                data: { removedCount: closing.length },
            };
        },
    );

    private removeVisualization = (id: string) => {
        this.applyVisualizations(this.currentVisualizations.filter((visualization) => visualization.id !== id));
    };

    private reorderVisualizations(sourceId: string, targetId: string) {
        if (!sourceId || !targetId || sourceId === targetId) {
            return;
        }

        const sourceIndex = this.currentVisualizations.findIndex((item) => item.id === sourceId);
        const targetIndex = this.currentVisualizations.findIndex((item) => item.id === targetId);
        if (sourceIndex === -1 || targetIndex === -1 || sourceIndex === targetIndex) {
            return;
        }

        const reordered = [...this.currentVisualizations];
        const [moved] = reordered.splice(sourceIndex, 1);
        reordered.splice(targetIndex, 0, moved);
        this.applyVisualizations(reordered);
    }

    private handleDragStart = (event: React.DragEvent, id: string) => {
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', id);
        this.setState({ draggingId: id });
    };

    private handleDragOver = (event: React.DragEvent, id: string) => {
        event.preventDefault();
        if (id !== this.state.dropTargetId) {
            this.setState({ dropTargetId: id });
        }
    };

    private handleDrop = (event: React.DragEvent, targetId: string) => {
        event.preventDefault();
        const sourceId = event.dataTransfer.getData('text/plain') || this.state.draggingId;
        if (sourceId) {
            this.reorderVisualizations(sourceId, targetId);
        }
        this.setState({ draggingId: null, dropTargetId: null });
    };

    private handleDragEnd = () => {
        this.setState({ draggingId: null, dropTargetId: null });
    };

    private handleResizeStart = (event: React.PointerEvent, instance: TInstance) => {
        if (event.button !== 0) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const startHeight = this.getPanelHeight(instance);
        this.resizeState = {
            id: instance.id,
            pointerId: event.pointerId,
            startY: event.clientY,
            startHeight,
            currentHeight: startHeight,
        };

        event.currentTarget.setPointerCapture(event.pointerId);
        this.setState({ resizingId: instance.id });
    };

    private handleResizeMove = (event: React.PointerEvent) => {
        if (!this.resizeState || this.resizeState.pointerId !== event.pointerId) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const nextHeight = Math.max(
            VisualizationPanelManager.MIN_PANEL_HEIGHT,
            Math.min(
                VisualizationPanelManager.MAX_PANEL_HEIGHT,
                this.resizeState.startHeight + event.clientY - this.resizeState.startY,
            ),
        );
        this.resizeState.currentHeight = nextHeight;

        const container = (event.currentTarget as HTMLElement)
            .closest('.visualization-container') as HTMLElement | null;
        if (container) {
            container.style.height = `${nextHeight}px`;
        }
    };

    private handleResizeEnd = (event: React.PointerEvent) => {
        if (!this.resizeState || this.resizeState.pointerId !== event.pointerId) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }

        const container = (event.currentTarget as HTMLElement)
            .closest('.visualization-container') as HTMLElement | null;
        if (container) {
            container.style.height = '';
        }

        const { id, currentHeight } = this.resizeState;
        const updated = this.currentVisualizations.map((instance) => (
            instance.id === id ? this.setPanelHeight(instance, currentHeight) : instance
        ));
        this.resizeState = null;
        this.setState({ resizingId: null });
        this.applyVisualizations(updated);
    };

    private renderVisualization = (instance: TInstance) => {
        const panelName = this.getPanelName(instance.type);
        if (!panelName) {
            return null;
        }

        const { draggingId, dropTargetId, resizingId } = this.state;
        return (
            <Box
                key={instance.name}
                className={`visualization-container${draggingId === instance.id ? ' is-dragging' : ''}${dropTargetId === instance.id ? ' is-drop-target' : ''}${resizingId === instance.id ? ' is-resizing' : ''}`}
                style={{ height: `${this.getPanelHeight(instance)}px` }}
                onDragOver={(event) => this.handleDragOver(event, instance.id)}
                onDrop={(event) => this.handleDrop(event, instance.id)}
            >
                <Box
                    className="visualization-header"
                    draggable
                    onDragStart={(event) => this.handleDragStart(event, instance.id)}
                    onDragEnd={this.handleDragEnd}
                >
                    <Flex align="center" gap="2">
                        <DragHandleDots2Icon className="drag-handle" />
                        <Text size="2" weight={this.getPanelTitleWeight()}>{panelName}</Text>
                    </Flex>
                    <IconButton
                        size="1"
                        variant="ghost"
                        onClick={() => this.removeVisualization(instance.id)}
                        aria-label={this.getRemoveButtonAriaLabel(instance)}
                    >
                        <Cross2Icon />
                    </IconButton>
                </Box>
                <Box
                    data-testid={this.getPanelContentTestId(instance)}
                    style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}
                >
                    {this.renderPanelContent(instance)}
                </Box>
                <Box
                    className="visualization-resize-handle"
                    onPointerDown={(event) => this.handleResizeStart(event, instance)}
                    onPointerMove={this.handleResizeMove}
                    onPointerUp={this.handleResizeEnd}
                    onPointerCancel={this.handleResizeEnd}
                />
            </Box>
        );
    };

    render() {
        const { visualizations } = this.state;
        const availableTypes = this.getAvailableTypes();

        return (
            <Box className={this.getManagerClassName()} data-testid={this.getManagerTestId()}>
                <Flex justify="between" align="center" className="manager-header">
                    <Text size="3" weight="bold">{this.getManagerTitle()}</Text>
                    <DropdownMenu.Root>
                        <DropdownMenu.Trigger>
                            <Button size="2" variant="soft" disabled={availableTypes.length === 0}>
                                <PlusIcon />
                                Add Visualization
                            </Button>
                        </DropdownMenu.Trigger>
                        <DropdownMenu.Content>
                            {availableTypes.map((type) => (
                                <DropdownMenu.Item key={type} onClick={() => this.addVisualization(type)}>
                                    {this.getPanelName(type) || type}
                                </DropdownMenu.Item>
                            ))}
                            {availableTypes.length === 0 && (
                                <DropdownMenu.Item disabled>
                                    All visualizations already added
                                </DropdownMenu.Item>
                            )}
                        </DropdownMenu.Content>
                    </DropdownMenu.Root>
                </Flex>

                <Box className={`visualization-workspace${visualizations.length === 0 ? ' visualization-workspace--map-only' : ''}`}>
                    <Box className="static-map-container">
                        <Box className="static-map-header">
                            <Text size="2" weight="medium">{this.getStaticMapTitle()}</Text>
                        </Box>
                        <Box className="static-map-body">
                            {this.renderStaticMap()}
                        </Box>
                    </Box>

                    {visualizations.length > 0 && (
                        <Box className="visualizations-container">
                            {visualizations.map(this.renderVisualization)}
                        </Box>
                    )}
                </Box>
            </Box>
        );
    }
}

export default VisualizationPanelManager;
