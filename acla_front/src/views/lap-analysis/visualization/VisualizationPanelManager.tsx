import React from 'react';
import { Box, Button, DropdownMenu, Flex, IconButton, Text } from '@radix-ui/themes';
import { Cross2Icon, DragHandleDots2Icon, PlusIcon } from '@radix-ui/react-icons';
import { visualizationController } from './VisualizationController';
import type { VisualizationInstance } from './VisualizationRegistry';
import './DynamicVisualizationManager.css';

interface ManagedVisualizationInstance {
    id: string;
    type: string;
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
    TProps,
    TInstance extends ManagedVisualizationInstance,
> extends React.Component<TProps, VisualizationPanelManagerState<TInstance>> {
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

    componentDidMount() {
        visualizationController.setUpdateCallback(this.handleControllerUpdate);
        this.synchronizeController(this.currentVisualizations);
    }

    componentDidUpdate(
        _previousProps: TProps,
        previousState: VisualizationPanelManagerState<TInstance>,
    ) {
        if (previousState.visualizations !== this.state.visualizations) {
            this.synchronizeController(this.currentVisualizations);
        }
    }

    componentWillUnmount() {
        visualizationController.setUpdateCallback(() => { });
    }

    protected abstract getManagerTitle(): React.ReactNode;
    protected abstract getStaticMapTitle(): React.ReactNode;
    protected abstract getPanelTypes(): string[];
    protected abstract getPanelName(type: string): string | undefined;
    protected abstract createPanelInstance(type: string): TInstance | undefined;
    protected abstract getPanelHeight(instance: TInstance): number;
    protected abstract setPanelHeight(instance: TInstance, height: number): TInstance;
    protected abstract deserializeControllerInstances(instances: VisualizationInstance[]): TInstance[];
    protected abstract serializeControllerInstances(instances: TInstance[]): VisualizationInstance[];
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

    private handleControllerUpdate = (instances: VisualizationInstance[]) => {
        this.applyVisualizations(this.deserializeControllerInstances(instances));
    };

    private synchronizeController(instances: TInstance[]) {
        visualizationController.setCurrentInstances(this.serializeControllerInstances(instances));
    }

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
        if (this.currentVisualizations.some((visualization) => visualization.type === type)) {
            return;
        }

        const instance = this.createPanelInstance(type);
        if (!instance) {
            return;
        }

        this.applyVisualizations([...this.currentVisualizations, instance]);
    };

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
                key={instance.id}
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
