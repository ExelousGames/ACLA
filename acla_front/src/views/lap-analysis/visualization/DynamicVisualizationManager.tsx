import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Box, Button, DropdownMenu, IconButton, Flex, Text } from '@radix-ui/themes';
import { PlusIcon, Cross2Icon, DragHandleDots2Icon } from '@radix-ui/react-icons';
import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';
import { visualizationController } from './VisualizationController';
import './DynamicVisualizationManager.css';

interface DynamicVisualizationManagerProps {
    onLayoutChange?: (instances: VisualizationInstance[]) => void;
}

const DynamicVisualizationManager: React.FC<DynamicVisualizationManagerProps> = ({
    onLayoutChange
}) => {
    const MIN_CHART_HEIGHT = 180;
    const MAX_CHART_HEIGHT = 900;

    const [visualizations, setVisualizations] = useState<VisualizationInstance[]>([]);
    const [draggingId, setDraggingId] = useState<string | null>(null);
    const [dropTargetId, setDropTargetId] = useState<string | null>(null);
    const [resizingId, setResizingId] = useState<string | null>(null);
    const visualizationsRef = useRef<VisualizationInstance[]>([]);
    const resizeStateRef = useRef<{
        id: string;
        pointerId: number;
        startY: number;
        startHeight: number;
        currentHeight: number;
    } | null>(null);

    useEffect(() => {
        visualizationsRef.current = visualizations;
    }, [visualizations]);

    // Setup controller callback on mount
    useEffect(() => {
        const handleControllerUpdate = (instances: VisualizationInstance[]) => {
            setVisualizations(instances);
        };

        visualizationController.setUpdateCallback(handleControllerUpdate);

        return () => {
            visualizationController.setUpdateCallback(() => { });
        };
    }, []);

    useEffect(() => {
        visualizationController.setCurrentInstances(visualizations);
    }, [visualizations]);

    const applyVisualizations = useCallback((next: VisualizationInstance[]) => {
        setVisualizations(next);
        onLayoutChange?.(next);
    }, [onLayoutChange]);

    const getInstanceHeight = useCallback((instance: VisualizationInstance): number => {
        if (!instance.position) {
            return 280;
        }

        const { height } = instance.position;
        return typeof height === 'number' ? height : 280;
    }, []);

    const updateVisualizationHeight = useCallback((id: string, height: number) => {
        const nextHeight = Math.max(MIN_CHART_HEIGHT, Math.min(MAX_CHART_HEIGHT, height));
        const updated = visualizationsRef.current.map(item => {
            if (item.id !== id) {
                return item;
            }

            return {
                ...item,
                position: {
                    ...(item.position || { x: 0, y: 0, width: '100%', height: nextHeight }),
                    height: nextHeight
                }
            };
        });

        applyVisualizations(updated);
    }, [applyVisualizations]);

    const reorderVisualizations = useCallback((sourceId: string, targetId: string) => {
        if (!sourceId || !targetId || sourceId === targetId) {
            return;
        }

        const current = visualizationsRef.current;
        const sourceIndex = current.findIndex(item => item.id === sourceId);
        const targetIndex = current.findIndex(item => item.id === targetId);

        if (sourceIndex === -1 || targetIndex === -1 || sourceIndex === targetIndex) {
            return;
        }

        const reordered = [...current];
        const [moved] = reordered.splice(sourceIndex, 1);
        reordered.splice(targetIndex, 0, moved);
        applyVisualizations(reordered);
    }, [applyVisualizations]);

    // Add a new visualization
    const addVisualization = useCallback((type: string) => {
        const component = visualizationRegistry.getComponent(type);
        if (!component) return;

        const newVisualization: VisualizationInstance = {
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
            type,
            config: component.defaultConfig || {},
            position: { x: 0, y: 0, width: '100%', height: 280 }
        };

        const current = visualizationsRef.current;
        if (current.some(visualization => visualization.type === type)) {
            return;
        }

        applyVisualizations([...current, newVisualization]);
    }, [applyVisualizations]);

    // Remove a visualization
    const removeVisualization = useCallback((id: string) => {
        const updated = visualizationsRef.current.filter(v => v.id !== id);
        applyVisualizations(updated);
    }, [applyVisualizations]);

    const handleDragStart = useCallback((event: React.DragEvent, id: string) => {
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', id);
        setDraggingId(id);
    }, []);

    const handleDragOver = useCallback((event: React.DragEvent, id: string) => {
        event.preventDefault();
        if (id !== dropTargetId) {
            setDropTargetId(id);
        }
    }, [dropTargetId]);

    const handleDrop = useCallback((event: React.DragEvent, targetId: string) => {
        event.preventDefault();
        const sourceId = event.dataTransfer.getData('text/plain') || draggingId;
        if (sourceId) {
            reorderVisualizations(sourceId, targetId);
        }
        setDropTargetId(null);
        setDraggingId(null);
    }, [draggingId, reorderVisualizations]);

    const handleDragEnd = useCallback(() => {
        setDropTargetId(null);
        setDraggingId(null);
    }, []);

    const handleResizeStart = useCallback((event: React.PointerEvent, instance: VisualizationInstance) => {
        if (event.button !== 0) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const startHeight = getInstanceHeight(instance);
        resizeStateRef.current = {
            id: instance.id,
            pointerId: event.pointerId,
            startY: event.clientY,
            startHeight,
            currentHeight: startHeight
        };

        event.currentTarget.setPointerCapture(event.pointerId);
        setResizingId(instance.id);
    }, [getInstanceHeight]);

    const handleResizeMove = useCallback((event: React.PointerEvent) => {
        const resizeState = resizeStateRef.current;
        if (!resizeState || resizeState.pointerId !== event.pointerId) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const deltaY = event.clientY - resizeState.startY;
        const nextHeight = Math.max(
            MIN_CHART_HEIGHT,
            Math.min(MAX_CHART_HEIGHT, resizeState.startHeight + deltaY)
        );

        resizeState.currentHeight = nextHeight;

        const container = (event.currentTarget as HTMLElement).closest('.visualization-container') as HTMLElement | null;
        if (container) {
            container.style.height = `${nextHeight}px`;
        }
    }, []);

    const handleResizeEnd = useCallback((event: React.PointerEvent) => {
        const resizeState = resizeStateRef.current;
        if (!resizeState || resizeState.pointerId !== event.pointerId) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }

        const container = (event.currentTarget as HTMLElement).closest('.visualization-container') as HTMLElement | null;
        if (container) {
            container.style.height = '';
        }

        updateVisualizationHeight(resizeState.id, resizeState.currentHeight);
        resizeStateRef.current = null;
        setResizingId(null);
    }, [updateVisualizationHeight]);

    const columnCount = useMemo(() => {
        if (visualizations.length <= 1) return 1;
        if (visualizations.length <= 4) return 2;
        return 3;
    }, [visualizations.length]);

    // Render a single visualization
    const renderVisualization = (instance: VisualizationInstance) => {
        const component = visualizationRegistry.getComponent(instance.type);
        if (!component) return null;

        const Component = component.component;

        return (
            <Box
                key={instance.id}
                className={`visualization-container${draggingId === instance.id ? ' is-dragging' : ''}${dropTargetId === instance.id ? ' is-drop-target' : ''}${resizingId === instance.id ? ' is-resizing' : ''}`}
                style={{
                    height: `${getInstanceHeight(instance)}px`
                }}
                onDragOver={(event) => handleDragOver(event, instance.id)}
                onDrop={(event) => handleDrop(event, instance.id)}
            >
                <Box
                    className="visualization-header"
                    draggable
                    onDragStart={(event) => handleDragStart(event, instance.id)}
                    onDragEnd={handleDragEnd}
                >
                    <Flex align="center" gap="2">
                        <DragHandleDots2Icon className="drag-handle" />
                        <Text size="2" weight="medium">{component.name}</Text>
                    </Flex>
                    <IconButton
                        size="1"
                        variant="ghost"
                        onClick={() => removeVisualization(instance.id)}
                    >
                        <Cross2Icon />
                    </IconButton>
                </Box>
                <Box style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}>
                    <Component
                        id={instance.id}
                        data={instance.data}
                        config={instance.config}
                        width="100%"
                        height="100%"
                    />
                </Box>
                <Box
                    className="visualization-resize-handle"
                    onPointerDown={(event) => handleResizeStart(event, instance)}
                    onPointerMove={handleResizeMove}
                    onPointerUp={handleResizeEnd}
                    onPointerCancel={handleResizeEnd}
                />
            </Box>
        );
    };

    const availableTypes = useMemo(() => (
        visualizationRegistry.getAllTypes().filter(type =>
            !visualizations.some(visualization => visualization.type === type)
        )
    ), [visualizations]);

    return (
        <Box className="dynamic-visualization-manager">
            <Flex justify="between" align="center" className="manager-header">
                <Text size="3" weight="bold">Data Visualizations</Text>
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger>
                        <Button size="2" variant="soft" disabled={availableTypes.length === 0}>
                            <PlusIcon />
                            Add Visualization
                        </Button>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                        {availableTypes.map(type => {
                            const component = visualizationRegistry.getComponent(type);
                            return (
                                <DropdownMenu.Item
                                    key={type}
                                    onClick={() => addVisualization(type)}
                                >
                                    {component?.name || type}
                                </DropdownMenu.Item>
                            );
                        })}
                        {availableTypes.length === 0 && (
                            <DropdownMenu.Item disabled>
                                All visualizations already added
                            </DropdownMenu.Item>
                        )}
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            </Flex>

            <Box
                className="visualizations-container"
                style={{
                    minHeight: '150px',
                    gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))`
                }}
            >
                {visualizations.length === 0 ? (
                    <Box className="empty-state">
                        <Text color="gray">No visualizations added yet. Click "Add Visualization" to get started.</Text>
                    </Box>
                ) : (
                    visualizations.map(renderVisualization)
                )}
            </Box>
        </Box>
    );
};

export default DynamicVisualizationManager;
