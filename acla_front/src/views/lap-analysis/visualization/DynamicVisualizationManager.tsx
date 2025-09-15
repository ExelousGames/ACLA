import React, { useState, useEffect, useCallback, useContext } from 'react';
import { Box, Button, DropdownMenu, IconButton, Flex, Text } from '@radix-ui/themes';
import { PlusIcon, Cross2Icon, DragHandleDots2Icon } from '@radix-ui/react-icons';
import { visualizationRegistry, VisualizationInstance } from './VisualizationRegistry';
import { visualizationController } from './VisualizationController';
import { AnalysisContext } from '../session-analysis';
import './DynamicVisualizationManager.css';

interface DynamicVisualizationManagerProps {
    onLayoutChange?: (instances: VisualizationInstance[]) => void;
}

const DynamicVisualizationManager: React.FC<DynamicVisualizationManagerProps> = ({
    onLayoutChange
}) => {
    const [visualizations, setVisualizations] = useState<VisualizationInstance[]>([]);
    const [isInitialized, setIsInitialized] = useState(false);
    const analysisContext = useContext(AnalysisContext);

    // Setup controller callback on mount
    useEffect(() => {
        const handleControllerUpdate = (instances: VisualizationInstance[]) => {
            setVisualizations(instances);
        };

        visualizationController.setUpdateCallback(handleControllerUpdate);
        setIsInitialized(true);

        return () => {
            visualizationController.setUpdateCallback(() => { });
        };
    }, []);

    // Calculate optimal layout based on available space
    const calculateOptimalLayout = useCallback((count: number, containerWidth: number, containerHeight: number, hasMap: boolean) => {
        const padding = 12; // Increased padding for better separation
        const headerHeight = 40;
        const containerPadding = 6; // Account for container padding from CSS
        const availableWidth = containerWidth - (containerPadding * 2) - (padding * 2);
        const availableHeight = containerHeight - headerHeight - (containerPadding * 2) - padding;

        if (count === 1) {
            return [{
                x: containerPadding + padding,
                y: containerPadding + padding,
                width: availableWidth,
                height: Math.min(availableHeight - 20, hasMap ? 350 : 300)
            }];
        }

        if (count === 2) {
            const cellWidth = Math.floor((availableWidth - padding) / 2);
            const cellHeight = Math.min(availableHeight - 20, hasMap ? 280 : 250);

            return [
                { x: containerPadding + padding, y: containerPadding + padding, width: cellWidth, height: cellHeight },
                { x: containerPadding + padding + cellWidth + padding, y: containerPadding + padding, width: cellWidth, height: cellHeight }
            ];
        }

        if (count === 3) {
            const topHeight = Math.min(availableHeight * 0.5, hasMap ? 200 : 180);
            const bottomHeight = Math.min(availableHeight - topHeight - (padding * 3), hasMap ? 180 : 160);
            const bottomCellWidth = Math.floor((availableWidth - padding) / 2);

            return [
                { x: containerPadding + padding, y: containerPadding + padding, width: availableWidth, height: topHeight },
                { x: containerPadding + padding, y: containerPadding + padding + topHeight + padding, width: bottomCellWidth, height: bottomHeight },
                { x: containerPadding + padding + bottomCellWidth + padding, y: containerPadding + padding + topHeight + padding, width: bottomCellWidth, height: bottomHeight }
            ];
        }

        if (count === 4) {
            const cellWidth = Math.floor((availableWidth - padding) / 2);
            const cellHeight = Math.floor((availableHeight - padding) / 2);
            const actualCellHeight = Math.min(cellHeight, hasMap ? 180 : 160);

            return [
                { x: containerPadding + padding, y: containerPadding + padding, width: cellWidth, height: actualCellHeight },
                { x: containerPadding + padding + cellWidth + padding, y: containerPadding + padding, width: cellWidth, height: actualCellHeight },
                { x: containerPadding + padding, y: containerPadding + padding + actualCellHeight + padding, width: cellWidth, height: actualCellHeight },
                { x: containerPadding + padding + cellWidth + padding, y: containerPadding + padding + actualCellHeight + padding, width: cellWidth, height: actualCellHeight }
            ];
        }

        // For 5+ items, calculate optimal grid
        const cols = Math.min(3, Math.ceil(Math.sqrt(count))); // Max 3 columns to maintain readability
        const rows = Math.ceil(count / cols);
        const cellWidth = Math.floor((availableWidth - (padding * (cols - 1))) / cols);
        const cellHeight = Math.min(Math.floor((availableHeight - (padding * (rows - 1))) / rows), hasMap ? 160 : 140);

        const layout = [];
        for (let i = 0; i < count; i++) {
            const row = Math.floor(i / cols);
            const col = i % cols;

            layout.push({
                x: containerPadding + padding + (col * (cellWidth + padding)),
                y: containerPadding + padding + (row * (cellHeight + padding)),
                width: cellWidth,
                height: cellHeight
            });
        }

        return layout;
    }, []);

    // Auto-correct layout when visualizations change
    const autoCorrectLayout = useCallback((instances: VisualizationInstance[]) => {
        if (instances.length === 0) return instances;

        const correctedInstances = [...instances];

        // Use a more reliable way to get container dimensions with fallbacks
        const getContainerDimensions = () => {
            const container = document.querySelector('.visualizations-container');
            if (container) {
                const rect = container.getBoundingClientRect();
                const computedStyle = window.getComputedStyle(container);
                const paddingLeft = parseInt(computedStyle.paddingLeft) || 0;
                const paddingRight = parseInt(computedStyle.paddingRight) || 0;
                const paddingTop = parseInt(computedStyle.paddingTop) || 0;
                const paddingBottom = parseInt(computedStyle.paddingBottom) || 0;

                return {
                    width: Math.max(rect.width - paddingLeft - paddingRight, container.clientWidth - paddingLeft - paddingRight, 800),
                    height: Math.max(rect.height - paddingTop - paddingBottom, container.clientHeight - paddingTop - paddingBottom, 600)
                };
            }
            return { width: 800, height: 600 }; // Increased fallback dimensions
        };

        const { width: containerWidth, height: containerHeight } = getContainerDimensions();

        // Check if any visualization is a map (needs more square space)
        const hasMap = instances.some(instance => instance.type === 'map-visualization');

        // Calculate optimal layout based on available space and content
        const layoutConfig = calculateOptimalLayout(instances.length, containerWidth, containerHeight, hasMap);

        correctedInstances.forEach((instance, index) => {
            const config = layoutConfig[index];
            if (config) {
                instance.position = {
                    x: config.x,
                    y: config.y,
                    width: config.width,
                    height: config.height
                };
            }
        });

        return correctedInstances;
    }, [calculateOptimalLayout]); // Add dependency to ensure it's available

    // Apply layout corrections when visualizations change (fallback for edge cases)
    useEffect(() => {
        if (visualizations.length > 0 && isInitialized) {
            // Only apply layout correction if positions are invalid or missing
            const needsCorrection = visualizations.some(v =>
                !v.position ||
                v.position.x === 0 && v.position.y === 0 &&
                visualizations.filter(viz => viz.position?.x === 0 && viz.position?.y === 0).length > 1
            );

            if (needsCorrection) {
                const layoutTimeout = setTimeout(() => {
                    const corrected = autoCorrectLayout(visualizations);
                    if (JSON.stringify(corrected) !== JSON.stringify(visualizations)) {
                        setVisualizations(corrected);
                        onLayoutChange?.(corrected);
                    }
                }, 100);

                return () => clearTimeout(layoutTimeout);
            }
        }
    }, [visualizations, isInitialized, autoCorrectLayout, onLayoutChange]); // Watch for changes that need correction

    // Calculate the required container height based on visualizations
    const calculateRequiredHeight = useCallback((instances: VisualizationInstance[]) => {
        if (instances.length === 0) return 150;

        let maxY = 0;
        instances.forEach(instance => {
            if (instance.position) {
                const y = typeof instance.position.y === 'number' ? instance.position.y : 0;
                const height = typeof instance.position.height === 'number' ? instance.position.height : 200;
                maxY = Math.max(maxY, y + height);
            }
        });

        return Math.max(maxY + 20, 200); // Add padding and minimum height
    }, []);

    // Add a new visualization
    const addVisualization = useCallback((type: string) => {
        const component = visualizationRegistry.getComponent(type);
        if (!component) return;

        const newVisualization: VisualizationInstance = {
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
            type,
            config: component.defaultConfig || {},
            position: { x: 0, y: 0, width: 300, height: 300 }
        };

        // Add the new visualization and immediately apply layout correction
        setVisualizations(prev => {
            const updated = [...prev, newVisualization];
            // Apply layout correction immediately to prevent overlapping
            const corrected = autoCorrectLayout(updated);
            return corrected;
        });
    }, [autoCorrectLayout]);

    // Remove a visualization
    const removeVisualization = useCallback((id: string) => {
        setVisualizations(prev => {
            const updated = prev.filter(v => v.id !== id);
            // Apply layout correction immediately after removal
            const corrected = autoCorrectLayout(updated);
            return corrected;
        });
    }, [autoCorrectLayout]);

    // Auto-correct layout when context data changes (less frequent)
    useEffect(() => {
        if (visualizations.length > 0) {
            const timeoutId = setTimeout(() => {
                const corrected = autoCorrectLayout(visualizations);
                setVisualizations(corrected);
            }, 300); // Debounce context changes

            return () => clearTimeout(timeoutId);
        }
    }, [analysisContext.recordedSessionDataFilePath, analysisContext.liveData]); // Watch file path and live data instead

    // Handle resize events with proper debouncing
    useEffect(() => {
        let resizeTimeout: NodeJS.Timeout;

        const handleResize = () => {
            if (visualizations.length > 0) {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(() => {
                    const corrected = autoCorrectLayout(visualizations);
                    setVisualizations(corrected);
                }, 500); // Longer debounce for resize events
            }
        };

        // Use both ResizeObserver and window resize for better coverage
        const containerElement = document.querySelector('.visualizations-container');
        let resizeObserver: ResizeObserver | null = null;

        if (containerElement && 'ResizeObserver' in window) {
            resizeObserver = new ResizeObserver(handleResize);
            resizeObserver.observe(containerElement);
        }

        window.addEventListener('resize', handleResize);

        return () => {
            clearTimeout(resizeTimeout);
            window.removeEventListener('resize', handleResize);
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
        };
    }, [visualizations.length, autoCorrectLayout]); // Only re-setup when count changes

    // Render a single visualization
    const renderVisualization = (instance: VisualizationInstance) => {
        const component = visualizationRegistry.getComponent(instance.type);
        if (!component) return null;

        const Component = component.component;
        const position = instance.position || { x: 0, y: 0, width: 300, height: 300 };

        return (
            <Box
                key={instance.id}
                className="visualization-container"
                style={{
                    position: 'absolute',
                    left: typeof position.x === 'number' ? `${position.x}px` : position.x,
                    top: typeof position.y === 'number' ? `${position.y}px` : position.y,
                    width: typeof position.width === 'number' ? `${position.width}px` : position.width,
                    height: typeof position.height === 'number' ? `${position.height}px` : position.height,
                    zIndex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                    minHeight: 0
                }}
            >
                <Box className="visualization-header" style={{ flex: '0 0 auto' }}>
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
            </Box>
        );
    };

    return (
        <Box className="dynamic-visualization-manager">
            <Flex justify="between" align="center" className="manager-header">
                <Text size="3" weight="bold">Data Visualizations</Text>
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger>
                        <Button size="2" variant="soft">
                            <PlusIcon />
                            Add Visualization
                        </Button>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                        {visualizationRegistry.getAllTypes().map(type => {
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
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            </Flex>

            <Box
                className="visualizations-container"
                style={{
                    position: 'relative',
                    height: visualizations.length > 0 ? `${calculateRequiredHeight(visualizations)}px` : '100%',
                    minHeight: '150px',
                    maxHeight: '90vh' // Prevent overflow
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
