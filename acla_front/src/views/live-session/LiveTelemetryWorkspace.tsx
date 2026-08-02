import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Button, DropdownMenu, Flex, IconButton, Text } from '@radix-ui/themes';
import { Cross2Icon, DragHandleDots2Icon, PlusIcon } from '@radix-ui/react-icons';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationController';
import type { VisualizationInstance } from 'views/lap-analysis/visualization/VisualizationRegistry';
import AnalysisResultsChart from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import { LiveSessionContext } from './LiveSessionContext';
import { LiveVisualizationInstance } from './live-session-types';
import LiveTrajectoryMap from './LiveTrajectoryMap';
import LiveTelemetryOverview from './LiveTelemetryOverview';
import LiveEventLog from './LiveEventLog';
import 'views/lap-analysis/visualization/DynamicVisualizationManager.css';

const OPTIONAL_VISUALIZATIONS = {
    'telemetry-overview': { name: 'Live Telemetry Overview' },
    'event-log': { name: 'Live Event Log' },
    'analysis-results': { name: 'Analysis Results' },
} as const;

const MIN_HEIGHT = 180;
const MAX_HEIGHT = 900;

const LiveTelemetryWorkspace = () => {
    const liveSession = useContext(LiveSessionContext);
    const [visualizations, setVisualizations] = useState<LiveVisualizationInstance[]>([]);
    const [draggingId, setDraggingId] = useState<string | null>(null);
    const [dropTargetId, setDropTargetId] = useState<string | null>(null);
    const resizeRef = useRef<{ id: string; pointerId: number; startY: number; startHeight: number; height: number } | null>(null);

    useEffect(() => {
        const handleControllerUpdate = (instances: VisualizationInstance[]) => {
            const next = instances
                .filter((instance): instance is VisualizationInstance & { type: LiveVisualizationInstance['type'] } => (
                    Object.prototype.hasOwnProperty.call(OPTIONAL_VISUALIZATIONS, instance.type)
                ))
                .map((instance): LiveVisualizationInstance => ({
                    id: instance.id,
                    type: instance.type,
                    height: typeof instance.position?.height === 'number' ? instance.position.height : 280,
                    data: instance.data,
                }));
            setVisualizations(next);
        };

        visualizationController.setUpdateCallback(handleControllerUpdate);

        return () => {
            visualizationController.setUpdateCallback(() => { });
        };
    }, []);

    useEffect(() => {
        visualizationController.setCurrentInstances(visualizations.map((instance) => ({
            id: instance.id,
            type: instance.type,
            data: instance.data,
            position: { x: 0, y: 0, width: '100%', height: instance.height },
        })));
    }, [visualizations]);

    const addVisualization = useCallback((type: LiveVisualizationInstance['type']) => {
        setVisualizations((current) => current.some((item) => item.type === type)
            ? current
            : [...current, { id: `${type}-${Date.now()}`, type, height: 280 }]);
    }, []);

    const availableTypes = useMemo(() => (
        (Object.keys(OPTIONAL_VISUALIZATIONS) as LiveVisualizationInstance['type'][])
            .filter((type) => !visualizations.some((item) => item.type === type))
    ), [visualizations]);

    const removeVisualization = useCallback((id: string) => {
        setVisualizations((current) => current.filter((item) => item.id !== id));
    }, []);

    const reorder = useCallback((sourceId: string, targetId: string) => {
        setVisualizations((current) => {
            const sourceIndex = current.findIndex((item) => item.id === sourceId);
            const targetIndex = current.findIndex((item) => item.id === targetId);
            if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) return current;
            const next = [...current];
            const [moved] = next.splice(sourceIndex, 1);
            next.splice(targetIndex, 0, moved);
            return next;
        });
    }, []);

    const resizeStart = (event: React.PointerEvent, item: LiveVisualizationInstance) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        resizeRef.current = { id: item.id, pointerId: event.pointerId, startY: event.clientY, startHeight: item.height, height: item.height };
    };

    const resizeMove = (event: React.PointerEvent) => {
        const resize = resizeRef.current;
        if (!resize || resize.pointerId !== event.pointerId) return;
        resize.height = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, resize.startHeight + event.clientY - resize.startY));
        const container = (event.currentTarget as HTMLElement).closest('.visualization-container') as HTMLElement | null;
        if (container) container.style.height = `${resize.height}px`;
    };

    const resizeEnd = (event: React.PointerEvent) => {
        const resize = resizeRef.current;
        if (!resize || resize.pointerId !== event.pointerId) return;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        const container = (event.currentTarget as HTMLElement).closest('.visualization-container') as HTMLElement | null;
        if (container) container.style.height = '';
        setVisualizations((current) => current.map((item) => item.id === resize.id ? { ...item, height: resize.height } : item));
        resizeRef.current = null;
    };

    const renderPanel = (item: LiveVisualizationInstance) => {
        const events = liveSession.sessionIntelligence.getAllEvents();
        return (
            <Box
                key={item.id}
                className={`visualization-container${draggingId === item.id ? ' is-dragging' : ''}${dropTargetId === item.id ? ' is-drop-target' : ''}`}
                style={{ height: `${item.height}px` }}
                onDragOver={(event) => { event.preventDefault(); setDropTargetId(item.id); }}
                onDrop={(event) => {
                    event.preventDefault();
                    const sourceId = event.dataTransfer.getData('text/plain') || draggingId;
                    if (sourceId) reorder(sourceId, item.id);
                    setDraggingId(null);
                    setDropTargetId(null);
                }}
            >
                <Box
                    className="visualization-header"
                    draggable
                    onDragStart={(event) => {
                        event.dataTransfer.setData('text/plain', item.id);
                        setDraggingId(item.id);
                    }}
                    onDragEnd={() => { setDraggingId(null); setDropTargetId(null); }}
                >
                    <Flex align="center" gap="2"><DragHandleDots2Icon className="drag-handle" /><Text size="2">{OPTIONAL_VISUALIZATIONS[item.type].name}</Text></Flex>
                    <IconButton size="1" variant="ghost" onClick={() => removeVisualization(item.id)} aria-label={`Remove ${OPTIONAL_VISUALIZATIONS[item.type].name}`}><Cross2Icon /></IconButton>
                </Box>
                <Box
                    data-testid={`live-visualization-content-${item.type}`}
                    style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}
                >
                    {item.type === 'telemetry-overview'
                        ? <LiveTelemetryOverview telemetry={liveSession.currentTelemetry} />
                        : item.type === 'event-log'
                            ? <LiveEventLog events={events} />
                            : <AnalysisResultsChart
                                id={item.id}
                                data={item.data}
                                width="100%"
                                height="100%"
                                showElementId={false}
                            />}
                </Box>
                <Box className="visualization-resize-handle" onPointerDown={(event) => resizeStart(event, item)} onPointerMove={resizeMove} onPointerUp={resizeEnd} onPointerCancel={resizeEnd} />
            </Box>
        );
    };

    return (
        <Box className="dynamic-visualization-manager live-telemetry-workspace" data-testid="live-telemetry-workspace">
            <Flex justify="between" align="center" className="manager-header">
                <Text size="3" weight="bold">Live Data Visualizations</Text>
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger>
                        <Button size="2" variant="soft" disabled={availableTypes.length === 0}><PlusIcon />Add Visualization</Button>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                        {availableTypes.map((type) => (
                            <DropdownMenu.Item key={type} onClick={() => addVisualization(type)}>{OPTIONAL_VISUALIZATIONS[type].name}</DropdownMenu.Item>
                        ))}
                        {availableTypes.length === 0 ? <DropdownMenu.Item disabled>All visualizations already added</DropdownMenu.Item> : null}
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            </Flex>
            <Box className={`visualization-workspace${visualizations.length === 0 ? ' visualization-workspace--map-only' : ''}`}>
                <Box className="static-map-container">
                    <Box className="static-map-header"><Text size="2" weight="medium">Live 2D Telemetry Trajectory</Text></Box>
                    <Box className="static-map-body"><LiveTrajectoryMap /></Box>
                </Box>
                {visualizations.length > 0 ? <Box className="visualizations-container">{visualizations.map(renderPanel)}</Box> : null}
            </Box>
        </Box>
    );
};

export default LiveTelemetryWorkspace;
