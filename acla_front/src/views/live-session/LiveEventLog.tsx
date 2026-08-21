import React, {
    forwardRef,
    useCallback,
    useContext,
    useEffect,
    useImperativeHandle,
    useMemo,
    useRef,
    useState,
} from 'react';
import { Badge, Box, Flex, Table, Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { EventType, SessionEvent } from 'views/lap-analysis/session-intelligence/types';
import { NamedAiToolComponentHandle, useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import { runVisualizationBooleanCallback } from 'views/lap-analysis/visualization/visualization-component-callbacks';
import { ComponentDisableFailedError, VisualizationUpdateFailedError } from 'contexts/AiToolComponentError';
import { getTelemetryLap, getTelemetryTrack } from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import { LiveSessionContext } from './LiveSessionContext';
import { EventLog, EventSearchParams } from './event-log/EventLog';
import { SensorManager } from './event-log/SensorManager';

const EVENT_COLORS: Record<EventType, 'blue' | 'green' | 'red' | 'amber'> = {
    CORNER: 'blue',
    STRAIGHT: 'green',
    CRASHED: 'red',
    OVERTAKE: 'amber',
};
const EMPTY_EVENTS: SessionEvent[] = [];

export interface LiveEventLogHandle extends NamedAiToolComponentHandle {
    updateLiveEvents(events: SessionEvent[]): true;
    disableLiveEventLog(): true;
    findEvents(params: EventSearchParams): SessionEvent[];
    getAllEvents(): SessionEvent[];
}

interface LiveEventLogProps {
    name: string;
    initialEvents?: SessionEvent[];
    onUpdate?: (events: SessionEvent[]) => boolean;
    onDisable?: () => boolean;
}

const LiveEventLog = forwardRef<LiveEventLogHandle, LiveEventLogProps>(({
    name,
    initialEvents = EMPTY_EVENTS,
    onUpdate,
    onDisable,
}, forwardedRef) => {
    const { currentTelemetry, currentTelemetrySampleIndex } = useContext(LiveSessionContext);
    const [search, setSearch] = useState('');
    const [events, setEvents] = useState<SessionEvent[]>(() => initialEvents.slice());
    const eventLogRef = useRef(new EventLog(initialEvents));
    const sensorManagerRef = useRef(new SensorManager());
    const trackRef = useRef('');
    const currentLapRef = useRef(0);
    const lastSampleIndexRef = useRef(-1);
    const initialEventsRef = useRef(initialEvents);

    const resetTracking = useCallback(() => {
        eventLogRef.current.reset();
        sensorManagerRef.current.reset();
        trackRef.current = '';
        currentLapRef.current = 0;
        lastSampleIndexRef.current = -1;
        setEvents([]);
    }, []);

    useEffect(() => {
        if (initialEventsRef.current === initialEvents) return;
        initialEventsRef.current = initialEvents;
        eventLogRef.current.replace(initialEvents);
        setEvents(eventLogRef.current.all());
    }, [initialEvents]);

    useEffect(() => {
        const hasTelemetry = currentTelemetry && Object.keys(currentTelemetry).length > 0;
        if (!hasTelemetry || currentTelemetrySampleIndex < 0) {
            if (lastSampleIndexRef.current >= 0) resetTracking();
            return;
        }

        if (currentTelemetrySampleIndex === lastSampleIndexRef.current) return;
        if (currentTelemetrySampleIndex < lastSampleIndexRef.current) resetTracking();

        const track = getTelemetryTrack(currentTelemetry);
        if (track && track !== trackRef.current) {
            trackRef.current = track;
            sensorManagerRef.current.setTrack(track);
        }

        currentLapRef.current = getTelemetryLap(currentTelemetry);
        const eventCount = eventLogRef.current.length;
        sensorManagerRef.current.tick(currentTelemetry, currentTelemetrySampleIndex, eventLogRef.current);
        lastSampleIndexRef.current = currentTelemetrySampleIndex;
        if (eventLogRef.current.length !== eventCount) {
            setEvents(eventLogRef.current.all());
        }
    }, [currentTelemetry, currentTelemetrySampleIndex, resetTracking]);

    const handle = useMemo<LiveEventLogHandle>(() => ({
        getComponentName: () => name,
        updateLiveEvents: (data) => {
            const updated = runVisualizationBooleanCallback(
                name,
                VisualizationUpdateFailedError,
                `Failed to update chart '${name}'.`,
                onUpdate ? () => onUpdate(data) : undefined,
            );
            eventLogRef.current.replace(data);
            setEvents(eventLogRef.current.all());
            return updated;
        },
        disableLiveEventLog: () => runVisualizationBooleanCallback(
            name,
            ComponentDisableFailedError,
            `Component '${name}' could not be disabled.`,
            onDisable,
        ),
        findEvents: (params) => eventLogRef.current.find({
            ...params,
            currentLap: currentLapRef.current,
        }),
        getAllEvents: () => eventLogRef.current.all(),
    }), [name, onDisable, onUpdate]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = React.useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterAiToolComponentRef(registeredHandleRef);
    const filtered = useMemo(() => {
        const term = search.trim().toLowerCase();
        return events.filter((event) => !term || JSON.stringify(event).toLowerCase().includes(term)).slice().reverse();
    }, [events, search]);

    return (
        <Box className="live-optional-panel">
            <Flex justify="between" align="center" gap="2">
                <Text size="1" color="gray">{events.length} detected events</Text>
                <TextField.Root placeholder="Search events..." value={search} onChange={(event) => setSearch(event.target.value)}>
                    <TextField.Slot><MagnifyingGlassIcon /></TextField.Slot>
                </TextField.Root>
            </Flex>
            <Box className="live-optional-panel__scroll">
                {filtered.length === 0 ? <Text color="gray">No live events detected yet</Text> : (
                    <Table.Root size="1">
                        <Table.Header><Table.Row><Table.ColumnHeaderCell>Time</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Type</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Lap</Table.ColumnHeaderCell></Table.Row></Table.Header>
                        <Table.Body>
                            {filtered.map((event) => (
                                <Table.Row key={event.id}>
                                    <Table.Cell>{new Date(event.timestamp).toLocaleTimeString()}</Table.Cell>
                                    <Table.Cell><Badge color={EVENT_COLORS[event.type]}>{event.type}</Badge></Table.Cell>
                                    <Table.Cell>{event.lap}</Table.Cell>
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table.Root>
                )}
            </Box>
        </Box>
    );
});

LiveEventLog.displayName = 'LiveEventLog';

export default LiveEventLog;
