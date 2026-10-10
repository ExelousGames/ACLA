import React, {
    forwardRef,
    useCallback,
    useEffect,
    useImperativeHandle,
    useMemo,
    useRef,
    useState,
} from 'react';
import { Badge, Box, Flex, Table, Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { LiveSessionEvent } from 'views/session-shared/session-intelligence/types';
import { NamedOperationComponentHandle, useOptionalOperationComponentRefDirectory, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { runVisualizationBooleanCallback } from 'views/session-shared/visualization/visualization-component-callbacks';
import { ComponentDisableFailedError, VisualizationUpdateFailedError } from 'contexts/OperationComponentError';
import { getTelemetryLap } from 'views/live-session/session-intelligence/live-performance-analyst';
import { EventLog, EventSearchParams } from './event-log/EventLog';
import { liveTelemetryStore } from './live-telemetry-store';
import type { LiveTrajectoryMapHandle } from './LiveTrajectoryMap';
import { getVisualizationComponentName } from 'views/session-shared/visualization/visualization-component-names';
import { CornerRecorder, DriverCornerRecord, formatCornerTime } from './event-log/CornerRecorder';
import { DriverCornerSummary, summarizeDriverCorners } from './event-log/corner-summary';
import './LiveEventLog.css';

const EMPTY_EVENTS: LiveSessionEvent[] = [];

export interface LiveEventLogHandle extends NamedOperationComponentHandle {
    updateLiveEvents(events: LiveSessionEvent[]): true;
    disableLiveEventLog(): true;
    findEvents(params: EventSearchParams): LiveSessionEvent[];
    getAllEvents(): LiveSessionEvent[];
    getCornerRecords(): DriverCornerRecord[];
}

interface LiveEventLogProps {
    name: string;
    initialEvents?: LiveSessionEvent[];
    onUpdate?: (events: LiveSessionEvent[]) => boolean;
    onDisable?: () => boolean;
}

const LiveEventLog = forwardRef<LiveEventLogHandle, LiveEventLogProps>(({
    name,
    initialEvents = EMPTY_EVENTS,
    onUpdate,
    onDisable,
}, forwardedRef) => {
    const directory = useOptionalOperationComponentRefDirectory();
    const liveMap = directory?.findComponentRef<LiveTrajectoryMapHandle>(getVisualizationComponentName('live-trajectory-map'))?.current ?? null;
    const [search, setSearch] = useState('');
    const cornerRecorderRef = useRef(new CornerRecorder());
    const cornerRecordsRef = useRef<DriverCornerRecord[]>([]);
    const [cornerRecords, setCornerRecords] = useState<DriverCornerRecord[]>([]);
    const [mappedCornerCount, setMappedCornerCount] = useState(0);
    const eventLogRef = useRef(new EventLog(initialEvents));
    const [events, setEvents] = useState<LiveSessionEvent[]>(() => eventLogRef.current.all());
    const currentLapRef = useRef(0);
    const lastSampleIndexRef = useRef(-1);
    const initialEventsRef = useRef(initialEvents);

    const resetTracking = useCallback(() => {
        eventLogRef.current.reset();
        currentLapRef.current = 0;
        lastSampleIndexRef.current = -1;
        cornerRecorderRef.current.reset();
        cornerRecordsRef.current = [];
        setCornerRecords([]);
        setEvents([]);
    }, []);

    useEffect(() => {
        const updateMap = () => {
            setMappedCornerCount(cornerRecorderRef.current.setMap(liveMap?.getCircuitMap() ?? null));
        };
        updateMap();
        return liveMap?.subscribeCircuitMap(updateMap);
    }, [liveMap]);

    useEffect(() => {
        if (initialEventsRef.current === initialEvents) return;
        initialEventsRef.current = initialEvents;
        eventLogRef.current.replace(initialEvents);
        setEvents(eventLogRef.current.all());
    }, [initialEvents]);

    useEffect(() => {
        return liveTelemetryStore.subscribeEvents((event) => {
            if (event.type !== 'frame') {
                if (event.type === 'session-reset') resetTracking();
                else {
                    currentLapRef.current = 0;
                    lastSampleIndexRef.current = -1;
                    cornerRecorderRef.current.reset();
                }
                return;
            }
            if (event.sampleIndex <= lastSampleIndexRef.current) return;

            currentLapRef.current = getTelemetryLap(event.sample);
            lastSampleIndexRef.current = event.sampleIndex;
            const completedCorners = cornerRecorderRef.current.tick(event.sample, event.sampleIndex)
                .map((record) => ({ ...record, id: `${event.streamGeneration}:${record.id}` }));
            if (completedCorners.length > 0) {
                cornerRecordsRef.current = [...cornerRecordsRef.current, ...completedCorners];
                setCornerRecords(cornerRecordsRef.current);
            }
        }, { replayLatest: true });
    }, [resetTracking]);

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
        getCornerRecords: () => cornerRecordsRef.current.slice(),
    }), [name, onDisable, onUpdate]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = React.useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterOperationComponentRef(registeredHandleRef);
    const filtered = useMemo(() => {
        const term = search.trim().toLowerCase();
        return events.filter((event) => !term || JSON.stringify(event).toLowerCase().includes(term)).slice().reverse();
    }, [events, search]);
    const cornerSummaries = useMemo(() => summarizeDriverCorners(cornerRecords), [cornerRecords]);
    const cornerRows = useMemo(() => {
        const term = search.trim().toLowerCase();
        const cars = new Map<string, DriverCornerSummary[]>();
        cornerSummaries.forEach((summary) => {
            if (term && !`${summary.isPlayer ? 'player' : ''} car ${summary.carId} ${summary.cornerName} ${summary.cornerId}`.toLowerCase().includes(term)) return;
            const corners = cars.get(summary.carId) ?? [];
            corners.push(summary);
            cars.set(summary.carId, corners);
        });
        return Array.from(cars, ([carId, corners]) => ({ carId,
            corners: corners.sort((a, b) => a.cornerName.localeCompare(b.cornerName, undefined, { numeric: true })),
        }))
            .sort((a, b) => a.carId.localeCompare(b.carId, undefined, { numeric: true }));
    }, [cornerSummaries, search]);

    return (
        <Box className="live-optional-panel">
            <Flex justify="between" align="center" gap="2">
                <Text size="1" color="gray">{events.length} detected events</Text>
                <TextField.Root placeholder="Search events..." value={search} onChange={(event) => setSearch(event.target.value)}>
                    <TextField.Slot><MagnifyingGlassIcon /></TextField.Slot>
                </TextField.Root>
            </Flex>
            <Box className="live-optional-panel__scroll">
                <section aria-label="Driver corner times">
                    <Flex justify="between" align="center" gap="2" mb="2">
                        <Text weight="bold">Driver corner times</Text>
                        <Text size="1" color="gray">{cornerRecords.length} completed corners</Text>
                    </Flex>
                    <Text as="p" size="1" color="gray">
                        Average time in each corner across passes, using the middle 75% of complete times.
                        Drops the fastest and slowest 12.5% (rounded down, at least one each).
                        Requires 3 complete passes. Times use MM:SS:mmm.
                    </Text>
                    {mappedCornerCount === 0 && <Text color="gray">Open Live Map with tagged corners to record driver corner timing.</Text>}
                    {cornerRows.length === 0 ? (mappedCornerCount > 0 || search.trim()) && <Text color="gray">{search.trim() ? 'No matching corner times' : 'Waiting for cars to cross a corner end'}</Text> : (
                            <Table.Root size="1" className="live-event-log__cars">
                                <Table.Header><Table.Row>
                                    <Table.ColumnHeaderCell>Car index</Table.ColumnHeaderCell>
                                    <Table.ColumnHeaderCell>Average corner time · middle 75%</Table.ColumnHeaderCell>
                                </Table.Row></Table.Header>
                                <Table.Body>{cornerRows.map(({ carId, corners }) => (
                                    <Table.Row key={carId}>
                                        <Table.RowHeaderCell className="live-event-log__car">
                                            <Text weight="bold">{corners.some((corner) => corner.isPlayer) ? 'Player' : 'Car'}{carId !== 'player' ? ` #${carId}` : ''}</Text>
                                        </Table.RowHeaderCell>
                                        <Table.Cell>
                                            <ol className="live-event-log__corners" aria-label={`Corner times for car ${carId}`}>
                                                {corners.map((corner) => (
                                                    <li key={corner.cornerId} className="live-event-log__corner">
                                                        <Text weight="bold"><span title={corner.cornerId}>{corner.cornerName}</span></Text>
                                                        <Text as="p" weight="bold">
                                                            {corner.averageTimeMs === null ? 'Collecting times' : formatCornerTime(corner.averageTimeMs)}
                                                        </Text>
                                                        <Text as="p" size="1" color="gray">
                                                            {corner.averageTimeMs === null
                                                                ? `${corner.sampleCount}/3 complete passes`
                                                                : `${corner.includedCount} of ${corner.sampleCount} times used`}
                                                        </Text>
                                                    </li>
                                                ))}
                                            </ol>
                                        </Table.Cell>
                                    </Table.Row>
                                ))}</Table.Body>
                            </Table.Root>
                        )}
                </section>
                <Box mt="4">
                {filtered.length === 0 ? <Text color="gray">No live events detected yet</Text> : (
                    <Table.Root size="1">
                        <Table.Header><Table.Row><Table.ColumnHeaderCell>Time</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Type</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Lap</Table.ColumnHeaderCell></Table.Row></Table.Header>
                        <Table.Body>
                            {filtered.map((event) => (
                                <Table.Row key={event.id}>
                                    <Table.Cell>{new Date(event.timestamp).toLocaleTimeString()}</Table.Cell>
                                    <Table.Cell><Badge color="green">{event.type}</Badge></Table.Cell>
                                    <Table.Cell>{event.lap}</Table.Cell>
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table.Root>
                )}
                </Box>
            </Box>
        </Box>
    );
});

LiveEventLog.displayName = 'LiveEventLog';

export default LiveEventLog;
