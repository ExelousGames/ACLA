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
import { LapRecorder, type DriverLapRecord } from './event-log/LapRecorder';
import { summarizeTimes } from './event-log/timing-summary';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';
import './LiveEventLog.css';

const EMPTY_EVENTS: LiveSessionEvent[] = [];

export interface LiveEventLogHandle extends NamedOperationComponentHandle {
    updateLiveEvents(events: LiveSessionEvent[]): true;
    disableLiveEventLog(): true;
    findEvents(params: EventSearchParams): LiveSessionEvent[];
    getAllEvents(): LiveSessionEvent[];
    getCornerRecords(): DriverCornerRecord[];
    getCornerSummaries(): DriverCornerSummary[];
    getLapRecords(): DriverLapRecord[];
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
    const cornerMapRef = useRef<CircuitMapDto | null>(null);
    const lapRecorderRef = useRef(new LapRecorder());
    const lapRecordsRef = useRef<DriverLapRecord[]>([]);
    const [lapRecords, setLapRecords] = useState<DriverLapRecord[]>([]);
    const lapCarsRef = useRef(new Map<string, boolean>());
    const [lapCars, setLapCars] = useState<Array<{ carId: string; isPlayer: boolean }>>([]);
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
        lapRecorderRef.current.reset();
        lapRecordsRef.current = [];
        lapCarsRef.current.clear();
        setLapRecords([]);
        setLapCars([]);
        setCornerRecords([]);
        setEvents([]);
    }, []);

    useEffect(() => {
        const updateMap = () => {
            cornerMapRef.current = liveMap?.getCircuitMap() ?? null;
            setMappedCornerCount(cornerRecorderRef.current.setMap(cornerMapRef.current));
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
                    lapRecorderRef.current.reset();
                }
                return;
            }
            if (event.sampleIndex <= lastSampleIndexRef.current) return;

            currentLapRef.current = getTelemetryLap(event.sample);
            lastSampleIndexRef.current = event.sampleIndex;
            const playerId = event.sample.Graphics_player_car_id;
            const playerKey = typeof playerId === 'number' && Number.isSafeInteger(playerId) && playerId >= 0 ? String(playerId) : 'player';
            let carsChanged = false;
            const observeCar = (carId: string, position: unknown) => {
                if (typeof position !== 'number' || !Number.isFinite(position) || position < 0 || position > 1) return;
                const isPlayer = carId === playerKey || lapCarsRef.current.get(carId) === true;
                if (lapCarsRef.current.get(carId) === isPlayer) return;
                lapCarsRef.current.set(carId, isPlayer);
                carsChanged = true;
            };
            Object.entries(event.sample.Graphics_normalized_positions ?? {}).forEach(([carId, position]) => observeCar(carId, position));
            observeCar(playerKey, event.sample.Graphics_normalized_car_position);
            if (carsChanged) setLapCars(Array.from(lapCarsRef.current, ([carId, isPlayer]) => ({ carId, isPlayer })));
            const completedLaps = lapRecorderRef.current.tick(event.sample, event.sampleIndex)
                .map((record) => ({ ...record, id: `${event.streamGeneration}:${record.id}` }));
            if (completedLaps.length > 0) {
                lapRecordsRef.current = [...lapRecordsRef.current, ...completedLaps];
                setLapRecords(lapRecordsRef.current);
            }
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
        getCornerSummaries: () => {
            const map = cornerMapRef.current;
            const corners = getCircuitMapCenterlineSegments(map);
            return summarizeDriverCorners(cornerRecordsRef.current.filter((record) => record.mapId === map?.id
                && corners.some((corner) => corner.id === record.cornerId && corner.end_position === record.exit.position
                    && (!record.entry || corner.start_position === record.entry.position))));
        },
        getLapRecords: () => lapRecordsRef.current.slice(),
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
    const lapSummaries = useMemo(() => {
        const recordsByCar = new Map<string, DriverLapRecord[]>();
        lapRecords.forEach((record) => {
            const records = recordsByCar.get(record.carId) ?? [];
            records.push(record);
            recordsByCar.set(record.carId, records);
        });
        return lapCars.map((car) => {
            const records = recordsByCar.get(car.carId) ?? [];
            return { ...car, ...summarizeTimes(records.map((record) => record.lapTimeMs)),
                lastLapTimeMs: records[records.length - 1]?.lapTimeMs ?? null };
        }).sort((a, b) => a.carId.localeCompare(b.carId, undefined, { numeric: true }));
    }, [lapCars, lapRecords]);
    const lapRows = useMemo(() => {
        const term = search.trim().toLowerCase();
        return lapSummaries.filter((car) => !term || `${car.isPlayer ? 'player' : ''} car ${car.carId} lap`.includes(term));
    }, [lapSummaries, search]);
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
                <section aria-label="Driver lap times" className="live-event-log__laps">
                    <Flex justify="between" align="center" gap="2" mb="2">
                        <Text weight="bold">Driver lap times</Text>
                        <Text size="1" color="gray">{lapRecords.length} completed laps</Text>
                    </Flex>
                    <Text as="p" size="1" color="gray">
                        Full laps with uninterrupted timing between finish-line crossings. Initial partial laps are excluded.
                        Average drops the fastest and slowest 12.5% (rounded down, at least one each).
                        Requires 3 complete laps. Times use MM:SS:mmm.
                    </Text>
                    {lapRows.length === 0 ? <Text color="gray">{search.trim() ? 'No matching lap times' : 'Waiting for car telemetry'}</Text> : (
                        <Table.Root size="1" className="live-event-log__cars">
                            <Table.Header><Table.Row>
                                <Table.ColumnHeaderCell>Car index</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Last lap</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Average lap time · middle 75%</Table.ColumnHeaderCell>
                            </Table.Row></Table.Header>
                            <Table.Body>{lapRows.map((car) => (
                                <Table.Row key={car.carId}>
                                    <Table.RowHeaderCell className="live-event-log__car">
                                        <Text weight="bold">{car.isPlayer ? 'Player' : 'Car'}{car.carId !== 'player' ? ` #${car.carId}` : ''}</Text>
                                    </Table.RowHeaderCell>
                                    <Table.Cell>{car.lastLapTimeMs === null ? 'Not recorded' : formatCornerTime(car.lastLapTimeMs)}</Table.Cell>
                                    <Table.Cell>
                                        <Text as="p" weight="bold">{car.averageTimeMs === null ? 'Collecting times' : formatCornerTime(car.averageTimeMs)}</Text>
                                        <Text as="p" size="1" color="gray">
                                            {car.averageTimeMs === null ? `${car.sampleCount}/3 complete laps` : `${car.includedCount} of ${car.sampleCount} laps used`}
                                        </Text>
                                    </Table.Cell>
                                </Table.Row>
                            ))}</Table.Body>
                        </Table.Root>
                    )}
                </section>
                <section aria-label="Driver corner times">
                    <Flex justify="between" align="center" gap="2" mb="2">
                        <Text weight="bold">Driver corner times</Text>
                        <Text size="1" color="gray">{cornerRecords.length} completed corners</Text>
                    </Flex>
                    <Text as="p" size="1" color="gray">
                        Average time in each corner across passes, using the middle 75% of complete times.
                        Drops the fastest and slowest 12.5% (rounded down, at least one each).
                        Requires 3 complete passes. Times use MM:SS:mmm.
                        Exit speed uses normalized lap position change per second at the corner end,
                        with the same trimming rule applied independently to exit observations.
                    </Text>
                    {mappedCornerCount === 0 && <Text color="gray">Open Live Map with tagged corners to record driver corner timing.</Text>}
                    {cornerRows.length === 0 ? (mappedCornerCount > 0 || search.trim()) && <Text color="gray">{search.trim() ? 'No matching corner times' : 'Waiting for cars to cross a corner end'}</Text> : (
                            <Table.Root size="1" className="live-event-log__cars">
                                <Table.Header><Table.Row>
                                    <Table.ColumnHeaderCell>Car index</Table.ColumnHeaderCell>
                                    <Table.ColumnHeaderCell>Average corner time and exit speed · middle 75%</Table.ColumnHeaderCell>
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
                                                        <Text as="p" size="1">
                                                            Exit speed: {corner.averageExitSpeed === null ? 'Collecting exits'
                                                                : `${(corner.averageExitSpeed * 100).toFixed(3)}% lap/s`}
                                                        </Text>
                                                        <Text as="p" size="1" color="gray">
                                                            {corner.averageExitSpeed === null
                                                                ? `${corner.exitSpeedSampleCount}/3 exit observations`
                                                                : `${corner.exitSpeedIncludedCount} of ${corner.exitSpeedSampleCount} exits used`}
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
