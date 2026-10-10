import React, { useLayoutEffect } from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import LiveEventLog, { LiveEventLogHandle } from '../LiveEventLog';
import { liveTelemetryStore } from '../live-telemetry-store';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { formatCornerTime } from '../event-log/CornerRecorder';
import type { LiveSessionEvent, SessionEvent } from 'views/session-shared/session-intelligence/types';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = ({ children, ...props }: any) => <div {...props}>{children}</div>;
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        Box: Div,
        Flex: Div,
        Table: {
            Root: ({ children }: any) => <table>{children}</table>,
            Header: ({ children }: any) => <thead>{children}</thead>,
            Body: ({ children }: any) => <tbody>{children}</tbody>,
            Row: ({ children }: any) => <tr>{children}</tr>,
            ColumnHeaderCell: ({ children }: any) => <th>{children}</th>,
            RowHeaderCell: ({ children }: any) => <th scope="row">{children}</th>,
            Cell: ({ children }: any) => <td>{children}</td>,
        },
        Text: ({ children }: any) => <span>{children}</span>,
        TextField: {
            Root: ({ children, ...props }: any) => <div><input {...props} />{children}</div>,
            Slot: Div,
        },
    };
});

const cornerMap: CircuitMapDto = {
    id: 'test-track', game: 'acc', circuit_name: 'Test circuit', resolution: 1000, samples: {},
    centerline_segments: [{ id: 'turn-1', tags: ['corner'], start_position: 0.2, end_position: 0.3 }],
};

const MapHarness = ({ map }: { map: CircuitMapDto | null }) => {
    const mapRef = React.useRef(map);
    mapRef.current = map;
    const listeners = React.useRef(new Set<() => void>());
    const handle = React.useRef({
        getComponentName: () => 'visualization:live-trajectory-map',
        getCircuitMap: () => mapRef.current,
        subscribeCircuitMap: (listener: () => void) => {
            listeners.current.add(listener);
            return () => { listeners.current.delete(listener); };
        },
    });
    useRegisterOperationComponentRef(handle);
    React.useEffect(() => { listeners.current.forEach((listener) => listener()); }, [map]);
    return null;
};

jest.mock('@radix-ui/react-icons', () => ({
    MagnifyingGlassIcon: () => <span>Search</span>,
}));

const telemetry = (speed: number) => ({
    Static_track: 'brands_hatch',
    Graphics_completed_lap: 2,
    Graphics_normalized_car_position: 0.4,
    Physics_speed_kmh: speed,
});

const Harness = ({
    open,
    speed,
    sampleIndex,
    eventLogRef,
}: {
    open: boolean;
    speed: number;
    sampleIndex: number;
    eventLogRef: React.RefObject<LiveEventLogHandle | null>;
}) => {
    useLayoutEffect(() => {
        liveTelemetryStore.publishFrame({
            type: 'frame',
            game: 'acc',
            sample: telemetry(speed),
            sequence: sampleIndex + 1,
            committedSequence: sampleIndex + 1,
            committedCount: sampleIndex + 1,
        }, { Static_track: 'brands_hatch' });
    }, [sampleIndex, speed]);
    return open ? <LiveEventLog ref={eventLogRef} name="visualization:event-log" /> : null;
};

describe('LiveEventLog telemetry ownership', () => {
    beforeEach(() => liveTelemetryStore.resetSession());

    it.each(['initial events', 'prop update', 'handle update'])('excludes retired event types from %s, counts, and searches', (source) => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const events: SessionEvent[] = ['CORNER', 'CRASHED', 'OVERTAKE', 'STRAIGHT'].map((type, index) => ({
            id: `event-${index}`, type: type as SessionEvent['type'], startSampleIdx: index,
            endSampleIdx: index, lap: 2, trackPosition: 0.4, timestamp: index,
        }));
        // External visualization data can still contain event types from older sessions.
        const incoming = events as LiveSessionEvent[];
        const { rerender } = render(<LiveEventLog ref={eventLogRef} name="visualization:event-log"
            initialEvents={source === 'initial events' ? incoming : []} onUpdate={() => true} />);
        if (source === 'prop update') {
            rerender(<LiveEventLog ref={eventLogRef} name="visualization:event-log" initialEvents={incoming} />);
        } else if (source === 'handle update') {
            act(() => { eventLogRef.current!.updateLiveEvents(incoming); });
        }

        expect(screen.getByText('1 detected events')).toBeInTheDocument();
        expect(screen.getByText('STRAIGHT')).toBeInTheDocument();
        expect(eventLogRef.current!.getAllEvents()).toEqual([events[3]]);
        for (const eventType of ['CORNER', 'CRASHED', 'OVERTAKE']) {
            expect(screen.queryByText(eventType)).not.toBeInTheDocument();
            expect(eventLogRef.current!.findEvents({ eventType, scope: 'all' } as any)).toEqual([]);
        }
        expect(eventLogRef.current!.findEvents({ eventType: 'STRAIGHT', scope: 'last' })).toEqual([events[3]]);
        act(() => { liveTelemetryStore.resetSession(); });
        expect(eventLogRef.current!.getAllEvents()).toEqual([]);
        expect(screen.getByText('0 detected events')).toBeInTheDocument();
    });

    it('does not log possible crashes from live telemetry when mounted or reopened', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const { rerender } = render(
            <Harness open speed={120} sampleIndex={10} eventLogRef={eventLogRef} />,
        );

        expect(screen.getByText('0 detected events')).toBeInTheDocument();

        rerender(<Harness open speed={50} sampleIndex={11} eventLogRef={eventLogRef} />);
        expect(screen.getByText('0 detected events')).toBeInTheDocument();
        expect(screen.queryByText('CRASHED')).not.toBeInTheDocument();
        expect(eventLogRef.current?.getAllEvents()).toEqual([]);

        rerender(<Harness open={false} speed={120} sampleIndex={12} eventLogRef={eventLogRef} />);
        expect(screen.queryByText(/detected events/)).not.toBeInTheDocument();

        rerender(<Harness open speed={50} sampleIndex={13} eventLogRef={eventLogRef} />);
        expect(screen.getByText('0 detected events')).toBeInTheDocument();
        expect(screen.queryByText('CRASHED')).not.toBeInTheDocument();
        expect(eventLogRef.current?.getAllEvents()).toEqual([]);
    });

    it('records each car after exit using the live map and the telemetry clock, and filters the section', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        render(<OperationComponentRefProvider>
            <MapHarness map={cornerMap} />
            <LiveEventLog ref={eventLogRef} name="visualization:event-log" />
        </OperationComponentRefProvider>);
        const section = within(screen.getByRole('region', { name: 'Driver corner times' }));
        expect(section.getByText('Waiting for cars to cross a corner end')).toBeInTheDocument();
        const positions = [0.1, 0.13, 0.16, 0.185, 0.205, 0.22, 0.23, 0.24, 0.255, 0.275, 0.3];
        positions.forEach((position, index) => {
            act(() => {
                liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: index + 1,
                    committedSequence: index + 1, committedCount: index + 1,
                    sample: { Graphics_current_time_str: formatCornerTime(2027758 + index * 1000),
                        Graphics_player_car_id: 7, Graphics_normalized_positions: { '7': position, '63': position } },
                });
            });
            if (index < positions.length - 1) expect(section.queryByRole('table')).not.toBeInTheDocument();
        });
        expect(section.getByText('2 completed corners')).toBeInTheDocument();
        expect(section.getByText('Player #7')).toBeInTheDocument();
        expect(section.getByText('Car #63')).toBeInTheDocument();
        expect(section.getAllByText('Collecting times')).toHaveLength(2);
        expect(section.getAllByText('1/3 complete passes')).toHaveLength(2);
        for (const label of ['Braking point', 'Acceleration start', 'Corner entry', 'Corner exit', 'Deceleration to exit']) {
            expect(section.queryByText(label)).not.toBeInTheDocument();
        }
        expect(section.queryByText('33:57:758')).not.toBeInTheDocument();
        expect(eventLogRef.current?.getCornerRecords()).toHaveLength(2);
        fireEvent.change(screen.getByPlaceholderText('Search events...'), { target: { value: '63' } });
        expect(section.queryByText('Player #7')).not.toBeInTheDocument();
        expect(section.getByText('Car #63')).toBeInTheDocument();
        act(() => { liveTelemetryStore.beginStream(); });
        expect(eventLogRef.current?.getCornerRecords()).toHaveLength(2);
        expect(section.getByText('Car #63')).toBeInTheDocument();
        act(() => { liveTelemetryStore.resetSession(); });
        expect(eventLogRef.current?.getCornerRecords()).toEqual([]);
        expect(section.getByText('0 completed corners')).toBeInTheDocument();
    });

    it('shows one average per car and corner across laps, missing cars, and searches', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const map = { ...cornerMap, centerline_segments: [
            ...cornerMap.centerline_segments!,
            { id: 'turn-2', tags: ['corner'], start_position: 0.4, end_position: 0.5 },
        ] };
        render(<OperationComponentRefProvider>
            <MapHarness map={map} />
            <LiveEventLog ref={eventLogRef} name="visualization:event-log" />
        </OperationComponentRefProvider>);
        act(() => {
            for (let index = 0; index < 60; index += 1) {
                const position = (0.18 + index * 0.05) % 1;
                liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: index + 1,
                    committedSequence: index + 1, committedCount: index + 1,
                    sample: { Graphics_current_time_str: formatCornerTime(2027758 + index * 1000),
                        Graphics_player_car_id: 7,
                        Graphics_normalized_positions: index < 8 ? { '7': position, '63': position } : { '7': position } },
                });
            }
        });

        const section = within(screen.getByRole('region', { name: 'Driver corner times' }));
        const rows = within(section.getByRole('table')).getAllByRole('row');
        expect(rows).toHaveLength(3);
        expect(within(rows[1]).getByRole('rowheader')).toHaveTextContent('Player #7');
        expect(within(rows[2]).getByRole('rowheader')).toHaveTextContent('Car #63');
        const playerCorners = within(rows[1]).getAllByRole('listitem');
        expect(playerCorners).toHaveLength(2);
        expect(within(rows[1]).getAllByText('00:02:000')).toHaveLength(2);
        expect(within(rows[1]).getAllByText('1 of 3 times used')).toHaveLength(2);
        expect(within(rows[2]).getAllByRole('listitem')).toHaveLength(2);
        expect(within(rows[2]).getAllByText('1/3 complete passes')).toHaveLength(2);
        expect(section.getByText('8 completed corners')).toBeInTheDocument();
        const records = eventLogRef.current!.getCornerRecords();
        expect(records).toHaveLength(8);
        expect(playerCorners[0]).toHaveTextContent('Corner 1');
        expect(playerCorners[1]).toHaveTextContent('Corner 2');

        fireEvent.change(screen.getByPlaceholderText('Search events...'), { target: { value: 'turn-2' } });
        expect(section.queryByText('Corner 1')).not.toBeInTheDocument();
        expect(section.getAllByRole('listitem')).toHaveLength(2);
        expect(section.getByText('00:02:000')).toBeInTheDocument();
        expect(section.getByText('1 of 3 times used')).toBeInTheDocument();
        fireEvent.change(screen.getByPlaceholderText('Search events...'), { target: { value: 'missing' } });
        expect(section.getByText('No matching corner times')).toBeInTheDocument();
        fireEvent.change(screen.getByPlaceholderText('Search events...'), { target: { value: '' } });
        expect(section.getAllByRole('row')).toHaveLength(3);
        expect(section.getAllByRole('listitem')).toHaveLength(4);
        expect(eventLogRef.current!.getCornerRecords()).toEqual(records);
    });

    it('updates the corner average after excluding the fastest and slowest complete passes', () => {
        render(<OperationComponentRefProvider>
            <MapHarness map={cornerMap} />
            <LiveEventLog name="visualization:event-log" />
        </OperationComponentRefProvider>);
        let timeMs = 2027758;
        act(() => {
            for (let index = 0; index < 150; index += 1) {
                timeMs += [200, 800, 1800][Math.floor(index / 50)];
                liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: index + 1,
                    committedSequence: index + 1, committedCount: index + 1,
                    sample: { Graphics_current_time_str: formatCornerTime(timeMs), Graphics_player_car_id: 7,
                        Graphics_normalized_positions: { '7': (0.18 + index * 0.02) % 1 } },
                });
            }
        });
        const section = within(screen.getByRole('region', { name: 'Driver corner times' }));
        expect(section.getAllByRole('listitem')).toHaveLength(1);
        expect(section.getByText('00:04:000')).toBeInTheDocument();
        expect(section.getByText('1 of 3 times used')).toBeInTheDocument();
        expect(section.queryByText('00:01:000')).not.toBeInTheDocument();
        expect(section.queryByText('00:09:000')).not.toBeInTheDocument();
    });

    it('retains completed records through map changes and stream restarts until the session resets', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const view = (map: CircuitMapDto | null) => <OperationComponentRefProvider>
            <MapHarness map={map} />
            <LiveEventLog ref={eventLogRef} name="visualization:event-log" />
        </OperationComponentRefProvider>;
        const { rerender } = render(view(cornerMap));
        const publish = (sequence: number, position: number) => act(() => {
            liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence,
                committedSequence: sequence, committedCount: sequence,
                sample: { Graphics_current_time_str: formatCornerTime(2027758 + sequence * 1000),
                    Graphics_normalized_positions: { '7': position } },
            });
        });
        publish(1, 0.28);
        publish(2, 0.31);
        const firstRecord = eventLogRef.current!.getCornerRecords()[0];
        expect(firstRecord).toBeDefined();
        rerender(view(null));
        expect(screen.getByText('Open Live Map with tagged corners to record driver corner timing.')).toBeInTheDocument();
        expect(screen.getAllByRole('listitem')).toHaveLength(1);
        expect(screen.getByText('0/3 complete passes')).toBeInTheDocument();
        rerender(view({ ...cornerMap }));
        expect(eventLogRef.current!.getCornerRecords()).toEqual([firstRecord]);
        expect(screen.getAllByRole('listitem')).toHaveLength(1);

        publish(3, 0.28);
        act(() => { liveTelemetryStore.beginStream(); });
        publish(1, 0.31);
        expect(eventLogRef.current!.getCornerRecords()).toEqual([firstRecord]);
        act(() => { liveTelemetryStore.beginStream(); });
        publish(1, 0.28);
        publish(2, 0.31);
        const records = eventLogRef.current!.getCornerRecords();
        expect(records).toHaveLength(2);
        expect(records[0]).toEqual(firstRecord);
        expect(records[1].id).not.toBe(firstRecord.id);
        expect(screen.getAllByRole('row')).toHaveLength(2);
        expect(screen.getAllByRole('listitem')).toHaveLength(1);
        expect(screen.getByText('0/3 complete passes')).toBeInTheDocument();

        act(() => { liveTelemetryStore.resetSession(); });
        expect(eventLogRef.current!.getCornerRecords()).toEqual([]);
        expect(screen.queryByRole('table')).not.toBeInTheDocument();
    });

    it('waits for a mapped corner and discards unfinished history when the map is removed', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const view = (map: CircuitMapDto | null) => <OperationComponentRefProvider>
            <MapHarness map={map} />
            <LiveEventLog ref={eventLogRef} name="visualization:event-log" />
        </OperationComponentRefProvider>;
        const { rerender } = render(view(null));
        expect(screen.getByText('Open Live Map with tagged corners to record driver corner timing.')).toBeInTheDocument();
        rerender(view(cornerMap));
        expect(screen.getByText('Waiting for cars to cross a corner end')).toBeInTheDocument();
        act(() => { liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1,
            sample: { Graphics_current_time_str: '33:47:758', Graphics_normalized_positions: { '7': 0.28 } } }); });
        rerender(view(null));
        rerender(view(cornerMap));
        act(() => { liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: 2, committedSequence: 2, committedCount: 2,
            sample: { Graphics_current_time_str: '33:48:758', Graphics_normalized_positions: { '7': 0.31 } } }); });
        expect(eventLogRef.current?.getCornerRecords()).toEqual([]);
    });
});
