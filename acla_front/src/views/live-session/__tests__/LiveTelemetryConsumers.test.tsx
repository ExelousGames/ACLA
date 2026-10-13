import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { LiveSessionContext } from '../LiveSessionContext';
import LiveTelemetryOverview from '../LiveTelemetryOverview';
import LiveTrajectoryMap from '../LiveTrajectoryMap';
import { liveTelemetryStore } from '../live-telemetry-store';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: any) => (
        <div ref={ref} {...props}>{children}</div>
    ));
    const TextFieldRoot = ({ children, ...props }: any) => <div {...props}>{children}</div>;
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        Box: Div,
        Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
        Card: Div,
        Flex: Div,
        Grid: Div,
        Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        TextField: {
            Root: TextFieldRoot,
            Slot: Div,
        },
    };
});

jest.mock('@radix-ui/react-icons', () => ({
    MagnifyingGlassIcon: () => <span>Search</span>,
}));

const mockGetCircuitMapByTrack = jest.fn();
let mockMapLookup = mockGetCircuitMapByTrack;
jest.mock('contexts/CircuitMapsContext', () => ({
    useCircuitMaps: () => ({
        getCircuitMapByTrack: mockMapLookup,
    }),
}));

const publishFrame = (sequence: number) => liveTelemetryStore.publishFrame({
    type: 'frame',
    game: 'acc',
    sample: {
        Graphics_status: ACC_STATUS.ACC_LIVE,
        Graphics_packed_id: sequence,
        Physics_speed_kmh: sequence,
        Graphics_clock: sequence / 60,
    },
    sequence,
    committedSequence: sequence,
    committedCount: sequence,
}, { Static_track: 'monza' });

const circuitMap: CircuitMapDto = {
    id: 'monza-map', game: 'acc', circuit_name: 'Monza', source_track_key: 'monza', resolution: 1000,
    samples: { middle_line: [
        { bin: 0, normalized_position: 0, x: 0, y: 0, z: 0, sample_count: 1, updated_at: '' },
        { bin: 500, normalized_position: 0.5, x: 100, y: 0, z: 100, sample_count: 1, updated_at: '' },
    ] },
};

const renderMap = (game: 'acc' | 'iracing' = 'acc', track = 'monza') => render(
    <LiveSessionContext.Provider value={{ sessionGame: game, staticData: { Static_track: track } } as any}>
        <LiveTrajectoryMap name="live map" />
    </LiveSessionContext.Provider>,
);

const publishPositions = (sequence: number, sample: Record<string, any>, game: 'acc' | 'iracing' = 'acc') => (
    liveTelemetryStore.publishFrame({
        type: 'frame', game, sample: { Graphics_status: ACC_STATUS.ACC_LIVE, ...sample },
        sequence, committedSequence: sequence, committedCount: sequence,
    })
);

describe('live telemetry latest-value and map consumers', () => {
    beforeEach(() => {
        liveTelemetryStore.resetSession();
        mockGetCircuitMapByTrack.mockReset().mockResolvedValue(circuitMap);
        mockMapLookup = mockGetCircuitMapByTrack;
        (global as any).ResizeObserver = class {
            observe = jest.fn();
            disconnect = jest.fn();
        };
        HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
            arc: jest.fn(),
            beginPath: jest.fn(),
            clearRect: jest.fn(),
            closePath: jest.fn(),
            createRadialGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
            fill: jest.fn(),
            fillRect: jest.fn(),
            lineTo: jest.fn(),
            moveTo: jest.fn(),
            setTransform: jest.fn(),
            stroke: jest.fn(),
        })) as any;
    });

    it('shows only the newest merged dataset row', () => {
        render(<LiveTelemetryOverview name="latest telemetry" />);

        act(() => {
            publishFrame(1);
            publishFrame(2);
        });

        expect(screen.getByText('Static_track')).toBeInTheDocument();
        expect(screen.getByText('monza')).toBeInTheDocument();
        expect(screen.getByText('Graphics_packed_id')).toBeInTheDocument();
        expect(screen.getAllByText('2').length).toBeGreaterThan(0);
        expect(screen.queryByText('1')).not.toBeInTheDocument();
    });

    it.each(['acc', 'iracing'] as const)('plots the latest normalized positions on the downloaded %s map', async (game) => {
        renderMap(game);
        await waitFor(() => expect(screen.getByText('Waiting for live positions')).toBeInTheDocument());
        expect(mockGetCircuitMapByTrack).toHaveBeenCalledWith(game, 'monza');
        act(() => {
            for (let sequence = 1; sequence <= 120; sequence += 1) {
                publishPositions(sequence, {
                    Graphics_player_car_id: 1052,
                    Graphics_normalized_car_position: sequence / 240,
                    Graphics_normalized_positions: { 63: 0, 1052: 0.9 },
                }, game);
            }
        });
        expect(screen.queryByText('Waiting for live positions')).not.toBeInTheDocument();
        expect(screen.getByText('1 opponents')).toBeInTheDocument();
        const context = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results.slice(-1)[0].value;
        expect(context.arc).toHaveBeenCalledTimes(2);
        // The fit projection uses only the downloaded 100 x 100 middle line.
        const padding = 520 * 0.08;
        const playerMarker = context.arc.mock.calls.find((call: number[]) => call[2] === 6);
        expect(playerMarker[0]).toBeCloseTo(400 + (520 - padding * 2) / 2);
        expect(playerMarker[1]).toBeCloseTo(game === 'acc' ? 520 - padding : padding);
    });

    it('clears missing positions, paused telemetry, and stream/session resets', async () => {
        renderMap();
        await screen.findByText('Waiting for live positions');
        act(() => { publishPositions(1, { Graphics_normalized_car_position: 0.5 }); });
        expect(screen.queryByText('Waiting for live positions')).not.toBeInTheDocument();
        act(() => { publishPositions(2, {}); });
        expect(screen.getByText('Waiting for live positions')).toBeInTheDocument();
        act(() => { publishPositions(3, { Graphics_normalized_car_position: 0.5, Graphics_status: ACC_STATUS.ACC_PAUSE }); });
        expect(screen.getByText('Waiting for live positions')).toBeInTheDocument();
        act(() => { publishPositions(4, { Graphics_normalized_car_position: 0.5 }); });
        act(() => { liveTelemetryStore.beginStream(); });
        expect(screen.getByText('Waiting for live positions')).toBeInTheDocument();
        act(() => { publishPositions(1, { Graphics_normalized_car_position: 0.5 }); });
        act(() => { liveTelemetryStore.resetSession(); });
        expect(screen.getByText('Waiting for live positions')).toBeInTheDocument();
    });

    it('uses telemetry received before the map download completes', async () => {
        let resolve!: (map: CircuitMapDto) => void;
        mockGetCircuitMapByTrack.mockReturnValue(new Promise((done) => { resolve = done; }));
        renderMap();
        act(() => { publishPositions(1, { Graphics_normalized_car_position: 0.5 }); });
        expect(screen.getByText('Loading circuit map')).toBeInTheDocument();
        await act(async () => { resolve(circuitMap); });
        expect(screen.queryByText('Loading circuit map')).not.toBeInTheDocument();
        expect(screen.queryByText('Waiting for live positions')).not.toBeInTheDocument();
    });

    it.each([null, { ...circuitMap, samples: { left_boundary: circuitMap.samples.middle_line } }])('requires a middle line to plot normalized positions', async (map) => {
        mockGetCircuitMapByTrack.mockResolvedValue(map);
        renderMap();
        act(() => { publishPositions(1, { Graphics_normalized_car_position: 0.5 }); });
        expect(await screen.findByText('Circuit middle line unavailable')).toBeInTheDocument();
        const context = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results.slice(-1)[0].value;
        expect(context.arc).not.toHaveBeenCalled();
    });

    it('discards the previous circuit and ignores late downloads after changing tracks', async () => {
        let resolveOld!: (map: CircuitMapDto) => void;
        mockGetCircuitMapByTrack.mockImplementation((_game, track) => track === 'monza'
            ? new Promise((done) => { resolveOld = done; })
            : Promise.resolve({ ...circuitMap, circuit_name: 'Spa', source_track_key: 'spa' }));
        const view = renderMap();
        view.rerender(
            <LiveSessionContext.Provider value={{ sessionGame: 'acc', staticData: { Static_track: 'spa' } } as any}>
                <LiveTrajectoryMap name="live map" />
            </LiveSessionContext.Provider>,
        );
        await screen.findByText('Spa');
        await act(async () => { resolveOld(circuitMap); });
        expect(screen.getByText('Spa')).toBeInTheDocument();
        expect(screen.queryByText('Monza')).not.toBeInTheDocument();
    });

    it('reports a failed download', async () => {
        mockGetCircuitMapByTrack.mockRejectedValue(new Error('offline'));
        renderMap();
        expect(await screen.findByText('Unable to load circuit map')).toBeInTheDocument();
    });

    it('does not repeat a missing-map lookup when the provider updates its list', async () => {
        mockGetCircuitMapByTrack.mockResolvedValue(null);
        const view = renderMap();
        await screen.findByText('Circuit middle line unavailable');
        mockMapLookup = jest.fn().mockResolvedValue(null);
        view.rerender(
            <LiveSessionContext.Provider value={{ sessionGame: 'acc', staticData: { Static_track: 'monza' } } as any}>
                <LiveTrajectoryMap name="live map" />
            </LiveSessionContext.Provider>,
        );
        expect(screen.getByText('Circuit middle line unavailable')).toBeInTheDocument();
        expect(mockMapLookup).not.toHaveBeenCalled();
    });

    it.each(['acc', 'iracing'] as const)('shows per-car positions from %s and removes unavailable values', (game) => {
        render(<LiveTelemetryOverview name="latest telemetry" />);
        const positions = { 0: 0, 63: 0.75, 1052: 1 };
        act(() => {
            expect(liveTelemetryStore.publishFrame({
                type: 'frame', game,
                sample: { Graphics_status: ACC_STATUS.ACC_LIVE, Graphics_normalized_positions: positions },
                sequence: 1, committedSequence: 1, committedCount: 1,
            })).toBe(true);
        });
        expect(screen.getByText('Graphics_normalized_positions')).toBeInTheDocument();
        expect(screen.getByText(JSON.stringify(positions))).toBeInTheDocument();

        act(() => {
            liveTelemetryStore.publishFrame({
                type: 'frame', game, sample: { Graphics_status: ACC_STATUS.ACC_LIVE },
                sequence: 2, committedSequence: 2, committedCount: 2,
            });
        });
        expect(screen.queryByText('Graphics_normalized_positions')).not.toBeInTheDocument();
        expect(screen.getByText('Graphics_status')).toBeInTheDocument();
    });
});
