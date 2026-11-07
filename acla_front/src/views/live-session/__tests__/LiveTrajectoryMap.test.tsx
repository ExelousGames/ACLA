import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import LiveTrajectoryMap, { LiveTrajectoryMapHandle } from '../LiveTrajectoryMap';
import { circuitMap } from '../live-phrases/test-fixtures';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { useLiveTelemetrySelector } from '../live-telemetry-store';
import { loadTopLapReferenceModel } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import fixtures from 'services/__tests__/fixtures/top-lap-reference.json';

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });
jest.mock('contexts/CircuitMapsContext', () => ({ useCircuitMaps: jest.fn() }));
jest.mock('../live-telemetry-store', () => ({ useLiveTelemetrySelector: jest.fn() }));
jest.mock('services/topLapReferenceService', () => ({ loadTopLapReferenceModel: jest.fn() }));
jest.mock('../LiveSessionContext', () => ({
    LiveSessionContext: require('react').createContext({ sessionGame: null, staticData: {} }),
}));

it('publishes the loaded map and clears it while a different circuit loads', async () => {
    const originalObserver = window.ResizeObserver;
    window.ResizeObserver = jest.fn(() => ({ observe: jest.fn(), disconnect: jest.fn() })) as any;
    const canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const first = circuitMap();
    const second = { ...circuitMap('fast'), id: 'second', source_track_key: 'spa' };
    let resolveFirst!: (map: typeof first) => void;
    let resolveSecond!: (map: typeof first) => void;
    const lookup = jest.fn()
        .mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve; }))
        .mockReturnValueOnce(new Promise((resolve) => { resolveSecond = resolve; }));
    (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: lookup });
    const telemetry = { currentTelemetry: { Static_track: 'test' }, telemetryStatus: 2, game: 'acc' };
    (useLiveTelemetrySelector as jest.Mock).mockImplementation(() => telemetry);
    const ref = React.createRef<LiveTrajectoryMapHandle>();
    const view = render(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
    try {
        const listener = jest.fn();
        const unsubscribe = ref.current!.subscribeCircuitMap(listener);
        expect(ref.current!.getCircuitMap()).toBeNull();
        await act(async () => { resolveFirst(first); });
        expect(ref.current!.getCircuitMap()).toBe(first);
        expect(listener).toHaveBeenCalledTimes(1);

        telemetry.currentTelemetry.Static_track = 'spa';
        view.rerender(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
        expect(ref.current!.getCircuitMap()).toBeNull();
        expect(listener).toHaveBeenCalledTimes(2);
        await act(async () => { resolveSecond(second); });
        expect(ref.current!.getCircuitMap()).toBe(second);
        expect(listener).toHaveBeenCalledTimes(3);
        expect(lookup).toHaveBeenNthCalledWith(2, 'acc', 'spa');
        unsubscribe();
        telemetry.currentTelemetry.Static_track = '';
        view.rerender(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
        expect(ref.current!.getCircuitMap()).toBeNull();
        expect(listener).toHaveBeenCalledTimes(3);
    } finally {
        view.unmount();
        canvas.mockRestore();
        window.ResizeObserver = originalObserver;
    }
});

describe('reference lap overlay', () => {
    const originalObserver = window.ResizeObserver;
    let canvas: jest.SpyInstance;
    let context: any;
    let path: number[][];
    let strokes: { color: string | { stops: [number, string][] }; points: number[][]; width: number }[];
    let telemetry: {
        currentTelemetry: StandardTelemetrySample;
        telemetryStatus: number;
        game: string;
    };
    const referenceStrokes = () => strokes.filter(({ color }) => color === '#e879f9');
    const pedalStrokes = (pedal: 'brake' | 'throttle') => strokes.filter(({ color }) => typeof color !== 'string'
        && color.stops.some(([, value]) => value.startsWith(pedal === 'brake' ? '#ef5350' : '#00e676')));

    beforeEach(() => {
        window.ResizeObserver = jest.fn(() => ({ observe: jest.fn(), disconnect: jest.fn() })) as any;
        path = [];
        strokes = [];
        context = {
            setTransform: jest.fn(), clearRect: jest.fn(() => { strokes = []; }),
            createRadialGradient: () => ({ addColorStop: jest.fn() }), fillRect: jest.fn(),
            createLinearGradient: jest.fn(() => {
                const stops: [number, string][] = [];
                return { stops, addColorStop: (offset: number, color: string) => stops.push([offset, color]) };
            }),
            beginPath: jest.fn(() => { path = []; }),
            moveTo: jest.fn((x, y) => path.push([x, y])), lineTo: jest.fn((x, y) => path.push([x, y])),
            stroke: jest.fn(() => strokes.push({ color: context.strokeStyle, points: path, width: context.lineWidth })),
            arc: jest.fn(), fill: jest.fn(),
        };
        canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue(circuitMap()) });
        telemetry = { currentTelemetry: { Static_track: 'spa', Static_car_model: 'car-a',
            Graphics_track_grip_status: 2, Graphics_normalized_car_position: 0.5 }, telemetryStatus: 2, game: 'acc' };
        (useLiveTelemetrySelector as jest.Mock).mockImplementation(() => telemetry);
        (loadTopLapReferenceModel as jest.Mock).mockReset().mockResolvedValue([fixtures.expected]);
    });

    afterEach(() => {
        canvas.mockRestore();
        window.ResizeObserver = originalObserver;
    });

    it('toggles pedal strength on the reference line and keeps both inputs visible through camera changes', async () => {
        const reference = { ...fixtures.expected, x: [0, 0.3, 0.7, 1],
            target_features: ['expert_optimal_throttle', 'expert_optimal_player_pos_z', 'expert_optimal_brake',
                'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y'],
            y: [[1, 20, 0, 10, 1], [0.5, 40, 0.5, 30, 2], [0, 70, 1, 55, 8], [1, 20, 0, 10, 1]],
        };
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference]);
        render(<LiveTrajectoryMap name="live-map" />);
        const brake = screen.getByRole('checkbox', { name: 'Reference brake' });
        const throttle = screen.getByRole('checkbox', { name: 'Reference throttle' });
        expect(brake).toBeDisabled();
        expect(throttle).toBeDisabled();
        await screen.findByText('Reference lap · car-a · grip 2');
        expect(brake).toBeEnabled();
        expect(throttle).toBeEnabled();
        expect(brake).not.toBeChecked();
        expect(throttle).not.toBeChecked();
        expect(pedalStrokes('brake')).toHaveLength(0);
        expect(pedalStrokes('throttle')).toHaveLength(0);

        fireEvent.click(brake);
        expect(brake).toBeChecked();
        expect(pedalStrokes('brake')).toHaveLength(3);
        expect(pedalStrokes('brake')[0].color).toMatchObject({ stops: [[0, '#ef535000'], [1, '#ef535080']] });
        expect(pedalStrokes('throttle')).toHaveLength(0);
        fireEvent.click(throttle);
        expect(throttle).toBeChecked();
        expect(pedalStrokes('throttle')).toHaveLength(3);
        expect(pedalStrokes('throttle')[0].color).toMatchObject({ stops: [[0, '#00e676ff'], [1, '#00e67680']] });
        expect(pedalStrokes('throttle')[0].width).toBeGreaterThan(pedalStrokes('brake')[0].width);
        expect(strokes.indexOf(pedalStrokes('throttle')[0])).toBeLessThan(strokes.indexOf(pedalStrokes('brake')[0]));
        expect(screen.getByText('Brighter = more input · Brake inside, throttle outside')).toBeVisible();

        const expectAligned = () => {
            const line = referenceStrokes()[0].points;
            for (const pedal of ['brake', 'throttle'] as const) {
                pedalStrokes(pedal).forEach((stroke, index) => expect(stroke.points).toEqual(line.slice(index, index + 2)));
            }
            expect(context.stroke.mock.invocationCallOrder.slice(-1)[0]).toBeLessThan(context.fill.mock.invocationCallOrder.slice(-1)[0]);
        };
        expectAligned();
        for (const name of ['X', 'Y', '+', 'Driver', 'Fit']) {
            fireEvent.click(screen.getByRole('button', { name }));
            expectAligned();
        }
        fireEvent.click(brake);
        expect(pedalStrokes('brake')).toHaveLength(0);
        expect(pedalStrokes('throttle')).toHaveLength(3);
        fireEvent.click(throttle);
        expect(pedalStrokes('throttle')).toHaveLength(0);
        expect(referenceStrokes()).toHaveLength(1);
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });

    it('refreshes selected pedals with the reference lap and disables missing channels without drawing stale inputs', async () => {
        const reference = { ...fixtures.expected,
            target_features: [...fixtures.expected.target_features, 'expert_optimal_brake', 'expert_optimal_throttle'],
            y: fixtures.expected.y.map((row) => [...row, 0.5, 0]),
        };
        const wet = { ...reference, avg_grip_int: 5,
            target_features: [...fixtures.expected.target_features, 'expert_optimal_throttle'],
            y: fixtures.expected.y.map((row) => [...row, 1]),
        };
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference, wet]);
        const view = render(<LiveTrajectoryMap name="live-map" />);
        await screen.findByText('Reference lap · car-a · grip 2');
        const brake = screen.getByRole('checkbox', { name: 'Reference brake' });
        const throttle = screen.getByRole('checkbox', { name: 'Reference throttle' });
        fireEvent.click(brake);
        fireEvent.click(throttle);
        expect(pedalStrokes('brake')).toHaveLength(2);
        expect(pedalStrokes('throttle')).toHaveLength(0);
        expect(throttle).toBeEnabled();
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Graphics_track_grip_status: 5 };
        view.rerender(<LiveTrajectoryMap name="live-map" />);
        expect(brake).toBeDisabled();
        expect(screen.getByText('Reference brake (unavailable)')).toBeVisible();
        expect(pedalStrokes('brake')).toHaveLength(0);
        expect(pedalStrokes('throttle')).toHaveLength(2);
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'missing' };
        view.rerender(<LiveTrajectoryMap name="live-map" />);
        expect(throttle).toBeDisabled();
        expect(pedalStrokes('throttle')).toHaveLength(0);
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });

    it('disables pedal overlays when the reference has no racing line', async () => {
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([{ ...fixtures.expected,
            target_features: ['expert_optimal_brake', 'expert_optimal_throttle'], y: [[1, 0], [0, 1], [1, 0]],
        }]);
        render(<LiveTrajectoryMap name="live-map" />);
        await screen.findByText('Reference lap has no racing line');
        expect(screen.getByRole('checkbox', { name: 'Reference brake' })).toBeDisabled();
        expect(screen.getByRole('checkbox', { name: 'Reference throttle' })).toBeDisabled();
        expect(pedalStrokes('brake')).toHaveLength(0);
        expect(pedalStrokes('throttle')).toHaveLength(0);
    });

    it('shows interpolated pedal percentages on hover with scaled canvas coordinates and camera controls', async () => {
        const reference = { ...fixtures.expected, x: [0, 0.4, 0.8, 1],
            target_features: ['expert_optimal_throttle', 'expert_optimal_player_pos_z', 'expert_optimal_brake',
                'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y'],
            y: [[1, 20, 0, 0, 0], [0.2, 40, 1, 100, 0], [0, 70, 0.5, 100, 100], [1, 20, 0, 0, 0]],
        };
        const map = { ...circuitMap(), samples: { middle_line: reference.y.map((row, index) => ({
            bin: index, normalized_position: reference.x[index], x: row[3], y: row[4], z: row[1],
            sample_count: 1, updated_at: '',
        })) } };
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue(map) });
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference]);
        render(<LiveTrajectoryMap name="live-map" />);
        await screen.findByText('Reference lap · car-a · grip 2');
        const mapCanvas = screen.getByRole('img', { name: 'Live circuit map' });
        jest.spyOn(mapCanvas, 'getBoundingClientRect').mockReturnValue({ left: 120, top: 40, width: 400, height: 260 } as DOMRect);
        const hover = (index: number, fraction: number) => {
            const points = referenceStrokes()[0].points;
            const x = points[index][0] + (points[index + 1][0] - points[index][0]) * fraction;
            const y = points[index][1] + (points[index + 1][1] - points[index][1]) * fraction;
            // Preserve the exact subpixel location; jsdom's MouseEvent constructor truncates it.
            const event = new MouseEvent('mousemove', { bubbles: true });
            Object.defineProperties(event, {
                clientX: { value: 120 + x / 2 }, clientY: { value: 40 + y / 2 },
            });
            fireEvent(mapCanvas, event);
        };
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
        hover(0, 0.5);
        expect(screen.getByRole('tooltip')).toHaveTextContent('Reference lap · 20.0%');
        expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: 50.0%');
        expect(screen.getByRole('tooltip')).toHaveTextContent('Throttle: 60.0%');
        fireEvent.click(screen.getByRole('checkbox', { name: 'Reference brake' }));
        fireEvent.click(screen.getByRole('checkbox', { name: 'Reference throttle' }));
        expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: 50.0%');
        hover(0, 0);
        expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: 0.0%');
        expect(screen.getByRole('tooltip')).toHaveTextContent('Throttle: 100.0%');
        for (const name of ['X', 'Y', '+', 'Fit']) {
            fireEvent.click(screen.getByRole('button', { name }));
            hover(2, 0.5);
            expect(screen.getByRole('tooltip')).toHaveTextContent('Reference lap · 90.0%');
            expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: 25.0%');
            expect(screen.getByRole('tooltip')).toHaveTextContent('Throttle: 50.0%');
        }
        fireEvent.mouseMove(mapCanvas, { clientX: 120, clientY: 40 });
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
        hover(2, 0.5);
        expect(screen.getByRole('tooltip')).toBeVisible();
        fireEvent.mouseLeave(mapCanvas);
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });

    it('updates hover values under a stationary pointer and clears them when the reference or camera changes', async () => {
        const reference = { ...fixtures.expected,
            target_features: [...fixtures.expected.target_features, 'expert_optimal_brake', 'expert_optimal_throttle'],
            y: fixtures.expected.y.map((row) => [...row, 0.5, 1]),
        };
        const wet = { ...reference, avg_grip_int: 5,
            target_features: [...fixtures.expected.target_features, 'expert_optimal_throttle'],
            y: fixtures.expected.y.map((row) => [...row, 0]),
        };
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference, wet]);
        const view = render(<LiveTrajectoryMap name="live-map" />);
        await screen.findByText('Reference lap · car-a · grip 2');
        const mapCanvas = screen.getByRole('img', { name: 'Live circuit map' });
        jest.spyOn(mapCanvas, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 800, height: 520 } as DOMRect);
        const [x, y] = referenceStrokes()[0].points[0];
        fireEvent.mouseMove(mapCanvas, { clientX: x, clientY: y });
        expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: 50.0%');
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Graphics_track_grip_status: 5 };
        view.rerender(<LiveTrajectoryMap name="live-map" />);
        expect(screen.getByRole('tooltip')).toHaveTextContent('Brake: Unavailable');
        expect(screen.getByRole('tooltip')).toHaveTextContent('Throttle: 0.0%');
        fireEvent.click(screen.getByRole('button', { name: '+' }));
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Fit' }));
        expect(screen.getByRole('tooltip')).toHaveTextContent('Throttle: 0.0%');
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'missing' };
        view.rerender(<LiveTrajectoryMap name="live-map" />);
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });

    it('draws the downloaded racing line below car markers and applies the map camera controls', async () => {
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(await screen.findByText('Reference lap · car-a · grip 2')).toBeInTheDocument();
        const points = referenceStrokes()[0].points;
        expect(points).toHaveLength(3);
        expect(points[0]).toEqual(points[2]);
        expect(points).not.toEqual(strokes[0].points);
        expect(points[1][0]).toBeGreaterThan(points[0][0]);
        expect(points[1][1]).toBeGreaterThan(points[0][1]);
        expect(context.stroke.mock.invocationCallOrder.slice(-1)[0]).toBeLessThan(context.fill.mock.invocationCallOrder.slice(-1)[0]);

        fireEvent.click(screen.getByRole('button', { name: 'X' }));
        expect(referenceStrokes()[0].points[0][0]).toBeCloseTo(800 - points[0][0]);
        fireEvent.click(screen.getByRole('button', { name: 'Y' }));
        expect(referenceStrokes()[0].points[0][1]).toBeCloseTo(520 - points[0][1]);
        fireEvent.click(screen.getByRole('button', { name: '+' }));
        expect(referenceStrokes()[0].points[1][0] - referenceStrokes()[0].points[0][0])
            .toBeCloseTo(-(points[1][0] - points[0][0]) * 1.25);
        act(() => ref.current!.focusDriver());
        expect(referenceStrokes()[0].points[1][0] - referenceStrokes()[0].points[0][0])
            .toBeCloseTo(-(points[1][0] - points[0][0]) * 2.8);
        act(() => ref.current!.fitTrack());
        expect(referenceStrokes()[0].points[0][0]).toBeCloseTo(800 - points[0][0]);
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });

    it.each([0, 10000])('projects ACC X/Y consistently with altitude variation %s and preserves stored XYZ', async (altitude) => {
        const map = { ...circuitMap(), samples: { middle_line: [
            [0, 0, 0, 3], [0.25, 20, 0, 3 + altitude],
            [0.5, 20, 400, 4], [0.75, 0, 400, 4 - altitude],
        ].map(([position, x, y, z], bin) => ({
            bin, normalized_position: position, x, y, z, sample_count: 1, updated_at: '',
        })) } };
        const reference = { ...fixtures.expected,
            x: [0, 0.5, 1],
            target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z'],
            y: [[0, 0, 3], [20, 400, 4], [0, 0, 3 + altitude]],
        };
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue(map) });
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference]);
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        const view = render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        await screen.findByText('Reference lap · car-a · grip 2');
        // The 400 m Y span fills the 436.8 px height inside the canvas padding.
        const expectedPoints = [[389.08, 41.6], [410.92, 41.6], [410.92, 478.4], [389.08, 478.4], [389.08, 41.6]];
        expect(strokes[0].points).toHaveLength(expectedPoints.length);
        strokes[0].points.forEach((point, index) => {
            expect(point[0]).toBeCloseTo(expectedPoints[index][0]);
            expect(point[1]).toBeCloseTo(expectedPoints[index][1]);
        });
        expect(referenceStrokes()[0].points).toEqual([strokes[0].points[0], strokes[0].points[2], strokes[0].points[0]]);
        expect(screen.getByRole('button', { name: 'Y' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Z' })).not.toBeInTheDocument();
        const originalLines = strokes;
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry,
            Graphics_player_car_id: 1052,
            Graphics_car_id: [63, 1052],
            // Raw coordinates deliberately disagree with normalized map positions.
            Graphics_car_coordinates: [{ x: 900, y: 901, z: 902 }, { x: -200, y: 4 + altitude, z: -400 }],
            Graphics_normalized_positions: { 63: 0.25 },
        };
        context.arc.mockClear();
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(strokes).toEqual(originalLines);
        expect(context.arc).toHaveBeenNthCalledWith(1, strokes[0].points[1][0], strokes[0].points[1][1], 4, 0, Math.PI * 2);
        const referencePoint = referenceStrokes()[0].points[1];
        expect(context.arc).toHaveBeenLastCalledWith(referencePoint[0], referencePoint[1], 6, 0, Math.PI * 2);
        fireEvent.click(screen.getByRole('button', { name: 'Driver' }));
        expect(context.arc).toHaveBeenLastCalledWith(400, 260, 6, 0, Math.PI * 2);
        expect(referenceStrokes()[0].points[1]).toEqual([400, 260]);
        expect(referenceStrokes()[0].points[0][1]).toBeCloseTo(260 - 436.8 * 2.8);
        fireEvent.click(screen.getByRole('button', { name: 'Y' }));
        expect(referenceStrokes()[0].points[0][1]).toBeCloseTo(260 + 436.8 * 2.8);
        fireEvent.click(screen.getByRole('button', { name: 'Fit' }));
        expect(referenceStrokes()[0].points[0][1]).toBeCloseTo(478.4);
        expect(ref.current!.getCircuitMap()).toBe(map);
        expect(ref.current!.getReferenceLap()).toBe(reference);
        expect(ref.current!.getPositionReferences(0.5)).toEqual({
            reference: { x: 20, y: 400, z: 4 },
        });
        expect(map.samples.middle_line[1]).toMatchObject({ x: 20, y: 0, z: 3 + altitude });
        expect(reference.y).toEqual([[0, 0, 3], [20, 400, 4], [0, 0, 3 + altitude]]);
        expect(telemetry.currentTelemetry.Graphics_car_coordinates![1]).toEqual({ x: -200, y: 4 + altitude, z: -400 });
    });

    it('preserves the iRacing X/Z projection, vertical direction and Z flip control', async () => {
        telemetry.game = 'iracing';
        const map = { ...circuitMap(), game: 'iracing', samples: { middle_line: [
            { normalized_position: 0, x: 0, y: -10000, z: 0 },
            { normalized_position: 0.5, x: 20, y: 10000, z: 400 },
        ] } };
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue(map) });
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([]);
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        await waitFor(() => expect(ref.current!.getCircuitMap()).toBe(map));
        expect(strokes[0].points[0][0]).toBeCloseTo(389.08);
        expect(strokes[0].points[0][1]).toBeCloseTo(478.4);
        expect(strokes[0].points[1][0]).toBeCloseTo(410.92);
        expect(strokes[0].points[1][1]).toBeCloseTo(41.6);
        expect(screen.queryByRole('button', { name: 'Y' })).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Z' }));
        expect(strokes[0].points[0][1]).toBeCloseTo(41.6);
        expect(strokes[0].points[1][1]).toBeCloseTo(478.4);
        fireEvent.click(screen.getByRole('button', { name: 'Driver' }));
        expect(context.arc).toHaveBeenLastCalledWith(400, 260, 6, 0, Math.PI * 2);
    });

    it('changes the matching line locally and immediately removes unmatched references', async () => {
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([
            fixtures.expected, { ...fixtures.expected, avg_grip_int: 5, y: fixtures.expected.y.map((row) => [row[0] + 20, ...row.slice(1)]) },
        ]);
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        const view = render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        await screen.findByText('Reference lap · car-a · grip 2');
        const changed = jest.fn();
        const unsubscribe = ref.current!.subscribeCircuitMap(changed);
        const first = referenceStrokes()[0].points;
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Graphics_track_grip_status: 4 };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(screen.getByRole('status')).toHaveTextContent('grip 5');
        expect(changed).toHaveBeenCalledTimes(1);
        expect(referenceStrokes()[0].points).not.toEqual(first);
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'different-car' };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(referenceStrokes()).toEqual([]);
        expect(screen.getByRole('status')).toHaveTextContent('No reference lap');
        expect(changed).toHaveBeenCalledTimes(2);
        expect(ref.current!.getReferenceLap()).toBeNull();
        unsubscribe();
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'car-a', Static_track: 'monza' };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        await waitFor(() => expect(ref.current!.getCircuitMap()).not.toBeNull());
        expect(referenceStrokes()).toEqual([]);
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_track: 'spa' };
        telemetry.game = 'iracing';
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        await waitFor(() => expect(ref.current!.getCircuitMap()).not.toBeNull());
        expect(referenceStrokes()).toEqual([]);
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });

    it('uses the current circuit when a model download finishes after a track change', async () => {
        let finish!: (entries: typeof fixtures.expected[]) => void;
        (loadTopLapReferenceModel as jest.Mock).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
        const view = render(<LiveTrajectoryMap name="live-map" />);
        expect(screen.getByRole('status')).toHaveTextContent('Loading reference lap model');
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_track: 'monza' };
        view.rerender(<LiveTrajectoryMap name="live-map" />);
        await act(async () => { finish([fixtures.expected]); });
        expect(referenceStrokes()).toEqual([]);
        expect(screen.getByRole('status')).toHaveTextContent('No reference lap');
    });

    it('keeps cars visible after a model failure and supports retry', async () => {
        (loadTopLapReferenceModel as jest.Mock).mockRejectedValueOnce(new Error('offline'));
        render(<LiveTrajectoryMap name="live-map" />);
        expect(await screen.findByText('Reference lap unavailable')).toBeInTheDocument();
        expect(context.arc).toHaveBeenCalled();
        expect(referenceStrokes()).toEqual([]);
        fireEvent.click(screen.getByRole('button', { name: 'Retry reference lap' }));
        await screen.findByText('Reference lap · car-a · grip 2');
        expect(referenceStrokes()).toHaveLength(1);
        expect(screen.getByRole('status')).toHaveTextContent('Reference lap · car-a');
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(2);
    });

    it('does not download before the live car and circuit are known', async () => {
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: '' };
        render(<LiveTrajectoryMap name="live-map" />);
        await screen.findByText('test');
        expect(screen.getByRole('status')).toHaveTextContent('waiting for car and circuit');
        expect(loadTopLapReferenceModel).not.toHaveBeenCalled();
    });

    it('shares reference XYZ interpolated by lap progress without a centerline, including the seam', async () => {
        const reference = { ...fixtures.expected, x: [0.25, 0.75],
            target_features: ['expert_optimal_player_pos_z', 'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y'],
            y: [[30, 10, 20], [70, 50, 60]] };
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference]);
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue({
            ...circuitMap(), samples: { middle_line: [] },
        }) });
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        const view = render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(ref.current!.getPositionReferences(0.5)).toBeNull();
        await screen.findByText('Reference lap · car-a · grip 2');
        const points = { reference: { x: 30, y: 40, z: 50 } };
        expect(ref.current!.getReferenceLap()).toBe(reference);
        expect(ref.current!.getPositionReferences(0.5)).toEqual(points);
        expect(ref.current!.getPositionReferences(0)).toEqual(points);
        expect(ref.current!.getPositionReferences(1)).toEqual(points);
        expect(ref.current!.getPositionReferences(NaN)).toBeNull();
        fireEvent.click(screen.getByRole('button', { name: 'X' }));
        fireEvent.click(screen.getByRole('button', { name: '+' }));
        expect(ref.current!.getPositionReferences(0.5)).toEqual(points);
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'missing' };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(ref.current!.getPositionReferences(0.5)).toBeNull();
        expect(loadTopLapReferenceModel).toHaveBeenCalledTimes(1);
    });
});
