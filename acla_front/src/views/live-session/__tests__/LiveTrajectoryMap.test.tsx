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
    let strokes: { color: string; points: number[][] }[];
    let telemetry: {
        currentTelemetry: StandardTelemetrySample;
        telemetryStatus: number;
        game: string;
    };
    const referenceStrokes = () => strokes.filter(({ color }) => color === '#e879f9');

    beforeEach(() => {
        window.ResizeObserver = jest.fn(() => ({ observe: jest.fn(), disconnect: jest.fn() })) as any;
        path = [];
        strokes = [];
        context = {
            setTransform: jest.fn(), clearRect: jest.fn(() => { strokes = []; }),
            createRadialGradient: () => ({ addColorStop: jest.fn() }), fillRect: jest.fn(),
            beginPath: jest.fn(() => { path = []; }),
            moveTo: jest.fn((x, y) => path.push([x, y])), lineTo: jest.fn((x, y) => path.push([x, y])),
            stroke: jest.fn(() => strokes.push({ color: context.strokeStyle, points: path })),
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
            // Raw ACC Y is altitude; Z is the second ground-plane axis.
            Graphics_car_coordinates: [{ x: 900, y: 901, z: 902 }, { x: 20, y: 4 + altitude, z: 400 }],
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
            reference: { x: 20, y: 400, z: 4 }, centerline: { x: 20, y: 400, z: 4 },
        });
        expect(map.samples.middle_line[1]).toMatchObject({ x: 20, y: 0, z: 3 + altitude });
        expect(reference.y).toEqual([[0, 0, 3], [20, 400, 4], [0, 0, 3 + altitude]]);
        expect(telemetry.currentTelemetry.Graphics_car_coordinates![1]).toEqual({ x: 20, y: 4 + altitude, z: 400 });
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
        const first = referenceStrokes()[0].points;
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Graphics_track_grip_status: 4 };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(screen.getByRole('status')).toHaveTextContent('grip 5');
        expect(referenceStrokes()[0].points).not.toEqual(first);
        telemetry.currentTelemetry = { ...telemetry.currentTelemetry, Static_car_model: 'different-car' };
        view.rerender(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(referenceStrokes()).toEqual([]);
        expect(screen.getByRole('status')).toHaveTextContent('No reference lap');
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

    it('shares loaded reference and centerline XYZ interpolated by lap progress, including the seam', async () => {
        const reference = { ...fixtures.expected, x: [0.25, 0.75],
            target_features: ['expert_optimal_player_pos_z', 'expert_optimal_player_pos_x', 'expert_optimal_player_pos_y'],
            y: [[30, 10, 20], [70, 50, 60]] };
        (loadTopLapReferenceModel as jest.Mock).mockResolvedValue([reference]);
        (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: jest.fn().mockResolvedValue({
            ...circuitMap(), samples: { middle_line: [
                { normalized_position: 0.25, x: 100, y: 200, z: 300 },
                { normalized_position: 0.75, x: 500, y: 600, z: 700 },
            ] },
        }) });
        const ref = React.createRef<LiveTrajectoryMapHandle>();
        const view = render(<LiveTrajectoryMap name="live-map" ref={ref} />);
        expect(ref.current!.getPositionReferences(0.5)).toBeNull();
        await screen.findByText('Reference lap · car-a · grip 2');
        const points = { reference: { x: 30, y: 40, z: 50 }, centerline: { x: 300, y: 400, z: 500 } };
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
