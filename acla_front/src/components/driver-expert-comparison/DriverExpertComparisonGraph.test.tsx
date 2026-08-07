import React from 'react';
import { act, render, screen, within } from '@testing-library/react';
import {
    DRIVER_COMPARISON_COLOR,
    EXPERT_COMPARISON_COLOR,
    DriverExpertComparisonGraph,
    getDriverExpertReplayDurationMs,
    normalizeDriverExpertComparisonData,
} from './DriverExpertComparisonGraph';
import { useDesktopGame } from 'contexts/DesktopGameContext';
import type { DesktopGame } from 'contexts/DesktopGameContext';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: jest.fn(),
}));

const mockedUseDesktopGame = useDesktopGame as jest.Mock;

const completeData = {
    samples: [
        {
            driverTimeMs: 10_000,
            expertTimeMs: 50_000,
            trackPosition: 0.2,
            driverTrajectory: { x: 0, y: 0, z: 100 },
            expertTrajectory: { x: 0, y: 10, z: 1000 },
            driverGas: 0,
            expertGas: 0.2,
            driverBrake: 1,
            expertBrake: 0.8,
            driverGear: 2,
            expertGear: 3,
        },
        {
            driverTimeMs: 11_000,
            expertTimeMs: 52_000,
            trackPosition: 0.25,
            driverTrajectory: { x: 50, y: 25, z: 200 },
            expertTrajectory: { x: 50, y: 35, z: 2000 },
            driverGas: 0.5,
            expertGas: 0.6,
            driverBrake: 0.5,
            expertBrake: 0.4,
            driverGear: 3,
            expertGear: 4,
        },
        {
            driverTimeMs: 13_000,
            expertTimeMs: 52_500,
            trackPosition: 0.3,
            driverTrajectory: { x: 100, y: 50, z: 300 },
            expertTrajectory: { x: 100, y: 60, z: 3000 },
            driverGas: 1.2,
            expertGas: 1,
            driverBrake: -0.2,
            expertBrake: 0,
            driverGear: 5,
            expertGear: 6,
        },
    ],
};

describe('DriverExpertComparisonGraph', () => {
    let detectedGame: DesktopGame | null;
    let nextFrameId = 1;
    let pendingFrames: Map<number, FrameRequestCallback>;
    let requestAnimationFrameMock: jest.Mock;
    let cancelAnimationFrameMock: jest.Mock;

    const runAnimationFrame = (timestamp: number) => {
        const callbacks = Array.from(pendingFrames.values());
        pendingFrames.clear();
        act(() => callbacks.forEach((callback) => callback(timestamp)));
    };

    const setReducedMotion = (matches: boolean) => {
        Object.defineProperty(window, 'matchMedia', {
            configurable: true,
            value: jest.fn().mockReturnValue({
                matches,
                media: '(prefers-reduced-motion: reduce)',
                onchange: null,
                addListener: jest.fn(),
                removeListener: jest.fn(),
                addEventListener: jest.fn(),
                removeEventListener: jest.fn(),
                dispatchEvent: jest.fn(),
            }),
        });
    };

    beforeEach(() => {
        detectedGame = null;
        mockedUseDesktopGame.mockImplementation(() => ({
            detectedGame,
            detectionStatus: detectedGame ? 'detected' : 'not-detected',
            error: null,
        }));
        nextFrameId = 1;
        pendingFrames = new Map();
        requestAnimationFrameMock = jest.fn((callback: FrameRequestCallback) => {
            const frameId = nextFrameId;
            nextFrameId += 1;
            pendingFrames.set(frameId, callback);
            return frameId;
        });
        cancelAnimationFrameMock = jest.fn((frameId: number) => pendingFrames.delete(frameId));
        Object.defineProperty(window, 'requestAnimationFrame', {
            configurable: true,
            value: requestAnimationFrameMock,
        });
        Object.defineProperty(window, 'cancelAnimationFrame', {
            configurable: true,
            value: cancelAnimationFrameMock,
        });
        setReducedMotion(false);
    });

    it('renders a compact HUD with no conventional telemetry charts or axes', () => {
        const { container } = render(
            <DriverExpertComparisonGraph data={completeData} title="Comparison" />,
        );

        expect(screen.getByRole('img', {
            name: 'Track replay showing Driver and Expert trajectories',
        })).toBeInTheDocument();
        expect(screen.getByTestId('driver-track-path')).toBeInTheDocument();
        expect(screen.getByTestId('expert-track-path')).toBeInTheDocument();
        expect(screen.getByTestId('driver-panel')).toHaveStyle({
            '--identity-color': DRIVER_COMPARISON_COLOR,
        });
        expect(screen.getByTestId('expert-panel')).toHaveStyle({
            '--identity-color': EXPERT_COMPARISON_COLOR,
        });
        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(container.querySelector('canvas')).not.toBeInTheDocument();
        expect(container.querySelector('[data-testid^="comparison-graph-"]')).not.toBeInTheDocument();
        expect(screen.queryByText('Segment progress (%)')).not.toBeInTheDocument();
        expect(screen.queryByText('Track X')).not.toBeInTheDocument();
    });

    it.each(['ac', 'iracing', null] as const)(
        'uses driver X/Y coordinates when the detected game is %s',
        (game) => {
            detectedGame = game;
            setReducedMotion(true);
            render(<DriverExpertComparisonGraph data={completeData} />);

            expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
            expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '50');
        },
    );

    it('uses driver X/Z for ACC while expert paths and markers remain on X/Y', () => {
        detectedGame = 'acc';
        setReducedMotion(true);
        const { rerender } = render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '300');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '60');
        const expertPath = screen.getByTestId('expert-track-path').getAttribute('d');
        const expertMarkerY = screen.getByTestId('expert-position-marker')
            .querySelector('circle:last-child')?.getAttribute('cy');

        const changedExpertZ = {
            samples: completeData.samples.map((sample) => ({
                ...sample,
                expertTrajectory: {
                    ...sample.expertTrajectory,
                    z: sample.expertTrajectory.z * -100,
                },
            })),
        };
        rerender(<DriverExpertComparisonGraph data={changedExpertZ} />);

        expect(screen.getByTestId('expert-track-path')).toHaveAttribute('d', expertPath);
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '60');
        expect(screen.getByTestId('expert-position-marker')
            .querySelector('circle:last-child')).toHaveAttribute('cy', expertMarkerY);
    });

    it('reprojects an existing comparison payload when the detected game changes', () => {
        setReducedMotion(true);
        const { rerender } = render(<DriverExpertComparisonGraph data={completeData} />);
        const xyPath = screen.getByTestId('driver-track-path').getAttribute('d');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '50');

        detectedGame = 'acc';
        rerender(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '300');
        expect(screen.getByTestId('driver-track-path').getAttribute('d')).not.toBe(xyPath);
    });

    it('requires finite coordinates for the active driver plane and expert X/Y plane', () => {
        const xyOnlyData = {
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrajectory: { x: 1, y: 2 },
                expertTrajectory: { x: 3, y: 4, z: Number.NaN },
            }],
        };
        const { rerender } = render(<DriverExpertComparisonGraph data={xyOnlyData} />);
        expect(screen.getByTestId('comparison-track-map')).toBeInTheDocument();

        detectedGame = 'acc';
        rerender(<DriverExpertComparisonGraph data={xyOnlyData} />);
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent(
            'Track data unavailable',
        );

        rerender(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, z: 5 },
            }],
        }} />);
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent(
            'Track data unavailable',
        );
    });

    it('normalizes complete timed payloads and preserves every finite source axis independently', () => {
        expect(normalizeDriverExpertComparisonData({
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 1_000,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, y: 5, z: Number.POSITIVE_INFINITY },
            }, {
                driverTimeMs: 200,
                expertTimeMs: 1_100,
                driverTrajectory: { x: Number.NaN, y: 6, z: 7 },
                expertTrajectory: { x: 8, y: Number.NaN, z: 9 },
            }],
        })).toEqual({
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 1_000,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, y: 5 },
            }, {
                driverTimeMs: 200,
                expertTimeMs: 1_100,
                expertTrajectory: { x: 8, z: 9 },
            }],
        });
    });

    it.each([
        ['missing driver time', [
            { expertTimeMs: 1_000 },
        ]],
        ['missing expert time', [
            { driverTimeMs: 100 },
        ]],
        ['non-finite driver time', [
            { driverTimeMs: Number.POSITIVE_INFINITY, expertTimeMs: 1_000 },
        ]],
        ['non-finite expert time', [
            { driverTimeMs: 100, expertTimeMs: Number.NaN },
        ]],
        ['repeated driver time', [
            { driverTimeMs: 100, expertTimeMs: 1_000 },
            { driverTimeMs: 100, expertTimeMs: 1_100 },
        ]],
        ['decreasing expert time', [
            { driverTimeMs: 100, expertTimeMs: 1_000 },
            { driverTimeMs: 200, expertTimeMs: 900 },
        ]],
    ])('rejects the complete payload for %s', (_case, samples) => {
        expect(normalizeDriverExpertComparisonData({ samples })).toBeUndefined();
    });

    it('reports the slower normalized clock duration without including absolute offsets', () => {
        expect(getDriverExpertReplayDurationMs(completeData)).toBe(3_000);
    });

    it('maps pedal values over the exact clockwise 1-to-3-o’clock sweep with clamping', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-gauge-angle', '0');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-gauge-angle', '-60');
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-gauge-angle', '0');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-gauge-angle', '-60');
    });

    it('uses each normalized irregular clock at 1x, interpolates continuous values, and steps gears', () => {
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '0');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '10');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');

        runAnimationFrame(1000);
        runAnimationFrame(1750);

        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '750');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.75s / 3.00s');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.375');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '37.5');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '19.375');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('3');

        runAnimationFrame(2500);

        expect(screen.getByTestId('replay-progress')).toHaveTextContent('1.50s / 3.00s');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.675');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0.325');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '62.5');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '28.75');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('3');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('3');

        runAnimationFrame(3500);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '60');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('6');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '87.5');

        runAnimationFrame(4000);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('3.00s / 3.00s');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '60');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('5');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('6');
        expect(pendingFrames.size).toBe(0);
    });

    it('holds the final state, restarts on remount, and cancels a pending frame on unmount', () => {
        const firstMount = render(<DriverExpertComparisonGraph data={completeData} />);
        runAnimationFrame(0);
        runAnimationFrame(3000);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(requestAnimationFrameMock).toHaveBeenCalledTimes(2);

        firstMount.unmount();
        const secondMount = render(<DriverExpertComparisonGraph data={completeData} />);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0');

        const pendingId = nextFrameId - 1;
        secondMount.unmount();
        expect(cancelAnimationFrameMock).toHaveBeenCalledWith(pendingId);
        expect(pendingFrames.size).toBe(0);
    });

    it('renders available pedals while showing muted placeholders for missing channels', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: [{ driverTimeMs: 100, expertTimeMs: 500, driverGas: 0.5, expertGas: 0.6 }],
        }} />);

        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent('Track data unavailable');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.5');
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-value', '0.6');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-state', 'unavailable');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-state', 'unavailable');
        expect(screen.getByTestId('driver-gear')).toHaveAttribute('data-state', 'unavailable');
        expect(screen.queryByText(/^Expert comparison unavailable$/)).not.toBeInTheDocument();
    });

    it('handles a single sample and an empty normalized payload without scheduling movement', () => {
        const single = render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 4_000,
                expertTimeMs: 8_000,
                driverGas: 0.4,
                expertGas: 0.5,
                driverGear: 3,
                expertGear: 4,
            }],
        }} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 0.00s');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.4');
        expect(requestAnimationFrameMock).not.toHaveBeenCalled();
        single.unmount();

        const normalized = normalizeDriverExpertComparisonData({
            samples: [{ driverTimeMs: 0, expertTimeMs: 0, Physics_gas: 1, expert_optimal_throttle: 1 }],
        });
        const unavailable = render(<DriverExpertComparisonGraph data={normalized!} />);

        expect(normalized).toEqual({ samples: [{ driverTimeMs: 0, expertTimeMs: 0 }] });
        expect(screen.getByText(/^Expert comparison unavailable$/)).toBeInTheDocument();
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.queryAllByRole('meter')).toHaveLength(0);

        unavailable.unmount();
        render(<DriverExpertComparisonGraph data={{ samples: [] }} />);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('No data');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 0.00s');
    });

    it('skips directly to the final sample when reduced motion is requested', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('3.00s / 3.00s');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('5');
        expect(requestAnimationFrameMock).not.toHaveBeenCalled();
    });

    it('honors the existing width and layout compatibility props', () => {
        render(
            <DriverExpertComparisonGraph
                data={completeData}
                width={720}
                layout={{ chartHeight: 160, trajectoryHeight: 200, minColumnWidth: 320 }}
            />,
        );

        expect(screen.getByTestId('driver-expert-comparison')).toHaveStyle({ width: '720px' });
        expect(screen.getByTestId('driver-expert-comparison')).toHaveStyle(
            '--driver-expert-min-column-width: 320px',
        );
        expect(screen.getByLabelText('Track replay')).toHaveStyle({ height: '200px' });
        expect(screen.getByTestId('pedal-panel-region')).toHaveStyle({ minHeight: '160px' });
        expect(within(screen.getByTestId('driver-panel')).getByText('Driver')).toBeInTheDocument();
    });
});
