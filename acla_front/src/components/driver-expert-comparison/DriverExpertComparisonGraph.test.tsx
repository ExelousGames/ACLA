import React from 'react';
import { act, render, screen, within } from '@testing-library/react';
import {
    DRIVER_COMPARISON_COLOR,
    EXPERT_COMPARISON_COLOR,
    DriverExpertComparisonGraph,
    normalizeDriverExpertComparisonData,
} from './DriverExpertComparisonGraph';

const completeData = {
    samples: [
        {
            progress: 0,
            trackPosition: 0.2,
            driverTrajectory: { x: 0, z: 0 },
            expertTrajectory: { x: 0, z: 10 },
            driverGas: 0,
            expertGas: 0.2,
            driverBrake: 1,
            expertBrake: 0.8,
            driverGear: 2,
            expertGear: 3,
        },
        {
            progress: 50,
            trackPosition: 0.25,
            driverTrajectory: { x: 50, z: 25 },
            expertTrajectory: { x: 50, z: 35 },
            driverGas: 0.5,
            expertGas: 0.6,
            driverBrake: 0.5,
            expertBrake: 0.4,
            driverGear: 3,
            expertGear: 4,
        },
        {
            progress: 100,
            trackPosition: 0.3,
            driverTrajectory: { x: 100, z: 50 },
            expertTrajectory: { x: 100, z: 60 },
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

    it('maps pedal values over the exact clockwise 2-to-6-o’clock sweep with clamping', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-gauge-angle', '90');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-gauge-angle', '-30');
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-gauge-angle', '90');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-gauge-angle', '-30');
    });

    it('interpolates pedals and both markers while stepping gears at sample boundaries', () => {
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '0');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');

        runAnimationFrame(1000);
        runAnimationFrame(1750);

        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '25');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.25');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '25');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-z', '22.5');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('3');

        runAnimationFrame(2500);

        expect(screen.getByTestId('replay-progress')).toHaveTextContent('50%');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.5');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0.5');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '50');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-z', '35');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('3');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('4');

        runAnimationFrame(4000);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '100');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-z', '60');
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
            samples: [{ progress: 0, driverGas: 0.5, expertGas: 0.6 }],
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
                progress: 40,
                driverGas: 0.4,
                expertGas: 0.5,
                driverGear: 3,
                expertGear: 4,
            }],
        }} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('40%');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.4');
        expect(requestAnimationFrameMock).not.toHaveBeenCalled();
        single.unmount();

        const normalized = normalizeDriverExpertComparisonData({
            samples: [{ progress: 0, Physics_gas: 1, expert_optimal_throttle: 1 }],
        });
        const unavailable = render(<DriverExpertComparisonGraph data={normalized!} />);

        expect(normalized).toEqual({ samples: [{ progress: 0 }] });
        expect(screen.getByText(/^Expert comparison unavailable$/)).toBeInTheDocument();
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.queryAllByRole('meter')).toHaveLength(0);

        unavailable.unmount();
        render(<DriverExpertComparisonGraph data={{ samples: [] }} />);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('No data');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0%');
    });

    it('skips directly to the final sample when reduced motion is requested', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('100%');
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
