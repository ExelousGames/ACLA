import React from 'react';
import { act, render, screen } from '@testing-library/react';
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
            driverTrackPosition: 0.2,
            expertTrackPosition: 0.2,
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
            driverTrackPosition: 0.25,
            expertTrackPosition: 0.26,
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
            expertTimeMs: 53_000,
            driverTrackPosition: 0.3,
            expertTrackPosition: 0.32,
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

const parseMatrix = (element: Element): [number, number, number, number, number, number] => {
    const match = element.getAttribute('transform')?.match(/^matrix\(([^)]+)\)$/);
    if (!match) throw new Error('Expected an SVG matrix transform');
    const values = match[1].trim().split(/[ ,]+/).map(Number);
    if (values.length !== 6 || values.some((value) => !Number.isFinite(value))) {
        throw new Error('Expected a finite six-value SVG matrix transform');
    }
    return values as [number, number, number, number, number, number];
};

const parseTranslate = (element: Element): { x: number; y: number } => {
    const match = element.getAttribute('transform')?.match(/^translate\(([^)]+)\)$/);
    if (!match) throw new Error('Expected an SVG translate transform');
    const [x, y] = match[1].trim().split(/[ ,]+/).map(Number);
    return { x, y };
};

const expectCameraLockedOn = (identity: 'driver' | 'expert') => {
    const camera = screen.getByTestId('comparison-camera-layer');
    const marker = screen.getByTestId(`${identity}-position-marker`).querySelector('circle');
    if (!marker) throw new Error(`Expected a ${identity} marker`);

    expect(camera).toHaveAttribute('data-camera-target', identity);
    expect(Number(marker.getAttribute('cx'))).toBeCloseTo(
        Number(camera.getAttribute('data-camera-anchor-x')),
        3,
    );
    expect(Number(marker.getAttribute('cy'))).toBeCloseTo(
        Number(camera.getAttribute('data-camera-anchor-y')),
        3,
    );
};

const expectCameraFacingDriverDirection = () => {
    const camera = screen.getByTestId('comparison-camera-layer');
    const [a, b, c, d] = parseMatrix(camera);
    const headingX = Number(camera.getAttribute('data-heading-x'));
    const headingY = Number(camera.getAttribute('data-heading-y'));
    if (![headingX, headingY].every(Number.isFinite)) {
        throw new Error('Expected a finite driver heading');
    }

    // Plotting Y is inverted during SVG projection. The camera should rotate that
    // projected tangent onto the negative screen Y axis (straight ahead/up).
    const screenHeadingX = (a * headingX) + (c * -headingY);
    const screenHeadingY = (b * headingX) + (d * -headingY);
    expect(screenHeadingX).toBeCloseTo(0, 3);
    expect(screenHeadingY).toBeLessThan(0);
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
        setReducedMotion(true);
        const { container } = render(
            <DriverExpertComparisonGraph data={completeData} title="Comparison" />,
        );

        expect(screen.getByRole('img', {
            name: 'Track replay showing Driver and Expert trajectories',
        })).toBeInTheDocument();
        expect(screen.getByTestId('driver-track-path')).toBeInTheDocument();
        expect(screen.getByTestId('expert-track-path')).toBeInTheDocument();
        expect(screen.getByTestId('driver-telemetry-pod')).toHaveStyle({
            '--identity-color': DRIVER_COMPARISON_COLOR,
        });
        expect(screen.getByTestId('expert-telemetry-pod')).toHaveStyle({
            '--identity-color': EXPERT_COMPARISON_COLOR,
        });
        expect(screen.getByTestId('driver-telemetry-pod')).toHaveAttribute(
            'data-card-width',
            '360',
        );
        expect(screen.getByTestId('driver-telemetry-pod')).toHaveAttribute(
            'data-card-height',
            '229.5',
        );
        expect(
            screen.getByTestId('driver-telemetry-pod').querySelector('g[transform="scale(2.25)"]'),
        ).toBeInTheDocument();
        expect(screen.getByTestId('driver-telemetry-pod').closest('svg')).toBe(
            screen.getByTestId('comparison-track-map'),
        );
        const cameraLayer = screen.getByTestId('comparison-camera-layer');
        const cameraOverlay = screen.getByTestId('comparison-camera-overlay');
        expect(cameraLayer).toContainElement(screen.getByTestId('driver-track-path'));
        expect(cameraLayer).toContainElement(screen.getByTestId('expert-track-path'));
        expect(cameraOverlay).toContainElement(screen.getByTestId('driver-telemetry-leader'));
        expect(cameraOverlay).toContainElement(screen.getByTestId('expert-position-marker'));
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();
        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(screen.queryByTestId('pedal-panel-region')).not.toBeInTheDocument();
        expect(container.querySelector('canvas')).not.toBeInTheDocument();
        expect(container.querySelector('[data-testid^="comparison-graph-"]')).not.toBeInTheDocument();
        expect(screen.queryByText('Driver / Expert')).not.toBeInTheDocument();
        expect(screen.queryByText('Track replay')).not.toBeInTheDocument();
        expect(screen.queryByText('Driver trace')).not.toBeInTheDocument();
        expect(screen.queryByText('Expert trace')).not.toBeInTheDocument();
        expect(screen.queryByText('Segment progress (%)')).not.toBeInTheDocument();
        expect(screen.queryByText('Track X')).not.toBeInTheDocument();
    });

    it('establishes the full trajectory before zooming and fading in competitor status', () => {
        render(<DriverExpertComparisonGraph data={completeData} />);

        const camera = screen.getByTestId('comparison-camera-layer');
        const overlay = screen.getByTestId('comparison-camera-overlay');
        const driverPath = screen.getByTestId('driver-track-path').getAttribute('d');
        const expertPath = screen.getByTestId('expert-track-path').getAttribute('d');

        expect(camera).toHaveAttribute('transform', 'matrix(1 0 0 1 0 0)');
        expect(camera).toHaveAttribute('data-camera-phase', 'overview');
        expect(camera).toHaveAttribute('data-camera-progress', '0');
        expect(overlay).toHaveAttribute('data-status-visibility', 'hidden');
        expect(overlay).toHaveAttribute('aria-hidden', 'true');
        expect(overlay).toHaveStyle({ opacity: '0' });
        expect(screen.queryAllByRole('meter')).toHaveLength(0);

        runAnimationFrame(0);
        runAnimationFrame(1_000);

        expect(camera).toHaveAttribute('transform', 'matrix(1 0 0 1 0 0)');
        expect(camera).toHaveAttribute('data-camera-phase', 'overview');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');

        runAnimationFrame(1_375);

        expect(camera).toHaveAttribute('data-camera-phase', 'focusing');
        expect(camera).toHaveAttribute('data-camera-progress', '0.5');
        expect(camera).not.toHaveAttribute('transform', 'matrix(1 0 0 1 0 0)');
        expect(overlay).toHaveAttribute('data-status-visibility', 'fading');
        expect(overlay).not.toHaveAttribute('aria-hidden');
        expect(overlay).toHaveStyle({ opacity: '0.5' });
        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(screen.getByTestId('driver-track-path')).toHaveAttribute('d', driverPath);
        expect(screen.getByTestId('expert-track-path')).toHaveAttribute('d', expertPath);

        runAnimationFrame(1_750);

        expect(camera).toHaveAttribute('data-camera-phase', 'following');
        expect(camera).toHaveAttribute('data-camera-progress', '1');
        expect(overlay).toHaveAttribute('data-status-visibility', 'visible');
        expect(overlay).toHaveStyle({ opacity: '1' });
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
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

    it('flips both vertical axes for ACC using driver X/Z and expert X/Y', () => {
        detectedGame = 'acc';
        setReducedMotion(true);
        const { rerender } = render(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-300');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '-60');
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
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '-60');
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

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-300');
        expect(screen.getByTestId('driver-track-path').getAttribute('d')).not.toBe(xyPath);
    });

    it('uses an explicit session game instead of detector updates', () => {
        detectedGame = 'iracing';
        setReducedMotion(true);
        const view = render(<DriverExpertComparisonGraph data={completeData} game="acc" />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-300');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '-60');

        detectedGame = 'ac';
        view.rerender(<DriverExpertComparisonGraph data={completeData} game="acc" />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-300');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '-60');
    });

    it('renders only competitors with finite coordinates on their active trajectory plane', () => {
        const xyOnlyData = {
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 1, y: 2 },
                expertTrajectory: { x: 3, y: 4, z: Number.NaN },
            }],
        };
        const { rerender } = render(<DriverExpertComparisonGraph data={xyOnlyData} />);
        expect(screen.getByTestId('comparison-track-map')).toBeInTheDocument();

        detectedGame = 'acc';
        rerender(<DriverExpertComparisonGraph data={xyOnlyData} />);
        expect(screen.queryByTestId('driver-position-marker')).not.toBeInTheDocument();
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '-4');
        expect(screen.getByTestId('expert-telemetry-pod')).toBeInTheDocument();

        rerender(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, z: 5 },
            }],
        }} />);
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-3');
        expect(screen.getByTestId('driver-telemetry-pod')).toBeInTheDocument();
        expect(screen.queryByTestId('expert-position-marker')).not.toBeInTheDocument();
    });

    it('normalizes complete timed payloads and preserves every finite source axis independently', () => {
        expect(normalizeDriverExpertComparisonData({
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.1,
                expertTrackPosition: 0.15,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, y: 5, z: Number.POSITIVE_INFINITY },
            }, {
                driverTimeMs: 200,
                expertTimeMs: 1_100,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.25,
                driverTrajectory: { x: Number.NaN, y: 6, z: 7 },
                expertTrajectory: { x: 8, y: Number.NaN, z: 9 },
            }],
        })).toEqual({
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.1,
                expertTrackPosition: 0.15,
                driverTrajectory: { x: 1, y: 2, z: 3 },
                expertTrajectory: { x: 4, y: 5 },
            }, {
                driverTimeMs: 200,
                expertTimeMs: 1_100,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.25,
                expertTrajectory: { x: 8, z: 9 },
            }],
        });
    });

    it('preserves each service-aligned stream from its first sample', () => {
        const data = {
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 100,
                driverTrackPosition: 0.1,
                expertTrackPosition: 0.2,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory: { x: 100, y: 100 },
                driverGas: 0,
                expertGas: 0.2,
                driverBrake: 1,
                expertBrake: 0.8,
                driverGear: 2,
                expertGear: 3,
            }, {
                driverTimeMs: 1_000,
                expertTimeMs: 600,
                driverTrackPosition: 0.3,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 20, y: 20 },
                expertTrajectory: { x: 120, y: 120 },
                driverGas: 1,
                expertGas: 0.4,
                driverBrake: 0,
                expertBrake: 0.6,
                driverGear: 4,
                expertGear: 4,
            }, {
                driverTimeMs: 2_000,
                expertTimeMs: 1_100,
                driverTrackPosition: 0.5,
                expertTrackPosition: 0.6,
                driverTrajectory: { x: 40, y: 40 },
                expertTrajectory: { x: 140, y: 140 },
                driverGas: 0,
                expertGas: 0.6,
                driverBrake: 0.4,
                expertBrake: 0.4,
                driverGear: 5,
                expertGear: 5,
            }],
        };

        expect(normalizeDriverExpertComparisonData(data)?.samples[0]).toMatchObject({
            driverTrackPosition: 0.1,
            expertTrackPosition: 0.2,
        });
        expect(getDriverExpertReplayDurationMs(data)).toBe(2_000);
        render(<DriverExpertComparisonGraph data={data} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '0');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.1');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.2');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');

        runAnimationFrame(0);
        runAnimationFrame(2_750);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '20');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '140');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.3');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.6');

        runAnimationFrame(3_250);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '30');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '140');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.4');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.6');

        runAnimationFrame(3_750);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '40');
    });

    it('freezes the driver at its endpoint while the expert timeline continues', () => {
        const data = {
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory: { x: 100, y: 100 },
            }, {
                driverTimeMs: 1_000,
                expertTimeMs: 2_000,
                driverTrackPosition: 0.6,
                expertTrackPosition: 0.6,
                driverTrajectory: { x: 40, y: 40 },
                expertTrajectory: { x: 140, y: 140 },
            }],
        };
        expect(getDriverExpertReplayDurationMs(data)).toBe(2_000);
        render(<DriverExpertComparisonGraph data={data} />);
        runAnimationFrame(0);
        runAnimationFrame(2_750);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '40');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '120');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.4');
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(pendingFrames.size).toBe(1);

        runAnimationFrame(3_750);
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '40');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '140');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(pendingFrames.size).toBe(0);
    });

    it('unwraps each finish-line crossing independently and accepts repeated positions', () => {
        expect(normalizeDriverExpertComparisonData({
            samples: [
                { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: 0.98, expertTrackPosition: 0.01 },
                { driverTimeMs: 100, expertTimeMs: 100, driverTrackPosition: 0.01, expertTrackPosition: 0.04 },
                { driverTimeMs: 200, expertTimeMs: 200, driverTrackPosition: 0.04, expertTrackPosition: 0.07 },
            ],
        })).toBeDefined();
        expect(normalizeDriverExpertComparisonData({
            samples: [
                { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: 0.2, expertTrackPosition: 0.2 },
                { driverTimeMs: 100, expertTimeMs: 100, driverTrackPosition: 0.2, expertTrackPosition: 0.25 },
                { driverTimeMs: 200, expertTimeMs: 200, driverTrackPosition: 0.3, expertTrackPosition: 0.3 },
            ],
        })).toBeDefined();
    });

    it('accepts lap-timer resets that occur at a finish-line crossing', () => {
        const data = {
            samples: [
                {
                    driverTimeMs: 99_800,
                    expertTimeMs: 49_800,
                    driverTrackPosition: 0.98,
                    expertTrackPosition: 0.98,
                    driverTrajectory: { x: 0, y: 0 },
                    expertTrajectory: { x: 0, y: 1 },
                    driverGas: 0.4,
                    expertGas: 0.5,
                },
                {
                    driverTimeMs: 50,
                    expertTimeMs: 50_000,
                    driverTrackPosition: 0.01,
                    expertTrackPosition: 0.01,
                    driverTrajectory: { x: 1, y: 1 },
                    expertTrajectory: { x: 1, y: 2 },
                    driverGas: 0.6,
                    expertGas: 0.7,
                },
                {
                    driverTimeMs: 150,
                    expertTimeMs: 50_100,
                    driverTrackPosition: 0.04,
                    expertTrackPosition: 0.04,
                    driverTrajectory: { x: 2, y: 2 },
                    expertTrajectory: { x: 2, y: 3 },
                    driverGas: 0.8,
                    expertGas: 0.9,
                },
            ],
        };

        expect(normalizeDriverExpertComparisonData(data)).toBeDefined();
        expect(getDriverExpertReplayDurationMs(data)).toBeCloseTo(300);

        render(<DriverExpertComparisonGraph data={data} />);
        expect(screen.queryByText(/^Expert comparison unavailable$/)).not.toBeInTheDocument();
        expect(screen.getByTestId('driver-throttle-gauge')).toBeInTheDocument();
        expect(screen.getByTestId('expert-throttle-gauge')).toBeInTheDocument();
    });

    it.each([
        ['legacy singular position', [
            { driverTimeMs: 0, expertTimeMs: 0, trackPosition: 0.2 },
        ]],
        ['partial competitor position', [
            { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: 0.2, expertTrackPosition: 0.2 },
            { driverTimeMs: 100, expertTimeMs: 100, driverTrackPosition: 0.3 },
        ]],
        ['non-finite position', [
            { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: Number.NaN, expertTrackPosition: 0.2 },
        ]],
        ['out-of-range position', [
            { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: 1.1, expertTrackPosition: 0.2 },
        ]],
        ['unexplained backward motion', [
            { driverTimeMs: 0, expertTimeMs: 0, driverTrackPosition: 0.4, expertTrackPosition: 0.4 },
            { driverTimeMs: 100, expertTimeMs: 100, driverTrackPosition: 0.3, expertTrackPosition: 0.5 },
        ]],
    ])('rejects %s', (_case, samples) => {
        expect(normalizeDriverExpertComparisonData({ samples })).toBeUndefined();
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

    it('reports the slower endpoint after aligning both normalized clocks', () => {
        expect(getDriverExpertReplayDurationMs(completeData)).toBe(3_000);
    });

    it('maps clamped pedal values to curved gauge angles and percentages', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={completeData} />);

        const throttleGauge = screen.getByTestId('driver-throttle-gauge');
        const brakeGauge = screen.getByTestId('driver-brake-gauge');
        const [throttleLabel, throttleValue] = Array.from(
            throttleGauge.querySelectorAll('text'),
        );
        const [brakeLabel, brakeValue] = Array.from(brakeGauge.querySelectorAll('text'));

        expect(throttleGauge).toHaveAttribute('data-value', '1');
        expect(throttleGauge).toHaveAttribute('data-gauge-angle', '0');
        expect(throttleGauge).toHaveTextContent('100%');
        expect(throttleGauge).toHaveAttribute('transform', 'translate(12 27) scale(0.62)');
        expect(brakeGauge).toHaveAttribute('data-value', '0');
        expect(brakeGauge).toHaveAttribute('data-gauge-angle', '-60');
        expect(brakeGauge).toHaveTextContent('0%');
        expect(brakeGauge).toHaveAttribute('transform', 'translate(66 27) scale(0.62)');
        [throttleLabel, brakeLabel].forEach((label) => {
            expect(label).toHaveAttribute('x', '66');
            expect(label).toHaveAttribute('text-anchor', 'middle');
        });
        [throttleValue, brakeValue].forEach((value) => {
            expect(value).toHaveAttribute('x', '66');
            expect(value).toHaveAttribute('y', '58');
            expect(value).toHaveAttribute('text-anchor', 'middle');
            expect(value).toHaveAttribute('dominant-baseline', 'middle');
        });
        expect(screen.getByTestId('expert-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('expert-brake-gauge')).toHaveAttribute('data-value', '0');
    });

    it('uses each normalized irregular clock at 1x, interpolates continuous values, and steps gears', () => {
        render(<DriverExpertComparisonGraph data={completeData} />);

        const initialCameraTransform = screen.getByTestId('comparison-camera-layer')
            .getAttribute('transform');
        const completeDriverPath = screen.getByTestId('driver-track-path').getAttribute('d');
        const completeExpertPath = screen.getByTestId('expert-track-path').getAttribute('d');
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '0');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '10');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.2');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.2');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');
        expectCameraLockedOn('driver');

        runAnimationFrame(0);
        runAnimationFrame(1_750);
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();

        runAnimationFrame(2_500);

        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '750');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.75s / 3.00s');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.375');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-gauge-angle', '-37.5');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveTextContent('38%');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '37.5');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '19.375');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.237');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.223');
        expect(screen.getByTestId('comparison-camera-layer')).not.toHaveAttribute(
            'transform',
            initialCameraTransform,
        );
        expect(screen.getByTestId('driver-track-path')).toHaveAttribute('d', completeDriverPath);
        expect(screen.getByTestId('expert-track-path')).toHaveAttribute('d', completeExpertPath);
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('3');
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();

        runAnimationFrame(3_250);

        expect(screen.getByTestId('replay-progress')).toHaveTextContent('1.50s / 3.00s');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.675');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0.325');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '62.5');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '28.75');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('3');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('3');

        runAnimationFrame(4_250);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '47.5');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('4');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '87.5');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.287');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.29');
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();

        runAnimationFrame(4_750);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('3.00s / 3.00s');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-y', '60');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('5');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('6');
        expect(pendingFrames.size).toBe(0);
    });

    it('smoothly rotates the camera toward the next driver forward axis', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0,
                expertTrackPosition: 0,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory: { x: 0, y: 10 },
            }, {
                driverTimeMs: 1_000,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.5,
                expertTrackPosition: 0.5,
                driverTrajectory: { x: 100, y: 0 },
                expertTrajectory: { x: 100, y: 10 },
            }, {
                driverTimeMs: 2_000,
                expertTimeMs: 2_000,
                driverTrackPosition: 1,
                expertTrackPosition: 1,
                driverTrajectory: { x: 100, y: 100 },
                expertTrajectory: { x: 100, y: 110 },
            }],
        }} />);

        runAnimationFrame(0);
        runAnimationFrame(1_750);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-rotation',
            '-90',
        );

        runAnimationFrame(2_250);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-rotation',
            '-45',
        );

        runAnimationFrame(2_750);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-rotation',
            '0',
        );
        expectCameraFacingDriverDirection();
    });

    it('holds the final state, restarts on remount, and cancels a pending frame on unmount', () => {
        const firstMount = render(<DriverExpertComparisonGraph data={completeData} />);
        runAnimationFrame(0);
        runAnimationFrame(4_750);
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

    it('does not render telemetry pods when trajectory data is unavailable', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 500,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverGas: 0.5,
                expertGas: 0.6,
            }],
        }} />);

        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent('Trajectory data unavailable');
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryByTestId('expert-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryByTestId('comparison-camera-layer')).not.toBeInTheDocument();
        expect(screen.queryAllByRole('meter')).toHaveLength(0);
        expect(screen.queryByText(/^Expert comparison unavailable$/)).not.toBeInTheDocument();
    });

    it('keeps telemetry-only replays pod-free while their clock advances', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
                driverGas: 0,
                expertGas: 0.2,
            }, {
                driverTimeMs: 1_000,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverGas: 1,
                expertGas: 0.8,
            }],
        }} />);

        runAnimationFrame(0);
        runAnimationFrame(500);

        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.50s / 1.00s');
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryByTestId('expert-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryAllByRole('meter')).toHaveLength(0);
    });

    it('keeps Driver above-right and Expert below-left without flipping or clamping', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory: { x: 0, y: 0 },
            }, {
                driverTimeMs: 1_000,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 100, y: 0 },
                expertTrajectory: { x: 100, y: 0 },
            }],
        }} />);

        const driverPod = screen.getByTestId('driver-telemetry-pod');
        const expertPod = screen.getByTestId('expert-telemetry-pod');
        const driverMarker = screen.getByTestId('driver-position-marker').querySelector('circle');
        const expertMarker = screen.getByTestId('expert-position-marker').querySelector('circle');
        if (!driverMarker || !expertMarker) throw new Error('Expected both position markers');
        const driverPosition = parseTranslate(driverPod);
        const expertPosition = parseTranslate(expertPod);

        expect(driverPosition.x - Number(driverMarker.getAttribute('cx'))).toBeCloseTo(41, 3);
        expect(driverPosition.y - Number(driverMarker.getAttribute('cy'))).toBeCloseTo(-243.5, 3);
        expect(expertPosition.x - Number(expertMarker.getAttribute('cx'))).toBeCloseTo(-401, 3);
        expect(expertPosition.y - Number(expertMarker.getAttribute('cy'))).toBeCloseTo(14, 3);
        expect(driverPod).not.toHaveAttribute('data-placement');
        expect(driverPod).not.toHaveAttribute('data-clamped');
        expect(expertPod).not.toHaveAttribute('data-placement');
        expect(expertPod).not.toHaveAttribute('data-clamped');
    });

    it('keeps the camera locked on the driver regardless of expert separation', () => {
        setReducedMotion(true);
        const comparisonData = (expertX: number) => ({
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory: { x: expertX, y: 0 },
                driverGas: 0.4,
                expertGas: 0.6,
                driverBrake: 0.3,
                expertBrake: 0.2,
                driverGear: 3,
                expertGear: 4,
            }],
        });
        const view = render(<DriverExpertComparisonGraph data={comparisonData(0)} />);
        const initialTransform = screen.getByTestId('comparison-camera-layer')
            .getAttribute('transform');

        expectCameraLockedOn('driver');
        view.rerender(<DriverExpertComparisonGraph data={comparisonData(10_000)} />);

        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'transform',
            initialTransform,
        );
        expectCameraLockedOn('driver');
    });

    it('updates the driver anchor for the rendered panel aspect ratio without changing zoom', () => {
        setReducedMotion(true);
        let notifyResize: ((width: number, height: number) => void) | undefined;
        const originalResizeObserver = window.ResizeObserver;
        const observe = jest.fn();
        const disconnect = jest.fn();
        const ResizeObserverMock = jest.fn(function mockResizeObserver(
            callback: ResizeObserverCallback,
        ) {
            notifyResize = (width, height) => callback([{
                contentRect: { width, height },
            } as ResizeObserverEntry], this as unknown as ResizeObserver);
            return { observe, disconnect, unobserve: jest.fn() };
        });
        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: ResizeObserverMock,
        });

        try {
            render(<DriverExpertComparisonGraph data={{
                samples: [{
                    driverTimeMs: 0,
                    expertTimeMs: 0,
                    driverTrackPosition: 0.4,
                    expertTrackPosition: 0.4,
                    driverTrajectory: { x: 0, y: 0 },
                    expertTrajectory: { x: 0, y: 0 },
                    driverGas: 0.4,
                    expertGas: 0.6,
                    driverBrake: 0.3,
                    expertBrake: 0.2,
                    driverGear: 3,
                    expertGear: 4,
                }],
            }} />);

            const initialMatrix = parseMatrix(screen.getByTestId('comparison-camera-layer'));
            const initialScale = Math.hypot(initialMatrix[0], initialMatrix[1]);
            expect(initialScale).toBeCloseTo(4, 6);
            act(() => notifyResize?.(400, 260));

            expect(screen.getByTestId('comparison-track-map')).toHaveAttribute(
                'viewBox',
                '0 0 760 494',
            );
            const resizedMatrix = parseMatrix(screen.getByTestId('comparison-camera-layer'));
            expect(Math.hypot(resizedMatrix[0], resizedMatrix[1])).toBeCloseTo(initialScale, 6);
            expectCameraLockedOn('driver');
            expect(observe).toHaveBeenCalledWith(screen.getByTestId('comparison-track-map'));
        } finally {
            Object.defineProperty(window, 'ResizeObserver', {
                configurable: true,
                value: originalResizeObserver,
            });
        }
    });

    it.each(['driver', 'expert'] as const)(
        'follows a lone %s marker and card when the other trajectory is unavailable',
        (identity) => {
            setReducedMotion(true);
            const otherIdentity = identity === 'driver' ? 'expert' : 'driver';
            render(<DriverExpertComparisonGraph data={{
                samples: [{
                    driverTimeMs: 0,
                    expertTimeMs: 0,
                    driverTrackPosition: 0.4,
                    expertTrackPosition: 0.4,
                    ...(identity === 'driver'
                        ? { driverTrajectory: { x: 20, y: 40 } }
                        : { expertTrajectory: { x: 20, y: 40 } }),
                    driverGas: 0.4,
                    expertGas: 0.6,
                    driverBrake: 0.3,
                    expertBrake: 0.2,
                    driverGear: 3,
                    expertGear: 4,
                }],
            }} />);

            expect(screen.getByTestId('comparison-track-map')).toBeInTheDocument();
            expect(screen.getByTestId(`${identity}-track-path`)).toBeInTheDocument();
            expect(screen.getByTestId(`${identity}-position-marker`)).toBeInTheDocument();
            expect(screen.getByTestId(`${identity}-telemetry-pod`)).toBeInTheDocument();
            expect(screen.queryByTestId(`${otherIdentity}-track-path`)).not.toBeInTheDocument();
            expect(screen.queryByTestId(`${otherIdentity}-position-marker`)).not.toBeInTheDocument();
            expect(screen.queryByTestId(`${otherIdentity}-telemetry-pod`)).not.toBeInTheDocument();
            expectCameraLockedOn(identity);
        },
    );

    it('keeps the default viewport when neither trajectory marker is available', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverGas: 0.4,
                expertGas: 0.6,
            }],
        }} />);

        expect(screen.getByTestId('comparison-trajectory-unavailable')).toHaveAttribute(
            'viewBox',
            '0 0 760 220',
        );
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent(
            'Trajectory data unavailable',
        );
        expect(screen.queryByTestId('comparison-camera-layer')).not.toBeInTheDocument();
    });

    it('handles a single sample and an empty normalized payload without scheduling movement', () => {
        const single = render(<DriverExpertComparisonGraph data={{
            samples: [{
                driverTimeMs: 4_000,
                expertTimeMs: 8_000,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverGas: 0.4,
                expertGas: 0.5,
                driverGear: 3,
                expertGear: 4,
            }],
        }} />);

        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 0.00s');
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();
        expect(requestAnimationFrameMock).not.toHaveBeenCalled();
        single.unmount();

        const normalized = normalizeDriverExpertComparisonData({
            samples: [{ driverTimeMs: 0, expertTimeMs: 0, Physics_gas: 1, expert_optimal_throttle: 1 }],
        });
        const unavailable = render(<DriverExpertComparisonGraph data={{ samples: [] }} />);

        expect(normalized).toBeUndefined();
        expect(screen.getByText(/^Expert comparison unavailable$/)).toBeInTheDocument();
        expect(screen.getByTestId('replay-status')).toHaveTextContent('No data');
        expect(screen.queryAllByRole('meter')).toHaveLength(0);
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();

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

    it('uses trajectoryHeight while retaining deprecated layout fields as no-ops', () => {
        render(
            <DriverExpertComparisonGraph
                data={completeData}
                width={720}
                layout={{ chartHeight: 160, trajectoryHeight: 200, minColumnWidth: 320 }}
            />,
        );

        const comparison = screen.getByTestId('driver-expert-comparison');
        expect(comparison).toHaveStyle({ width: '720px' });
        expect(comparison.style.getPropertyValue('--driver-expert-min-column-width')).toBe('');
        expect(screen.getByLabelText('Track replay')).toHaveStyle({ height: '200px' });
        expect(screen.queryByTestId('pedal-panel-region')).not.toBeInTheDocument();
        expect(screen.getByTestId('driver-telemetry-pod')).toHaveTextContent('Driver');
    });
});
