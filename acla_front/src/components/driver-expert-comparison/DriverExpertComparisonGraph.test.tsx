import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import {
    DRIVER_COMPARISON_COLOR,
    EXPERT_COMPARISON_COLOR,
    DriverExpertComparisonGraph,
    driverExpertComparisonOverlayRenderer,
    getDriverExpertReplayDurationMs,
    getDriverExpertComparisonUnavailableDiagnostics,
    hasComparableDriverExpertData,
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
    const match = element.getAttribute('data-camera-transform')?.match(/^matrix\(([^)]+)\)$/);
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

const expectTelemetryPodWithinViewport = (identity: 'driver' | 'expert') => {
    const pod = screen.getByTestId(`${identity}-telemetry-pod`);
    const body = pod.querySelector('rect');
    if (!body) throw new Error('Expected a telemetry card body');
    const { x, y } = parseTranslate(pod);
    const width = Number(body.getAttribute('width'));
    const height = Number(body.getAttribute('height'));
    const [, , viewportWidth, viewportHeight] = screen.getByTestId('comparison-track-map')
        .getAttribute('viewBox')!.split(' ').map(Number);
    const right = x + width;
    const bottom = y + height;
    expect(width).toBeGreaterThan(0);
    expect(height).toBeGreaterThan(0);
    expect(width / height).toBeCloseTo(160 / 102, 5);
    expect(x).toBeGreaterThanOrEqual(12 - 0.001);
    expect(y).toBeGreaterThanOrEqual(12 - 0.001);
    expect(right).toBeLessThanOrEqual(viewportWidth - 12 + 0.001);
    expect(bottom).toBeLessThanOrEqual(viewportHeight - 12 + 0.001);

    const contentScale = Number(pod.querySelector('g')?.getAttribute('transform')
        ?.match(/^scale\(([^)]+)\)$/)?.[1]);
    expect(contentScale * 160).toBeCloseTo(width, 5);
    expect(contentScale * 102).toBeCloseTo(height, 5);

    const leader = screen.getByTestId(`${identity}-telemetry-leader`);
    const marker = screen.getByTestId(`${identity}-position-marker`).querySelector('circle')!;
    expect(x + Number(leader.getAttribute('x1'))).toBeCloseTo(Number(marker.getAttribute('cx')), 3);
    expect(y + Number(leader.getAttribute('y1'))).toBeCloseTo(Number(marker.getAttribute('cy')), 3);
    const endX = Number(leader.getAttribute('x2'));
    const endY = Number(leader.getAttribute('y2'));
    expect(endX).toBeGreaterThanOrEqual(0);
    expect(endX).toBeLessThanOrEqual(width);
    expect(endY).toBeGreaterThanOrEqual(0);
    expect(endY).toBeLessThanOrEqual(height);
    expect(Math.min(endX, width - endX, endY, height - endY)).toBeCloseTo(0, 5);
    return { x, y, width, height, right, bottom };
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

    it('plays stored narration on the first animation frame, waits for speech, and stops on unmount', () => {
        const play = jest.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
        const pause = jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
        const onReplayComplete = jest.fn();
        const voice = { text: 'Brake smoothly.', audioDataUrl: 'data:audio/wav;base64,UklGRg==', durationMs: 8000 };
        const view = render(<DriverExpertComparisonGraph data={completeData} voice={voice} onReplayComplete={onReplayComplete} />);
        expect(play).not.toHaveBeenCalled();
        runAnimationFrame(0);
        expect(play).toHaveBeenCalledTimes(1);
        view.rerender(<DriverExpertComparisonGraph data={completeData} voice={voice} onReplayComplete={onReplayComplete} />);
        runAnimationFrame(4750);
        expect(play).toHaveBeenCalledTimes(1);
        expect(onReplayComplete).not.toHaveBeenCalled();
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Finishing narration');
        fireEvent.ended(view.container.querySelector('audio')!);
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        fireEvent.click(screen.getByRole('button', { name: 'Replay comparison' }));
        expect(pause).toHaveBeenCalled();
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        runAnimationFrame(5_000);
        expect(play).toHaveBeenCalledTimes(2);
        runAnimationFrame(9_750);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Finishing narration');
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        fireEvent.ended(view.container.querySelector('audio')!);
        expect(onReplayComplete).toHaveBeenCalledTimes(2);
        view.unmount();
        expect(pause).toHaveBeenCalled();
        play.mockRestore();
        pause.mockRestore();
    });

    it('finishes the graph if audio playback is blocked', async () => {
        const play = jest.spyOn(HTMLMediaElement.prototype, 'play').mockRejectedValue(new Error('Autoplay blocked'));
        const pause = jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
        const onReplayComplete = jest.fn();
        const voice = { text: 'Brake smoothly.', audioDataUrl: 'data:audio/wav;base64,UklGRg==', durationMs: 8000 };
        const view = render(<DriverExpertComparisonGraph data={completeData} voice={voice} onReplayComplete={onReplayComplete} />);
        await act(async () => runAnimationFrame(0));
        expect(screen.getByText('Narration unavailable')).toBeInTheDocument();
        runAnimationFrame(4750);
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        view.unmount();
        play.mockRestore();
        pause.mockRestore();
    });

    it('stops narration when the graph is replaced and starts the new replay once', () => {
        const play = jest.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
        const pause = jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
        const voice = { text: 'Brake smoothly.', audioDataUrl: 'data:audio/wav;base64,UklGRg==', durationMs: 8000 };
        const view = render(<DriverExpertComparisonGraph data={completeData} voice={voice} />);
        runAnimationFrame(0);
        view.rerender(<DriverExpertComparisonGraph data={{ samples: [...completeData.samples] }} voice={voice} />);
        expect(pause).toHaveBeenCalled();
        runAnimationFrame(1000);
        expect(play).toHaveBeenCalledTimes(2);
        view.unmount();
        play.mockRestore();
        pause.mockRestore();
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
        expectTelemetryPodWithinViewport('driver');
        expectTelemetryPodWithinViewport('expert');
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

    it('establishes the full trajectory before zooming, tilting, and fading in competitor status', () => {
        render(<DriverExpertComparisonGraph data={completeData} />);

        const camera = screen.getByTestId('comparison-camera-layer');
        const overlay = screen.getByTestId('comparison-camera-overlay');
        const driverPath = screen.getByTestId('driver-track-path').getAttribute('d');
        const expertPath = screen.getByTestId('expert-track-path').getAttribute('d');

        expect(camera).toHaveAttribute('data-camera-transform', 'matrix(1 0 0 1 0 0)');
        expect(camera).toHaveAttribute('data-camera-phase', 'overview');
        expect(camera).toHaveAttribute('data-camera-progress', '0');
        expect(overlay).toHaveAttribute('data-status-visibility', 'hidden');
        expect(overlay).toHaveAttribute('aria-hidden', 'true');
        expect(overlay).toHaveStyle({ opacity: '0' });
        expect(screen.queryAllByRole('meter')).toHaveLength(0);

        runAnimationFrame(0);
        runAnimationFrame(1_000);

        expect(camera).toHaveAttribute('data-camera-transform', 'matrix(1 0 0 1 0 0)');
        expect(camera).toHaveAttribute('data-camera-phase', 'overview');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');

        runAnimationFrame(1_375);

        expect(camera).toHaveAttribute('data-camera-phase', 'focusing');
        expect(camera).toHaveAttribute('data-camera-progress', '0.5');
        expect(camera).not.toHaveAttribute('data-camera-transform', 'matrix(1 0 0 1 0 0)');
        const [midA, midB, midC, midD] = parseMatrix(camera);
        expect(Math.hypot(midB, midD) / Math.hypot(midA, midC)).toBeCloseTo(Math.cos(Math.PI / 6), 6);
        expect(overlay).toHaveAttribute('data-status-visibility', 'fading');
        expect(overlay).not.toHaveAttribute('aria-hidden');
        expect(overlay).toHaveStyle({ opacity: '0.5' });
        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(screen.getByTestId('driver-track-path')).not.toHaveAttribute('d', driverPath);
        expect(screen.getByTestId('expert-track-path')).not.toHaveAttribute('d', expertPath);

        runAnimationFrame(1_750);

        expect(camera).toHaveAttribute('data-camera-phase', 'following');
        expect(camera).toHaveAttribute('data-camera-progress', '1');
        const [a, b, c, d] = parseMatrix(camera);
        expect(Math.hypot(a, c)).toBeCloseTo(4, 6);
        expect(Math.hypot(b, d)).toBeCloseTo(2, 6);
        expect(camera).toHaveAttribute('data-camera-projection', 'perspective');
        expect(camera).not.toHaveAttribute('transform');
        expect(screen.getByTestId('comparison-ground-grid').getAttribute('d')).toContain('L');
        expect(overlay).toHaveAttribute('data-status-visibility', 'visible');
        expect(overlay).toHaveStyle({ opacity: '1' });
        expectCameraLockedOn('driver');
        expectCameraFacingDriverDirection();
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
    });

    it('shrinks lateral separation with distance and projects track endpoints onto their markers', () => {
        setReducedMotion(true);
        const dataAtDepth = (y: number) => ({ samples: [{
            driverTimeMs: 0, expertTimeMs: 0,
            driverTrackPosition: 0.4, expertTrackPosition: 0.4,
            driverTrajectory: { x: 0, y: 0 }, expertTrajectory: { x: 20, y },
        }] });
        const view = render(<DriverExpertComparisonGraph data={dataAtDepth(0)} />);
        const readSeparation = () => {
            const marker = screen.getByTestId('expert-position-marker').querySelector('circle')!;
            const x = Number(marker.getAttribute('cx'));
            const y = Number(marker.getAttribute('cy'));
            const [, pathX, pathY] = screen.getByTestId('expert-track-path').getAttribute('d')!.split(' ');
            expect(Number(pathX)).toBeCloseTo(x, 3);
            expect(Number(pathY)).toBeCloseTo(y, 3);
            expectCameraLockedOn('driver');
            expectTelemetryPodWithinViewport('expert');
            return x - Number(screen.getByTestId('comparison-camera-layer').getAttribute('data-camera-anchor-x'));
        };
        const atDriver = readSeparation();
        view.rerender(<DriverExpertComparisonGraph data={dataAtDepth(40)} />);
        const distant = readSeparation();
        view.rerender(<DriverExpertComparisonGraph data={dataAtDepth(-40)} />);
        const nearby = readSeparation();
        expect(distant).toBeGreaterThan(0);
        expect(distant).toBeLessThan(atDriver);
        expect(nearby).toBeGreaterThan(atDriver);
    });

    it('clips trajectories and the driver ribbon at the near plane without connecting across the camera', () => {
        setReducedMotion(true);
        render(<DriverExpertComparisonGraph data={{ samples: [0, -1_000, 0].map((y, index) => ({
            driverTimeMs: index * 1_000, expertTimeMs: index * 1_000,
            driverTrackPosition: index / 4, expertTrackPosition: index / 4,
            driverTrajectory: { x: 0, y }, expertTrajectory: { x: 20, y: y + 100 },
        })) }} />);
        const path = screen.getByTestId('expert-track-path').getAttribute('d')!;
        expect(path.match(/[ML]/g)).toEqual(['M', 'L', 'M', 'L']);
        const ribbon = screen.getByTestId('driver-track-path').previousElementSibling!.getAttribute('d')!;
        expect(ribbon).toMatch(/ Z$/);
        for (const value of `${path} ${ribbon}`.replace(/[MLZ]/g, '').trim().split(/\s+/).map(Number)) {
            expect(Number.isFinite(value)).toBe(true);
            expect(Math.abs(value)).toBeLessThan(20_000);
        }
        expectCameraLockedOn('driver');
        expectTelemetryPodWithinViewport('expert');
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
        const xyTransform = screen.getByTestId('comparison-camera-layer').getAttribute('data-camera-transform');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '50');

        detectedGame = 'acc';
        rerender(<DriverExpertComparisonGraph data={completeData} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-y', '-300');
        expect(screen.getByTestId('comparison-camera-layer').getAttribute('data-camera-transform')).not.toBe(xyTransform);
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

    it('preserves each service-aligned stream and stops both when the expert finishes first', () => {
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
        expect(getDriverExpertReplayDurationMs(data)).toBe(1_000);
        render(<DriverExpertComparisonGraph data={data} />);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '0');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.1');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.2');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');

        runAnimationFrame(0);
        runAnimationFrame(2_250);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '10');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '120');
        expect(pendingFrames.size).toBe(1);

        runAnimationFrame(2_750);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('1.00s / 1.00s');
        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuemax', '1000');
        expect(screen.getByTestId('replay-progress')).toHaveAttribute('aria-valuenow', '1000');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '20');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '140');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.3');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(pendingFrames.size).toBe(0);

        runAnimationFrame(3_250);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '20');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '140');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.3');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.6');

        runAnimationFrame(3_750);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '20');
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '1');
        expect(screen.getByTestId('driver-brake-gauge')).toHaveAttribute('data-value', '0');
        expect(screen.getByTestId('driver-gear')).toHaveTextContent('4');
    });

    it('stops both trajectories and completes the replay when the driver finishes first', () => {
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
        const onReplayComplete = jest.fn();
        expect(getDriverExpertReplayDurationMs(data)).toBe(1_000);
        render(<DriverExpertComparisonGraph data={data} onReplayComplete={onReplayComplete} />);
        runAnimationFrame(0);
        runAnimationFrame(2_250);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
        expect(onReplayComplete).not.toHaveBeenCalled();

        runAnimationFrame(2_750);

        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '40');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '120');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.4');
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        expect(pendingFrames.size).toBe(0);

        runAnimationFrame(3_750);
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '40');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-x', '120');
        expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-track-position', '0.6');
        expect(screen.getByTestId('expert-position-marker')).toHaveAttribute('data-track-position', '0.4');
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
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
        expect(getDriverExpertReplayDurationMs(data)).toBeCloseTo(150);

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

    it('preserves the duration when both normalized clocks finish together', () => {
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
            .getAttribute('data-camera-transform');
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
            'data-camera-transform',
            initialCameraTransform,
        );
        expect(screen.getByTestId('driver-track-path')).not.toHaveAttribute('d', completeDriverPath);
        expect(screen.getByTestId('expert-track-path')).not.toHaveAttribute('d', completeExpertPath);
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

    it('maintains a steady camera turn through telemetry sample boundaries', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: Array.from({ length: 21 }, (_, index) => {
                const timeMs = index * 200;
                const angle = timeMs * Math.PI / 6_000;
                const trajectory = { x: 100 * Math.sin(angle), y: 100 * (1 - Math.cos(angle)) };
                return {
                    driverTimeMs: timeMs,
                    expertTimeMs: timeMs,
                    driverTrackPosition: index / 20,
                    expertTrackPosition: index / 20,
                    driverTrajectory: trajectory,
                    expertTrajectory: trajectory,
                };
            }),
        }} />);
        runAnimationFrame(0);

        let previousRotation: number | undefined;
        for (let timeMs = 1_000; timeMs <= 2_000; timeMs += 25) {
            runAnimationFrame(1_750 + timeMs);
            const rotation = Number(screen.getByTestId('comparison-camera-layer')
                .getAttribute('data-camera-rotation'));
            if (previousRotation !== undefined) {
                // A 30 degrees/second bend should not pause at each telemetry point.
                expect(rotation - previousRotation).toBeGreaterThan(0.6);
                expect(rotation - previousRotation).toBeLessThan(0.9);
            }
            previousRotation = rotation;
            expectCameraLockedOn('driver');
        }
    });

    it('filters high frequency trajectory noise out of camera rotation', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: Array.from({ length: 301 }, (_, index) => ({
                driverTimeMs: index * 10,
                expertTimeMs: index * 10,
                driverTrackPosition: index / 300,
                expertTrackPosition: index / 300,
                driverTrajectory: { x: index * 0.4, y: 0.12 * Math.sin(index * 1.7) },
                expertTrajectory: { x: index * 0.4, y: 0 },
            })),
        }} />);
        runAnimationFrame(0);

        for (let timeMs = 1_000; timeMs <= 1_600; timeMs += 16) {
            runAnimationFrame(1_750 + timeMs);
            const rotation = Number(screen.getByTestId('comparison-camera-layer')
                .getAttribute('data-camera-rotation'));
            expect(Math.abs(rotation + 90)).toBeLessThan(0.2);
            expectCameraLockedOn('driver');
        }
    });

    it('keeps camera rotation independent of telemetry sampling density', () => {
        const makeData = (times: number[]) => ({
            samples: times.map((timeMs) => {
                const trajectory = { x: Math.min(timeMs, 1_000) / 10, y: Math.max(0, timeMs - 1_000) / 10 };
                return {
                    driverTimeMs: timeMs,
                    expertTimeMs: timeMs,
                    driverTrackPosition: timeMs / 3_000,
                    expertTrackPosition: timeMs / 3_000,
                    driverTrajectory: trajectory,
                    expertTrajectory: trajectory,
                };
            }),
        });
        const view = render(<DriverExpertComparisonGraph data={makeData([0, 1_000, 2_000, 3_000])} />);
        const readRotations = () => {
            runAnimationFrame(0);
            return [650, 900, 1_000, 1_100, 1_350].map((timeMs) => {
                runAnimationFrame(1_750 + timeMs);
                return Number(screen.getByTestId('comparison-camera-layer')
                    .getAttribute('data-camera-rotation'));
            });
        };
        const sparseRotations = readRotations();
        view.rerender(<DriverExpertComparisonGraph data={makeData([
            0, 200, 700, 950, 1_000, 1_010, 1_040, 1_500, 2_000, 3_000,
        ])} />);
        readRotations().forEach((rotation, index) => {
            expect(rotation).toBeCloseTo(sparseRotations[index], 3);
        });
    });

    it('holds its heading through a stop and turns smoothly as motion resumes', () => {
        const points = [
            { x: 0, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 100 },
        ];
        render(<DriverExpertComparisonGraph data={{
            samples: points.map((trajectory, index) => ({
                driverTimeMs: index * 1_000,
                expertTimeMs: index * 1_000,
                driverTrackPosition: index / 4,
                expertTrackPosition: index / 4,
                driverTrajectory: trajectory,
                expertTrajectory: trajectory,
            })),
        }} />);
        runAnimationFrame(0);
        runAnimationFrame(3_750);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute('data-camera-rotation', '-90');
        let previousRotation = -90;
        for (let timeMs = 2_400; timeMs <= 3_600; timeMs += 16) {
            runAnimationFrame(1_750 + timeMs);
            const rotation = Number(screen.getByTestId('comparison-camera-layer')
                .getAttribute('data-camera-rotation'));
            expect(rotation - previousRotation).toBeGreaterThanOrEqual(0);
            expect(rotation - previousRotation).toBeLessThan(4);
            previousRotation = rotation;
        }
        expect(previousRotation).toBeCloseTo(0);
    });

    it('rotates smoothly across the 180 degree angle boundary', () => {
        render(<DriverExpertComparisonGraph data={{
            samples: Array.from({ length: 21 }, (_, index) => {
                const angle = 4 * Math.PI / 3 + index * Math.PI / 30;
                const trajectory = { x: 100 * Math.sin(angle), y: 100 * (1 - Math.cos(angle)) };
                return {
                    driverTimeMs: index * 200,
                    expertTimeMs: index * 200,
                    driverTrackPosition: index / 20,
                    expertTrackPosition: index / 20,
                    driverTrajectory: trajectory,
                    expertTrajectory: trajectory,
                };
            }),
        }} />);
        runAnimationFrame(0);
        let previousAngle: number | undefined;
        for (let timeMs = 800; timeMs <= 1_200; timeMs += 16) {
            runAnimationFrame(1_750 + timeMs);
            const [a, b] = parseMatrix(screen.getByTestId('comparison-camera-layer'));
            const angle = Math.atan2(b, a);
            if (previousAngle !== undefined) {
                const turn = Math.atan2(Math.sin(angle - previousAngle), Math.cos(angle - previousAngle));
                expect(turn).toBeGreaterThan(0);
                expect(turn).toBeLessThan(Math.PI / 180);
            }
            previousAngle = angle;
            expectCameraLockedOn('driver');
        }
    });

    it('smooths a corner within a fixed time window', () => {
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
                driverTrackPosition: 0.5,
                expertTrackPosition: 0.5,
                driverTrajectory: { x: 100, y: 100 },
                expertTrajectory: { x: 100, y: 110 },
            }, {
                driverTimeMs: 3_000,
                expertTimeMs: 3_000,
                driverTrackPosition: 0.75,
                expertTrackPosition: 0.75,
                driverTrajectory: { x: 100, y: 200 },
                expertTrajectory: { x: 100, y: 210 },
            }, {
                driverTimeMs: 4_000,
                expertTimeMs: 4_000,
                driverTrackPosition: 1,
                expertTrackPosition: 1,
                driverTrajectory: { x: 100, y: 300 },
                expertTrajectory: { x: 100, y: 310 },
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
            '-90',
        );

        runAnimationFrame(2_500);
        const enteringRotation = Number(screen.getByTestId('comparison-camera-layer')
            .getAttribute('data-camera-rotation'));
        expect(enteringRotation).toBeGreaterThan(-90);
        expect(enteringRotation).toBeLessThan(-45);

        runAnimationFrame(2_750);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-rotation',
            '-45',
        );

        runAnimationFrame(3_000);
        const leavingRotation = Number(screen.getByTestId('comparison-camera-layer')
            .getAttribute('data-camera-rotation'));
        expect(leavingRotation).toBeGreaterThan(-45);
        expect(leavingRotation).toBeLessThan(0);

        runAnimationFrame(3_250);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-rotation',
            '0',
        );

        runAnimationFrame(4_750);
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

    it('replays the same comparison repeatedly without remounting', () => {
        const onReplayComplete = jest.fn();
        render(<DriverExpertComparisonGraph data={completeData} onReplayComplete={onReplayComplete} />);
        const button = screen.getByRole('button', { name: 'Replay comparison' });
        button.focus();

        for (let cycle = 0; cycle < 3; cycle += 1) {
            if (cycle > 0) fireEvent.click(button);
            expect(screen.getByTestId('replay-status')).toHaveTextContent('Replaying');
            expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
            expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute('data-camera-phase', 'overview');
            expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '0');
            expect(screen.getByTestId('driver-gear')).toHaveTextContent('2');
            expect(onReplayComplete).toHaveBeenCalledTimes(cycle);
            expect(button).toHaveFocus();

            runAnimationFrame(cycle * 5_000);
            runAnimationFrame(cycle * 5_000 + 4_750);
            expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
            expect(screen.getByTestId('replay-progress')).toHaveTextContent('3.00s / 3.00s');
            expect(screen.getByTestId('driver-position-marker')).toHaveAttribute('data-x', '100');
            expect(screen.getByTestId('driver-gear')).toHaveTextContent('5');
            expect(onReplayComplete).toHaveBeenCalledTimes(cycle + 1);
            expect(pendingFrames.size).toBe(0);
        }
    });

    it('cancels active playback when Replay is clicked and cleans up the restarted replay', () => {
        const view = render(<DriverExpertComparisonGraph data={completeData} />);
        runAnimationFrame(0);
        runAnimationFrame(2_750);
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('1.00s / 3.00s');
        const previousFrameId = nextFrameId - 1;

        fireEvent.click(screen.getByRole('button', { name: 'Replay comparison' }));
        expect(cancelAnimationFrameMock).toHaveBeenCalledWith(previousFrameId);
        expect(pendingFrames.size).toBe(1);
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('0.00s / 3.00s');
        runAnimationFrame(3_000);
        runAnimationFrame(5_750);
        expect(screen.getByTestId('replay-progress')).toHaveTextContent('1.00s / 3.00s');

        view.unmount();
        expect(pendingFrames.size).toBe(0);
    });

    it('fires replay completion exactly once after the full timeline and never after unmount', () => {
        const onReplayComplete = jest.fn();
        const first = render(
            <DriverExpertComparisonGraph
                data={completeData}
                onReplayComplete={onReplayComplete}
            />,
        );

        runAnimationFrame(0);
        runAnimationFrame(1_000);
        runAnimationFrame(1_750);
        runAnimationFrame(4_749);
        expect(onReplayComplete).not.toHaveBeenCalled();

        runAnimationFrame(4_750);
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        runAnimationFrame(5_000);
        expect(onReplayComplete).toHaveBeenCalledTimes(1);
        first.unmount();

        const neverComplete = jest.fn();
        const second = render(
            <DriverExpertComparisonGraph
                data={completeData}
                onReplayComplete={neverComplete}
            />,
        );
        runAnimationFrame(0);
        second.unmount();
        runAnimationFrame(4_750);
        expect(neverComplete).not.toHaveBeenCalled();
    });

    it('converts graph completion to the replay_complete renderer event', () => {
        const emitRendererEvent = jest.fn();
        render(driverExpertComparisonOverlayRenderer.renderOverlay({
            title: 'Comparison',
            comparison: completeData,
            labelGroups: [
                { category: 'mistakes', subLabels: ['Late braking', 'Late turn-in'] },
                { category: 'expert', subLabels: ['Matches expert line'] },
                { category: 'recovery', subLabels: ['Merge back to expert line'] },
            ],
        }, 'expanded', {
            componentName: 'comparison',
            revision: 1,
            emitRendererEvent,
        }));

        const mistakes = within(screen.getByRole('region', { name: 'Mistakes labels' }));
        expect(mistakes.getByRole('heading', { name: 'Mistakes' })).toBeInTheDocument();
        expect(mistakes.getAllByRole('listitem').map((item) => item.textContent))
            .toEqual(['Late braking', 'Late turn-in']);
        expect(screen.getByRole('region', { name: 'Expert labels' }))
            .toHaveTextContent('Matches expert line');
        expect(screen.getByRole('region', { name: 'Recovery labels' }))
            .toHaveTextContent('Merge back to expert line');
        expect(screen.queryByRole('button', { name: 'Replay comparison' })).not.toBeInTheDocument();

        runAnimationFrame(0);
        runAnimationFrame(4_750);

        expect(emitRendererEvent).toHaveBeenCalledTimes(1);
        expect(emitRendererEvent).toHaveBeenCalledWith('replay_complete');
        runAnimationFrame(9_500);
        expect(emitRendererEvent).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('replay-status')).toHaveTextContent('Replay complete');
        expect(pendingFrames.size).toBe(0);
    });

    it('shows parent-only labels and removes stale groups when the segment changes', () => {
        const view = render(<DriverExpertComparisonGraph
            data={completeData}
            labelGroups={[{ category: 'expert', subLabels: [] }]}
        />);
        expect(screen.getByRole('region', { name: 'Expert labels' }))
            .toHaveTextContent('No sublabels provided');
        expect(screen.queryByRole('region', { name: 'Mistakes labels' })).not.toBeInTheDocument();
        view.rerender(<DriverExpertComparisonGraph data={completeData} />);
        expect(screen.queryByLabelText('Segment analysis labels')).not.toBeInTheDocument();
    });

    it('picks up trajectory signs, merges active labels, and releases them at their end points', () => {
        const data = { samples: completeData.samples.map((sample, index) => ({
            ...sample, driverSourceIndex: 100 + index * 10,
        })) };
        expect(normalizeDriverExpertComparisonData(data)?.samples.map((sample) => sample.driverSourceIndex))
            .toEqual([100, 110, 120]);
        const range = { label: 'Late braking', category: 'mistakes' as const, startIndex: 104, endIndex: 116 };
        const ranges = [range, { label: 'Smooth recovery', category: 'recovery' as const, startIndex: 112, endIndex: 120 }];
        const labelGroups = [
            { category: 'mistakes' as const, subLabels: ['Late braking'] },
            { category: 'recovery' as const, subLabels: ['Smooth recovery'] },
        ];
        const view = render(<DriverExpertComparisonGraph data={data} labelGroups={labelGroups} labelRanges={ranges} />);
        const following = () => view.container.querySelector('[data-testid="comparison-road-sign"][data-state="following"]');
        expect(screen.getAllByTestId('comparison-label-sign')).toHaveLength(2);
        expect(following()).toBeNull();
        runAnimationFrame(0);
        runAnimationFrame(2_149);
        expect(following()).toBeNull();
        runAnimationFrame(2_150);
        expect(following()).toHaveAttribute('aria-label', 'Late braking');
        const marker = screen.getByTestId('driver-position-marker').querySelector('circle')!;
        expect(Number(following()!.getAttribute('data-anchor-x'))).toBeCloseTo(Number(marker.getAttribute('cx')));
        expect(Number(following()!.getAttribute('data-anchor-y'))).toBeCloseTo(Number(marker.getAttribute('cy')));
        runAnimationFrame(3_150);
        expect(screen.getAllByTestId('comparison-label-sign')).toHaveLength(1);
        expect(screen.getByTestId('comparison-label-sign')).toHaveAttribute('aria-label', 'Late braking, Smooth recovery');
        expect(screen.getByTestId('comparison-label-sign').getAttribute('transform')).not.toMatch(/NaN|Infinity/);
        runAnimationFrame(3_950);
        expect(following()).toHaveAttribute('aria-label', 'Smooth recovery');
        const released = view.container.querySelector('[data-testid="comparison-road-sign"][data-state="released"]')!;
        expect(released).toHaveAttribute('aria-label', 'Late braking');
        const endX = released.getAttribute('data-world-x');
        const endY = released.getAttribute('data-world-y');
        const previousAnchorY = released.getAttribute('data-anchor-y');
        runAnimationFrame(4_150);
        expect(released).toHaveAttribute('data-world-x', endX);
        expect(released).toHaveAttribute('data-world-y', endY);
        expect(released.getAttribute('data-anchor-y')).not.toBe(previousAnchorY);
        runAnimationFrame(4_750);
        expect(following()).toBeNull();
        expect(view.container.querySelector('[data-state="released"][aria-label="Smooth recovery"]')).toBeInTheDocument();
        expect(screen.queryByTestId('comparison-label-range')).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Replay comparison' }));
        expect(screen.getAllByTestId('comparison-label-sign')).toHaveLength(2);
        expect(following()).toBeNull();
        runAnimationFrame(5_000);
        runAnimationFrame(7_150);
        expect(following()).toHaveAttribute('aria-label', 'Late braking');
        view.rerender(<DriverExpertComparisonGraph data={data} labelRanges={[]} />);
        expect(screen.queryByTestId('comparison-road-signs')).not.toBeInTheDocument();
    });

    it('validates and renders label ranges in overlay snapshots', () => {
        const labelRange = { label: 'Late braking', startIndex: 104, endIndex: 116 };
        const snapshot = { title: 'Comparison', comparison: {
            samples: completeData.samples.map((sample, index) => ({ ...sample, driverSourceIndex: 100 + index * 10 })),
        }, labelRanges: [labelRange], labelGroups: [{ category: 'mistakes' as const, subLabels: ['Late braking'] }] };
        expect(driverExpertComparisonOverlayRenderer.validateSnapshot(snapshot)).toBe(true);
        for (const labelRanges of [null, {}, [null], [{ ...labelRange, endIndex: 104 }], [{ ...labelRange, startIndex: -1 }]]) {
            expect(driverExpertComparisonOverlayRenderer.validateSnapshot({ ...snapshot, labelRanges })).toBe(false);
        }
        render(driverExpertComparisonOverlayRenderer.renderOverlay(snapshot, 'expanded', {
            componentName: 'comparison', revision: 1, emitRendererEvent: jest.fn(),
        }));
        expect(screen.getByTestId('comparison-road-sign')).toHaveAttribute('data-state', 'waiting');
        runAnimationFrame(0);
        runAnimationFrame(2_150);
        expect(screen.getByTestId('comparison-label-sign')).toHaveAttribute('aria-label', 'Late braking');
        expect(screen.getByTestId('comparison-road-sign')).toHaveAttribute('data-state', 'following');
    });

    it('only shows road signs for labels listed above the graph and updates them when the list changes', () => {
        const data = { samples: completeData.samples.map((sample, index) => ({
            ...sample, driverSourceIndex: 100 + index * 10,
        })) };
        const labelRanges = [' Late braking ', 'Smooth recovery', 'Mistake (Practice)', 'Unlisted label']
            .map((label) => ({ label, startIndex: 104, endIndex: 116 }));
        const view = render(<DriverExpertComparisonGraph data={data} labelRanges={labelRanges}
            labelGroups={[{ category: 'mistakes', subLabels: ['Late braking'] }]} />);
        const signLabels = () => screen.queryAllByTestId('comparison-road-sign')
            .map((sign) => sign.getAttribute('aria-label'));
        expect(signLabels()).toEqual(['Late braking']);
        runAnimationFrame(0);
        runAnimationFrame(2_150);
        expect(signLabels()).toEqual(['Late braking']);
        view.rerender(<DriverExpertComparisonGraph data={data} labelRanges={labelRanges}
            labelGroups={[{ category: 'recovery', subLabels: ['Smooth recovery'] }]} />);
        expect(signLabels()).toEqual(['Smooth recovery']);
        view.rerender(<DriverExpertComparisonGraph data={data} labelRanges={labelRanges}
            labelGroups={[{ category: 'mistakes', subLabels: [] }]} />);
        expect(signLabels()).toEqual([]);
        view.rerender(<DriverExpertComparisonGraph data={data} labelRanges={labelRanges} labelGroups={[]} />);
        expect(signLabels()).toEqual([]);
        view.rerender(<DriverExpertComparisonGraph data={data} labelRanges={labelRanges} />);
        expect(signLabels()).toEqual([]);
    });

    it('uses the track camera depth to shrink distant road signs', () => {
        const data = { samples: completeData.samples.map((sample, index) => ({
            ...sample, driverSourceIndex: 100 + index * 10,
        })) };
        render(<DriverExpertComparisonGraph data={data}
            labelGroups={[{ category: 'mistakes', subLabels: ['Nearby', 'Distant'] }]}
            labelRanges={[
                { label: 'Nearby', startIndex: 100, endIndex: 110 },
                { label: 'Distant', startIndex: 115, endIndex: 120 },
            ]} />);
        runAnimationFrame(0);
        runAnimationFrame(1_750);
        const signs = screen.getAllByTestId('comparison-road-sign');
        const nearby = signs.find((sign) => sign.getAttribute('aria-label') === 'Nearby')!;
        const distant = signs.find((sign) => sign.getAttribute('aria-label') === 'Distant')!;
        const width = (sign: HTMLElement) => Number(within(sign).getByTestId('comparison-label-sign-board').getAttribute('width'));
        const cardWidth = Number(screen.getByTestId('driver-telemetry-pod').querySelector('rect')!.getAttribute('width'));
        expect(width(nearby)).toBeCloseTo(cardWidth * 0.63);
        expect(width(distant)).toBeLessThan(width(nearby));
        expect(signs.indexOf(distant)).toBeLessThan(signs.indexOf(nearby));
    });

    it('validates optional label groups while accepting existing overlay snapshots', () => {
        const snapshot = { title: 'Comparison', comparison: completeData };
        expect(driverExpertComparisonOverlayRenderer.validateSnapshot(snapshot)).toBe(true);
        expect(driverExpertComparisonOverlayRenderer.validateSnapshot({
            ...snapshot, labelGroups: [{ category: 'expert', subLabels: [] }],
        })).toBe(true);
        for (const labelGroups of [null, {}, [null], [
            { category: 'other', subLabels: [] },
        ], [
            { category: 'mistakes', subLabels: [7] },
        ]]) {
            expect(driverExpertComparisonOverlayRenderer.validateSnapshot({
                ...snapshot, labelGroups,
            })).toBe(false);
        }
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

    it('keeps cards beside their markers and slides the Expert card up to the bottom edge', () => {
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

        const driverMarker = screen.getByTestId('driver-position-marker').querySelector('circle');
        const expertMarker = screen.getByTestId('expert-position-marker').querySelector('circle');
        if (!driverMarker || !expertMarker) throw new Error('Expected both position markers');
        const driver = expectTelemetryPodWithinViewport('driver');
        const expert = expectTelemetryPodWithinViewport('expert');

        expect(driver.x).toBeGreaterThan(Number(driverMarker.getAttribute('cx')));
        expect(driver.bottom).toBeLessThan(Number(driverMarker.getAttribute('cy')));
        expect(expert.right).toBeLessThan(Number(expertMarker.getAttribute('cx')));
        expect(expert.y).toBeLessThan(Number(expertMarker.getAttribute('cy')) + 14);
        expect(expert.bottom).toBeCloseTo(208, 3);
    });

    it.each([
        { edge: 'left', from: { x: -1_000, y: 60 }, to: { x: -1_000, y: 65 } },
        { edge: 'right', from: { x: 1_000, y: 60 }, to: { x: 1_000, y: 65 } },
        { edge: 'bottom', from: { x: 0, y: -1_000 }, to: { x: 0.25, y: -1_000 } },
    ])('slides along the $edge edge while the Expert marker is offscreen', ({ edge, from, to }) => {
        setReducedMotion(true);
        const comparisonData = (expertTrajectory: { x: number; y: number }) => ({
            samples: [{
                driverTimeMs: 0,
                expertTimeMs: 0,
                driverTrackPosition: 0.4,
                expertTrackPosition: 0.4,
                driverTrajectory: { x: 0, y: 0 },
                expertTrajectory,
            }],
        });
        const view = render(<DriverExpertComparisonGraph data={comparisonData(from)} />);
        const before = expectTelemetryPodWithinViewport('expert');
        const markerY = Number(screen.getByTestId('expert-position-marker').querySelector('circle')!.getAttribute('cy'));
        view.rerender(<DriverExpertComparisonGraph data={comparisonData(to)} />);
        const after = expectTelemetryPodWithinViewport('expert');
        expectTelemetryPodWithinViewport('driver');
        expectCameraLockedOn('driver');

        if (edge === 'left' || edge === 'right') {
            expect(after.x).toBe(before.x);
            const nextMarkerY = Number(screen.getByTestId('expert-position-marker').querySelector('circle')!.getAttribute('cy'));
            expect(after.y - before.y).toBeCloseTo(nextMarkerY - markerY, 2);
            expect(after.y).toBeLessThan(before.y);
            expect(edge === 'left' ? after.x : after.right).toBeCloseTo(edge === 'left' ? 12 : 748, 3);
        } else {
            expect(after.y).toBe(before.y);
            expect(after.x - before.x).toBeCloseTo(20, 3);
            expect(after.bottom).toBeCloseTo(208, 3);
        }
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
            .getAttribute('data-camera-transform');

        expectCameraLockedOn('driver');
        view.rerender(<DriverExpertComparisonGraph data={comparisonData(10_000)} />);

        expect(screen.getAllByRole('meter')).toHaveLength(4);
        expect(screen.getByTestId('comparison-camera-layer')).toHaveAttribute(
            'data-camera-transform',
            initialTransform,
        );
        expectCameraLockedOn('driver');
    });

    it('resizes road signs, both cards and the driver anchor with the panel while preserving camera zoom', () => {
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
            const view = render(<DriverExpertComparisonGraph data={{
                samples: [{
                    driverSourceIndex: 0,
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
            }} labelGroups={[{ category: 'mistakes', subLabels: ['Late braking'] }]}
                labelRanges={[{ label: 'Late braking', startIndex: 0, endIndex: 1 }]} />);

            const initialMatrix = parseMatrix(screen.getByTestId('comparison-camera-layer'));
            const initialScale = Math.hypot(initialMatrix[0], initialMatrix[2]);
            const initialTiltedScale = Math.hypot(initialMatrix[1], initialMatrix[3]);
            expect(initialScale).toBeCloseTo(4, 6);
            expect(initialTiltedScale).toBeCloseTo(2, 6);
            const cardWidths: number[] = [];
            const signWidths: number[] = [];
            for (const [width, height] of [[320, 640], [1280, 640], [1280, 160], [2400, 130]]) {
                act(() => notifyResize?.(width, height));
                const driver = expectTelemetryPodWithinViewport('driver');
                const expert = expectTelemetryPodWithinViewport('expert');
                expect(driver.width).toBe(expert.width);
                expect(driver.height).toBe(expert.height);
                // Convert SVG units to pixels to check the visible card size after each resize.
                const [, , svgWidth, svgHeight] = screen.getByTestId('comparison-track-map')
                    .getAttribute('viewBox')!.split(' ').map(Number);
                const pixelScale = Math.min(width / svgWidth, height / svgHeight);
                cardWidths.push(driver.width * pixelScale);
                const sign = screen.getByTestId('comparison-label-sign-board');
                signWidths.push(Number(sign.getAttribute('width')) * pixelScale);
                expect(signWidths[signWidths.length - 1] / signWidths[0])
                    .toBeCloseTo(cardWidths[cardWidths.length - 1] / cardWidths[0], 5);
                expect(driver.width * pixelScale).toBeLessThan(width / 2);
                expect(driver.height * pixelScale).toBeLessThan(height / 2);
                expectCameraLockedOn('driver');
            }
            expect(cardWidths[1]).toBeGreaterThan(cardWidths[0]);
            expect(cardWidths[2]).toBeLessThan(cardWidths[1]);
            expect(cardWidths[3]).toBeLessThan(cardWidths[2]);
            act(() => notifyResize?.(400, 260));

            expect(screen.getByTestId('comparison-track-map')).toHaveAttribute(
                'viewBox',
                '0 0 760 494',
            );
            const resizedMatrix = parseMatrix(screen.getByTestId('comparison-camera-layer'));
            expect(Math.hypot(resizedMatrix[0], resizedMatrix[2])).toBeCloseTo(initialScale, 6);
            expect(Math.hypot(resizedMatrix[1], resizedMatrix[3])).toBeCloseTo(initialTiltedScale, 6);
            expectCameraLockedOn('driver');
            expectTelemetryPodWithinViewport('driver');
            expectTelemetryPodWithinViewport('expert');
            expect(observe).toHaveBeenCalledWith(screen.getByTestId('comparison-track-map'));
            view.unmount();
            expect(disconnect).toHaveBeenCalledTimes(1);
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
        expect(screen.getByRole('button', { name: 'Replay comparison' })).toBeDisabled();
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
        expect(screen.getByRole('button', { name: 'Replay comparison' })).toBeDisabled();
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
        expect(screen.getByRole('button', { name: 'Replay comparison' })).toBeDisabled();
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

describe('Driver/Expert comparison availability', () => {
    const reasonCodes = (
        data: Parameters<typeof getDriverExpertComparisonUnavailableDiagnostics>[0],
        game: DesktopGame | null = null,
    ) => getDriverExpertComparisonUnavailableDiagnostics(data, game)
        .map((diagnostic) => diagnostic.code);

    it('only reports missing data or a completely empty sample list', () => {
        expect(reasonCodes(undefined)).toEqual(['comparison_data_missing']);
        expect(reasonCodes({ samples: [] })).toEqual(['comparison_samples_missing']);
        const backendPayload = {
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 100,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
            }, {
                driverTimeMs: 100,
                expertTimeMs: 200,
                driverTrackPosition: 0.3,
                expertTrackPosition: 0.3,
            }],
        };

        expect(reasonCodes(backendPayload)).toEqual([]);
        expect(hasComparableDriverExpertData(backendPayload)).toBe(true);
    });

    it('keeps a non-empty backend payload available when optional channels are absent', () => {
        (useDesktopGame as jest.Mock).mockReturnValue({ detectedGame: null });
        Object.defineProperty(window, 'matchMedia', {
            configurable: true,
            value: jest.fn().mockReturnValue({ matches: false }),
        });
        const backendPayload = {
            samples: [{
                driverTimeMs: 100,
                expertTimeMs: 1_000,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
            }],
        };

        expect(reasonCodes(backendPayload)).toEqual([]);
        expect(hasComparableDriverExpertData(backendPayload)).toBe(true);

        render(<DriverExpertComparisonGraph data={backendPayload} />);
        expect(screen.queryByText(/^Expert comparison unavailable$/)).not.toBeInTheDocument();
    });
});
