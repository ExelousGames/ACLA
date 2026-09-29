import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import LocalTrackView from './LocalTrackView';
import { reconstructTrack } from './track-position-analysis';
import { vision } from './test-fixtures';
import type { CameraParameters, LocalTrackScene, TrackSceneMemory } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';

it('aligns the rendered grid with near road depth, preserves car distances and hides stale guides', () => {
    const frame = vision(Date.now(), { camera: { pitchDeg: 0 } });
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    depth.values = depth.values.map((value) => value * 0.15);
    const scene = reconstructTrack(frame)!;
    expect(scene.geometry).toBeNull();
    const { rerender } = render(<LocalTrackView frame={frame} scene={scene} camera={frame.calibration!} applied />);
    fireEvent.click(screen.getByRole('button', { name: 'Camera view' }));
    const grid = screen.getByLabelText('Depth distance grid');
    const paths = Array.from(grid.querySelectorAll('path')).map((path) => path.getAttribute('d')).join(' ');
    const coordinates = Array.from(paths.matchAll(/[ML](-?[\d.]+),(-?[\d.]+)/g));
    // At 3 m the measured road projects to SVG y=249; the old z=0 plane put it at y=385.
    expect(coordinates.some((match) => Math.abs(Number(match[2]) - 249) < 0.5)).toBe(true);
    expect(grid).not.toHaveTextContent('20 m');
    const labels = Array.from(grid.querySelectorAll('text'));
    expect(labels.length).toBeGreaterThan(1);
    // The 3, 4 and 5 m labels would otherwise overlap within ten vertical pixels.
    expect(labels.filter((label) => {
        const y = Number(label.getAttribute('y'));
        return y >= 230 && y <= 248;
    })).toHaveLength(1);
    expect(screen.getByText(`Car · ${scene.cars[0].center.y.toFixed(1)} m`)).toBeInTheDocument();
    rerender(<LocalTrackView frame={{ ...frame, capturedAt: Date.now() - VISION_MAX_AGE_MS - 1 }}
        scene={scene} camera={frame.calibration!} applied />);
    expect(grid).toBeEmptyDOMElement();
});

it.each(['left', 'right'] as const)('renders a supported %s edge when the other edge is missing', (side) => {
    const frame = vision(Date.now(), { cars: [] });
    const scene = reconstructTrack(frame)!;
    scene[side === 'left' ? 'rightBoundary' : 'leftBoundary'] = [];
    scene.geometry = null;
    render(<LocalTrackView frame={frame} scene={scene} camera={frame.calibration!} applied />);
    expect(screen.getByLabelText('Reconstruction status')).toHaveTextContent(
        `${scene.leftBoundary.length} left / ${scene.rightBoundary.length} right edge points`);
    const edges = screen.getByLabelText('Reconstructed track edges').querySelectorAll('path');
    expect(edges[side === 'left' ? 0 : 1].getAttribute('d')).toContain('L');
    expect(screen.getByLabelText('Road fit status')).toHaveTextContent('Road geometry unresolved');
});

it('draws occluded sections as dashed estimates and replaces them when the edge is visible again', () => {
    const frame = vision(Date.now(), { cars: [] });
    const scene = reconstructTrack(frame)!;
    scene.leftBoundary = scene.leftBoundary.map((point, i) => i >= 2 && i <= 5 ? { ...point, estimated: true } : point);
    const props = { frame, scene, camera: frame.calibration!, applied: true };
    const { rerender } = render(<LocalTrackView {...props} />);
    const estimated = screen.getByLabelText('Estimated track edges');
    expect(estimated).toHaveAttribute('stroke-dasharray', '5 4');
    const left = screen.getByLabelText('Estimated left track edge');
    expect(left.getAttribute('d')!.match(/M/g)).toHaveLength(1);
    expect(left.getAttribute('d')!.match(/L/g)).toHaveLength(5);
    expect(screen.getByLabelText('Estimated right track edge')).toHaveAttribute('d', '');
    const measured = screen.getByLabelText('Observed left track edge');
    expect(measured.getAttribute('d')!.match(/M/g)).toHaveLength(2);
    expect(screen.getByLabelText('Reconstruction status')).toHaveTextContent('4 edge points estimated behind traffic');
    rerender(<LocalTrackView {...props} scene={reconstructTrack(frame)} />);
    expect(screen.getByLabelText('Estimated left track edge')).toHaveAttribute('d', '');
    expect(screen.getByLabelText('Reconstruction status')).not.toHaveTextContent('estimated behind traffic');
    rerender(<LocalTrackView {...props} frame={{ ...frame, capturedAt: Date.now() - VISION_MAX_AGE_MS - 1 }} />);
    expect(estimated).toBeEmptyDOMElement();
});

it.each<[Partial<CameraParameters>, string]>([
    [{ heightM: 4 }, 'M480.00,385.00'],
    [{ pitchDeg: 45 }, 'M494.28,-41.67'],
    [{ yawDeg: 45 }, 'M133.33,319.28'],
    [{ horizontalFovDeg: 53.13010235415598 }, 'M560.00,385.00'],
    [{ lateralOffsetM: 1 }, 'M400.00,305.00'],
    [{ forwardOffsetM: -5 }, 'M440.00,265.00'],
])('uses the supplied camera and follows draft changes %j', (change, expectedStart) => {
    const frame = vision(Date.now(), { cars: [] });
    const camera = { ...frame.calibration!, heightM: 2, pitchDeg: 0, yawDeg: 0,
        horizontalFovDeg: 90, lateralOffsetM: -1, forwardOffsetM: 5 };
    const scene: LocalTrackScene = { leftBoundary: [{ x: 1, y: 15, z: 0 }, { x: 1, y: 25, z: 0 }],
        rightBoundary: [], cars: [], geometry: null };
    const { rerender } = render(<LocalTrackView frame={frame} scene={scene} camera={camera} applied />);
    fireEvent.click(screen.getByRole('button', { name: 'Camera view' }));
    const edge = () => screen.getByLabelText('Reconstructed track edges').querySelector('path')!.getAttribute('d');
    expect(screen.getByLabelText('Perspective 3D track edges and cars')).toHaveAttribute('viewBox', '0 0 800 450');
    expect(edge()).toBe('M480.00,305.00 L440.00,265.00');
    // A level camera ahead of the car cannot see the origin or the zero-meter guide.
    expect(screen.queryByText('Your car')).not.toBeInTheDocument();
    expect(screen.queryByText('0 m')).not.toBeInTheDocument();
    rerender(<LocalTrackView frame={frame} scene={scene} camera={{ ...camera, ...change }} applied={false} />);
    expect(edge()!.startsWith(expectedStart)).toBe(true);
});

it('hides the scene for invalid camera settings without substituting a camera', () => {
    const frame = vision(Date.now(), { cars: [] });
    const scene = reconstructTrack(frame)!;
    const camera = frame.calibration!;
    const { rerender } = render(<LocalTrackView frame={frame} scene={scene} camera={{ ...camera, heightM: NaN }} applied={false} />);
    expect(screen.queryByLabelText('Perspective 3D track edges and cars')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Reconstruction status')).toHaveTextContent('Enter valid camera settings');
    rerender(<LocalTrackView frame={frame} scene={scene} camera={camera} applied={false} />);
    expect(screen.getByLabelText('Perspective 3D track edges and cars')).toBeInTheDocument();
});

it('shows the reconstructed world without the capture camera or its field of view', () => {
    const frame = vision(Date.now(), { cars: [] });
    const props = { frame, scene: reconstructTrack(frame), camera: frame.calibration!, applied: false };
    const { rerender } = render(<LocalTrackView {...props} />);
    expect(screen.getByRole('button', { name: '3D overview' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByLabelText('Capture camera')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Camera field of view')).not.toBeInTheDocument();
    expect(screen.getByText('Your car')).toBeInTheDocument();
    const edge = () => screen.getByLabelText('Observed left track edge').getAttribute('d');
    const original = edge();
    // Capture pose must not affect world framing when reconstructed geometry is unchanged.
    rerender(<LocalTrackView {...props} camera={{ ...props.camera, lateralOffsetM: -3, forwardOffsetM: 5,
        yawDeg: 45, pitchDeg: -15, horizontalFovDeg: 150 }} />);
    expect(edge()).toBe(original);
    fireEvent.click(screen.getByRole('button', { name: 'Camera view' }));
    expect(edge()).not.toBe(original);
    expect(screen.queryByRole('button', { name: 'Reset view' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '3D overview' }));
    expect(edge()).toBe(original);
    expect(screen.queryByLabelText('Capture camera')).not.toBeInTheDocument();
});

it.each([
    ['mouse', 'pointerup'], ['touch', 'pointercancel'], ['pen', 'lostpointercapture'],
])('rotates the world with %s dragging and stops on %s', (pointerType, endEvent) => {
    const frame = vision(Date.now(), { cars: [] });
    render(<LocalTrackView frame={frame} scene={reconstructTrack(frame)} camera={frame.calibration!} applied />);
    const world = screen.getByLabelText('Perspective 3D track edges and cars');
    const captured = new Set<number>();
    Object.assign(world, {
        setPointerCapture: jest.fn((id: number) => captured.add(id)),
        hasPointerCapture: (id: number) => captured.has(id),
        releasePointerCapture: jest.fn((id: number) => captured.delete(id)),
    });
    // JSDOM has no PointerEvent constructor or pointer capture implementation.
    const pointer = (type: string, x: number, y: number, pointerId = 1, button = 0) => fireEvent(world,
        Object.assign(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y, button }), { pointerId, pointerType }));
    const edge = () => screen.getByLabelText('Observed left track edge').getAttribute('d');
    const original = edge();
    pointer('pointermove', 200, 200);
    pointer('pointerdown', 100, 100, 1, 2);
    pointer('pointermove', 200, 200);
    expect(edge()).toBe(original);
    pointer('pointerdown', 100, 100);
    expect(world).toHaveFocus();
    expect(world.setPointerCapture).toHaveBeenCalledWith(1);
    pointer('pointerdown', 300, 300, 2);
    pointer('pointermove', 400, 400, 2);
    expect(edge()).toBe(original);
    pointer('pointermove', 200, 100);
    const horizontal = edge();
    expect(horizontal).not.toBe(original);
    pointer('pointermove', 200, 160);
    const rotated = edge();
    expect(rotated).not.toBe(horizontal);
    pointer(endEvent, 200, 160);
    expect(world.releasePointerCapture).toHaveBeenCalledWith(1);
    pointer('pointermove', 300, 250);
    expect(edge()).toBe(rotated);
    fireEvent.click(screen.getByRole('button', { name: 'Reset view' }));
    expect(edge()).toBe(original);

    fireEvent.click(screen.getByRole('button', { name: 'Camera view' }));
    const cameraView = edge();
    pointer('pointerdown', 100, 100);
    pointer('pointermove', 200, 200);
    expect(edge()).toBe(cameraView);
});

it('supports keyboard rotation, keeps the orbit across frames and mode switches, and resets it', () => {
    const frame = vision(Date.now(), { cars: [] });
    const props = { frame, scene: reconstructTrack(frame), camera: frame.calibration!, applied: true };
    const { rerender } = render(<LocalTrackView {...props} />);
    const world = screen.getByLabelText('Perspective 3D track edges and cars');
    const edge = () => screen.getByLabelText('Observed left track edge').getAttribute('d');
    const original = edge();
    expect(world).toHaveAttribute('tabindex', '0');
    expect(world).toHaveAccessibleDescription(/Drag to rotate the 3D world/);
    fireEvent.keyDown(world, { key: 'ArrowLeft' });
    expect(edge()).not.toBe(original);
    fireEvent.keyDown(world, { key: 'ArrowRight' });
    expect(edge()).toBe(original);
    fireEvent.keyDown(world, { key: 'ArrowUp' });
    expect(edge()).not.toBe(original);
    fireEvent.keyDown(world, { key: 'ArrowDown' });
    expect(edge()).toBe(original);
    fireEvent.keyDown(world, { key: 'ArrowRight' });
    const rotated = edge();
    rerender(<LocalTrackView {...props} frame={{ ...frame, capturedAt: Date.now() + 1 }} scene={{ ...props.scene! }} />);
    expect(edge()).toBe(rotated);
    fireEvent.click(screen.getByRole('button', { name: 'Camera view' }));
    expect(world).not.toHaveAttribute('tabindex');
    const cameraView = edge();
    fireEvent.keyDown(world, { key: 'ArrowRight' });
    expect(edge()).toBe(cameraView);
    fireEvent.click(screen.getByRole('button', { name: '3D overview' }));
    expect(edge()).toBe(rotated);
    fireEvent.keyDown(world, { key: 'Home' });
    expect(edge()).toBe(original);
});

it('renders aligned memory only alongside its fresh frame with applied calibration', () => {
    const frame = vision(Date.now(), { cars: [] });
    const scene = reconstructTrack(frame)!;
    const memory: TrackSceneMemory = { capturedAt: frame.capturedAt, status: 'aligned', reason: 'Visual motion aligned.',
        matchedFeatures: 20, inliers: 18, alignmentErrorM: 0.12,
        points: [{ x: 1, y: 10, z: 0, surface: 'road', lastSeenAt: frame.capturedAt - 200, observations: 2 }] };
    const props = { frame, scene, camera: frame.calibration!, applied: true, memory };
    const { rerender } = render(<LocalTrackView {...props} />);
    expect(screen.getByLabelText('Rolling scene memory')).not.toBeEmptyDOMElement();
    expect(screen.getByLabelText('Scene memory status')).toHaveTextContent('18/20 features · 0.12 m');
    rerender(<LocalTrackView {...props} applied={false} />);
    expect(screen.getByLabelText('Rolling scene memory')).toBeEmptyDOMElement();
    rerender(<LocalTrackView {...props} memory={{ ...memory, capturedAt: frame.capturedAt - 200 }} />);
    expect(screen.getByLabelText('Rolling scene memory')).toBeEmptyDOMElement();
    const capturedAt = Date.now() - VISION_MAX_AGE_MS - 1;
    rerender(<LocalTrackView {...props} frame={{ ...frame, capturedAt }} memory={{ ...memory, capturedAt }} />);
    expect(screen.getByLabelText('Rolling scene memory')).toBeEmptyDOMElement();
});
