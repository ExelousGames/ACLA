import React from 'react';
import { render, screen } from '@testing-library/react';
import LocalTrackView from './LocalTrackView';
import { reconstructTrack } from './track-position-analysis';
import { vision } from './test-fixtures';
import type { CameraParameters, LocalTrackScene } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';

it('aligns the rendered grid with near road depth, preserves car distances and hides stale guides', () => {
    const frame = vision(Date.now(), { camera: { pitchDeg: 0 } });
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    depth.values = depth.values.map((value) => value * 0.15);
    const scene = reconstructTrack(frame)!;
    expect(scene.geometry).toBeNull();
    const { rerender } = render(<LocalTrackView frame={frame} scene={scene} camera={frame.calibration!} applied />);
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

it('shows measured start positions independently and clears them when the frame expires', () => {
    const frame = vision(Date.now(), { cars: [], corner: 'straight', player: 'middle', camera: { yawDeg: 15, forwardOffsetM: 1 } });
    frame.boundaryStartDistanceM = 12;
    const scene = reconstructTrack(frame)!;
    const props = { frame, scene, camera: frame.calibration!, applied: true, boundaryStartDistanceM: 12, onBoundaryStartChange: jest.fn() };
    const { rerender } = render(<LocalTrackView {...props} />);
    const positions = screen.getByLabelText('Boundary starting positions');
    for (const point of [scene.leftBoundary[0], scene.rightBoundary[0]]) {
        expect(positions).toHaveTextContent(`${point.x.toFixed(2)}, 13.00, ${point.z.toFixed(2)} m`);
    }
    expect(screen.getByLabelText('Left boundary start')).toBeInTheDocument();
    expect(screen.getByLabelText('Right boundary start')).toBeInTheDocument();
    rerender(<LocalTrackView {...props} scene={{ ...scene, leftBoundary: [] }} />);
    expect(screen.queryByLabelText('Left boundary start')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Right boundary start')).toBeInTheDocument();
    expect(positions).toHaveTextContent('Unobserved at start line');
    rerender(<LocalTrackView {...props} frame={{ ...frame, capturedAt: Date.now() - VISION_MAX_AGE_MS - 1 }} />);
    expect(screen.queryByLabelText('Left boundary start')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Right boundary start')).not.toBeInTheDocument();
    expect(positions).not.toHaveTextContent('13.00');
});
