import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TrackBoundaryCutoff from './TrackBoundaryCutoff';
import { createCameraProjection, DEFAULT_CAMERA } from './camera-projection';

const camera = { ...DEFAULT_CAMERA, yawDeg: 20, forwardOffsetM: 2, lateralOffsetM: -0.4, imageWidth: 1600, imageHeight: 900 };
afterEach(() => { jest.restoreAllMocks(); });

it.each([
    { width: 800, height: 600 }, // Image has 75 px of top and bottom padding.
    { width: 1200, height: 450 }, // Image has 200 px of left and right padding.
])('drags an elevated line in local meters inside a $width × $height view', ({ width, height }) => {
    const onChange = jest.fn();
    render(<svg viewBox="0 0 800 450"><TrackBoundaryCutoff camera={camera} width={800} value={12}
        left={{ x: -5, y: 14, z: 0.3 }} right={{ x: 5, y: 14, z: 0.3 }} onChange={onChange} /></svg>);
    jest.spyOn(SVGSVGElement.prototype, 'getBoundingClientRect').mockReturnValue({ top: 75, left: 20, width, height } as DOMRect);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    const setPointerCapture = jest.fn(), releasePointerCapture = jest.fn();
    Object.assign(handle, { setPointerCapture, releasePointerCapture, focus: jest.fn() });
    const pointer = (type: string, distance: number, pointerId = 1) => {
        const pixel = createCameraProjection(camera).localToImage({ x: 0, y: camera.forwardOffsetM + distance, z: 0.3 })!;
        const scale = Math.min(width / 1600, height / 900);
        const event = new MouseEvent(type, { bubbles: true, button: 0,
            clientX: 20 + (width - 1600 * scale) / 2 + pixel.u * 1600 * scale,
            clientY: 75 + (height - 900 * scale) / 2 + pixel.v * 900 * scale });
        Object.defineProperties(event, { pointerId: { value: pointerId },
            clientX: { value: 20 + (width - 1600 * scale) / 2 + pixel.u * 1600 * scale },
            clientY: { value: 75 + (height - 900 * scale) / 2 + pixel.v * 900 * scale } });
        fireEvent(handle, event);
    };
    pointer('pointermove', 20);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointerdown', 12);
    expect(setPointerCapture).toHaveBeenCalledWith(1);
    pointer('pointermove', 20, 2);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointermove', 12);
    expect(onChange.mock.calls.at(-1)[0]).toBeCloseTo(12, 5);
    pointer('pointermove', 20);
    expect(onChange.mock.calls.at(-1)[0]).toBeCloseTo(20, 5);
    pointer('pointerup', 150);
    expect(onChange).toHaveBeenLastCalledWith(100);
    expect(releasePointerCapture).toHaveBeenCalledWith(1);
    onChange.mockClear();
    pointer('pointermove', 20);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointerdown', 12);
    pointer('pointercancel', 12);
    onChange.mockClear();
    pointer('pointermove', 20);
    expect(onChange).not.toHaveBeenCalled();
});

it('projects independent start markers at the same local Y with different image rows', () => {
    const left = { x: -5, y: 14, z: 0.2 }, right = { x: 5, y: 14, z: 0.4 };
    render(<svg><TrackBoundaryCutoff camera={camera} width={800} value={12} left={left} right={right} onChange={jest.fn()} /></svg>);
    for (const [label, point] of [['Left', left], ['Right', right]] as const) {
        const pixel = createCameraProjection(camera).localToImage(point)!;
        expect(screen.getByLabelText(`${label} boundary start`)).toHaveAttribute('cx', String(pixel.u * 800));
        expect(screen.getByLabelText(`${label} boundary start`)).toHaveAttribute('cy', String(pixel.v * 450));
    }
    expect(screen.getByLabelText('Left boundary start').getAttribute('cy')).not.toBe(screen.getByLabelText('Right boundary start').getAttribute('cy'));
});

it('supports keyboard adjustment in meters without consuming unrelated keys', () => {
    const onChange = jest.fn();
    render(<svg><TrackBoundaryCutoff camera={camera} width={800} value={12} left={null} right={null} onChange={onChange} /></svg>);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    expect(handle).toHaveAttribute('aria-valuetext', '12.0 meters forward from camera; track edges detected beyond this line');
    fireEvent.keyDown(handle, { key: 'ArrowUp' });
    expect(onChange).toHaveBeenLastCalledWith(12.5);
    fireEvent.keyDown(handle, { key: 'ArrowDown' });
    expect(onChange).toHaveBeenLastCalledWith(11.5);
    fireEvent.keyDown(handle, { key: 'Home' });
    expect(onChange).toHaveBeenLastCalledWith(0.5);
    fireEvent.keyDown(handle, { key: 'End' });
    expect(onChange).toHaveBeenLastCalledWith(100);
    onChange.mockClear();
    fireEvent.keyDown(handle, { key: 'Tab' });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByLabelText('Left boundary start')).not.toBeInTheDocument();
});
