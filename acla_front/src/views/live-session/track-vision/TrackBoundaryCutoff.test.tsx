import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TrackBoundaryCutoff from './TrackBoundaryCutoff';

afterEach(() => { jest.restoreAllMocks(); });

it.each([
    { width: 800, height: 600, imageWidth: 1600, imageHeight: 900 },
    { width: 1200, height: 450, imageWidth: 1600, imageHeight: 900 },
    { width: 800, height: 600, imageWidth: 900, imageHeight: 1600 },
])('drags in screen rows inside a $width × $height view of a $imageWidth × $imageHeight capture', ({ width, height, imageWidth, imageHeight }) => {
    const onChange = jest.fn();
    render(<svg viewBox={`0 0 ${imageWidth} ${imageHeight}`}><TrackBoundaryCutoff
        width={imageWidth} height={imageHeight} value={0.6} onChange={onChange} /></svg>);
    jest.spyOn(SVGSVGElement.prototype, 'getBoundingClientRect').mockReturnValue({ top: 75, left: 20, width, height } as DOMRect);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    const setPointerCapture = jest.fn(), releasePointerCapture = jest.fn();
    Object.assign(handle, { setPointerCapture, releasePointerCapture, focus: jest.fn() });
    const pointer = (type: string, v: number, pointerId = 1) => {
        const scale = Math.min(width / imageWidth, height / imageHeight);
        const event = new MouseEvent(type, { bubbles: true, button: 0 });
        Object.defineProperties(event, { pointerId: { value: pointerId },
            clientY: { value: 75 + (height - imageHeight * scale) / 2 + v * imageHeight * scale } });
        fireEvent(handle, event);
    };
    pointer('pointermove', 0.8);
    expect(onChange).not.toHaveBeenCalled();
    // Grabbing near the line retains the offset instead of jumping to the pointer.
    pointer('pointerdown', 0.61);
    expect(setPointerCapture).toHaveBeenCalledWith(1);
    pointer('pointermove', 0.81, 2);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointermove', 0.61);
    expect(onChange.mock.calls.at(-1)[0]).toBeCloseTo(0.6, 5);
    pointer('pointermove', 0.81);
    expect(onChange.mock.calls.at(-1)[0]).toBeCloseTo(0.8, 5);
    pointer('pointermove', -0.2);
    expect(onChange).toHaveBeenLastCalledWith(0);
    pointer('pointerup', 1.5);
    expect(onChange).toHaveBeenLastCalledWith(1);
    expect(releasePointerCapture).toHaveBeenCalledWith(1);
    onChange.mockClear();
    pointer('pointermove', 0.8);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointerdown', 0.6);
    pointer('pointercancel', 0.6);
    pointer('pointermove', 0.8);
    expect(onChange).not.toHaveBeenCalled();
});

it('moves up and down by screen percentage without consuming unrelated keys', () => {
    const onChange = jest.fn();
    render(<svg><TrackBoundaryCutoff width={800} height={450} value={0.6} onChange={onChange} /></svg>);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    expect(handle).toHaveAttribute('aria-valuetext', '60% from top of capture; boundary detection scans upward');
    fireEvent.keyDown(handle, { key: 'ArrowUp' });
    expect(onChange).toHaveBeenLastCalledWith(0.59);
    fireEvent.keyDown(handle, { key: 'ArrowDown' });
    expect(onChange).toHaveBeenLastCalledWith(0.61);
    fireEvent.keyDown(handle, { key: 'Home' });
    expect(onChange).toHaveBeenLastCalledWith(0);
    fireEvent.keyDown(handle, { key: 'End' });
    expect(onChange).toHaveBeenLastCalledWith(1);
    onChange.mockClear();
    fireEvent.keyDown(handle, { key: 'Tab' });
    expect(onChange).not.toHaveBeenCalled();
});
