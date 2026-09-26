import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TrackBoundaryCutoff from './TrackBoundaryCutoff';

afterEach(() => { jest.restoreAllMocks(); });

it.each([
    { width: 800, height: 600, clientY: 330 }, // Image has 75 px of top and bottom padding.
    { width: 1200, height: 450, clientY: 255 }, // Image has 200 px of left and right padding.
])('drags in source-image coordinates inside a $width × $height preview', ({ width, height, clientY }) => {
    const onChange = jest.fn();
    render(<TrackBoundaryCutoff width={1600} height={900} value={0.75} onChange={onChange} />);
    jest.spyOn(SVGSVGElement.prototype, 'getBoundingClientRect').mockReturnValue({ top: 75, left: 0, width, height } as DOMRect);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    const setPointerCapture = jest.fn(), releasePointerCapture = jest.fn();
    Object.assign(handle, { setPointerCapture, releasePointerCapture, focus: jest.fn() });
    const pointer = (type: string, y: number, pointerId = 1) => {
        const event = new MouseEvent(type, { bubbles: true, clientY: y, button: 0 });
        Object.defineProperty(event, 'pointerId', { value: pointerId });
        fireEvent(handle, event);
    };
    pointer('pointermove', clientY);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointerdown', clientY);
    expect(setPointerCapture).toHaveBeenCalledWith(1);
    expect(onChange).toHaveBeenLastCalledWith(0.4);
    pointer('pointermove', clientY + 100, 2);
    expect(onChange).toHaveBeenCalledTimes(1);
    pointer('pointermove', -100);
    expect(onChange).toHaveBeenLastCalledWith(0);
    pointer('pointerup', 2000);
    expect(onChange).toHaveBeenLastCalledWith(1);
    expect(releasePointerCapture).toHaveBeenCalledWith(1);
    onChange.mockClear();
    pointer('pointermove', clientY);
    expect(onChange).not.toHaveBeenCalled();
    pointer('pointerdown', clientY);
    pointer('pointercancel', clientY);
    onChange.mockClear();
    pointer('pointermove', clientY);
    expect(onChange).not.toHaveBeenCalled();
});

it('supports keyboard adjustment without consuming unrelated keys', () => {
    const onChange = jest.fn();
    render(<TrackBoundaryCutoff width={1600} height={900} value={0.75} onChange={onChange} />);
    const handle = screen.getByRole('slider', { name: 'Boundary start line' });
    fireEvent.keyDown(handle, { key: 'ArrowUp' });
    expect(onChange).toHaveBeenLastCalledWith(0.74);
    fireEvent.keyDown(handle, { key: 'ArrowDown' });
    expect(onChange).toHaveBeenLastCalledWith(0.76);
    fireEvent.keyDown(handle, { key: 'Home' });
    expect(onChange).toHaveBeenLastCalledWith(0);
    fireEvent.keyDown(handle, { key: 'End' });
    expect(onChange).toHaveBeenLastCalledWith(1);
    onChange.mockClear();
    fireEvent.keyDown(handle, { key: 'Tab' });
    expect(onChange).not.toHaveBeenCalled();
});
