import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { copyPhraseReference } from '../phrase-driving-reference';
import { referenceRangeSegments } from '../phrase-reference-guidance';
import { variableBrakingReference } from '../test-fixtures';
import ReferenceRangeMap from './ReferenceRangeMap';

it.each(['acc', 'iracing'])('maps %s deceleration and direction through start/finish in the ground plane', (game) => {
    const reference = variableBrakingReference();
    reference.game = game;
    if (game === 'iracing') reference.y.forEach((row) => { [row[1], row[2]] = [row[2], row[1]]; });
    reference.y[reference.y.length - 1][6] = reference.y[0][6] + 10;
    const ranges = referenceRangeSegments(reference)!;
    expect(ranges.speedAvailable).toBe(true);
    expect(ranges.segments).toHaveLength(500);
    expect(ranges.segments.every(({ changesDirection }) => changesDirection)).toBe(true);
    expect(ranges.segments[0].start).toEqual({ x: 400, y: 0 });
    expect(ranges.segments[125].start.y).toBeCloseTo(400);
    const last = ranges.segments[ranges.segments.length - 1];
    expect(last.end).toEqual(ranges.segments[0].start);
    expect(last.deceleration).toBeGreaterThan(0);
    expect(ranges.segments[10].deceleration).toBeLessThan(0);
    expect(ranges.segments[20].deceleration).toBeGreaterThan(0);
});

it('keeps direction ranges when speed is missing and clears the map when geometry is invalid', () => {
    const reference = variableBrakingReference();
    reference.target_features[6] = 'unavailable';
    const { rerender } = render(<ReferenceRangeMap reference={reference} />);
    expect(screen.getByLabelText('Deceleration ranges')).toHaveAttribute('d', '');
    expect(screen.getByLabelText('Direction-change ranges').getAttribute('d')).not.toBe('');
    expect(screen.getByText(/Reference speed unavailable/)).toBeVisible();
    const invalid = { ...reference, y: reference.y.map((row) => [...row]) };
    invalid.y[0][0] = NaN;
    rerender(<ReferenceRangeMap reference={invalid} />);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText(/Open Live Map with a reference lap/)).toBeVisible();
});

const interactiveMap = () => {
    const svg = screen.getByRole('img');
    svg.getBoundingClientRect = jest.fn(() => ({
        x: 100, y: 50, left: 100, top: 50, width: 280, height: 150, right: 380, bottom: 200, toJSON: () => ({}),
    }));
    svg.setPointerCapture = jest.fn();
    svg.hasPointerCapture = jest.fn(() => true);
    svg.releasePointerCapture = jest.fn();
    return svg;
};
const pointer = (svg: HTMLElement, type: string, clientX: number, clientY: number, pointerId = 1, button = 0) => {
    const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY, button });
    Object.defineProperty(event, 'pointerId', { value: pointerId });
    fireEvent(svg, event);
};
const transform = () => screen.getByTestId('reference-range-map-viewport').getAttribute('transform');

it('zooms around the pointer at the displayed map size and prevents sidebar scrolling', () => {
    const { rerender } = render(<ReferenceRangeMap reference={null} />);
    expect(screen.queryByRole('button', { name: 'Zoom in' })).not.toBeInTheDocument();
    rerender(<ReferenceRangeMap reference={variableBrakingReference()} />);
    const svg = interactiveMap();
    const wheel = { clientX: 170, clientY: 90, deltaY: -100 };
    expect(fireEvent.wheel(svg, wheel)).toBe(false);
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%');
    expect(transform()).toBe('translate(-35 -20) scale(1.25)');
    fireEvent.wheel(svg, { ...wheel, deltaY: 100 });
    expect(transform()).toBe('translate(0 0) scale(1)');
    fireEvent.wheel(svg, { ...wheel, deltaY: 0 });
    expect(transform()).toBe('translate(0 0) scale(1)');
});

it.each(['pointerup', 'pointercancel', 'lostpointercapture'])('pans all ranges together and ends dragging on %s', (endEvent) => {
    const reference = variableBrakingReference();
    const original = JSON.stringify(reference);
    render(<ReferenceRangeMap reference={reference} />);
    const svg = interactiveMap();
    const path = screen.getByLabelText('Deceleration ranges').getAttribute('d');
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    pointer(svg, 'pointerdown', 170, 90, 1, 2);
    pointer(svg, 'pointermove', 190, 100);
    expect(transform()).toBe('translate(-70 -37.5) scale(1.25)');
    pointer(svg, 'pointerdown', 170, 90);
    expect(svg.setPointerCapture).toHaveBeenCalledWith(1);
    expect(svg).toHaveAttribute('data-dragging', 'true');
    pointer(svg, 'pointerdown', 170, 90, 2);
    pointer(svg, 'pointermove', 270, 130, 2);
    pointer(svg, 'pointerup', 270, 130, 2);
    expect(transform()).toBe('translate(-70 -37.5) scale(1.25)');
    pointer(svg, 'pointermove', 190, 100);
    expect(transform()).toBe('translate(-30 -17.5) scale(1.25)');
    pointer(svg, 'pointermove', 200, 105);
    expect(transform()).toBe('translate(-10 -7.5) scale(1.25)');
    pointer(svg, endEvent, 200, 105);
    expect(svg.releasePointerCapture).toHaveBeenCalledWith(1);
    expect(svg).toHaveAttribute('data-dragging', 'false');
    pointer(svg, 'pointermove', 240, 150);
    expect(transform()).toBe('translate(-10 -7.5) scale(1.25)');
    expect(screen.getByLabelText('Deceleration ranges')).toHaveAttribute('d', path);
    const viewport = within(screen.getByTestId('reference-range-map-viewport'));
    expect(viewport.getByLabelText('Reference line')).toBeInTheDocument();
    expect(viewport.getByLabelText('Direction-change ranges')).toBeInTheDocument();
    expect(viewport.getByText('Lap start')).toBeInTheDocument();
    expect(JSON.stringify(reference)).toBe(original);
    fireEvent.click(screen.getByRole('button', { name: 'Fit map' }));
    expect(transform()).toBe('translate(0 0) scale(1)');
});

it('bounds zoom and supports keyboard navigation and fit', () => {
    render(<ReferenceRangeMap reference={variableBrakingReference()} />);
    const svg = interactiveMap();
    for (let i = 0; i < 20; i++) fireEvent.wheel(svg, { clientX: 240, clientY: 125, deltaY: -100 });
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('600%');
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
    for (let i = 0; i < 30; i++) fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('35%');
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeDisabled();
    fireEvent.keyDown(svg, { key: 'Home' });
    expect(transform()).toBe('translate(0 0) scale(1)');
    expect(fireEvent.keyDown(svg, { key: 'ArrowLeft' })).toBe(false);
    fireEvent.keyDown(svg, { key: 'ArrowDown' });
    expect(transform()).toBe('translate(28 -28) scale(1)');
    fireEvent.keyDown(svg, { key: '+' });
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%');
    fireEvent.keyDown(svg, { key: '-' });
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
    fireEvent.keyDown(svg, { key: '0' });
    expect(transform()).toBe('translate(0 0) scale(1)');
});

it('preserves the view across aggression updates and fits a changed or reloaded map', () => {
    const reference = variableBrakingReference();
    const { rerender } = render(<ReferenceRangeMap reference={reference} />);
    const svg = interactiveMap();
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    pointer(svg, 'pointerdown', 170, 90);
    pointer(svg, 'pointermove', 190, 100);
    pointer(svg, 'pointerup', 190, 100);
    const pannedView = transform();
    const deceleration = screen.getByLabelText('Deceleration ranges').getAttribute('d');
    rerender(<ReferenceRangeMap reference={copyPhraseReference(reference, 100)} />);
    expect(transform()).toBe(pannedView);
    expect(screen.getByLabelText('Deceleration ranges').getAttribute('d')).not.toBe(deceleration);
    rerender(<ReferenceRangeMap reference={{ ...reference, track: 'Another track' }} />);
    expect(transform()).toBe('translate(0 0) scale(1)');
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    rerender(<ReferenceRangeMap reference={null} />);
    rerender(<ReferenceRangeMap reference={reference} />);
    expect(transform()).toBe('translate(0 0) scale(1)');
    fireEvent.wheel(interactiveMap(), { clientX: 170, clientY: 90, deltaY: -100 });
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%');
});
