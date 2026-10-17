import React from 'react';
import { render, screen } from '@testing-library/react';
import BirdsEyeView from './BirdsEyeView';
import type { TrackVisionFrame } from './track-vision-types';

const position: NonNullable<TrackVisionFrame['position']> = {
    referenceDelta: [-2, 3, -4], centerlineDelta: [5, -6, 7], inferenceMs: 20,
    normalizedPosition: 0.25,
    references: { reference: { x: 100, y: 200, z: 300 }, centerline: { x: 102, y: 205, z: 301 } },
};

it('plots signed XY offsets without adding absolute references or using Z as a ground axis', () => {
    const view = render(<BirdsEyeView scene={null} hasReconstructedScene={false} position={position} />);
    const reference = screen.getByLabelText('Reference lap position marker').querySelector('line')!;
    const centerline = screen.getByLabelText('Centerline position marker').querySelector('line')!;
    const originX = Number(reference.getAttribute('x1')), originY = Number(reference.getAttribute('y1'));
    expect(Number(reference.getAttribute('x2'))).toBeLessThan(originX);
    expect(Number(centerline.getAttribute('x2'))).toBeGreaterThan(originX);
    expect(Number(reference.getAttribute('y2'))).toBeLessThan(originY);
    expect(Number(centerline.getAttribute('y2'))).toBeGreaterThan(originY);
    expect(screen.getByLabelText('Position add-on result')).toHaveTextContent('Reference ΔXYZ: -2.000, 3.000, -4.000');
    expect(screen.getByLabelText('Position add-on result')).toHaveTextContent('Centerline ΔXYZ: 5.000, -6.000, 7.000');
    expect(screen.getByLabelText('Reference lap position marker').querySelector('title')).toHaveTextContent('Reference lap ΔXYZ: -2.000, 3.000, -4.000 · Z is height');
    const before = reference.outerHTML;
    view.rerender(<BirdsEyeView scene={null} hasReconstructedScene={false} position={{ ...position,
        referenceDelta: [-2, 3, 1000], centerlineDelta: [5, -6, -2000] }} />);
    expect(screen.getByLabelText('Reference lap position marker').querySelector('line')!.outerHTML).toBe(before);
    expect(screen.getByLabelText('Position add-on result')).toHaveTextContent('Reference ΔXYZ: -2.000, 3.000, 1000.000');
});

it.each([0, 10000])('keeps zero and large offsets inside the bird’s-eye view (%s)', (offset) => {
    render(<BirdsEyeView scene={null} hasReconstructedScene={false} position={{ ...position,
        referenceDelta: [offset, -offset, 0], centerlineDelta: [-offset, offset, 0] }} />);
    for (const label of ['Reference lap', 'Centerline']) {
        const line = screen.getByLabelText(`${label} position marker`).querySelector('line')!;
        expect(Number(line.getAttribute('x2'))).toBeGreaterThan(0);
        expect(Number(line.getAttribute('x2'))).toBeLessThan(720);
        expect(Number(line.getAttribute('y2'))).toBeGreaterThan(0);
        expect(Number(line.getAttribute('y2'))).toBeLessThan(460);
    }
    expect(screen.getByRole('img').querySelectorAll('line').length).toBeLessThan(30);
});
