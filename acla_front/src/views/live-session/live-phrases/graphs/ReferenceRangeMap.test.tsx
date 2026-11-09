import React from 'react';
import { render, screen } from '@testing-library/react';
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
