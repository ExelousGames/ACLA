import React from 'react';
import { render, screen } from '@testing-library/react';
import { isJsonSafe } from 'views/floating-chat/ai-overlay-types';
import { buildSteeringCornerOverlay } from './steering-corner-overlay';
import { SteeringCornerDisplay, steeringCornerOverlayRenderer } from './graphs/SteeringCornerDisplay';
import { steeringCornerFixture } from './test-fixtures';

const corner = { sectionId: 'turn-1', cornerStartPosition: 0.275, cornerEndPosition: 0.435 };

it.each(['acc', 'iracing'] as const)('clips and interpolates the reference and both saved edges in %s', (game) => {
    const { map, reference } = steeringCornerFixture(game);
    const original = JSON.stringify({ map, reference });
    // Saved boundaries need not arrive sorted.
    map.samples.left_boundary!.reverse();
    const display = buildSteeringCornerOverlay(map, reference, corner)!;
    const vertical = (index: number) => reference.y[index][game === 'acc' ? 1 : 2] * (game === 'acc' ? -1 : 1);
    expect(display.title).toBe('Steering · Turn 1');
    expect(display.referenceLine).toHaveLength(18);
    expect(display.referenceLine[0].x).toBeCloseTo((reference.y[27][0] + reference.y[28][0]) / 2);
    expect(display.referenceLine[0].y).toBeCloseTo((vertical(27) + vertical(28)) / 2);
    expect(display.referenceLine[17].x).toBeCloseTo((reference.y[43][0] + reference.y[44][0]) / 2);
    expect(display.leftBoundary).toHaveLength(18);
    expect(display.rightBoundary).toHaveLength(18);
    expect(display.leftBoundary[0].x).toBeCloseTo(display.referenceLine[0].x - 5);
    expect(display.rightBoundary[0].x).toBeCloseTo(display.referenceLine[0].x + 5);
    expect(isJsonSafe(display)).toBe(true);
    expect(steeringCornerOverlayRenderer.validateSnapshot(display)).toBe(true);
    map.samples.left_boundary!.reverse();
    expect(JSON.stringify({ map, reference })).toBe(original);
});

it('orders a corner across start/finish without drawing the rest of the lap', () => {
    const { map, reference } = steeringCornerFixture();
    const display = buildSteeringCornerOverlay(map, reference, { ...corner, cornerStartPosition: 0.98, cornerEndPosition: 0.02 })!;
    expect(display.referenceLine).toEqual([98, 99, 0, 1, 2].map((i) => ({ x: reference.y[i][0], y: -reference.y[i][1] })));
    expect(display.leftBoundary).toHaveLength(5);
    expect(display.rightBoundary).toHaveLength(5);
});

it('renders the vehicle view in focus and expanded states, and just the title when folded', () => {
    const { map, reference } = steeringCornerFixture();
    const display = buildSteeringCornerOverlay(map, reference, corner)!;
    for (const status of ['focus', 'expanded', 'folded'] as const) {
        const view = render(<>{steeringCornerOverlayRenderer.renderOverlay(display, status, {
            componentName: 'steering', revision: 1, emitRendererEvent: jest.fn(),
        })}</>);
        expect(screen.getByText('Steering · Turn 1')).toBeVisible();
        if (status !== 'folded') {
            expect(screen.getByRole('img', { name: 'Vehicle view of incoming corner reference line and track boundaries' })).toBeVisible();
            for (const label of ['Reference line', 'Left boundary', 'Right boundary']) {
                expect(screen.getByLabelText(label).getAttribute('d')).toMatch(/^M /);
            }
            expect(screen.getByText('Entry')).toBeVisible();
            expect(screen.getByText('Vehicle view')).toBeVisible();
        } else expect(screen.queryByRole('img')).not.toBeInTheDocument();
        view.unmount();
    }
});

it('only labels the exit when it is in front of the vehicle and inside the view', () => {
    const display = { title: 'Steering · Turn 1', circuitName: 'Test circuit',
        referenceLine: [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 2, y: 30 }],
        leftBoundary: [], rightBoundary: [] };
    const view = render(<SteeringCornerDisplay snapshot={display} />);
    expect(screen.getByText('Exit')).toBeVisible();
    for (const exit of [{ x: 0, y: -10 }, { x: 100, y: 10 }]) {
        view.rerender(<SteeringCornerDisplay snapshot={{ ...display,
            referenceLine: [...display.referenceLine.slice(0, 2), exit] }} />);
        expect(screen.queryByText('Exit')).not.toBeInTheDocument();
    }
});

it('reports unavailable saved boundaries without substituting a centerline', () => {
    const { map, reference } = steeringCornerFixture();
    delete map.samples.right_boundary;
    const display = buildSteeringCornerOverlay(map, reference, corner)!;
    expect(display.rightBoundary).toEqual([]);
    render(<SteeringCornerDisplay snapshot={display} />);
    expect(screen.getByRole('status')).toHaveTextContent('Right boundary unavailable in Circuit Map.');
});

it('rejects unavailable or invalid reference geometry and invalid renderer snapshots', () => {
    const { map, reference } = steeringCornerFixture();
    expect(buildSteeringCornerOverlay(map, null, corner)).toBeNull();
    expect(buildSteeringCornerOverlay(map, { ...reference, game: 'iracing' }, corner)).toBeNull();
    expect(buildSteeringCornerOverlay(map, reference, { ...corner, cornerEndPosition: NaN })).toBeNull();
    expect(buildSteeringCornerOverlay(map, { ...reference, target_features: [] }, corner)).toBeNull();
    const display = buildSteeringCornerOverlay(map, reference, corner)!;
    for (const bad of [null, { ...display, referenceLine: [] }, { ...display, leftBoundary: [{ x: NaN, y: 0 }, { x: 1, y: 1 }] }]) {
        expect(steeringCornerOverlayRenderer.validateSnapshot(bad)).toBe(false);
    }
});
