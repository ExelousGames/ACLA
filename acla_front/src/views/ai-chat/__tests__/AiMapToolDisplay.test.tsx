import { render, screen } from '@testing-library/react';
import type { CircuitMapBinSample, CircuitMapDto, CircuitMapGame } from 'views/circuit-maps/circuit-map-types';
import AiMapToolDisplay from '../AiMapToolDisplay';

const makeMap = (game: CircuitMapGame, altitude = 0): CircuitMapDto => {
    const sample = (bin: number, x: number, ground: number): CircuitMapBinSample => ({
        bin,
        normalized_position: bin / 4,
        x,
        y: game === 'acc' ? ground : altitude * (bin + 1),
        z: game === 'acc' ? altitude * (bin + 1) : ground,
        sample_count: 1,
        updated_at: '2026-10-26T00:00:00Z',
    });
    return {
        id: 'test-map', game, circuit_name: 'Test circuit', resolution: 1000,
        samples: {
            left_boundary: [sample(0, 0, 0), sample(1, 0, 100), sample(2, 100, 100)],
            right_boundary: [sample(0, 10, 0), sample(1, 10, 100), sample(2, 110, 100)],
            pit_lane: [sample(0, 5, 0), sample(1, 5, 50)],
        },
    };
};

it.each(['acc', 'iracing'] as const)('projects the %s ground plane for every map layer and ignores altitude', (game) => {
    const display = { status: 'ready' as const, map: makeMap(game), section: { start: 0, end: 0.25 } };
    const { rerender } = render(<AiMapToolDisplay display={display} />);
    const paths = () => Array.from(screen.getByRole('img', { name: 'Circuit map' }).querySelectorAll('polyline'))
        .map((line) => line.getAttribute('points'));
    const expected = [
        '192.8,332.0 192.8,28.0 496.8,28.0',
        '223.2,332.0 223.2,28.0 527.2,28.0',
        '208.0,332.0 208.0,180.0',
        '208.0,332.0 208.0,28.0 512.0,28.0',
        '208.0,332.0 208.0,28.0',
    ];
    expect(paths()).toEqual(expected);

    rerender(<AiMapToolDisplay display={{ ...display, map: makeMap(game, 10000) }} />);
    expect(paths()).toEqual(expected);
});
