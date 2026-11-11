import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { createConsecutiveCornerPositioning } from './phrase-consecutive-corners';
import type { PhraseMapContext } from './phrase-map-context';

const wrap = (position: number) => ((position % 1) + 1) % 1;
function fixture(game: 'acc' | 'iracing' = 'acc', fraction: number | ((position: number) => number) = 0.8, shift = 0) {
    const line = (side: number | ((position: number) => number)) => Array.from({ length: 200 }, (_, index) => {
        const position = index / 200, angle = position * Math.PI * 2;
        const lateral = (typeof side === 'number' ? side : side(position)) * 20 - 10;
        const radius = 100 + lateral * (game === 'acc' ? -1 : 1);
        return { bin: index, normalized_position: wrap(position + shift), x: radius * Math.cos(angle),
            y: game === 'acc' ? radius * Math.sin(angle) : 5000,
            z: game === 'iracing' ? radius * Math.sin(angle) : 5000, sample_count: 1, updated_at: '' };
    }).sort((a, b) => a.normalized_position - b.normalized_position);
    const target = line(fraction);
    const map: CircuitMapDto = { id: 'positioning-map', game, circuit_name: 'test', resolution: 200,
        samples: { middle_line: line(0.5), left_boundary: line(0), right_boundary: line(1) } };
    const reference: TopLapReferenceEntry = { game, track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z'],
        x: target.map((point) => point.normalized_position), y: target.map(({ x, y, z }) => [x, y, z]) };
    const corner: PhraseMapContext = { sequenceId: 'sequence', sequenceRemaining: 1,
        cornerStartPosition: wrap(0.05 + shift), nextCornerStartPosition: wrap(0.123 + shift), nextCornerDirection: 'left' };
    return { map, reference, corner };
}

describe.each(['acc', 'iracing'] as const)('reference positioning in %s', (game) => {
    it.each([[0.15, 'left'], [0.5, 'middle'], [0.85, 'right']] as const)(
        'uses reference fraction %s as the %s side, independent of the next turn', (fraction, side) => {
            const { map, reference, corner } = fixture(game, fraction);
            const position = createConsecutiveCornerPositioning(map, reference);
            expect(position(corner)).toBe(`${side}-for-left`);
            expect(position({ ...corner, nextCornerDirection: 'right' })).toBe(`${side}-for-right`);
        });

    it('samples the next entry rather than the current reference side', () => {
        const { map, reference, corner } = fixture(game, (position) => position < 0.1 ? 0.1 : 0.9);
        expect(createConsecutiveCornerPositioning(map, reference)(corner)).toBe('right-for-left');
    });

    it('interpolates the reference and cross-section across start/finish', () => {
        const { map, reference, corner } = fixture(game, 0.1, 0.88);
        expect(corner.cornerStartPosition).toBeGreaterThan(corner.nextCornerStartPosition!);
        expect(createConsecutiveCornerPositioning(map, reference)(corner)).toBe('left-for-left');
        expect(createConsecutiveCornerPositioning(map, reference)({ ...corner, nextCornerStartPosition: 0.999 }))
            .toBe('left-for-left');
    });
});

it.each(['missing reference', 'game mismatch', 'missing coordinates', 'nonfinite coordinates', 'unsorted positions',
    'duplicate positions', 'mismatched rows', 'missing boundary', 'degenerate width', 'longitudinal edges', 'missing tangent'])(
    'withholds positioning with %s', (failure) => {
        const { map, reference, corner } = fixture();
        if (failure === 'game mismatch') reference.game = 'iracing';
        if (failure === 'missing coordinates') reference.target_features[0] = 'missing';
        if (failure === 'nonfinite coordinates') reference.y[20][0] = NaN;
        if (failure === 'unsorted positions') reference.x.reverse();
        if (failure === 'duplicate positions') reference.x[1] = reference.x[0];
        if (failure === 'mismatched rows') reference.y.pop();
        if (failure === 'missing boundary') delete map.samples.right_boundary;
        if (failure === 'degenerate width') map.samples.right_boundary = map.samples.left_boundary;
        if (failure === 'longitudinal edges') {
            map.samples.left_boundary = map.samples.middle_line!.map((point) => {
                const angle = point.normalized_position * Math.PI * 2;
                return { ...point, x: point.x + 10 * Math.sin(angle), y: point.y - 10 * Math.cos(angle) };
            });
            map.samples.right_boundary = map.samples.middle_line!.map((point) => {
                const angle = point.normalized_position * Math.PI * 2;
                return { ...point, x: point.x - 10 * Math.sin(angle), y: point.y + 10 * Math.cos(angle) };
            });
        }
        if (failure === 'missing tangent') map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point, x: 0, y: 0 }));
        expect(createConsecutiveCornerPositioning(map, failure === 'missing reference' ? null : reference)(corner)).toBeUndefined();
    });

it.each([-0.2, 1.2])('rejects a reference outside the track at fraction %s', (fraction) => {
    const { map, reference, corner } = fixture('acc', fraction);
    expect(createConsecutiveCornerPositioning(map, reference)(corner)).toBeUndefined();
});

it('rejects an entry reference displaced along the track beyond the cross-section', () => {
    const { map, reference, corner } = fixture();
    reference.y.forEach((row, index) => {
        const angle = reference.x[index] * Math.PI * 2;
        row[0] -= 30 * Math.sin(angle);
        row[1] += 30 * Math.cos(angle);
    });
    expect(createConsecutiveCornerPositioning(map, reference)(corner)).toBeUndefined();
});

it('requires an identified next turn in the same sequence', () => {
    const { map, reference, corner } = fixture();
    const position = createConsecutiveCornerPositioning(map, reference);
    expect(position(undefined)).toBeUndefined();
    for (const patch of [{ sequenceId: undefined }, { sequenceRemaining: 0 }, { nextCornerStartPosition: undefined },
        { nextCornerDirection: undefined }, { nextCornerStartPosition: NaN }]) {
        expect(position({ ...corner, ...patch })).toBeUndefined();
    }
});
