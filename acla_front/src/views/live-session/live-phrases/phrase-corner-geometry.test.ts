import { createPhraseMapContext } from './phrase-map-context';
import { shapedCornerMap } from './test-fixtures';

describe('centerline corner shape', () => {
    it.each(['bend', 'hairpin', 's-bend', 'tightening', 'opening'] as const)('measures a %s from its full path', (shape) => {
        const result = createPhraseMapContext(shapedCornerMap(shape)).at(0.2);
        expect(result.cornerShape).toBe(shape);
        expect(result.cornerGeometry?.lengthM).toBeCloseTo(405);
        if (shape === 's-bend') {
            expect(result.cornerGeometry?.turnSign).toBeUndefined();
            expect(result.cornerGeometry?.headingChangeDeg).toBeCloseTo(0, 0);
        } else {
            expect(result.cornerGeometry?.turnSign).toBe(1);
            expect(result.cornerGeometry!.headingChangeDeg).toBeGreaterThan(shape === 'hairpin' ? 170 : 85);
        }
    });

    it.each(['bend', 'hairpin', 's-bend', 'tightening', 'opening'] as const)('preserves %s shape with mirrored coordinates and uneven capture density', (shape) => {
        const map = shapedCornerMap(shape);
        const line = map.samples.middle_line!;
        map.samples.middle_line = line.flatMap((point, index) => {
            const next = line[index + 1];
            return next && index < 30 ? [point, ...[0.1, 0.2, 0.3].map((fraction) => ({
                ...point,
                normalized_position: point.normalized_position + (next.normalized_position - point.normalized_position) * fraction,
                x: point.x + (next.x - point.x) * fraction,
                y: point.y + (next.y - point.y) * fraction,
            }))] : [point];
        }).map((point) => ({ ...point, y: -point.y })).reverse();
        const result = createPhraseMapContext(map).at(0.2);
        expect(result.cornerShape).toBe(shape);
        if (shape !== 's-bend') expect(result.cornerGeometry?.turnSign).toBe(-1);
    });

    it('orders the path across start/finish and ignores geometry outside the corner', () => {
        const map = shapedCornerMap('hairpin');
        map.samples.middle_line!.forEach((point) => {
            point.normalized_position = (point.normalized_position + 0.85) % 1;
        });
        map.samples.middle_line!.push({ ...map.samples.middle_line![0], normalized_position: 0.5, x: -5000, y: 5000 });
        map.centerline_segments![0].start_position = 0.95;
        map.centerline_segments![0].end_position = 0.15;
        expect(createPhraseMapContext(map).at(0.05)).toMatchObject({ cornerShape: 'hairpin', phase: 'middle' });
    });

    it('withholds shape for straight, missing or degenerate geometry', () => {
        for (const kind of ['straight', 'missing', 'duplicate', 'tiny'] as const) {
            const map = shapedCornerMap('bend');
            if (kind === 'missing') map.samples.middle_line = map.samples.middle_line!.filter((_, index) => index === 0 || index === 81);
            else map.samples.middle_line!.forEach((point, index) => {
                if (kind === 'straight') { point.x = index * 5; point.y = 0; }
                if (kind === 'duplicate') { point.x = 1; point.y = 1; }
                if (kind === 'tiny') { point.x *= 0.001; point.y *= 0.001; }
            });
            expect(createPhraseMapContext(map).at(0.2).cornerShape).toBeUndefined();
        }
    });

    it('does not classify small capture noise as a direction change', () => {
        const map = shapedCornerMap('bend');
        map.samples.middle_line!.forEach((point, index) => { point.y += index % 2 ? 0.05 : -0.05; });
        expect(createPhraseMapContext(map).at(0.2).cornerShape).toBe('bend');
    });

    it.each(['acc', 'iracing'] as const)('measures %s corners on the ground plane regardless of altitude', (game) => {
        const map = shapedCornerMap('hairpin');
        map.game = game;
        map.samples.middle_line = map.samples.middle_line!.map((point, index) => game === 'acc'
            ? { ...point, z: index % 2 ? 1000 : -1000 }
            : { ...point, y: index % 2 ? 1000 : -1000, z: point.y });
        const original = JSON.stringify(map);
        const result = createPhraseMapContext(map).at(0.2);
        expect(result.cornerShape).toBe('hairpin');
        expect(result.cornerGeometry?.lengthM).toBeCloseTo(405);
        expect(result.cornerDirection).toBe(game === 'acc' ? 'right' : 'left');
        expect(JSON.stringify(map)).toBe(original);
    });
});
