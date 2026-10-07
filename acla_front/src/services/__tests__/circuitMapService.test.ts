import { normalizeCircuitMap, normalizeCircuitMapList } from '../circuitMapService';

describe('circuit map game normalization', () => {
    it('preserves segments with multiple tags and defaults older maps to no segments', () => {
        const segments = [{ id: 'corner', tags: ['corner', 'slow'], start_position: 0.95, end_position: 0.05 }];
        expect(normalizeCircuitMap({ centerline_segments: segments }).centerline_segments).toEqual(segments);
        expect(normalizeCircuitMap({}).centerline_segments).toEqual([]);
    });

    it('groups legacy tags with the same directed range without changing the input', () => {
        const tags = [
            { id: 'speed', label: 'slow', start_position: 0.95, end_position: 0.05 },
            { id: 'corner', label: 'corner', start_position: 0.95, end_position: 0.05 },
            { id: 'duplicate', label: 'slow', start_position: 0.95, end_position: 0.05 },
            { id: 'other', label: 'long straight', start_position: 0.05, end_position: 0.95 },
        ];
        const original = JSON.stringify(tags);
        const result = normalizeCircuitMap({ centerline_tags: tags });
        expect(result.centerline_segments).toEqual([
            { id: 'corner', tags: ['slow', 'corner'], start_position: 0.95, end_position: 0.05 },
            { id: 'other', tags: ['long straight'], start_position: 0.05, end_position: 0.95 },
        ]);
        expect(result).not.toHaveProperty('centerline_tags');
        expect(JSON.stringify(tags)).toBe(original);
        expect(normalizeCircuitMap({ centerline_segments: [], centerline_tags: tags }).centerline_segments).toEqual([]);
    });

    it.each(['acc', 'iracing'] as const)('preserves %s in saved maps and lists', (game) => {
        const map = { id: 'map-1', game, circuit_name: 'Circuit', source_track_key: 'track - layout' };
        expect(normalizeCircuitMapList({ list: [map] })).toEqual([expect.objectContaining(map)]);
        expect(normalizeCircuitMap(map, 'iracing')).toMatchObject(map);
    });

    it('uses the selected game for legacy details without a game', () => {
        expect(normalizeCircuitMap({ id: 'map-1' }, 'iracing').game).toBe('iracing');
    });
});
