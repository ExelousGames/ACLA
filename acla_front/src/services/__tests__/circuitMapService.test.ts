import { normalizeCircuitMap, normalizeCircuitMapList } from '../circuitMapService';

describe('circuit map game normalization', () => {
    it('preserves range tags and defaults legacy maps to no tags', () => {
        const tags = [{ id: 'corner', label: 'Turn 1', start_position: 0.95, end_position: 0.05 }];
        expect(normalizeCircuitMap({ centerline_tags: tags }).centerline_tags).toEqual(tags);
        expect(normalizeCircuitMap({}).centerline_tags).toEqual([]);
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
