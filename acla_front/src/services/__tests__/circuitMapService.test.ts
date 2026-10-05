import { normalizeCircuitMap, normalizeCircuitMapList } from '../circuitMapService';

describe('circuit map game normalization', () => {
    it.each(['acc', 'iracing', 'other'] as const)('preserves %s in saved maps and lists', (game) => {
        const map = { id: 'map-1', game, circuit_name: 'Circuit', source_track_key: 'track - layout' };
        expect(normalizeCircuitMapList({ list: [map] })).toEqual([expect.objectContaining(map)]);
        expect(normalizeCircuitMap(map, 'iracing')).toMatchObject(map);
    });

    it('uses the selected game for legacy details without a game', () => {
        expect(normalizeCircuitMap({ id: 'map-1' }, 'iracing').game).toBe('iracing');
    });
});
