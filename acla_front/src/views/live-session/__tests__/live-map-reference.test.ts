import { getLiveMapReferenceLine, selectLiveMapReference } from '../live-map-reference';
import fixtures from 'services/__tests__/fixtures/top-lap-reference.json';

const lap = fixtures.expected;
it('selects the exact game, track and car with the nearest grip bucket', () => {
    const wet = { ...lap, avg_grip_int: 5 };
    const otherGame = { ...wet, game: 'iracing' };
    const entries = [{ ...lap, car: 'other' }, { ...lap, track: 'monza' }, otherGame, lap, wet];
    expect(selectLiveMapReference(entries, 'acc', 'Circuit de Spa-Francorchamps', 'car-a', 2)).toBe(lap);
    expect(selectLiveMapReference(entries, 'acc', 'spa', 'car-a', 4)).toBe(wet);
    expect(selectLiveMapReference(entries, 'acc', 'spa', 'car-a', undefined)).toBe(lap);
    expect(selectLiveMapReference(entries, 'iracing', 'spa', 'car-a', 4)).toBe(otherGame);
    expect(selectLiveMapReference(entries, 'acc', 'spa', 'unknown', 2)).toBeNull();
    expect(selectLiveMapReference(entries, 'acc', 'unknown', 'car-a', 2)).toBeNull();
    expect(selectLiveMapReference(entries, null, 'spa', 'car-a', 2)).toBeNull();
});

it('keeps iRacing layouts distinct and preserves store order for grip ties', () => {
    const first = { ...lap, game: 'iracing', track: 'spa - layout', avg_grip_int: 1 };
    const second = { ...first, avg_grip_int: 3 };
    expect(selectLiveMapReference([first, second], 'iracing', 'spa - layout', 'car-a', 2)).toBe(first);
    expect(selectLiveMapReference([first], 'iracing', 'spa', 'car-a', 2)).toBeNull();
});

it('extracts the actual expert coordinates in lap order using feature names', () => {
    expect(getLiveMapReferenceLine(lap)).toEqual([
        { x: 10, y: 1, z: 20 }, { x: 30, y: 2, z: 40 }, { x: 10, y: 1, z: 20 },
    ]);
    expect(getLiveMapReferenceLine({ ...lap, target_features: [...lap.target_features].reverse(), y: lap.y.map((row) => [...row].reverse()) }))
        .toEqual(getLiveMapReferenceLine(lap));
    expect(getLiveMapReferenceLine({ ...lap, target_features: [] })).toEqual([]);
    expect(getLiveMapReferenceLine(null)).toEqual([]);
});
