import { getLiveMapReferenceHit, getLiveMapReferenceLine, getLiveMapReferencePedal, selectLiveMapReference } from '../live-map-reference';
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

it('extracts normalized reference pedals by feature name without changing the lap', () => {
    const reference = { ...lap, target_features: ['expert_optimal_throttle', 'expert_optimal_brake'],
        y: [[1, 0], [0.25, 0.75], [0, 1]] };
    expect(getLiveMapReferencePedal(reference, 'brake')).toEqual([0, 0.75, 1]);
    expect(getLiveMapReferencePedal(reference, 'throttle')).toEqual([1, 0.25, 0]);
    expect(getLiveMapReferencePedal({ ...reference, target_features: [...reference.target_features].reverse(),
        y: reference.y.map((row) => [...row].reverse()) }, 'brake')).toEqual([0, 0.75, 1]);
    expect(reference.y).toEqual([[1, 0], [0.25, 0.75], [0, 1]]);
});

it.each(['brake', 'throttle'] as const)('distinguishes zero %s input from unavailable data', (pedal) => {
    const reference = { ...lap, target_features: [`expert_optimal_${pedal}`], y: [[0], [0], [0]] };
    expect(getLiveMapReferencePedal(reference, pedal)).toEqual([0, 0, 0]);
    expect(getLiveMapReferencePedal(null, pedal)).toEqual([]);
    expect(getLiveMapReferencePedal(lap, pedal)).toEqual([]);
    expect(getLiveMapReferencePedal({ ...reference, x: [0], y: [[1]] }, pedal)).toEqual([]);
    for (const value of [NaN, Infinity, -0.1, 1.1]) {
        expect(getLiveMapReferencePedal({ ...reference, y: [[0], [value], [0]] }, pedal)).toEqual([]);
    }
});

it('finds the nearest point along a rendered segment within eight pixels', () => {
    const points = [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }];
    expect(getLiveMapReferenceHit(points, { x: 25, y: 8 })).toEqual({ index: 0, fraction: 0.25, x: 25, y: 0 });
    expect(getLiveMapReferenceHit(points, { x: 25, y: 9 })).toBeNull();
    const cornerHit = getLiveMapReferenceHit(points, { x: 98, y: 7 });
    expect(cornerHit).toMatchObject({ index: 1, fraction: 0.07, x: 100 });
    expect(cornerHit!.y).toBeCloseTo(7);
    expect(getLiveMapReferenceHit(points, { x: -3, y: 0 })).toEqual({ index: 0, fraction: 0, x: 0, y: 0 });
    expect(getLiveMapReferenceHit(points, { x: 100, y: 103 })).toEqual({ index: 1, fraction: 1, x: 100, y: 100 });
    // Do not create a closing segment that the reference line does not draw.
    expect(getLiveMapReferenceHit(points, { x: 50, y: 50 })).toBeNull();
});

it('ignores empty and zero-length reference segments and keeps the first segment on ties', () => {
    expect(getLiveMapReferenceHit([], { x: 0, y: 0 })).toBeNull();
    expect(getLiveMapReferenceHit([{ x: 0, y: 0 }], { x: 0, y: 0 })).toBeNull();
    const points = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 0 }];
    expect(getLiveMapReferenceHit(points, { x: 25, y: 25 })).toEqual({ index: 1, fraction: 0.25, x: 25, y: 25 });
});
