import { inspectDrivingReference, prepareDrivingReference, referenceSpeedTargets } from './driving-reference';
import { brakingFixtureSpeed, brakingReferenceFixture, referenceLapFixture } from './training-fixtures';

function speedLap(x: number[], y: number[][]) {
    return { ...referenceLapFixture(), x, y: y.map((row) => row.length === 4 ? [...row, 1, 0] : row),
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_speed', 'expert_optimal_throttle', 'expert_optimal_brake'] };
}

it('uses raw XYZ travel distance, interpolates speed, and caches immutable speed and pedals', () => {
    const lap = speedLap([0.1, 0.3, 0.7], [[0, 0, 0, 120], [3, 4, 12, 80], [6, 8, 24, 120]]);
    const reference = prepareDrivingReference(lap)!;
    expect(reference.points.map((point) => point.distance)).toEqual([0, 13, 26]);
    expect(reference.length).toBe(52);
    expect(referenceSpeedTargets(reference, 0.2, 50).reference).toBeCloseTo(100);
    expect(prepareDrivingReference(lap)).toBe(reference);
    expect(Object.isFrozen(reference.points[0])).toBe(true);
    lap.y[0][3] = 999;
    lap.y[0][4] = 0;
    expect(reference.points[0]).toMatchObject({ speed: 120, throttle: 1, brake: 0 });
});

it.each([0, 12.5, 25, 49, 50])('matches Live Phrases braking adjustment at training aggression %s', (aggression) => {
    const { lap, map } = brakingReferenceFixture();
    const original = JSON.stringify(lap);
    const reference = prepareDrivingReference(lap, map)!;
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    if (aggression < 50) expect(referenceSpeedTargets(reference, 0.2, aggression).target).toBeLessThan(lap.y[20][speed]);
    lap.x.forEach((position, i) => {
        expect(referenceSpeedTargets(reference, position, aggression).reference).toBe(lap.y[i][speed]);
        expect(referenceSpeedTargets(reference, position, aggression).target).toBeCloseTo(brakingFixtureSpeed(position, aggression), 8);
        if (i + 1 < lap.x.length) {
            const midpoint = (position + lap.x[i + 1]) / 2;
            expect(referenceSpeedTargets(reference, midpoint, aggression).target)
                .toBeCloseTo((brakingFixtureSpeed(position, aggression) + brakingFixtureSpeed(lap.x[i + 1], aggression)) / 2);
        }
    });
    expect(JSON.stringify(lap)).toBe(original);
});

it.each([[50, 1], [51, 1.01], [75, 1.25], [100, 1.5]])('increases every reference speed by the direct multiplier at aggression %s', (aggression, multiplier) => {
    const reference = prepareDrivingReference(referenceLapFixture())!;
    for (let i = 0; i <= 200; i++) {
        const result = referenceSpeedTargets(reference, i / 200, aggression);
        expect(result.target).toBeCloseTo(result.reference * multiplier);
    }
});

it('preserves pending target snapshots and rebuilds targets when the circuit map changes', () => {
    const { lap, map } = brakingReferenceFixture();
    const reference = prepareDrivingReference(lap, map)!;
    expect(prepareDrivingReference(lap, map)).toBe(reference);
    const withoutCorners = prepareDrivingReference(lap, { ...map, centerline_tags: [] })!;
    expect(withoutCorners).not.toBe(reference);
    expect(prepareDrivingReference(lap, map)).toBe(reference);
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    const i = 20;
    expect(referenceSpeedTargets(withoutCorners, lap.x[i], 0).target).toBe(lap.y[i][speed]);
    // Change the source before the old reference lazily prepares its first cautious profile.
    lap.y.forEach((row) => { row[speed] = 999; });
    map.centerline_tags = [];
    expect(referenceSpeedTargets(reference, lap.x[i], 0).target).toBeCloseTo(brakingFixtureSpeed(lap.x[i], 0));
});

it('uses the same unadjusted fallback as Live Phrases without tagged corners', () => {
    const { lap } = brakingReferenceFixture();
    const reference = prepareDrivingReference(lap)!;
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    lap.x.forEach((position, i) => expect(referenceSpeedTargets(reference, position, 0).target).toBe(lap.y[i][speed]));
});

it('interpolates speed across start/finish even with unchanged pedals', () => {
    const reference = prepareDrivingReference(speedLap([0.1, 0.5, 0.9],
        [[0, 0, 0, 120], [100, 0, 0, 100], [100, 100, 0, 80]]))!;
    for (const aggression of [0, 50, 100]) {
        const multiplier = 1 + Math.max(0, aggression - 50) / 100;
        expect(referenceSpeedTargets(reference, 0, aggression).target).toBeCloseTo(100 * multiplier);
        expect(referenceSpeedTargets(reference, 1, aggression).target).toBeCloseTo(100 * multiplier);
        expect(referenceSpeedTargets(reference, 0.95, aggression).target).toBeCloseTo(90 * multiplier);
    }
});

it.each([-1, 101, NaN, Infinity])('rejects invalid aggression %s', (aggression) => {
    expect(() => referenceSpeedTargets(prepareDrivingReference(referenceLapFixture())!, 0.25, aggression)).toThrow(/aggression/);
});

it('preserves different recorded start/finish speeds and includes their closing segment', () => {
    const lap = speedLap([0, 0.25, 0.5, 0.75, 1],
        [[0, 0, 0, 120], [100, 0, 0, 120], [100, 100, 0, 120], [0, 100, 0, 120], [0, 5, 0, 80]]);
    const original = JSON.stringify(lap);
    const reference = prepareDrivingReference(lap)!;
    expect(reference.points.map((point) => point.distance)).toEqual([0, 100, 200, 300, 395]);
    expect(reference.length).toBe(400);
    expect(referenceSpeedTargets(reference, 0, 50).target).toBe(120);
    expect(referenceSpeedTargets(reference, 1, 50).target).toBe(80);
    expect(JSON.stringify(lap)).toBe(original);
});

it.each([
    ['coordinates', /missing raw Y coordinates/], ['speed', /missing speed/],
    ['throttle', /missing throttle/], ['brake', /missing brake/],
    ['nonfinite', /invalid raw XYZ coordinates at sample 3/], ['range', /invalid speed at sample 3/],
    ['nanSpeed', /invalid speed at sample 3/], ['positions', /invalid normalized positions/],
    ['rows', /matching positions and telemetry rows/], ['stationary', /no usable travel distance/],
])('rejects unusable reference data and explains the failing check: %s', (fault, message) => {
    const lap = referenceLapFixture();
    if (fault === 'coordinates') lap.target_features[1] = 'unrelated';
    if (fault === 'speed') lap.target_features[5] = 'unrelated';
    if (fault === 'throttle') lap.target_features[3] = 'unrelated';
    if (fault === 'brake') lap.target_features[4] = 'unrelated';
    if (fault === 'nonfinite') lap.y[2][0] = NaN;
    if (fault === 'range') lap.y[2][5] = -1;
    if (fault === 'nanSpeed') lap.y[2][5] = NaN;
    if (fault === 'positions') lap.x[2] = lap.x[1];
    if (fault === 'rows') lap.y.pop();
    if (fault === 'stationary') lap.y.forEach((row) => { row[0] = row[1] = row[2] = 0; });
    expect(prepareDrivingReference(lap)).toBeNull();
    expect(inspectDrivingReference(lap).error).toMatch(message);
    expect(prepareDrivingReference(null)).toBeNull();
});

describe.each([['throttle', 3], ['brake', 4]] as const)('reference %s validation', (pedal, column) => {
    it.each([NaN, Infinity, undefined, -0.01, 1.01, '0'])('rejects invalid value %s', (value) => {
        const lap = referenceLapFixture();
        (lap.y[2] as unknown[])[column] = value;
        expect(inspectDrivingReference(lap).error).toContain(`invalid ${pedal} at sample 3`);
    });
});
