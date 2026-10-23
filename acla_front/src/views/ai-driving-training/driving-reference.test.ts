import { inspectDrivingReference, prepareDrivingReference, referencePedalTargets } from './driving-reference';
import { referenceLapFixture } from './training-fixtures';

it('accumulates consecutive raw XYZ differences and the closing segment without a track length', () => {
    const lap = referenceLapFixture();
    lap.x = [0.1, 0.3, 0.7];
    lap.y = [[0, 0, 0, 1, 0], [3, 4, 12, 0, 1], [6, 8, 24, 1, 0]];
    const reference = prepareDrivingReference(lap)!;
    expect(reference.points.map((point) => point.distance)).toEqual([0, 13, 26]);
    expect(reference.length).toBe(52);
    expect(referencePedalTargets(reference, 0.2, 50).reference.throttle).toBeCloseTo(0.5);
    expect(referencePedalTargets(reference, 0.2, 50).reference.brake).toBeCloseTo(0.5);
    expect(prepareDrivingReference(lap)).toBe(reference);
    expect(Object.isFrozen(reference.points[0])).toBe(true);
});

it('locates windows by raw distance instead of treating normalized lap position as uniform distance', () => {
    const lap = referenceLapFixture();
    lap.x = [0, 0.1, 0.9, 1];
    lap.y = [[0, 0, 0, 1, 0], [60, 0, 0, 0.8, 0.2], [60, 80, 0, 0.2, 0.8], [0, 0, 0, 1, 0]];
    const reference = prepareDrivingReference(lap)!;
    expect(reference.length).toBe(240);
    expect(referencePedalTargets(reference, 0.5, 0).target.brake).toBeCloseTo(0.65);
    expect(referencePedalTargets(reference, 0.5, 100).target.brake).toBeCloseTo(0.35);
});

it.each([
    [0.175, 0.25, 0, 0], [0.25, 1, 0.5, 0], [0.325, 1, 1, 0.75],
    [0.4, 1, 1, 1], [0.575, 0.75, 0.25, 0], [0.625, 0.25, 0, 0],
])('expands/original/contracts braking at position %s with complementary throttle', (position, cautious, neutral, aggressive) => {
    const reference = prepareDrivingReference(referenceLapFixture())!;
    [0, 50, 100].forEach((aggression, i) => {
        const result = referencePedalTargets(reference, position, aggression);
        expect(result.target.brake).toBeCloseTo([cautious, neutral, aggressive][i]);
        expect(result.target.throttle).toBeCloseTo(1 - [cautious, neutral, aggressive][i]);
        expect(result.radius).toBe(aggression === 50 ? 0 : 20);
    });
});

it('interpolates both window boundaries and leaves original pedals at 50', () => {
    const reference = prepareDrivingReference(referenceLapFixture())!;
    const cautious = referencePedalTargets(reference, 0.24, 25);
    expect(cautious.radius).toBe(10);
    expect(cautious.reference.brake).toBeCloseTo(0.4);
    expect(cautious.target.brake).toBeCloseTo(0.65);
    expect(referencePedalTargets(reference, 0.24, 75).target.brake).toBeCloseTo(0.15);
    const neutral = referencePedalTargets(reference, 0.24, 50);
    expect(neutral.target).toEqual(neutral.reference);
});

it('interpolates across start/finish for laps without explicit 0/1 endpoints', () => {
    const lap = referenceLapFixture();
    lap.x = [0.1, 0.5, 0.9];
    lap.y = [[0, 0, 0, 1, 0], [100, 0, 0, 0.5, 0.5], [100, 100, 0, 0, 1]];
    const reference = prepareDrivingReference(lap)!;
    const neutral = referencePedalTargets(reference, 0, 50);
    expect(neutral.target.brake).toBeCloseTo(0.5);
    expect(referencePedalTargets(reference, 1, 50).target.brake).toBeCloseTo(0.5);
    expect(referencePedalTargets(reference, 0.95, 50).target.brake).toBeCloseTo(0.75);
    const change = 20 / Math.hypot(100, 100);
    expect(referencePedalTargets(reference, 0, 0).target.brake).toBeCloseTo(0.5 + change);
    expect(referencePedalTargets(reference, 1, 100).target.brake).toBeCloseTo(0.5 - change);
});

it('wraps both window boundaries and includes extrema on the other side of start/finish', () => {
    const lap = referenceLapFixture();
    lap.y = lap.y.map((row, i) => [...row.slice(0, 3), lap.x[i] <= 0.1 ? 1 - lap.x[i] * 10 : 0,
        lap.x[i] <= 0.1 ? lap.x[i] * 10 : 1]);
    lap.y[lap.y.length - 1] = lap.y[0].slice();
    const reference = prepareDrivingReference(lap)!;
    expect(referencePedalTargets(reference, 0.99, 0).target.brake).toBeCloseTo(0.6);
    expect(referencePedalTargets(reference, 0.01, 100).target.brake).toBeCloseTo(0);
    expect(referencePedalTargets(reference, 0, 0)).toEqual(referencePedalTargets(reference, 1, 0));
});

it('handles coincident consecutive coordinates and windows larger than the lap', () => {
    const lap = referenceLapFixture();
    lap.x = [0, 0.2, 0.4, 0.8];
    lap.y = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [2, 0, 0, 0.5, 0.5], [1, 0, 0, 1, 0]];
    const reference = prepareDrivingReference(lap)!;
    expect(reference.length).toBe(4);
    expect(referencePedalTargets(reference, 0.1, 50).target).toEqual({ throttle: 0.5, brake: 0.5 });
    expect(referencePedalTargets(reference, 0.9, 0).target).toEqual({ throttle: 0, brake: 1 });
    expect(referencePedalTargets(reference, 0.9, 100).target).toEqual({ throttle: 1, brake: 0 });
});

it('accepts recorded 0/1 endpoints that differ and includes the closing segment in pedal windows', () => {
    const lap = referenceLapFixture();
    lap.x = [0, 0.25, 0.5, 0.75, 1];
    lap.y = [[0, 0, 0, 1, 0], [100, 0, 0, 1, 0], [100, 100, 0, 1, 0], [0, 100, 0, 1, 0], [0, 5, 0, 0, 1]];
    const original = JSON.stringify(lap);
    const reference = prepareDrivingReference(lap)!;
    expect(reference.points.map((point) => point.distance)).toEqual([0, 100, 200, 300, 395]);
    expect(reference.length).toBe(400);
    expect(referencePedalTargets(reference, 0, 50).target).toEqual({ throttle: 1, brake: 0 });
    expect(referencePedalTargets(reference, 1, 50).target).toEqual({ throttle: 0, brake: 1 });
    expect(referencePedalTargets(reference, 0, 0).target).toEqual({ throttle: 0, brake: 1 });
    expect(referencePedalTargets(reference, 1, 100).target).toEqual({ throttle: 1, brake: 0 });
    expect(JSON.stringify(lap)).toBe(original);
});

it.each([
    ['coordinates', /missing raw Y coordinates/], ['throttle', /missing throttle/], ['brake', /missing brake/],
    ['nonfinite', /invalid raw XYZ coordinates at sample 3/], ['range', /invalid throttle at sample 3 \(1.01\)/],
    ['positions', /invalid normalized positions/], ['rows', /matching positions and telemetry rows/],
    ['stationary', /no usable travel distance/],
])(
    'rejects unusable reference data and explains the failing check: %s', (fault, message) => {
        const lap = referenceLapFixture();
        if (fault === 'coordinates') lap.target_features[1] = 'unrelated';
        if (fault === 'throttle') lap.target_features[3] = 'unrelated';
        if (fault === 'brake') lap.target_features[4] = 'unrelated';
        if (fault === 'nonfinite') lap.y[2][0] = NaN;
        if (fault === 'range') lap.y[2][3] = 1.01;
        if (fault === 'positions') lap.x[2] = lap.x[1];
        if (fault === 'rows') lap.y.pop();
        if (fault === 'stationary') lap.y.forEach((row) => { row[0] = row[1] = row[2] = 0; });
        expect(prepareDrivingReference(lap)).toBeNull();
        expect(inspectDrivingReference(lap).error).toMatch(message);
        expect(prepareDrivingReference(null)).toBeNull();
    },
);
