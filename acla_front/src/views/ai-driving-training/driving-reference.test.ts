import { inspectDrivingReference, prepareDrivingReference, referenceSpeedTargets } from './driving-reference';
import { referenceLapFixture } from './training-fixtures';

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
    expect(Object.isFrozen(reference.pedalEvents)).toBe(true);
    lap.y[0][3] = 999;
    lap.y[0][4] = 0;
    expect(reference.points[0]).toMatchObject({ speed: 120, throttle: 1, brake: 0 });
});

it('anchors the speed profile to reference brake application and throttle pickup', () => {
    const reference = prepareDrivingReference(referenceLapFixture())!;
    expect(reference.pedalEvents).toEqual([
        { distance: 80, kind: 'deceleration', shiftLimit: 30 },
        { distance: 200, kind: 'acceleration', shiftLimit: 30 },
    ]);
    expect(reference.pedalEvents.every(Object.isFrozen)).toBe(true);
});

it.each([
    [0.175, 112.5, 120, 120], [0.25, 90, 100, 120], [0.275, 82.5, 90, 105],
    [0.4, 80, 80, 80], [0.525, 80, 90, 106.25], [0.575, 275 / 3, 110, 120], [0.625, 115, 120, 120],
])('shifts pedal-anchored slowing and speeding up according to aggression at %s',
    (position, cautious, neutral, aggressive) => {
        const reference = prepareDrivingReference(referenceLapFixture())!;
        [0, 50, 100].forEach((aggression, i) => {
            const result = referenceSpeedTargets(reference, position, aggression);
            expect(result.target).toBeCloseTo([cautious, neutral, aggressive][i]);
            expect(result.radius).toBe(aggression === 50 ? 0 : 20);
        });
    });

it('scales timing shifts with aggression and uses the configured maximum', () => {
    const reference = prepareDrivingReference(referenceLapFixture())!;
    const cautious = referenceSpeedTargets(reference, 0.24, 25);
    expect(cautious.radius).toBe(10);
    expect(cautious.reference).toBeCloseTo(104);
    expect(cautious.target).toBeCloseTo(120 - 26 * 6 / 7);
    expect(referenceSpeedTargets(reference, 0.24, 75).target).toBeCloseTo(112.8);
    const neutral = referenceSpeedTargets(reference, 0.24, 50);
    expect(neutral.target).toEqual(neutral.reference);
    expect(referenceSpeedTargets(reference, 0.24, 0, 10)).toEqual(cautious);
});

it('keeps cautious targets lower and aggressive targets higher throughout uneven pedal sections', () => {
    const lap = referenceLapFixture();
    // A speed recovery during braking and a dip during throttle pickup must not
    // turn cautious settings into a demand for more speed (or aggressive into less).
    lap.y[3][5] = 90;
    lap.y[4][5] = 110;
    lap.y[6][5] = 100;
    lap.y[7][5] = 85;
    const reference = prepareDrivingReference(lap)!;
    for (let i = 0; i <= 200; i++) {
        const position = i / 200;
        const neutral = referenceSpeedTargets(reference, position, 50).target;
        for (const aggression of [0, 25]) {
            expect(referenceSpeedTargets(reference, position, aggression).target).toBeLessThanOrEqual(neutral);
        }
        for (const aggression of [75, 100]) {
            expect(referenceSpeedTargets(reference, position, aggression).target).toBeGreaterThanOrEqual(neutral);
        }
    }
});

it('changes timing when reference pedals move even if the speed samples stay identical', () => {
    const lap = referenceLapFixture();
    // Move brake/lift onset from 0.2 to 0.1 without altering any recorded speeds.
    lap.y[2][3] = 0.75;
    lap.y[2][4] = 0.25;
    const original = prepareDrivingReference(referenceLapFixture())!;
    const moved = prepareDrivingReference(lap)!;
    expect(moved.pedalEvents[0].distance).toBe(40);
    expect(referenceSpeedTargets(moved, 0.25, 50)).toEqual(referenceSpeedTargets(original, 0.25, 50));
    expect(referenceSpeedTargets(moved, 0.25, 100).target).not.toBeCloseTo(referenceSpeedTargets(original, 0.25, 100).target);
});

it('uses reference pedal changes rather than speed extrema or brake release to define events', () => {
    const lap = referenceLapFixture();
    lap.y.forEach((row, i) => {
        const p = lap.x[i];
        row[3] = p <= 0.2 ? 0.8 : p <= 0.5 ? 0 : 0.8;
        row[4] = p <= 0.25 ? 0 : p <= 0.3 ? 1 : p <= 0.5 ? 0.4 : 0;
        row[5] = 100; // Pedal timing remains defined even before speed responds.
    });
    const reference = prepareDrivingReference(lap)!;
    expect(reference.pedalEvents.map(({ distance, kind }) => ({ distance, kind }))).toEqual([
        { distance: 80, kind: 'deceleration' }, { distance: 200, kind: 'acceleration' },
    ]);
    expect(referenceSpeedTargets(reference, 0.25, 0).target).toBe(100);
});

it('uses coordinate distance, not normalized position, to shift pedal events', () => {
    const lap = referenceLapFixture();
    const remapped = { ...lap, x: lap.x.map((p) => p * p) };
    const original = prepareDrivingReference(lap)!;
    const reference = prepareDrivingReference(remapped)!;
    expect(reference.pedalEvents).toEqual(original.pedalEvents);
    // These positions are halfway between the same two recorded samples.
    expect(referenceSpeedTargets(reference, (0.2 ** 2 + 0.25 ** 2) / 2, 0))
        .toEqual(referenceSpeedTargets(original, 0.225, 0));
});

it('keeps the original speed profile when pedals never change, including across start/finish', () => {
    const reference = prepareDrivingReference(speedLap([0.1, 0.5, 0.9],
        [[0, 0, 0, 120], [100, 0, 0, 100], [100, 100, 0, 80]]))!;
    expect(reference.pedalEvents).toEqual([]);
    for (const aggression of [0, 50, 100]) {
        expect(referenceSpeedTargets(reference, 0, aggression).target).toBeCloseTo(100);
        expect(referenceSpeedTargets(reference, 1, aggression).target).toBeCloseTo(100);
        expect(referenceSpeedTargets(reference, 0.95, aggression).target).toBeCloseTo(90);
    }
});

it('wraps pedal-event timing across start/finish', () => {
    const lap = referenceLapFixture();
    // Rotate the recorded lap so braking starts just after start/finish.
    const points = lap.x.slice(0, -1).map((p, i) => ({ p: (p + 0.85) % 1, row: lap.y[i].slice() }))
        .sort((a, b) => a.p - b.p);
    const reference = prepareDrivingReference({ ...lap, x: points.map(({ p }) => p), y: points.map(({ row }) => row) })!;
    const original = prepareDrivingReference(lap)!;
    for (const aggression of [0, 50, 100]) {
        expect(referenceSpeedTargets(reference, 0, aggression).target)
            .toBeCloseTo(referenceSpeedTargets(original, 0.15, aggression).target);
        expect(referenceSpeedTargets(reference, 1, aggression)).toEqual(referenceSpeedTargets(reference, 0, aggression));
        expect(referenceSpeedTargets(reference, 0.025, aggression).target)
            .toBeCloseTo(referenceSpeedTargets(original, 0.175, aggression).target);
    }
});

it('limits shifts between close events and handles coincident coordinates without reversing event order', () => {
    const lap = referenceLapFixture();
    lap.y.forEach((row) => { row[0] /= 100; row[1] /= 100; });
    lap.y[1][0] = lap.y[0][0];
    const reference = prepareDrivingReference(lap)!;
    expect(reference.pedalEvents.every((event) => event.shiftLimit < 1)).toBe(true);
    for (const aggression of [0, 50, 100]) {
        for (let i = 0; i <= 100; i++) {
            const { target } = referenceSpeedTargets(reference, i / 100, aggression);
            expect(target).toBeGreaterThanOrEqual(80);
            expect(target).toBeLessThanOrEqual(120);
        }
    }
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
