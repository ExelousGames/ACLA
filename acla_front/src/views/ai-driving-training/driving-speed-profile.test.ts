import { prepareDrivingReference, referenceSpeedTargets } from './driving-reference';
import { brakingFixtureSpeed, brakingReferenceFixture } from './training-fixtures';

it.each((['acc', 'iracing'] as const).flatMap((game) => [-1, 1].map((sign) => ({ game, sign }))))(
    'uses the correct ground plane for $game turns of sign $sign', ({ game, sign }) => {
        const { lap, map } = brakingReferenceFixture();
        lap.game = map.game = game;
        lap.y.forEach((row) => {
            row[1] *= sign;
            if (game === 'iracing') [row[1], row[2]] = [row[2], row[1]];
        });
        const reference = prepareDrivingReference(lap, map)!;
        for (const aggression of [0, 25, 50, 75, 100]) lap.x.forEach((position) => {
            expect(referenceSpeedTargets(reference, position, aggression).target).toBeCloseTo(brakingFixtureSpeed(position, aggression), 8);
        });
    });

it.each([false, true])('keeps braking targets continuous across start/finish (duplicate endpoint: %s)', (duplicate) => {
    const { lap, map } = brakingReferenceFixture();
    lap.y = lap.y.map((_, i, rows) => [...rows[(i + 28) % rows.length]]);
    map.centerline_tags = [{ id: 'turn', label: 'corner', start_position: 0.96, end_position: 0.22 }];
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const reference = prepareDrivingReference(lap, map)!;
    for (const aggression of [0, 25, 50, 100]) lap.x.forEach((position, i) => {
        expect(referenceSpeedTargets(reference, position, aggression).target)
            .toBeCloseTo(brakingFixtureSpeed(((i + 28) % 100) / 100, aggression), 8);
    });
});

it('honors explicit corner segments and excludes group tags and removed legacy corners', () => {
    const { lap, map } = brakingReferenceFixture();
    for (const segments of [[], [{ id: 'group', tags: ['corner', 'consecutive corners'], start_position: 0.24, end_position: 0.5 }]]) {
        const reference = prepareDrivingReference(lap, { ...map, centerline_segments: segments })!;
        expect(referenceSpeedTargets(reference, 0.2, 0).target).toBe(144);
    }
    const reference = prepareDrivingReference(lap, { ...map,
        centerline_segments: [{ id: 'corner', tags: [' SLOW CORNER '], start_position: 0.24, end_position: 0.5 }] })!;
    expect(referenceSpeedTargets(reference, 0.2, 0).target).toBeCloseTo(brakingFixtureSpeed(0.2, 0), 8);
    // Braking already reaches its lowest rate before a later turn-in; no extension is needed.
    const lateTurn = prepareDrivingReference(lap, { ...map,
        centerline_segments: [{ id: 'late', tags: ['corner'], start_position: 0.26, end_position: 0.5 }] })!;
    expect(referenceSpeedTargets(lateTurn, 0.2, 0).target).toBe(144);
});

it('limits braking extension at the previous minimum', () => {
    const { lap, map } = brakingReferenceFixture();
    lap.x[19] = 0.199;
    lap.y[19][0] = 199;
    lap.y[19][5] = 143.99;
    const reference = prepareDrivingReference(lap, map)!;
    expect(referenceSpeedTargets(reference, 0.199, 0).target).toBeCloseTo(143.99, 8);
    expect(referenceSpeedTargets(reference, 0.2, 0).target).toBeLessThan(144);
    expect(referenceSpeedTargets(reference, 0.3, 0).target).toBeCloseTo(108, 8);
    lap.x.forEach((position, i) => expect(referenceSpeedTargets(reference, position, 0).target).toBeLessThanOrEqual(lap.y[i][5] + 1e-8));
});

it.each([0, -0.5, 0.25])('retains signed braking work through a short interruption at %s m/s²', (gapRate) => {
    const { lap, map } = brakingReferenceFixture();
    const rates = [6, 6, 4, 2, gapRate, 3, 2, 1, 1, 1];
    let energy = 1600, x = 0, y = 0;
    lap.y.forEach((row, i) => {
        // A continuous arc keeps both braking inputs inside the same turn.
        row[0] = x; row[1] = y;
        const heading = (Math.min(20, Math.max(0, i - 24)) + (i === 24 ? 0.5 : 0)) * Math.PI / 60;
        x += 10 * Math.cos(heading); y += 10 * Math.sin(heading);
        if (i > 20 && i <= 30) energy -= 20 * rates[i - 21];
        row[5] = i <= 20 || i >= 70 ? 144 : i <= 60 ? Math.sqrt(energy) * 3.6
            : Math.sqrt(energy) * 3.6 + (144 - Math.sqrt(energy) * 3.6) * (i - 60) / 10;
    });
    const minimum = 1600 - 20 * (26 + gapRate), reference = prepareDrivingReference(lap, map)!;
    expect(referenceSpeedTargets(reference, 0.2, 0).target).toBeLessThan(144);
    for (let i = 24; i <= 30; i++) {
        expect(referenceSpeedTargets(reference, i / 100, 0).target).toBeCloseTo(Math.sqrt(minimum + 2 * (300 - i * 10)) * 3.6, 8);
    }
});

it('uses physical distance when normalized lap positions are nonuniform', () => {
    const { lap, map } = brakingReferenceFixture();
    const positions = [...lap.x];
    lap.x = lap.x.map((p) => p ** 2);
    map.centerline_tags!.forEach((tag) => { tag.start_position **= 2; tag.end_position **= 2; });
    const reference = prepareDrivingReference(lap, map)!;
    lap.x.forEach((position, i) => expect(referenceSpeedTargets(reference, position, 0).target).toBeCloseTo(brakingFixtureSpeed(positions[i], 0), 8));
});
