import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { readReferenceGuidance } from './phrase-reference-guidance';

// A 1000 m square with 10 m intervals and an independently specified braking episode.
function easingReference(tail = [2, 2, 2], start = 25, duplicate = false): TopLapReferenceEntry {
    const lap: TopLapReferenceEntry = {
        game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        x: Array.from({ length: 100 }, (_, i) => i / 100),
        y: Array.from({ length: 100 }, (_, i) => {
            const distance = i * 10;
            const xy = distance < 250 ? [distance, 0] : distance < 500 ? [250, distance - 250]
                : distance < 750 ? [750 - distance, 250] : [0, 1000 - distance];
            return [...xy, 0, 0.8, 0, 40 * 3.6];
        }),
    };
    let squaredSpeed = 40 ** 2;
    [4, ...tail].forEach((deceleration, offset) => {
        squaredSpeed -= 20 * deceleration;
        lap.y[(start + offset + 1) % 100][5] = Math.sqrt(squaredSpeed) * 3.6;
    });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    return lap;
}

function guidance(lap: TopLapReferenceEntry, position = 0.265, speed = 36, evaluationPosition = position, brake = 0.8) {
    const sample = { Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: position,
        Physics_speed_kmh: speed, Physics_brake: brake, Physics_gas: 0 };
    return readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, undefined, undefined, undefined, false, evaluationPosition);
}

it('integrates the remaining easing and lower steady braking using reference travel time', () => {
    const lap = easingReference([3, 2, 2]);
    const expectedMs = ((Math.sqrt(1490) - Math.sqrt(1460)) / 3 + (Math.sqrt(1460) - Math.sqrt(1380)) / 2) * 1000;
    for (const liveSpeed of [36, 180]) {
        expect(guidance(lap, 0.265, liveSpeed).brakeReleaseDurationMs).toBeCloseTo(expectedMs, 8);
    }
});

it('distinguishes different remaining braking lengths with the same easing rate', () => {
    const short = guidance(easingReference([2]));
    const long = guidance(easingReference([2, 2, 2]));
    expect(short.brakeReleaseRate).toBeCloseTo(long.brakeReleaseRate!, 8);
    expect(short.brakeReleaseDurationMs).toBeCloseTo((Math.sqrt(1500) - Math.sqrt(1480)) / 2 * 1000, 8);
    expect(long.brakeReleaseDurationMs).toBeCloseTo((Math.sqrt(1500) - Math.sqrt(1400)) / 2 * 1000, 8);
    expect(long.brakeReleaseDurationMs).toBeGreaterThan(short.brakeReleaseDurationMs!);
});

it('uses the evaluation position and shortens the remaining time as it advances', () => {
    const lap = easingReference();
    expect(guidance(lap, 0.245, 36, 0.265).brakeReleaseDurationMs).toBe(guidance(lap).brakeReleaseDurationMs);
    const before = guidance(lap, 0.261).brakeReleaseDurationMs!;
    const after = guidance(lap, 0.269).brakeReleaseDurationMs!;
    expect(before - after).toBeCloseTo((Math.sqrt(1516) - Math.sqrt(1484)) / 2 * 1000, 8);
});

it.each([false, true])('integrates across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = easingReference([2, 2, 2], 98, duplicate);
    expect(guidance(lap, 0.995).brakeReleaseDurationMs)
        .toBeCloseTo((Math.sqrt(1500) - Math.sqrt(1400)) / 2 * 1000, 8);
});

it('ends the easing episode when stronger braking resumes', () => {
    const lap = easingReference([2, 2, 3, 1]);
    expect(guidance(lap).brakeReleaseDurationMs)
        .toBeCloseTo((Math.sqrt(1500) - Math.sqrt(1440)) / 2 * 1000, 8);
});

it('ends the easing episode at constant speed', () => {
    const lap = easingReference([2, 0, 1]);
    expect(guidance(lap).brakeReleaseDurationMs)
        .toBeCloseTo((Math.sqrt(1500) - Math.sqrt(1480)) / 2 * 1000, 8);
});

it.each(['missing speed', 'invalid speed', 'released brake', 'constant deceleration', 'increasing deceleration'])(
    'withholds easing duration with %s', (kind) => {
        const lap = easingReference(kind === 'constant deceleration' ? [4, 4] : kind === 'increasing deceleration' ? [5, 5] : undefined);
        if (kind === 'missing speed') lap.target_features[5] = 'missing';
        if (kind === 'invalid speed') lap.y[28][5] = NaN;
        expect(guidance(lap, 0.265, 36, 0.265, kind === 'released brake' ? 0 : 0.8).brakeReleaseDurationMs).toBeUndefined();
    });
