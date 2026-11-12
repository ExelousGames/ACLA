import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { readReferenceGuidance, referenceBrakingDeceleration } from './phrase-reference-guidance';

// A 1000 m square: brake from 200 to 300 m, then accelerate from the 10 m/s minimum.
function reference(deceleration = 4): TopLapReferenceEntry {
    const entryEnergy = 100 + 200 * deceleration;
    return {
        game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        x: Array.from({ length: 100 }, (_, i) => i / 100),
        y: Array.from({ length: 100 }, (_, i) => {
            const d = i * 10;
            const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
            const energy = d < 200 ? entryEnergy : d <= 300 ? entryEnergy - 2 * deceleration * (d - 200)
                : d < 400 ? 100 + (entryEnergy - 100) * (d - 300) / 100 : entryEnergy;
            return [...xy, 0, i > 20 && i < 30 ? 0.8 : 0, i > 30 && i < 90 ? 0.8 : 0, Math.sqrt(energy) * 3.6];
        }),
    };
}

function guidance(lap: TopLapReferenceEntry, position: number, speed = 40, gas = 0) {
    return readReferenceGuidance({ Static_track: 'test', Static_car_model: 'car',
        Graphics_normalized_car_position: position, Physics_speed_kmh: speed * 3.6,
        Physics_brake: 0, Physics_gas: gas }, null, lap, 'acc', 1000);
}

it('calculates braking duration and time until onset from available deceleration', () => {
    const lap = reference(), before = guidance(lap, 0.1);
    expect(referenceBrakingDeceleration(lap)).toBeCloseTo(4, 8);
    // Slowing from 40 to 10 m/s takes 7.5 s and 187.5 m; braking starts at 112.5 m.
    expect(before.brakeDurationMs).toBeCloseTo(7500, 8);
    expect(before.brakeInMs).toBeCloseTo(312.5, 8);
    expect(before).toMatchObject({ brakeRequired: false, pedalZone: 'none', currentPedalZone: 'none' });
    expect(guidance(lap, 0.11251)).toMatchObject({ brakeInMs: 0, brakeRequired: true, brake: 2500,
        pedalZone: 'none', currentPedalZone: 'none' });
});

it.each([[20, 8125, 2500], [30, 10000 / 3, 5000], [40, 312.5, 7500]])(
    'uses an arriving speed of %s m/s for the required braking time', (speed, brakeInMs, duration) => {
        const result = guidance(reference(), 0.1, speed);
        expect(result.brakeInMs).toBeCloseTo(brakeInMs, 8);
        expect(result.brakeDurationMs).toBeCloseTo(duration, 8);
    });

it('needs less braking distance when the reference demonstrates stronger deceleration', () => {
    const stronger = reference(8);
    expect(referenceBrakingDeceleration(stronger)).toBeCloseTo(8, 8);
    expect(guidance(stronger, 0.1).brakeDurationMs).toBeCloseTo(3750, 8);
    expect(guidance(stronger, 0.1).brakeInMs).toBeCloseTo(2656.25, 8);
    expect(guidance(stronger, 0.12).brakeRequired).toBe(false);
    expect(guidance(reference(), 0.12).brakeRequired).toBe(true);
});

it.each([false, true])('calculates across start/finish with duplicate endpoint %s', (duplicate) => {
    const lap = reference();
    lap.y = [...lap.y.slice(25), ...lap.y.slice(0, 25)];
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    // The same braking episode now runs from 950 m through 50 m of the next lap.
    expect(guidance(lap, 0.85).brakeInMs).toBeCloseTo(312.5, 8);
    expect(guidance(lap, 0.85).brakeDurationMs).toBeCloseTo(7500, 8);
    expect(guidance(lap, 0.87)).toMatchObject({ brakeRequired: true, brakeInMs: 0 });
    expect(guidance(lap, 0.02, 20)).toMatchObject({ brakeRequired: true, brakeInMs: 0 });
    expect(guidance(lap, 0.06).brakeRequired).toBe(false);
});

it('uses physical distance when normalized reference positions are nonuniform', () => {
    const lap = reference();
    lap.x = lap.x.map((position) => position ** 2);
    expect(guidance(lap, 0.1 ** 2).brakeInMs).toBeCloseTo(312.5, 8);
    expect(guidance(lap, 0.12 ** 2).brakeRequired).toBe(true);
});

it.each(['missing', 'invalid', 'flat'])('does not invent braking capability with %s reference speed', (kind) => {
    const lap = reference();
    if (kind === 'missing') lap.target_features[5] = 'missing speed';
    else if (kind === 'invalid') lap.y[25][5] = NaN;
    else lap.y.forEach((row) => { row[5] = 108; });
    expect(referenceBrakingDeceleration(lap)).toBeUndefined();
    expect(guidance(lap, 0.12).brakeInMs).toBeUndefined();
    expect(guidance(lap, 0.12).brakeRequired).toBe(false);
    const inside = guidance(lap, 0.25, 30);
    expect(inside.brakeRequired).toBe(true);
    expect(inside.brakeDurationMs).toBeUndefined();
    expect(inside.brake).toBe(kind === 'flat' ? undefined : 2500);
});

it('does not reuse a passed minimum as an immediate braking target on the next lap', () => {
    const result = guidance(reference(), 0.31);
    expect(result.brakeRequired).toBe(false);
    expect(result.brakeInMs).toBeCloseTo(20062.5, 8);
});

it('ignores passed speed targets while a brake pedal zone continues after the minimum', () => {
    const lap = reference();
    lap.y.forEach((row, i) => { if (i >= 30 && i < 35) { row[3] = 0.8; row[4] = 0; } });
    const result = guidance(lap, 0.31);
    expect(result.pedalZone).toBe('brake');
    expect(result.brakeInMs).toBeUndefined();
    expect(result.brakeDurationMs).toBeUndefined();
});

it('suppresses conflicting throttle cues when braking is due before the nominal brake zone', () => {
    const lap = reference();
    lap.y.forEach((row, i) => { if (i > 5 && i < 20) row[4] = 0.8; });
    const result = guidance(lap, 0.12, 40, 0.8);
    expect(result).toMatchObject({ brakeRequired: true, pedalZone: 'throttle', brake: 2500 });
    expect(result.throttle).toBeUndefined();
    expect(result.moreThrottle).toBeUndefined();
    expect(result.lessThrottle).toBeUndefined();
});
