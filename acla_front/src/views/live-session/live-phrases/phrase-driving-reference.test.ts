import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression } from './phrase-driving-reference';
import { readReferenceGuidance, referenceDirectionChanges } from './phrase-reference-guidance';
import { circuitMap, variableBrakingReference } from './test-fixtures';

// A 1000 raw-coordinate-unit loop, sampled every 10 units, with its first bend at 240.
const reference = (): TopLapReferenceEntry => ({
    game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
    target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
        'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_steering', 'expert_optimal_speed'],
    x: Array.from({ length: 100 }, (_, i) => i / 100),
    y: Array.from({ length: 100 }, (_, i) => {
        const d = i * 10;
        const xy = d < 240 ? [d, 0] : d < 500 ? [240, d - 240] : d < 740 ? [740 - d, 260] : [0, 1000 - d];
        const speed = i < 20 || i >= 70 ? 120 : i < 30 ? 120 - (i - 20) * 4 : i <= 60 ? 80 : 80 + (i - 60) * 4;
        return [...xy, 0, i >= 21 && i <= 29 ? 0.8 : 0, i >= 61 && i <= 90 ? 1 : 0, 0.2, speed];
    }),
});
const rowAt = (lap: TopLapReferenceEntry, position: number) => lap.y[lap.x.findIndex((p) => Math.abs(p - position) < 1e-9)];
const speedAt = (lap: TopLapReferenceEntry, position: number) => rowAt(lap, position)[6];
const rates = (lap: TopLapReferenceEntry) => lap.y.flatMap((row, i) => {
    const next = lap.y[(i + 1) % lap.y.length];
    const distance = Math.hypot(...row.slice(0, 3).map((v, j) => next[j] - v));
    return distance < 1e-7 ? [] : [{ start: lap.x[i], distance, rate: (row[6] ** 2 - next[6] ** 2) / (2 * 3.6 ** 2 * distance) }];
});
const twoRates = () => {
    const lap = reference();
    // 40 m/s entry: 50 m at 6 m/s², then 50 m at 1 m/s², ending at 30 m/s.
    lap.y.forEach((row, i) => {
        row[6] = i < 20 || i >= 70 ? 144 : i <= 25 ? Math.sqrt(1600 - 120 * (i - 20)) * 3.6
            : i < 30 ? Math.sqrt(1000 - 20 * (i - 25)) * 3.6 : i <= 60 ? 108 : 108 + (i - 60) * 3.6;
    });
    return lap;
};
const turnTrajectory = (lap: TopLapReferenceEntry, entry: number, sign = 1) => {
    let x = 0, y = 0;
    lap.y.forEach((row, i) => {
        row[0] = x;
        row[lap.game === 'acc' ? 1 : 2] = y;
        row[lap.game === 'acc' ? 2 : 1] = 0;
        const heading = sign * (Math.min(20, Math.max(0, i - entry)) + (i === entry ? 0.5 : 0)) * Math.PI / 60;
        x += 10 * Math.cos(heading);
        y += 10 * Math.sin(heading);
    });
};
const interruptedBraking = (gapRate: number) => {
    const lap = reference();
    turnTrajectory(lap, 24);
    const deceleration = [6, 6, 4, 2, gapRate, 3, 2, 1, 1, 1];
    let energy = 1600;
    lap.y.forEach((row, i) => {
        if (i > 20 && i <= 30) energy -= 20 * deceleration[i - 21];
        row[6] = i <= 20 || i >= 70 ? 144 : i <= 60 ? Math.sqrt(energy) * 3.6
            : Math.sqrt(energy) * 3.6 + (144 - Math.sqrt(energy) * 3.6) * (i - 60) / 10;
        row[3] = i >= 20 && i < 30 ? Math.max(0, deceleration[i - 20]) / 6 : 0;
    });
    return lap;
};

it.each([0, -0.5, 0.25])('adjusts both braking inputs in one corner across a short gap at %s m/s²', (gapRate) => {
    const lap = interruptedBraking(gapRate), baseline = JSON.parse(JSON.stringify(lap));
    let previous = lap;
    for (const caution of [25, 50, 100]) {
        const copy = copyPhraseReference(lap, caution)!;
        // Both inputs contribute to the same fixed endpoint, including the brief acceleration.
        const minimum = 1600 - 20 * (26 + gapRate);
        const transition = 270 - caution / 100 * 30;
        expect(speedAt(copy, 0.2)).toBeLessThan(speedAt(previous, 0.2));
        if (caution === 100) expect(rowAt(copy, 0.19)[3]).toBeGreaterThan(0);
        expect(speedAt(copy, 0.3)).toBeCloseTo(Math.sqrt(minimum) * 3.6, 8);
        rates(copy).filter(({ start }) => start >= transition / 1000 - 1e-9 && start < 0.3)
            .forEach(({ rate }) => expect(rate).toBeCloseTo(1, 7));
        expect(Math.max(...rates(copy).map(({ rate }) => rate))).toBeLessThanOrEqual(6 + 1e-7);
        lap.y.forEach((row, i) => {
            expect(copy.y[i][6]).toBeLessThanOrEqual(row[6] + 1e-7);
            expect(copy.y[i].filter((_, j) => j !== 3 && j !== 6)).toEqual(row.filter((_, j) => j !== 3 && j !== 6));
            if (i >= 30) {
                expect(copy.y[i][6]).toBeCloseTo(row[6], 8);
                expect(copy.y[i][3]).toBe(row[3]);
            }
        });
        previous = copy;
    }
    expect(lap).toEqual(baseline);
    expect(copyPhraseReference(lap, 0)).toEqual(lap);
});

it.each([false, true])('joins interrupted corner braking across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const baseline = interruptedBraking(-0.5), adjusted = copyPhraseReference(baseline, 100)!;
    const lap = { ...baseline, x: [...baseline.x], y: baseline.y.map((_, i, rows) => [...rows[(i + 25) % rows.length]]) };
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const copy = copyPhraseReference(lap, 100)!;
    for (let i = 0; i < 100; i++) {
        expect(copy.y[i][6]).toBeCloseTo(adjusted.y[(i + 25) % 100][6], 8);
        expect(copy.y[i][3]).toBeCloseTo(adjusted.y[(i + 25) % 100][3], 8);
    }
    if (duplicate) expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
});

it.each([0, -0.5])('keeps a longer interruption at %s m/s² separate within the same turn', (gapRate) => {
    const lap = interruptedBraking(gapRate);
    // Lower speeds make the same 10 m gap last more than half a second.
    lap.y.forEach((row) => { row[6] /= 2; });
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.2)).toBeCloseTo(speedAt(lap, 0.2), 8);
    expect(speedAt(copy, 0.24)).toBeCloseTo(speedAt(lap, 0.24), 8);
    expect(speedAt(copy, 0.3)).toBeCloseTo(speedAt(lap, 0.3), 8);
});

it('keeps short-gap braking for distinct direction-change ranges separate', () => {
    const lap = interruptedBraking(-0.5);
    let x = 0, y = 0;
    lap.y.forEach((row, i) => {
        row[0] = x;
        row[1] = y;
        const heading = (i <= 24 ? Math.max(0, i - 20) : 4 - Math.min(20, i - 24)) * Math.PI / 60;
        x += 10 * Math.cos(heading);
        y += 10 * Math.sin(heading);
    });
    const turns = referenceDirectionChanges(lap).filter(({ start }) => start >= 190 && start < 300);
    expect(turns).toHaveLength(2);
    expect(turns[0].end).toBeLessThanOrEqual(250);
    expect(turns[1].start).toBeGreaterThanOrEqual(240);
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeCloseTo(speedAt(lap, 0.24), 8);
    expect(speedAt(copy, 0.3)).toBeCloseTo(speedAt(lap, 0.3), 8);
});

it('preserves the next brake application after a long coast', () => {
    const lap = reference();
    turnTrajectory(lap, 21);
    const minimum = Math.sqrt((104 / 3.6) ** 2 - 60) * 3.6;
    lap.y.forEach((row, i) => {
        row[6] = i < 20 || i >= 70 ? 120 : i < 24 ? 120 - (i - 20) * 4 : i < 27 ? 104
            : i < 30 ? Math.sqrt((104 / 3.6) ** 2 - 20 * (i - 27)) * 3.6 : i <= 60 ? minimum
                : minimum + (120 - minimum) * (i - 60) / 10;
        row[3] = i >= 20 && i < 24 ? 0.8 : i >= 27 && i < 30 ? 0.4 : 0;
    });
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.2)).toBeLessThan(speedAt(lap, 0.2));
    expect(speedAt(copy, 0.24)).toBeCloseTo(104, 8);
    for (const i of [27, 28, 29]) {
        expect(copy.y[i][3]).toBe(lap.y[i][3]);
        expect(copy.y[i][6]).toBeCloseTo(lap.y[i][6], 8);
    }
});

it('copies the complete reference and map without sharing editable arrays, samples or tags', () => {
    const lap = reference(), map = circuitMap();
    map.centerline_segments = [{ id: 'turn', start_position: 0.1, end_position: 0.2, tags: ['corner', 'slow'] }];
    const copied = copyPhraseReference(lap)!, copiedMap = copyPhraseMap(map)!;
    expect(copied).toEqual(lap);
    expect(copiedMap).toEqual(map);
    copied.x[0] = 0.001;
    copied.y[0][0] = 100;
    copied.target_features[0] = 'changed';
    copiedMap.samples.middle_line![0].x = 42;
    copiedMap.centerline_segments![0].tags.push('fast');
    copiedMap.centerline_tags![0].label = 'straight';
    expect(lap).toEqual(reference());
    expect(map.samples.middle_line![0].x).toBe(0);
    expect(map.centerline_segments![0].tags).toEqual(['corner', 'slow']);
    expect(map.centerline_tags![0].label).toBe('corner');
});

it.each(['acc', 'iracing'])('rebuilds the complete %s lap on its original samples across representative aggression settings', (game) => {
    const lap = variableBrakingReference();
    lap.game = game;
    if (game === 'iracing') lap.y.forEach((row) => { [row[1], row[2]] = [row[2], row[1]]; });
    const baseline = JSON.parse(JSON.stringify(lap));
    const unchangedChannels = lap.y.map((row) => [...row.slice(0, 3), ...row.slice(4, 6)]);
    for (const aggression of [0, 1, 25, 50, 75, 99, 100]) {
        const copy = copyPhraseReference(lap, aggression)!;
        expect(copy.x).toEqual(lap.x);
        expect(copy.y).toHaveLength(lap.y.length);
        expect(copy.target_features).toEqual(lap.target_features);
        expect(copy.y.map((row) => [...row.slice(0, 3), ...row.slice(4, 6)])).toEqual(unchangedChannels);
        expect(readReferenceGuidance({ Static_track: 'test', Static_car_model: 'car',
            Graphics_normalized_car_position: 0.05, Physics_speed_kmh: 220, Physics_brake: 0, Physics_gas: 0,
        }, null, copy, game, 1000).ready).toBe(true);
        if (aggression > 0) expect(copy.y.some((row, i) => row[6] < lap.y[i][6] - 1e-8)).toBe(true);
        lap.y.forEach((row, i) => {
            if (i > 0 && i + 1 < lap.y.length && row[6] < lap.y[i - 1][6] && row[6] <= lap.y[i + 1][6]) {
                expect(copy.y[i][6]).toBeCloseTo(row[6], 8);
                expect(copy.y[i - 1][6]).toBeGreaterThan(row[6]);
            }
        });
    }
    expect(lap).toEqual(baseline);
});

it.each([10, 25, 50, 75, 100])('keeps the minimum at its original position at %s%% while preserving braking work', (caution) => {
    const lap = reference(), copy = copyPhraseReference(lap, caution)!;
    const end = 0.3;
    expect(referenceDirectionChanges(lap)[0].start).toBe(240);
    expect(speedAt(copy, end)).toBeCloseTo(80, 8);
    expect(copy.x[copy.y.findIndex((row) => Math.abs(row[6] - 80) < 1e-8)]).toBe(end);
    expect(Math.min(...copy.y.map((row) => row[6]))).toBeCloseTo(80, 8);
    const originalRates = rates(lap), adjustedRates = rates(copy);
    // Intervals crossing adjusted boundaries average the rates without exceeding the original peak.
    expect(Math.max(...adjustedRates.map(({ rate }) => rate))).toBeLessThanOrEqual(Math.max(...originalRates.map(({ rate }) => rate)) + 1e-7);
    expect(adjustedRates.find(({ rate }) => rate > 1e-7)!.start).toBeLessThan(0.2);
    const area = (values: ReturnType<typeof rates>) => values.reduce((sum, { rate, distance }) => sum + Math.max(0, rate) * distance, 0);
    expect(area(adjustedRates)).toBeCloseTo(area(originalRates), 7);
    adjustedRates.filter(({ start }) => start >= end && start < 0.6).forEach(({ rate }) => expect(rate).toBeCloseTo(0, 7));
    lap.y.forEach((row, i) => {
        [0, 1, 2, 4, 5].forEach((channel) => expect(rowAt(copy, lap.x[i])[channel]).toBe(row[channel]));
        expect(speedAt(copy, lap.x[i])).toBeLessThanOrEqual(row[6] + 1e-8);
        if (i >= 30) expect(speedAt(copy, lap.x[i])).toBeCloseTo(row[6], 8);
    });
    expect(lap).toEqual(reference());
    expect(copyPhraseReference(lap, 0)).toEqual(lap);
});

it('does not advance braking for a uniformly near-straight trajectory', () => {
    const lap = variableBrakingReference();
    // A 1600 m radius changes heading by only about 0.36 degrees per 10 m.
    lap.y.forEach((row) => { row[0] *= 4; row[1] *= 4; });
    expect(referenceDirectionChanges(lap)).toEqual([]);
    expect(copyPhraseReference(lap, 100)).toEqual(lap);
});

it.each([25, 50, 100])('integrates adjusted deceleration backward from the original minimum at %s%%', (caution) => {
    const lap = twoRates(), copy = copyPhraseReference(lap, caution)!;
    const fraction = caution / 100, releaseRate = 1;
    const start = 200 - 25 * fraction / 3, transition = 250 - 10 * fraction, end = 300;
    // Extend the 1 m/s² section toward turn-in at 240 m, compensating with earlier braking.
    copy.x.forEach((position, i) => {
        if (position > 0.6) return;
        const remainingWork = 6 * Math.max(0, transition - Math.max(start, position * 1000))
            + releaseRate * Math.max(0, end - Math.max(transition, position * 1000));
        expect(copy.y[i][6]).toBeCloseTo(Math.sqrt(900 + 2 * remainingWork) * 3.6, 8);
    });
    expect(Math.max(...rates(copy).map(({ rate }) => rate))).toBeCloseTo(6, 7);
    expect(rates(copy).find(({ rate }) => rate > 1e-7)!.start).toBeLessThan(0.2);
    expect(rowAt(copy, Math.floor(start / 10) / 100)[3]).toBe(0);
    copy.x.forEach((p, i) => {
        if (p >= end / 1000 && p <= 0.6) expect(copy.y[i][3]).toBe(0);
        const index = Math.floor(p * 100), f = p * 100 - index;
        const a = lap.y[index % 100], b = lap.y[(index + 1) % 100];
        [0, 1, 2, 4, 5].forEach((j) => expect(copy.y[i][j]).toBeCloseTo(a[j] + (b[j] - a[j]) * f, 8));
    });
});

it.each([25, 50, 100])('moves the low-deceleration boundary toward turn-in at %s%%', (caution) => {
    for (const entry of [15, 20, 24]) {
        const lap = twoRates();
        turnTrajectory(lap, entry);
        expect(referenceDirectionChanges(lap).some(({ start }) => Math.abs(start - entry * 10) < 1e-7)).toBe(true);
        const copy = copyPhraseReference(lap, caution)!;
        const transition = 250 + caution / 100 * (entry * 10 - 250);
        rates(copy).filter(({ start }) => start * 1000 >= transition && start < 0.3).forEach(({ rate }) => {
            expect(rate).toBeCloseTo(1, 7);
        });
        const before = Math.ceil(transition / 10) - 1;
        expect(rates(copy)[before].rate).toBeGreaterThan(1 + 1e-7);
        expect(speedAt(copy, 0.3)).toBeCloseTo(108, 8);
        expect(copy.y[29][6]).toBeGreaterThan(108);
    }
});

it.each([25, 26, 29])('preserves a release already at minimum deceleration before turn-in at sample %s', (entry) => {
    const lap = twoRates();
    turnTrajectory(lap, entry);
    expect(copyPhraseReference(lap, 100)).toEqual(lap);
});

it.each([false, true])('keeps the minimum fixed across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = reference(), original = lap.y.map((row) => [...row]);
    // Rotate geometry and controls together: braking ends at 20 m, turn-in begins at 960 m.
    lap.y = original.map((_, i) => [...original[(i + 28) % 100]]);
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.92)).toBeLessThan(speedAt(lap, 0.92));
    expect(rowAt(copy, 0.92)[3]).toBeGreaterThan(0);
    expect(speedAt(copy, 0.95)).toBeGreaterThan(80);
    for (const p of [0.96, 0.99, 0, 0.01]) expect(speedAt(copy, p)).toBeGreaterThan(80);
    const floor = rates(reference()).find(({ start }) => start === 0.29)!.rate;
    rates(copy).filter(({ start }) => start >= 0.96 || start < 0.02)
        .forEach(({ rate }) => expect(rate).toBeCloseTo(floor, 7));
    for (const p of [0.02, 0.32]) {
        expect(speedAt(copy, p)).toBeCloseTo(80, 8);
        expect(rowAt(copy, p)[3]).toBe(0);
    }
    expect(speedAt(copy, 0.33)).toBeCloseTo(84);
    expect(Math.max(...rates(copy).map(({ rate }) => rate))).toBeLessThanOrEqual(Math.max(...rates(lap).map(({ rate }) => rate)) + 1e-7);
    if (duplicate) expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
});

it.each([false, true])('advances brake-zone entry into the previous lap (duplicated endpoint: %s)', (duplicate) => {
    const lap = reference(), original = lap.y;
    lap.y = original.map((_, i) => [...original[(i + 20) % 100]]);
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const copy = copyPhraseReference(lap, 100)!;
    const sample = { Static_track: 'test', Static_car_model: 'car',
        Graphics_normalized_car_position: 0.995, Physics_speed_kmh: 120, Physics_brake: 0, Physics_gas: 0 };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).pedalZone).toBe('none');
    expect(readReferenceGuidance(sample, null, copy, 'acc', 1000)).toMatchObject({ pedalZone: 'brake', brake: 2500 });
    expect(speedAt(copy, 0)).toBeLessThan(speedAt(lap, 0));
    expect(speedAt(copy, 0.09)).toBeGreaterThan(80);
    expect(speedAt(copy, 0.1)).toBeCloseTo(80, 8);
    if (duplicate) expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
});

it('limits the earlier extension at a preceding minimum and joins without exceeding peak deceleration', () => {
    const lap = twoRates();
    // A shallow preceding minimum only 1 m before braking leaves too little approach
    // for the full 8.33 m extension. Preserve it and reduce the requested compression.
    lap.x[19] = 0.199;
    lap.y[19][0] = 199;
    lap.y[19][6] = 143.99;
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.199)).toBeCloseTo(143.99, 8);
    expect(speedAt(copy, 0.2)).toBeLessThan(144);
    expect(speedAt(copy, 0.3)).toBeCloseTo(108, 8);
    expect(Math.max(...rates(copy).map(({ rate }) => rate))).toBeCloseTo(6, 7);
    lap.y.forEach((row, i) => expect(copy.y[i][6]).toBeLessThanOrEqual(row[6] + 1e-8));
});

it('uses physical distance with nonuniform lap positions and resampled braking data', () => {
    const lap = twoRates(), dense = twoRates();
    const row = lap.y[23].map((value, j) => (value + lap.y[24][j]) / 2);
    row[6] = Math.sqrt((lap.y[23][6] ** 2 + lap.y[24][6] ** 2) / 2);
    dense.x.splice(24, 0, 0.235);
    dense.y.splice(24, 0, row);
    dense.x = dense.x.map((position) => position ** 2);
    const original = copyPhraseReference(lap, 100)!, resampled = copyPhraseReference(dense, 100)!;
    lap.x.forEach((position) => {
        expect(speedAt(resampled, position ** 2)).toBeCloseTo(speedAt(original, position), 7);
        expect(rowAt(resampled, position ** 2)[3]).toBeCloseTo(rowAt(original, position)[3], 7);
    });
});

it.each([NaN, Infinity, -0.1, 1.1])('preserves an invalid brake channel (%s) while recalculating speed', (invalid) => {
    const lap = reference();
    lap.y[21][3] = invalid;
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeLessThan(speedAt(lap, 0.24));
    expect(speedAt(copy, 0.3)).toBeCloseTo(80);
    lap.x.forEach((position, i) => expect(rowAt(copy, position)[3]).toBe(lap.y[i][3]));
});

it.each(['missing', NaN, Infinity, -1])('leaves the reference unchanged when speed is %s', (invalid) => {
    const lap = reference();
    if (invalid === 'missing') lap.target_features[6] = 'missing speed';
    else lap.y[0][6] = invalid as number;
    expect(copyPhraseReference(lap, 100)).toEqual(lap);
});

it('adjusts speed independently of missing or invalid control channels', () => {
    const lap = reference();
    lap.target_features[3] = 'missing brake';
    lap.y[0][4] = NaN;
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeLessThan(speedAt(lap, 0.24));
    expect(speedAt(copy, 0.3)).toBeCloseTo(80);
    lap.y.forEach((row, i) => expect(rowAt(copy, lap.x[i]).slice(0, 6)).toEqual(row.slice(0, 6)));
    expect(copyPhraseReference(null, 0)).toBeNull();
    expect(copyPhraseMap(null)).toBeNull();
});

it.each([-10, 0, 10, 25, 50, 100, 150, NaN, Infinity])('clamps %s to 0–100%% with a reference fallback', (value) => {
    const normalized = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
    expect(normalizeDrivingAggression(value)).toBe(normalized);
    expect(copyPhraseReference(reference(), value)).toEqual(copyPhraseReference(reference(), normalized));
});

it('preserves constant deceleration and distinct corner minima at their original positions', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        row[6] = i >= 20 && i <= 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i > 30 && i < 40 ? 108 + (i - 30) * 3.6
                : i >= 40 && i <= 50 ? Math.sqrt(1600 - 120 * (i - 40)) * 3.6
                    : i > 50 && i < 60 ? 72 + (i - 50) * 7.2 : 144;
        row[3] = i >= 21 && i < 30 ? 0.4 : i >= 41 && i < 50 ? 0.8 : 0;
    });
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeCloseTo(speedAt(lap, 0.24));
    expect(speedAt(copy, 0.3)).toBeCloseTo(108);
    expect(speedAt(copy, 0.49)).toBeCloseTo(speedAt(lap, 0.49));
    expect(speedAt(copy, 0.5)).toBeCloseTo(72);
    const adjustedRates = rates(copy);
    expect(Math.max(...adjustedRates.filter(({ start }) => start < 0.3).map(({ rate }) => rate))).toBeCloseTo(3.5, 7);
    expect(Math.max(...adjustedRates.filter(({ start }) => start >= 0.3 && start < 0.5).map(({ rate }) => rate))).toBeCloseTo(6, 7);
});

it('preserves flat profiles, already completed braking, and finite zero-speed minima', () => {
    const flat = reference();
    flat.y.forEach((row) => { row[6] = 80; });
    expect(copyPhraseReference(flat, 100)).toEqual(flat);
    const early = reference();
    early.y.forEach((row, i) => { row[6] = i < 10 || i >= 70 ? 120 : i < 20 ? 120 - (i - 10) * 4 : i <= 60 ? 80 : 80 + (i - 60) * 4; });
    expect(copyPhraseReference(early, 100)).toEqual(early);
    const lap = reference();
    lap.y.forEach((row) => { row[6] -= 80; });
    const copy = copyPhraseReference(lap, 100)!;
    expect(copy.y.every((row) => Number.isFinite(row[6]) && row[6] >= 0)).toBe(true);
    expect(speedAt(copy, 0.29)).toBeGreaterThan(0);
    expect(speedAt(copy, 0.3)).toBe(0);
    expect(speedAt(copy, 0.61)).toBeCloseTo(4);
});

it('retains a coast inside the adjusted braking curve with its minimum fixed', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        // A coast within the release keeps its distance and zero deceleration.
        if (i >= 20 && i <= 30) row[6] = i < 22 ? 120 - (i - 20) * 10 : i <= 27 ? 100 : 100 - (i - 27) * 20 / 3;
    });
    const copy = copyPhraseReference(lap, 100)!;
    const intervals = rates(copy).filter(({ start }) => start > 0.2 && start < 0.29);
    expect(intervals.some(({ rate }) => Math.abs(rate) < 1e-7)).toBe(true);
    expect(speedAt(copy, 0.29)).toBeGreaterThan(80);
    expect(speedAt(copy, 0.3)).toBeCloseTo(80, 8);
});

it.each(['acc', 'iracing'])('keeps the minimum fixed when turn-in precedes reference braking in %s', (game) => {
    for (const sign of [-1, 1]) {
        const lap = twoRates();
        lap.game = game;
        turnTrajectory(lap, 15, sign);
        const range = referenceDirectionChanges(lap).find(({ start }) => Math.abs(start - 150) < 1e-7)!;
        expect(range).toBeDefined();
        const copy = copyPhraseReference(lap, 100)!;
        expect(speedAt(copy, 0.15)).toBeCloseTo(Math.sqrt(900 + 2 * 150) * 3.6, 8);
        expect(rowAt(copy, 0.15)[3]).toBeGreaterThan(0);
        const braking = rates(copy).filter(({ rate }) => rate > 1e-7);
        braking.filter(({ start }) => start >= 0.15 && start < 0.3)
            .forEach(({ rate }) => expect(rate).toBeCloseTo(1, 7));
        expect(Math.max(...braking.map(({ rate }) => rate))).toBeCloseTo(6, 7);
        expect(speedAt(copy, 0.29)).toBeGreaterThan(108);
        expect(speedAt(copy, 0.3)).toBeCloseTo(108, 8);
        braking.forEach(({ start }) => expect(start).toBeLessThan(0.3));
        expect(braking.reduce((sum, { rate, distance }) => sum + rate * distance, 0)).toBeCloseTo(350, 7);
    }
});

it.each([31, 35])('keeps consecutive corner minima fixed when turn-in starts at sample %s', (entry) => {
    const lap = reference();
    turnTrajectory(lap, entry);
    lap.y.forEach((row, i) => {
        row[6] = i >= 20 && i <= 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i > 30 && i < 40 ? 108 + (i - 30) * 3.6
                : i >= 40 && i <= 50 ? Math.sqrt(1600 - 120 * (i - 40)) * 3.6
                    : i > 50 && i < 60 ? 72 + (i - 50) * 7.2 : 144;
    });
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.3)).toBeCloseTo(108, 8);
    expect(speedAt(copy, 0.31)).toBeGreaterThan(72); // 10 m cannot dissipate the required energy at 6 m/s².
    expect(speedAt(copy, 0.4)).toBeCloseTo(144, 8);
    expect(speedAt(copy, 0.5)).toBeCloseTo(72, 8);
    expect(speedAt(copy, 0.35)).toBeCloseTo(speedAt(lap, 0.35), 8);
    expect(Math.max(...rates(copy).filter(({ start }) => start >= 0.3 && start < 0.5).map(({ rate }) => rate))).toBeCloseTo(6, 7);
    expect(copy.x.every((p, i) => Number.isFinite(p) && (!i || p > copy.x[i - 1]))).toBe(true);
});

it('recalculates a ramped braking profile while retaining its rise, peak and fixed minimum', () => {
    const lap = reference();
    const referenceRates = [1, 2, 4, 6, 6, 4, 3, 2, 1, 1];
    let energy = 1600;
    lap.y.forEach((row, i) => {
        if (i > 20 && i <= 30) energy -= 20 * referenceRates[i - 21];
        row[6] = i <= 20 || i >= 70 ? 144 : i <= 60 ? Math.sqrt(energy) * 3.6
            : Math.sqrt(energy) * 3.6 + (144 - Math.sqrt(energy) * 3.6) * (i - 60) / 10;
    });
    let previous = lap, previousReleaseSpan = 0.03;
    for (const caution of [25, 50, 75, 100]) {
        const copy = copyPhraseReference(lap, caution)!;
        const braking = rates(copy).filter(({ rate }) => rate > 1e-7);
        const transition = 0.28 - 0.04 * caution / 100;
        braking.filter(({ start }) => start >= transition - 1e-9 && start < 0.3)
            .forEach(({ rate }) => expect(rate).toBeCloseTo(1, 7));
        expect(braking.find(({ start }) => Math.abs(start - transition + 0.01) < 1e-9)!.rate).toBeGreaterThan(1);
        expect(braking[0].rate).toBeLessThanOrEqual(1 + 1e-7);
        expect(braking[0].start).toBeLessThan(0.2);
        expect(Math.max(...braking.map(({ rate }) => rate))).toBeCloseTo(6, 7);
        const peakIntervals = braking.filter(({ rate }) => Math.abs(rate - 6) < 1e-7);
        const releaseSpan = transition - (peakIntervals[peakIntervals.length - 1].start + 0.01);
        expect(releaseSpan).toBeLessThanOrEqual(previousReleaseSpan + 1e-9);
        if (caution === 100) expect(releaseSpan).toBeLessThan(0.03);
        previousReleaseSpan = releaseSpan;
        expect(speedAt(copy, 0.29)).toBeGreaterThan(lap.y[30][6]);
        expect(speedAt(copy, 0.3)).toBeCloseTo(lap.y[30][6], 7);
        lap.x.forEach((p) => expect(speedAt(copy, p)).toBeLessThanOrEqual(speedAt(previous, p) + 1e-7));
        previous = copy;
    }
});
