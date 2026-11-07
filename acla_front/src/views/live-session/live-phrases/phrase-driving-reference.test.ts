import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression } from './phrase-driving-reference';
import { referenceDirectionChanges } from './phrase-reference-guidance';
import { circuitMap } from './test-fixtures';

// A 1 km loop with known 10 m sample spacing and a direction-change range at 240 m.
const reference = (): TopLapReferenceEntry => ({
    game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
    target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
        'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_steering', 'expert_optimal_speed'],
    x: Array.from({ length: 100 }, (_, i) => i / 100),
    y: Array.from({ length: 100 }, (_, i) => {
        const d = i * 10;
        const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
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
        const heading = sign * Math.min(20, Math.max(0, i - entry)) * Math.PI / 60;
        x += 10 * Math.cos(heading);
        y += 10 * Math.sin(heading);
    });
};

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

it.each([10, 25, 50, 75, 100])('advances braking at %s%% without lowering the corner minimum or peak deceleration', (caution) => {
    const lap = reference(), copy = copyPhraseReference(lap, caution)!;
    const end = (300 - 60 * caution / 100) / 1000;
    expect(referenceDirectionChanges(lap)[0].start).toBe(240);
    expect(speedAt(copy, end)).toBeCloseTo(80, 8);
    expect(Math.min(...copy.y.map((row) => row[6]))).toBeCloseTo(80, 8);
    const originalRates = rates(lap), adjustedRates = rates(copy);
    expect(Math.max(...adjustedRates.map(({ rate }) => rate))).toBeCloseTo(Math.max(...originalRates.map(({ rate }) => rate)), 7);
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

it.each([25, 50, 100])('compresses the release and replaces lost braking work with an unchanged peak hold at %s%%', (caution) => {
    const lap = twoRates(), copy = copyPhraseReference(lap, caution)!;
    const fraction = caution / 100, start = 200 - 35 * fraction, transition = 250 - 30 * fraction, end = 300 - 60 * fraction;
    expect(speedAt(copy, start / 1000)).toBeCloseTo(144, 8);
    expect(speedAt(copy, transition / 1000)).toBeCloseTo(Math.sqrt(900 + 2 * (end - transition)) * 3.6, 8);
    expect(speedAt(copy, end / 1000)).toBeCloseTo(108, 8);
    rates(copy).filter(({ rate }) => rate > 1e-7).forEach(({ start: position, distance, rate }) => {
        expect(rate).toBeCloseTo(position * 1000 + distance / 2 < transition ? 6 : 1, 7);
    });
    expect(rowAt(copy, start / 1000)[3]).toBeCloseTo(0);
    copy.x.forEach((p, i) => {
        if (p >= end / 1000 && p <= 0.6) expect(copy.y[i][3]).toBe(0);
        const index = Math.floor(p * 100), f = p * 100 - index;
        const a = lap.y[index % 100], b = lap.y[(index + 1) % 100];
        [0, 1, 2, 4, 5].forEach((j) => expect(copy.y[i][j]).toBeCloseTo(a[j] + (b[j] - a[j]) * f, 8));
    });
});

it.each([false, true])('wraps earlier braking across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = reference(), original = lap.y.map((row) => [...row]);
    // Rotate geometry and controls together: braking ends at 20 m, turn-in begins at 960 m.
    lap.y = original.map((_, i) => [...original[(i + 28) % 100]]);
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.95)).toBeGreaterThan(80);
    expect(speedAt(copy, 0.96)).toBeCloseTo(80, 8);
    for (const p of [0.96, 0.99, 0, 0.02, 0.32]) {
        expect(speedAt(copy, p)).toBeCloseTo(80, 8);
        expect(rowAt(copy, p)[3]).toBe(0);
    }
    expect(speedAt(copy, 0.33)).toBeCloseTo(84);
    expect(Math.max(...rates(copy).map(({ rate }) => rate))).toBeCloseTo(Math.max(...rates(lap).map(({ rate }) => rate)), 7);
    if (duplicate) expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
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

it.each([NaN, Infinity, -0.1, 1.1])('preserves an invalid brake channel (%s) while shifting speed', (invalid) => {
    const lap = reference();
    lap.y[21][3] = invalid;
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeCloseTo(80);
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
    expect(speedAt(copy, 0.24)).toBeCloseTo(80);
    lap.y.forEach((row, i) => expect(rowAt(copy, lap.x[i]).slice(0, 6)).toEqual(row.slice(0, 6)));
    expect(copyPhraseReference(null, 0)).toBeNull();
    expect(copyPhraseMap(null)).toBeNull();
});

it.each([-10, 0, 10, 25, 50, 100, 150, NaN, Infinity])('clamps %s to 0–100%% with a reference fallback', (value) => {
    const normalized = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
    expect(normalizeDrivingAggression(value)).toBe(normalized);
    expect(copyPhraseReference(reference(), value)).toEqual(copyPhraseReference(reference(), normalized));
});

it('preserves distinct corner minima and their peaks when an earlier approach would overlap', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        row[6] = i >= 20 && i <= 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i > 30 && i < 40 ? 108 + (i - 30) * 3.6
                : i >= 40 && i <= 50 ? Math.sqrt(1600 - 120 * (i - 40)) * 3.6
                    : i > 50 && i < 60 ? 72 + (i - 50) * 7.2 : 144;
        row[3] = i >= 21 && i < 30 ? 0.4 : i >= 41 && i < 50 ? 0.8 : 0;
    });
    const copy = copyPhraseReference(lap, 100)!;
    expect(speedAt(copy, 0.24)).toBeCloseTo(108);
    expect(speedAt(copy, 0.3)).toBeCloseTo(108);
    expect(speedAt(copy, 0.49)).toBeCloseTo(72);
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
    expect(speedAt(copy, 0.24)).toBe(0);
    expect(speedAt(copy, 0.61)).toBeCloseTo(4);
});

it('retains a coast inside the earlier braking curve', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        if (i >= 20 && i <= 30) row[6] = i < 24 ? 120 - (i - 20) * 5 : i <= 26 ? 100 : 100 - (i - 26) * 5;
    });
    const copy = copyPhraseReference(lap, 100)!;
    const intervals = rates(copy).filter(({ start }) => start > 0.16 && start < 0.23);
    expect(intervals.some(({ rate }) => Math.abs(rate) < 1e-7)).toBe(true);
    expect(speedAt(copy, 0.24)).toBeCloseTo(80, 8);
});

it.each(['acc', 'iracing'])('finishes before a continuous direction change even when turn-in precedes reference braking in %s', (game) => {
    for (const sign of [-1, 1]) {
        const lap = twoRates();
        lap.game = game;
        turnTrajectory(lap, 15, sign);
        const range = referenceDirectionChanges(lap).find(({ start }) => Math.abs(start - 150) < 1e-7)!;
        expect(range).toBeDefined();
        const copy = copyPhraseReference(lap, 100)!;
        expect(speedAt(copy, 0.15)).toBeCloseTo(108, 8);
        expect(rowAt(copy, 0.15)[3]).toBe(0);
        const braking = rates(copy).filter(({ rate }) => rate > 1e-7);
        expect(Math.max(...braking.map(({ rate }) => rate))).toBeCloseTo(6, 7);
        braking.forEach(({ start }) => expect(start).toBeLessThan(0.15));
        expect(braking.reduce((sum, { rate, distance }) => sum + rate * distance, 0)).toBeCloseTo(350, 7);
    }
});

it.each([31, 35])('uses the available entry speed without changing either corner minimum or peak (turn-in sample %s)', (entry) => {
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
    expect(speedAt(copy, 0.4)).toBeCloseTo(72, 8);
    expect(speedAt(copy, 0.5)).toBeCloseTo(72, 8);
    if (entry === 35) expect(speedAt(copy, 0.35)).toBeCloseTo(72, 8);
    expect(Math.max(...rates(copy).filter(({ start }) => start >= 0.3 && start < 0.5).map(({ rate }) => rate))).toBeCloseTo(6, 7);
    expect(copy.x.every((p, i) => Number.isFinite(p) && (!i || p > copy.x[i - 1]))).toBe(true);
});

it('moves a ramped braking profile progressively earlier while retaining its rise and peak', () => {
    const lap = reference();
    const referenceRates = [1, 2, 4, 6, 6, 4, 3, 2, 1, 1];
    let energy = 1600;
    lap.y.forEach((row, i) => {
        if (i > 20 && i <= 30) energy -= 20 * referenceRates[i - 21];
        row[6] = i <= 20 || i >= 70 ? 144 : i <= 60 ? Math.sqrt(energy) * 3.6
            : Math.sqrt(energy) * 3.6 + (144 - Math.sqrt(energy) * 3.6) * (i - 60) / 10;
    });
    let previous = lap;
    for (const caution of [1, 10, 50, 90, 100]) {
        const copy = copyPhraseReference(lap, caution)!;
        const braking = rates(copy).filter(({ rate }) => rate > 1e-7);
        expect(braking[0].rate).toBeCloseTo(1, 7);
        expect(Math.max(...braking.map(({ rate }) => rate))).toBeCloseTo(6, 7);
        expect(speedAt(copy, (300 - caution * 0.6) / 1000)).toBeCloseTo(lap.y[30][6], 7);
        lap.x.forEach((p) => expect(speedAt(copy, p)).toBeLessThanOrEqual(speedAt(previous, p) + 1e-7));
        previous = copy;
    }
});
