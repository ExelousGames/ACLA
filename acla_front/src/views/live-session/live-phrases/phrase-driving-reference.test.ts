import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression } from './phrase-driving-reference';
import { circuitMap } from './test-fixtures';

// A 1 km loop with known 10 m sample spacing and a slower corner.
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
const rowAt = (lap: TopLapReferenceEntry, position: number) => lap.y[lap.x.indexOf(position)];
const speedAt = (lap: TopLapReferenceEntry, position: number) => rowAt(lap, position)[6];

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

it('derives braking distance from reduced local rates and the lower corner minimum', () => {
    const lap = reference(), cautious = copyPhraseReference(lap, 50)!, middle = copyPhraseReference(lap, 25)!;
    // At 50%, half the reference deceleration needs 320 m to reach 40 km/h.
    // At 25%, 75% of the reference rate needs 180 m to reach 60 km/h.
    expect(speedAt(cautious, 0.98)).toBeCloseTo(120);
    expect(speedAt(cautious, 0.99)).toBeLessThan(120);
    expect(speedAt(middle, 0.12)).toBeCloseTo(120);
    expect(speedAt(middle, 0.13)).toBeLessThan(120);
    expect(speedAt(cautious, 0.25)).toBeCloseTo(Math.sqrt(3268.8));
    expect(speedAt(middle, 0.25)).toBeCloseTo(Math.sqrt(6170.4));
    expect(speedAt(cautious, 0.3)).toBeCloseTo(40);
    expect(speedAt(middle, 0.3)).toBeCloseTo(60);
    expect(speedAt(copyPhraseReference(lap, 10)!, 0.3)).toBeCloseTo(72);
    // Acceleration still starts at 600 m and finishes at 700 m, with no delayed flat minimum.
    expect(speedAt(cautious, 0.6)).toBeCloseTo(40);
    expect(speedAt(cautious, 0.61)).toBeCloseTo(48);
    expect(speedAt(cautious, 0.65)).toBeCloseTo(80);
    expect(speedAt(middle, 0.65)).toBeCloseTo(90);
    expect(speedAt(cautious, 0.7)).toBeCloseTo(120);
    lap.y.forEach((row, i) => {
        for (const channel of [0, 1, 2, 4, 5]) {
            expect(rowAt(cautious, lap.x[i])[channel]).toBe(row[channel]);
            expect(rowAt(middle, lap.x[i])[channel]).toBe(row[channel]);
        }
        expect(speedAt(cautious, lap.x[i])).toBeLessThanOrEqual(speedAt(middle, lap.x[i]) + 1e-8);
        expect(speedAt(middle, lap.x[i])).toBeLessThanOrEqual(row[6] + 1e-8);
        expect(speedAt(cautious, lap.x[i])).toBeGreaterThanOrEqual(40 - 1e-8);
    });
    expect(lap).toEqual(reference());
    expect(copyPhraseReference(lap, 0)).toEqual(lap);
});

it.each([[25, 1.8], [50, 3.2]])('stretches the brake pressure pattern with braking distance at %s%% caution', (caution, stretch) => {
    const lap = reference();
    // Include a pressure ramp, a lift and a second application before release.
    const pressures = [0, 0.4, 0.8, 0.6, 0.2, 0, 0.3, 0.7, 0.5, 0.2, 0];
    pressures.forEach((pressure, i) => { lap.y[20 + i][3] = pressure; });
    const copy = copyPhraseReference(lap, caution)!;
    pressures.forEach((pressure, i) => {
        const position = ((300 - (100 - i * 10) * stretch + 1000) % 1000) / 1000;
        const index = copy.x.findIndex((p) => Math.abs(p - position) < 1e-8);
        expect(index).toBeGreaterThanOrEqual(0);
        expect(copy.y[index][3]).toBeCloseTo(pressure, 8);
    });
    expect(rowAt(copy, 0.3)[3]).toBe(0);
    expect(rowAt(copy, 0.6)[3]).toBe(0);
    expect(lap.y.slice(20, 31).map((row) => row[3])).toEqual(pressures);
    expect(copyPhraseReference(lap, 0)).toEqual(lap);
});

it.each([false, true])('wraps braking and lower corner speeds across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = reference(), original = lap.y.map((row) => [...row]);
    // Move the known corner 300 m back, so braking finishes at the start/finish line.
    lap.y.forEach((row, i) => { row[6] = original[(i + 30) % 100][6]; row[3] = original[(i + 30) % 100][3]; });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const copy = copyPhraseReference(lap, 50)!;
    expect(speedAt(copy, 0.68)).toBeCloseTo(120);
    expect(speedAt(copy, 0.69)).toBeLessThan(120);
    expect(speedAt(copy, 0.95)).toBeCloseTo(Math.sqrt(3268.8));
    expect(speedAt(copy, 0)).toBeCloseTo(40);
    expect(speedAt(copy, 0.3)).toBeCloseTo(40);
    expect(speedAt(copy, 0.31)).toBeCloseTo(48);
    expect(speedAt(copy, 0.4)).toBeCloseTo(120);
    expect(rowAt(copy, 0.68)[3]).toBeCloseTo(0);
    expect(rowAt(copy, 0.69)[3]).toBeCloseTo(0.25);
    expect(rowAt(copy, 0.9)[3]).toBeCloseTo(0.8);
    expect(rowAt(copy, 0)[3]).toBeCloseTo(0);
    lap.y.forEach((row, i) => [0, 1, 2, 4, 5].forEach((channel) => expect(rowAt(copy, lap.x[i])[channel]).toBe(row[channel])));
    if (duplicate) expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
});

it('uses physical distance rather than sample counts or normalized position spacing', () => {
    const lap = reference(), dense = reference();
    // Add samples on only one straight and give the same geometry nonuniform lap positions.
    const row = lap.y[60].map((value, j) => (value + lap.y[61][j]) / 2);
    dense.x.splice(61, 0, 0.605);
    dense.y.splice(61, 0, row);
    dense.x = dense.x.map((position) => position ** 2);
    const original = copyPhraseReference(lap, 50)!, resampled = copyPhraseReference(dense, 50)!;
    lap.x.forEach((position) => {
        expect(speedAt(resampled, position ** 2)).toBeCloseTo(speedAt(original, position));
        expect(rowAt(resampled, position ** 2)[3]).toBeCloseTo(rowAt(original, position)[3]);
    });
});

it.each([NaN, Infinity, -0.1, 1.1])('preserves an invalid brake channel (%s) while adjusting speed', (invalid) => {
    const lap = reference();
    lap.y[21][3] = invalid;
    const copy = copyPhraseReference(lap, 50)!;
    expect(speedAt(copy, 0.3)).toBeCloseTo(40);
    lap.x.forEach((position, i) => expect(rowAt(copy, position)[3]).toBe(lap.y[i][3]));
});

it.each(['missing', NaN, Infinity, -1])('leaves the reference unchanged when speed is %s', (invalid) => {
    const lap = reference();
    if (invalid === 'missing') lap.target_features[6] = 'missing speed';
    else lap.y[0][6] = invalid as number;
    expect(copyPhraseReference(lap, 50)).toEqual(lap);
});

it('adjusts speed independently of missing or invalid control channels', () => {
    const lap = reference();
    lap.target_features[3] = 'missing brake';
    lap.y[0][4] = NaN;
    const copy = copyPhraseReference(lap, 50)!;
    expect(speedAt(copy, 0.25)).toBeCloseTo(Math.sqrt(3268.8));
    lap.y.forEach((row, i) => expect(rowAt(copy, lap.x[i]).slice(0, 6)).toEqual(row.slice(0, 6)));
    expect(copyPhraseReference(null, 0)).toBeNull();
    expect(copyPhraseMap(null)).toBeNull();
});

it('handles a duplicated start/finish point without nonfinite speed values', () => {
    const lap = reference();
    lap.x.push(1);
    lap.y.push([...lap.y[0]]);
    const copy = copyPhraseReference(lap, 50)!;
    expect(copy.y.every((row) => Number.isFinite(row[6]))).toBe(true);
    expect(copy.y[copy.y.length - 1]).toEqual(copy.y[0]);
});

it.each([-10, 0, 10, 25, 50, 100, NaN, Infinity])('clamps caution %s to 0–50%% with a reference fallback', (value) => {
    const normalized = Number.isFinite(value) ? Math.max(0, Math.min(50, value)) : 0;
    expect(normalizeDrivingAggression(value)).toBe(normalized);
    expect(copyPhraseReference(reference(), value)).toEqual(copyPhraseReference(reference(), normalized));
});

it('lowers each distinct corner minimum relative to its own reference speed', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        if (i >= 75 && i <= 95) row[6] = 100 + Math.abs(i - 85) * 2;
    });
    const copy = copyPhraseReference(lap, 25)!;
    expect(speedAt(copy, 0.3)).toBeCloseTo(60);
    expect(speedAt(copy, 0.85)).toBeCloseTo(75);
    expect(speedAt(copy, 0.95)).toBeCloseTo(120);
});

it('preserves flat speed profiles and keeps zero-speed minima finite', () => {
    const flat = reference();
    flat.y.forEach((row) => { row[6] = 80; });
    expect(copyPhraseReference(flat, 50)).toEqual(flat);
    const lap = reference();
    lap.y.forEach((row) => { row[6] -= 80; });
    const copy = copyPhraseReference(lap, 50)!;
    expect(copy.y.every((row) => Number.isFinite(row[6]) && row[6] >= 0)).toBe(true);
    expect(speedAt(copy, 0.3)).toBe(0);
    expect(speedAt(copy, 0.61)).toBeCloseTo(4);
});

it.each([10, 25, 50])('caps each local braking rate at matching zone progress (%s%% caution)', (caution) => {
    const lap = reference();
    // 40 m/s entry: 50 m at 6 m/s², then 50 m at 1 m/s², ending at 30 m/s.
    lap.y.forEach((row, i) => {
        row[6] = i < 20 || i >= 70 ? 144 : i <= 25 ? Math.sqrt(1600 - 120 * (i - 20)) * 3.6
            : i < 30 ? Math.sqrt(1000 - 20 * (i - 25)) * 3.6 : i <= 60 ? 108 : 108 + (i - 60) * 3.6;
    });
    const copy = copyPhraseReference(lap, caution)!;
    const factor = 1 - caution / 100;
    const span = (1600 - (30 * factor) ** 2) / (7 * factor);
    const start = 300 - span, transition = 300 - span / 2;
    expect(copy.x.some((p) => Math.abs(p * 1000 - transition) < 1e-7)).toBe(true);
    let brakingIntervals = 0;
    copy.y.forEach((row, i) => {
        const next = copy.y[(i + 1) % copy.y.length];
        const ds = Math.hypot(...row.slice(0, 3).map((v, j) => next[j] - v));
        if (ds < 1e-8 || row[6] <= next[6] + 1e-8) return;
        let midpoint = copy.x[i] * 1000 + ds / 2;
        if (midpoint > 300) midpoint -= 1000;
        expect(midpoint).toBeGreaterThan(start);
        const expected = (midpoint < transition ? 6 : 1) * factor;
        const actual = (row[6] ** 2 - next[6] ** 2) / (2 * 3.6 ** 2 * ds);
        expect(actual).toBeCloseTo(expected, 7);
        brakingIntervals++;
    });
    expect(brakingIntervals).toBeGreaterThan(10);
    // Every added point stays on the original polyline with local throttle and steering.
    copy.x.forEach((p, i) => {
        const index = Math.floor(p * 100), fraction = p * 100 - index;
        const a = lap.y[index % 100], b = lap.y[(index + 1) % 100];
        [0, 1, 2, 4, 5].forEach((j) => expect(copy.y[i][j]).toBeCloseTo(a[j] + (b[j] - a[j]) * fraction, 8));
    });
});

it('satisfies both local rate limits when stretched braking zones overlap', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        row[6] = i >= 20 && i <= 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i > 30 && i < 35 ? 108 + (i - 30) * 7.2
                : i >= 40 && i <= 50 ? Math.sqrt(1600 - 120 * (i - 40)) * 3.6
                    : i > 50 && i < 60 ? 72 + (i - 50) * 7.2 : 144;
        row[3] = i >= 21 && i < 30 ? 0.4 : i >= 41 && i < 50 ? 0.8 : 0;
    });
    const copy = copyPhraseReference(lap, 50)!;
    copy.y.forEach((row, i) => {
        const next = copy.y[(i + 1) % copy.y.length];
        const ds = Math.hypot(...row.slice(0, 3).map((v, j) => next[j] - v));
        if (ds < 1e-8 || row[6] <= next[6] + 1e-8) return;
        const midpoint = copy.x[i] * 1000 + ds / 2;
        const firstCornerPosition = midpoint > 300 ? midpoint - 1000 : midpoint;
        const firstCap = firstCornerPosition >= 300 - 1375 / 3.5 ? 1.75 : Infinity;
        const secondCap = midpoint >= 250 && midpoint < 500 ? 3 : Infinity;
        const limit = Math.min(firstCap, secondCap);
        expect(Number.isFinite(limit)).toBe(true);
        const actual = (row[6] ** 2 - next[6] ** 2) / (2 * 3.6 ** 2 * ds);
        expect(actual).toBeLessThanOrEqual(limit + 1e-8);
    });
    expect(speedAt(copy, 0.35)).toBeLessThan(144);
    expect(speedAt(copy, 0.3)).toBeCloseTo(54);
    expect(speedAt(copy, 0.5)).toBeCloseTo(36);
    expect(rowAt(copy, 0.25)[3]).toBeCloseTo(0.4);
    expect(rowAt(copy, 0.28)[3]).toBeCloseTo(0.8);
    expect(rowAt(copy, 0.3)[3]).toBeCloseTo(0.8);
    expect(rowAt(copy, 0.5)[3]).toBeCloseTo(0);
});

it('preserves a zero-deceleration section inside the stretched braking approach', () => {
    const lap = reference();
    lap.y.forEach((row, i) => {
        if (i >= 20 && i <= 30) row[6] = i < 24 ? 120 - (i - 20) * 5 : i <= 26 ? 100 : 100 - (i - 26) * 5;
    });
    const copy = copyPhraseReference(lap, 50)!;
    // The original 240–260 m plateau maps to 108–172 m with the 3.2 stretch.
    const plateau = copy.x.map((p, i) => ({ p, speed: copy.y[i][6] })).filter(({ p }) => p >= 0.108 && p <= 0.172);
    expect(plateau.length).toBeGreaterThan(5);
    plateau.forEach(({ speed }) => expect(speed).toBeCloseTo(plateau[0].speed, 8));
});

it('keeps sparse and resampled braking data physically equivalent', () => {
    const sparse = reference(), dense = reference();
    const row = sparse.y[24].map((v, i) => (v + sparse.y[25][i]) / 2);
    row[6] = Math.sqrt((sparse.y[24][6] ** 2 + sparse.y[25][6] ** 2) / 2);
    dense.x.splice(25, 0, 0.245);
    dense.y.splice(25, 0, row);
    const a = copyPhraseReference(sparse, 35)!, b = copyPhraseReference(dense, 35)!;
    sparse.x.forEach((p) => expect(speedAt(b, p)).toBeCloseTo(speedAt(a, p), 8));
});

it('stays finite and continuous when the requested braking distance exceeds a lap', () => {
    const lap = reference();
    lap.y.forEach((row) => { row[6] = 119 + (row[6] - 80) / 40; });
    lap.x.push(1);
    lap.y.push([...lap.y[0]]);
    const copy = copyPhraseReference(lap, 50)!;
    expect(copy.x.every((p, i) => p >= 0 && p <= 1 && (!i || p > copy.x[i - 1]))).toBe(true);
    expect(copy.y.every((row) => Number.isFinite(row[6]) && row[6] >= 0 && row[6] <= 120)).toBe(true);
    expect(rowAt(copy, 1)).toEqual(rowAt(copy, 0));
    expect(speedAt(copy, 0.3)).toBeCloseTo(59.5);
});
