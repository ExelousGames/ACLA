import { conditionTestEngine } from './test-fixtures';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { readReferenceGuidance } from './phrase-reference-guidance';

// A 1000 m square, with a brake onset at 200 m and throttle onset at 210 m.
function lapWithShortZone(pedal: 'brake' | 'throttle', speed: number): TopLapReferenceEntry {
    return {
        game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed', 'expert_optimal_steering'],
        x: Array.from({ length: 100 }, (_, i) => i / 100),
        y: Array.from({ length: 100 }, (_, i) => {
            const d = i * 10;
            const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
            const short = i === 21 ? 0.001 : 0, long = i >= 22 && i < 90 ? 0.001 : 0;
            return [...xy, 0, pedal === 'brake' ? short : long, pedal === 'throttle' ? short : long, speed, 0];
        }),
    };
}
const sampleAt = (position: number, speed = 36): StandardTelemetrySample => ({
    Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: position,
    Physics_speed_kmh: speed, Physics_brake: 0, Physics_gas: 0,
});
const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});

describe.each(['brake', 'throttle'] as const)('%s zone onsets', (pedal) => {
    const opposite = pedal === 'brake' ? 'throttle' : 'brake';
    const name = pedal === 'brake' ? 'Brake zone' : 'Throttle zone';

    it.each([[90, false], [72, false], [71.99, true], [36, true]] as const)(
        'filters the entire reference interval at %s km/h (qualified: %s)', (referenceSpeed, qualified) => {
            const lap = lapWithShortZone(pedal, referenceSpeed);
            // The complete 10 m interval is respectively 400, 500, >500, or 1000 ms.
            // Live speed, pedals, and time since entry cannot change its qualification.
            for (const liveSpeed of [20, 36, 180]) {
                for (const position of [0.2, 0.200001, 0.209999]) {
                    const sample = { ...sampleAt(position, liveSpeed), Physics_brake: 1, Physics_gas: 1 };
                    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).pedalZone)
                        .toBe(qualified ? pedal : 'none');
                }
            }
            expect(readReferenceGuidance(sampleAt(0.21), null, lap, 'acc', 1000).pedalZone).toBe(opposite);
        });

    it(pedal === 'brake'
        ? 'enters immediately below the old pressure threshold and keeps Brake active at throttle onset'
        : 'enters immediately below the old pressure threshold and exits on the first brake rise', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(null, 0, lapWithShortZone(pedal, 36));
        // At 36 km/h the decision position leads telemetry by 5 m / 0.005 of this lap.
        expect(engine.receiveTelemetry(frame(sampleAt(0.195001)), 1000).state.path).toEqual(['root', name]);
        expect(engine.receiveTelemetry(frame(sampleAt(0.204999)), 1001).state.path).toEqual(['root', name]);
        const boundary = pedal === 'brake' ? 0.21 : 0.205001;
        if (pedal === 'brake') expect(engine.receiveTelemetry(frame(sampleAt(0.209999)), 1002).state.path).toEqual(['root', name]);
        expect(engine.receiveTelemetry(frame(sampleAt(boundary)), 1003).state.path)
            .toEqual(['root', 'Brake zone']);
    });

    it(opposite === 'brake'
        ? 'keeps the previous Brake zone when the new throttle interval is filtered out'
        : 'exits the previous Throttle zone even when the new brake interval is filtered out', () => {
        const engine = conditionTestEngine();
        const lap = lapWithShortZone(pedal, 72);
        // Keep the preceding pedal pressed until the opposite onset under test.
        lap.y.forEach((row, i) => { if (i < 20 || i >= 90) row[pedal === 'brake' ? 4 : 3] = 0.001; });
        engine.receiveMap(null, 0, lap);
        const otherName = opposite === 'brake' ? 'Brake zone' : 'Throttle zone';
        expect(engine.receiveTelemetry(frame(sampleAt(0.1949)), 1000).state.path).toEqual(['root', otherName]);
        const boundary = opposite === 'brake' ? 0.2 : 0.195001;
        expect(readReferenceGuidance(sampleAt(0.2), null, lap, 'acc', 1001).pedalZone).toBe('none');
        expect(engine.receiveTelemetry(frame(sampleAt(boundary)), 1001).state.path)
            .toEqual(opposite === 'brake' ? ['root', 'Brake zone'] : ['root']);
        expect(engine.receiveTelemetry(frame(sampleAt(0.210001)), 1002).state.path).toEqual(['root', otherName]);
    });

    it('tracks zone membership through release and reapplication of the same pedal', () => {
        const lap = lapWithShortZone(pedal, 36);
        const channel = pedal === 'brake' ? 3 : 4, other = pedal === 'brake' ? 4 : 3;
        lap.y.forEach((row, i) => {
            row[channel] = i === 21 || i === 25 ? 0.001 : 0;
            row[other] = i >= 41 && i < 90 ? 0.001 : 0;
        });
        for (const position of [0.200001, 0.219999, 0.240001, 0.259999]) {
            expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe(pedal);
        }
        for (const position of [0.22, 0.23, 0.26, 0.39]) {
            expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
        }
        expect(readReferenceGuidance(sampleAt(0.400001), null, lap, 'acc', 1000).pedalZone).toBe(opposite);
    });

    it('ends on opposite onset even while its own reference pedal remains pressed', () => {
        const lap = lapWithShortZone(pedal, 36), channel = pedal === 'brake' ? 3 : 4;
        lap.y.forEach((row, i) => { row[channel] = i >= 21 && i < 90 ? 0.8 : 0; });
        expect(readReferenceGuidance(sampleAt(0.209999), null, lap, 'acc', 1000).pedalZone).toBe(pedal);
        expect(readReferenceGuidance(sampleAt(0.210001), null, lap, 'acc', 1000).pedalZone).toBe(opposite);
    });

    it.each([false, true])('filters duration across start/finish with duplicate endpoint %s', (duplicate) => {
        for (const speed of [72, 36]) {
            const lap = lapWithShortZone(pedal, speed), channel = pedal === 'brake' ? 3 : 4, other = pedal === 'brake' ? 4 : 3;
            lap.y.forEach((row, i) => { row[channel] = i === 0 ? 0.001 : 0; row[other] = i >= 1 && i < 90 ? 0.001 : 0; });
            if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
            expect(readReferenceGuidance(sampleAt(0.995), null, lap, 'acc', 1000).pedalZone)
                .toBe(speed === 36 ? pedal : 'none');
            expect(readReferenceGuidance(sampleAt(1), null, lap, 'acc', 1000).pedalZone).toBe(opposite);
        }
    });

    it('uses live travel time when the reference has no speed channel', () => {
        const lap = lapWithShortZone(pedal, 36);
        lap.target_features[5] = 'missing speed';
        expect(readReferenceGuidance(sampleAt(0.205, 72), null, lap, 'acc', 1000).pedalZone).toBe('none');
        expect(readReferenceGuidance(sampleAt(0.205, 36), null, lap, 'acc', 1000).pedalZone).toBe(pedal);
    });
});

it.each([3, 4])('withholds zone membership when reference pedal channel %s is unavailable', (channel) => {
    const lap = lapWithShortZone('brake', 36);
    lap.y[50][channel] = NaN;
    expect(readReferenceGuidance(sampleAt(0.205), null, lap, 'acc', 1000).pedalZone).toBeUndefined();
});

it('integrates changing reference speed over the whole zone', () => {
    const lap = lapWithShortZone('brake', 36);
    lap.y[20][5] = 36; lap.y[21][5] = 72;
    // A 10 m linear acceleration from 10 to 20 m/s takes ln(2) seconds (>500 ms).
    expect(readReferenceGuidance(sampleAt(0.209999, 180), null, lap, 'acc', 1000).pedalZone).toBe('brake');
});

it.each([0, 0.5, 1])('ends the reference brake interval at release to zero but keeps Brake active with live pedal %s', (pedal) => {
    const lap = lapWithShortZone('brake', 36);
    lap.y.forEach((row, i) => {
        row[3] = i === 21 ? 0.8 : i === 22 ? 0.4 : i === 23 ? 0.001 : 0;
        row[4] = 0;
    });
    const engine = conditionTestEngine();
    engine.receiveMap(null, 0, lap);
    for (const position of [0.2, 0.21, 0.22, 0.23, 0.239999, 0.24, 0.3, 0.5]) {
        const sample = { ...sampleAt(position), Physics_brake: pedal };
        const inside = position < 0.24;
        expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).pedalZone).toBe(inside ? 'brake' : 'none');
        expect(engine.receiveTelemetry(frame(sample), 1000).state.path.slice(0, 2))
            .toEqual(['root', 'Brake zone']);
    }
});

it.each([false, true])('filters brake duration through release to zero, with reference speed missing: %s', (missingSpeed) => {
    for (const speed of [180, 144, 143.99, 72]) {
        const lap = lapWithShortZone('brake', speed);
        lap.y.forEach((row, i) => { row[3] = i === 21 ? 0.8 : 0; row[4] = 0; });
        if (missingSpeed) lap.target_features[5] = 'missing speed';
        // The complete rise and release spans 20 m, exactly 500 ms at 144 km/h.
        for (const position of [0.2, 0.21, 0.219999]) {
            expect(readReferenceGuidance(sampleAt(position, missingSpeed ? speed : 180), null, lap, 'acc', 1000).pedalZone)
                .toBe(speed < 144 ? 'brake' : 'none');
        }
        expect(readReferenceGuidance(sampleAt(0.22), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each([false, true])('ends braking at release across start/finish (duplicate endpoint: %s)', (duplicate) => {
    const lap = lapWithShortZone('brake', 36);
    lap.y.forEach((row, i) => { row[3] = i === 99 ? 0.5 : 0; row[4] = 0; });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const position of [0.98, 0.999999]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('brake');
    }
    for (const position of [0, 1, 0.01, 0.5]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it('keeps the throttle zone active when an overlapping reference brake releases to zero', () => {
    const lap = lapWithShortZone('throttle', 36);
    lap.y.forEach((row, i) => {
        row[3] = i >= 11 && i < 24 ? 0.5 : 0;
        row[4] = i >= 21 && i < 50 ? 0.5 : 0;
    });
    for (const position of [0.2, 0.239999, 0.24, 0.3]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('throttle');
    }
});

function lapReachingFullThrottle(): TopLapReferenceEntry {
    const lap = lapWithShortZone('throttle', 36);
    lap.y.forEach((row, i) => {
        row[3] = i >= 61 && i < 90 ? 0.8 : 0;
        row[4] = i === 21 ? 0.4 : i === 22 ? 0.7 : i === 23 ? 0.999
            : i >= 24 && i <= 40 ? 1 : i > 40 && i < 50 ? 0.5 : 0;
    });
    return lap;
}

it.each([0, 1])('waits for the last steering change before throttle reaches %s', (endpoint) => {
    for (const change of [-0.2, -0.001, 0.001, 0.2]) {
        const lap = lapReachingFullThrottle();
        lap.y.forEach((row, i) => {
            if (i >= 24 && i <= 40) row[4] = endpoint;
            row[6] = i < 22 ? 0 : change;
        });
        // The current section is flat, but steering will change between 210 and 220 m.
        for (const position of [0.2, 0.205, 0.21, 0.219999, 0.22, 0.239999, 0.24]) {
            const sample = { ...sampleAt(position), Physics_steer_angle: -change };
            expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).pedalZone)
                .toBe(position >= 0.22 && position < 0.24 ? 'throttle' : 'none');
        }
    }
});

it.each([0, 1])('blocks an increase followed by a reduction even with no net steering change before throttle %s', (endpoint) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row) => { if (row[4] === 1) row[4] = endpoint; });
    lap.y[22][6] = 0.2;
    for (const position of [0.2, 0.21, 0.22, 0.229999]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
    expect(readReferenceGuidance(sampleAt(0.23), null, lap, 'acc', 1000).pedalZone).toBe('throttle');
});

it.each([0, 1])('checks steering all the way through the throttle endpoint %s', (endpoint) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row) => { if (row[4] === 1) row[4] = endpoint; });
    lap.y[24][6] = 0.2;
    for (const position of [0.2, 0.23, 0.239999, 0.24]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each([-0.2, 0, 0.2])('allows constant steering %s and ignores changes after the throttle endpoint', (steering) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[6] = i <= 24 ? steering : 0.4; });
    for (const position of [0.2, 0.23, 0.239999]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('throttle');
    }
});

it.each([[false, 0], [true, 0], [false, 1], [true, 1]] as const)(
    'delays throttle across start/finish (duplicate: %s, steering settles at sample %s)', (duplicate, settledIndex) => {
        const lap = lapReachingFullThrottle();
        lap.y.forEach((row, i) => {
            row[4] = i >= 99 || i <= 1 ? 0.5 : i === 2 ? 1 : 0;
            row[6] = i >= 99 ? 0.1 : i <= 2 ? 0.2 + (i >= settledIndex ? 0.1 : 0) : 0;
        });
        if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
        for (const position of [0.98, 0.99, 0.999999, 0, 1, 0.01, 0.019999]) {
            expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone)
                .toBe(position % 1 >= settledIndex / 100 && position % 1 < 0.02 ? 'throttle' : 'none');
        }
        expect(readReferenceGuidance(sampleAt(0.02), null, lap, 'acc', 1000).pedalZone).toBe('none');
    });

it.each(['missing', 'invalid', 'out of range'])('withholds throttle with %s reference steering while preserving brake zones', (kind) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { if (i <= 20) row[3] = 0.001; });
    if (kind === 'missing') lap.target_features[6] = 'missing steering';
    else lap.y[22][6] = kind === 'invalid' ? NaN : 1.1;
    expect(readReferenceGuidance(sampleAt(0.2), null, lap, 'acc', 1000).pedalZone).toBe('none');
    expect(readReferenceGuidance(sampleAt(0.6), null, lap, 'acc', 1000).pedalZone).toBe('brake');
    const engine = conditionTestEngine();
    engine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sampleAt(0.19)), 1000).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.2)), 1100).state.path).toEqual(['root', 'Brake zone']);
});

it('uses active-reference steering rather than the source reference for throttle entry', () => {
    const source = lapReachingFullThrottle(), active = lapReachingFullThrottle();
    active.y[22][6] = 0.2;
    expect(readReferenceGuidance(sampleAt(0.2), null, active, 'acc', 1000, source).pedalZone).toBe('none');
    expect(readReferenceGuidance(sampleAt(0.2), null, source, 'acc', 1000, active).pedalZone).toBe('throttle');
});

it('keeps Brake active at throttle onset while throttle entry still requires steady steering', () => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[6] = i < 22 ? 0 : 0.2; if (i <= 20) row[3] = 0.001; });
    const engine = conditionTestEngine(), throttleEngine = conditionTestEngine();
    engine.receiveMap(null, 0, lap);
    throttleEngine.receiveMap(null, 0, lap);
    expect(engine.receiveTelemetry(frame(sampleAt(0.19)), 1000).state.path).toEqual(['root', 'Brake zone']);
    for (const position of [0.2, 0.21]) {
        expect(engine.receiveTelemetry(frame(sampleAt(position)), 1100).state.path).toEqual(['root', 'Brake zone']);
        expect(throttleEngine.receiveTelemetry(frame(sampleAt(position)), 1100).state.path).toEqual(['root']);
    }
    expect(engine.receiveTelemetry(frame(sampleAt(0.22)), 1200).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.receiveTelemetry(frame(sampleAt(0.24)), 1300).state.path).toEqual(['root', 'Brake zone']);
    expect(throttleEngine.receiveTelemetry(frame(sampleAt(0.22)), 1200).state.path).toEqual(['root', 'Throttle zone']);
    expect(throttleEngine.receiveTelemetry(frame(sampleAt(0.24)), 1300).state.path).toEqual(['root']);
});

it.each([0, 0.5, 1])('ends throttle at reference full throttle regardless of live pedal %s', (pedal) => {
    const lap = lapReachingFullThrottle();
    const zoneAt = (position: number) => readReferenceGuidance({ ...sampleAt(position), Physics_gas: pedal }, null, lap, 'acc', 1000).pedalZone;
    for (const position of [0.2, 0.23, 0.239999]) expect(zoneAt(position)).toBe('throttle');
    // Stay outside after full throttle, even when the reference later lifts or coasts.
    for (const position of [0.24, 0.3, 0.415, 0.5, 0.599999]) expect(zoneAt(position)).toBe('none');
    expect(zoneAt(0.6)).toBe('brake');
});

it.each([0, 0.5, 1])('ends throttle when the reference releases to zero regardless of live pedal %s', (pedal) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[4] = i === 21 ? 0.8 : i === 22 ? 0.4 : i === 23 ? 0.001 : 0; });
    const zoneAt = (position: number) => readReferenceGuidance({ ...sampleAt(position), Physics_gas: pedal }, null, lap, 'acc', 1000).pedalZone;
    for (const position of [0.2, 0.21, 0.22, 0.23, 0.239999]) expect(zoneAt(position)).toBe('throttle');
    for (const position of [0.24, 0.3, 0.5, 0.599999]) expect(zoneAt(position)).toBe('none');
    expect(zoneAt(0.6)).toBe('brake');
});

it.each([false, true])('filters duration through throttle release to zero, with reference speed missing: %s', (missingSpeed) => {
    for (const speed of [180, 144, 143.99, 72]) {
        const lap = lapReachingFullThrottle();
        lap.y.forEach((row, i) => { row[4] = i === 21 ? 0.8 : 0; row[5] = speed; });
        if (missingSpeed) lap.target_features[5] = 'missing speed';
        // The complete rise and release spans 20 m, exactly 500 ms at 144 km/h.
        for (const position of [0.2, 0.21, 0.219999]) {
            expect(readReferenceGuidance(sampleAt(position, missingSpeed ? speed : 180), null, lap, 'acc', 1000).pedalZone)
                .toBe(speed < 144 ? 'throttle' : 'none');
        }
        expect(readReferenceGuidance(sampleAt(0.22), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each([false, true])('filters only the ramp to full throttle, with reference speed missing: %s', (missingSpeed) => {
    for (const speed of [90, 72, 71.99, 36]) {
        const lap = lapReachingFullThrottle();
        lap.y.forEach((row, i) => { row[4] = i >= 21 && i <= 40 ? 1 : 0; row[5] = speed; });
        if (missingSpeed) lap.target_features[5] = 'missing speed';
        for (const position of [0.2, 0.209999]) {
            expect(readReferenceGuidance(sampleAt(position, missingSpeed ? speed : 180), null, lap, 'acc', 1000).pedalZone)
                .toBe(speed < 72 ? 'throttle' : 'none');
        }
        expect(readReferenceGuidance(sampleAt(0.21), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each([[false, 1], [true, 1], [false, 0], [true, 0]] as const)('ends throttle at start/finish (duplicate: %s, throttle: %s)', (duplicate, throttle) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[4] = i === 99 ? 0.5 : i <= 40 ? throttle : 0; });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const position of [0.98, 0.999999]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('throttle');
    }
    for (const position of [0, 1, 0.01, 0.5]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it('allows a new throttle onset after full throttle without intervening braking', () => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[3] = 0; row[4] = i === 21 || i === 41 ? 0.5 : i === 22 || i === 42 ? 1 : 0; });
    for (const position of [0.2, 0.219999, 0.4, 0.419999]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('throttle');
    }
    for (const position of [0.1, 0.22, 0.3, 0.42, 0.8]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each(['full lap', 'partial lift'])('has no throttle zone without an onset after full throttle: %s', (profile) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[3] = 0; row[4] = profile === 'partial lift' && i < 50 ? 0.5 : 1; });
    for (const position of [0, 0.25, 0.5, 0.75]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('none');
    }
});

it.each([0, 1])('does not end braking when an already-pressed reference throttle reaches %s', (throttle) => {
    const lap = lapReachingFullThrottle();
    lap.y.forEach((row, i) => { row[4] = i >= 70 && i < 80 ? throttle : 0.5; });
    for (const position of [0.6, 0.7, 0.75]) {
        expect(readReferenceGuidance(sampleAt(position), null, lap, 'acc', 1000).pedalZone).toBe('brake');
    }
});
