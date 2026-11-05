import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { readReferenceGuidance } from './phrase-reference-guidance';
import { copyPhraseReference } from './phrase-driving-reference';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { circuitMap, vision as trafficVision } from './test-fixtures';

beforeEach(() => jest.spyOn(Math, 'random').mockReturnValue(0));
afterEach(() => jest.restoreAllMocks());

// A 1000 m square lap, sampled every 10 m. Independent, known control onsets.
function referenceLap(): TopLapReferenceEntry {
    return { game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_steering', 'expert_optimal_brake', 'expert_optimal_throttle'],
        x: Array.from({ length: 100 }, (_, i) => i / 100),
        y: Array.from({ length: 100 }, (_, i) => {
            const d = i * 10;
            const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
            return [...xy, 0, i >= 41 && i <= 49 ? 0.4 : 0, i >= 21 && i <= 29 ? 0.8 : 0, i >= 61 && i <= 90 ? 0.8 : 0];
        }),
    };
}
function addSpeed(lap: TopLapReferenceEntry) {
    lap.target_features.push('expert_optimal_speed');
    lap.y.forEach((row, i) => row.push(i < 20 || i >= 70 ? 120
        : i < 30 ? 120 - (i - 20) * 4 : i <= 60 ? 80 : 80 + (i - 60) * 4));
}
const driving = (position: number): StandardTelemetrySample => ({ Static_track: 'test', Static_car_model: 'car',
    Graphics_normalized_car_position: position, Physics_speed_kmh: 36, Physics_steer_angle: 0, Physics_brake: 0, Physics_gas: 0 });
function detection(lap: TopLapReferenceEntry, sample: StandardTelemetrySample, now = 1000): TrackVisionDetection {
    return { capturedAt: now, width: 100, height: 100, detections: {}, reconstruction: null, geometry: null,
        telemetry: { ...createLiveTelemetryStore().getSnapshot(), game: 'acc', receivedAt: now, currentTelemetry: sample,
            telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0 },
        position: { coordinateFrame: 'car_local', inferenceMs: 10, referenceDelta: [0, 0, 0],
            normalizedPosition: sample.Graphics_normalized_car_position!, referenceLap: lap, references: { reference: { x: 0, y: 0, z: 0 } } } };
}
const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({ type: 'frame', sample, sampleIndex: 0,
    telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } });

const steeringGuidance = (sample: StandardTelemetrySample, lap: TopLapReferenceEntry, cornerTurnSign: -1 | 1 = 1) =>
    readReferenceGuidance(sample, null, lap, lap.game, 1000, lap, undefined, undefined, undefined, cornerTurnSign);

it.each([20, 36, 180])('requires a nonzero forward steering sum at %s km/h', (speed) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[3] = i >= 40 && i <= 55 ? (i - 40) * 0.04 : 0; });
    for (const position of [0.3, 0.35]) {
        const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_steer_angle: 0.5 };
        const result = steeringGuidance(sample, lap);
        expect(result.ready).toBe(true);
        expect(result.steering).toBeUndefined();
        expect(result.steeringPhase).toBeUndefined();
    }
    const sample = { ...driving(0.401), Physics_speed_kmh: speed };
    expect(steeringGuidance(sample, lap))
        .toMatchObject({ steering: 2500, steeringPhase: 'prepare' });
});

it.each([
    ['increasing steering', [0.1, 0.2, 0.3], 1, 'turn'],
    ['decreasing steering', [-0.1, -0.2, -0.3], -1, 'turn'],
    ['a tiny positive sum', [0, 0.000001, 0], 1, 'prepare'],
    ['a tiny negative sum', [0, -0.000001, 0], -1, 'prepare'],
    ['below the turn threshold', [0.02, 0.04, 0.06], 1, 'prepare'],
    ['unwinding in the corner direction', [0.1, 0.05, 0], 1, 'turn'],
    ['a reversal ahead', [0.1, 0.2, 0.15], 1, 'turn'],
    ['a plateau ahead', [0.1, 0.2, 0.2], 1, 'turn'],
    ['a flat segment before turn-in', [0, 0, 0.2], 1, 'prepare'],
    ['held steering above the threshold', [0.2, 0.2, 0.2], 1, 'turn'],
    ['a zero sum', [0.125, -0.25, 0.125], 1, undefined],
    ['steering opposite to the corner', [0.1, 0.2, 0.3], -1, undefined],
    ['negative steering opposite to the corner', [-0.1, -0.2, -0.3], 1, undefined],
    ['local steering opposite to a matching sum', [-0.1, 0.2, 0.3], 1, 'prepare'],
] as const)('adds every forward steering value with %s', (_name, steering, direction, phase) => {
    const lap = referenceLap();
    steering.forEach((value, i) => { lap.y[25 + i][3] = value; });
    // At 90 km/h the 0.8 s window covers exactly 250–270 m; boundaries count once.
    const guidance = steeringGuidance({ ...driving(0.25), Physics_speed_kmh: 90 }, lap, direction);
    expect(guidance.steeringPhase).toBe(phase);
    expect(guidance.steering).toBe(phase ? 2500 : undefined);
});

it('includes intermediate samples even when their direction differs from both window boundaries', () => {
    const lap = referenceLap();
    lap.y[25][3] = 0.1;
    lap.y[26][3] = 0.1;
    lap.x.splice(26, 0, 0.255, 0.258);
    lap.y.splice(26, 0, [250, 5, 0, -0.5, 0.8, 0], [250, 8, 0, 0.1, 0.8, 0]);
    expect(steeringGuidance(driving(0.25), lap).steering).toBeUndefined();
    expect(steeringGuidance(driving(0.25), lap, -1).steeringPhase).toBe('prepare');
});

it.each([[20, undefined], [36, undefined], [45, undefined], [46, 'prepare'], [90, 'prepare'], [180, 'prepare']] as const)(
    'bounds the steering sum to 0.8 seconds at %s km/h', (speed, phase) => {
        const lap = referenceLap();
        [0, 0, 0.3, 0.2].forEach((value, i) => { lap.y[25 + i][3] = value; });
        lap.y[24][3] = 0.5; // Outside the window, behind the driver.
        const sample = { ...driving(0.25), Physics_speed_kmh: speed };
        expect(steeringGuidance(sample, lap).steeringPhase).toBe(phase);
    });

it.each(['missing', 'invalid', 'out of range'])('withholds steering with %s forward reference data', (kind) => {
    const lap = referenceLap();
    [0.1, 0.2, 0.3].forEach((value, i) => { lap.y[25 + i][3] = value; });
    if (kind === 'missing') lap.target_features[3] = 'missing';
    else lap.y[27][3] = kind === 'invalid' ? NaN : 1.1;
    expect(steeringGuidance(driving(0.25), lap).steering).toBeUndefined();
});

it('withholds steering when the corner direction is unknown', () => {
    const lap = referenceLap();
    expect(readReferenceGuidance(driving(0.45), null, lap, 'acc', 1000).steering).toBeUndefined();
});

it.each([[0.25, 'brake', 'Physics_brake', 0.8],
    [0.7, 'throttle', 'Physics_gas', 0.8]] as const)('corrects low input at %s, but accepts matching input', (position, cue, field, value) => {
    const lap = referenceLap(), sample = driving(position), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000)[cue]).toBe(2500);
    expect(readReferenceGuidance({ ...sample, [field]: value }, view, lap, 'acc', 1000)[cue]).toBeUndefined();
    expect(readReferenceGuidance({ ...sample, [field]: undefined }, view, lap, 'acc', 1000)[cue]).toBeUndefined();
});

it('bases corrections on interpolated speed even when pedals already match the reference', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const brake = { ...driving(0.255), Physics_speed_kmh: 106, Physics_brake: 0.8 };
    const throttle = { ...driving(0.655), Physics_speed_kmh: 94, Physics_gas: 0.8 };
    expect(readReferenceGuidance(brake, detection(lap, brake), lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    expect(readReferenceGuidance(throttle, detection(lap, throttle), lap, 'acc', 1000).moreThrottle).toBe(2500);
    for (const [sample, speed] of [[brake, 98], [throttle, 102]] as const) {
        const matching = { ...sample, Physics_speed_kmh: speed };
        const cues = readReferenceGuidance(matching, detection(lap, matching), lap, 'acc', 1000, lap, 1);
        expect(cues.moreBrake).toBeUndefined();
        expect(cues.moreThrottle).toBeUndefined();
    }
    expect(readReferenceGuidance({ ...brake, Physics_brake: 1 }, detection(lap, brake), lap, 'acc', 1000, lap, 1).moreBrake).toBeUndefined();
    expect(readReferenceGuidance({ ...throttle, Physics_gas: 1 }, detection(lap, throttle), lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it('does not announce throttle before pickup', () => {
    const lap = referenceLap(), sample = driving(0.59);
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance).not.toHaveProperty('throttleSoon');
    expect(guidance.throttle).toBeUndefined();
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([
    [102, 0, 'throttle'], [94, 0.8, 'moreThrottle'], [102, 1, undefined], [120, 0.8, undefined],
    [102, 0.8, undefined], [94, 1, undefined], [120, 0, undefined],
] as const)('selects one throttle instruction at %s km/h with pedal %s: %s', (speed, pedal, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: speed, Physics_gas: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    const cues = Object.entries(guidance).filter(([key]) => ['throttle', 'moreThrottle', 'lessThrottle', 'releaseThrottle'].includes(key));
    expect(cues).toEqual(cue ? [[cue, 2500]] : []);
});

it.each([0.4, 0.8])('uses reference throttle %s to estimate demand at matching speed', (pedal) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[5]) row[5] = pedal; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.6 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance.moreThrottle).toBe(pedal === 0.8 ? 2500 : undefined);
    expect(guidance.lessThrottle).toBeUndefined();
});

it.each([0.2, 1.2])('scales throttle correction using reference acceleration (%s km/h per metre)', (slope) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => { row[6] = i < 60 || i > 70 ? 100 : 102 + (i * 10 - 655) * slope; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 97, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(slope === 0.2 ? 2500 : undefined);
});

it('preserves maintenance throttle on a flat speed profile', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.75), Physics_speed_kmh: 120, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toEqual({ ready: true, lateralOffset: 0, braking: false, pedalZone: 'throttle', speedAtOrBelowReference: true });
    expect(readReferenceGuidance({ ...sample, Physics_gas: 0.4 }, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
});

it.each(['missing throttle', 'invalid throttle', 'missing pedal', 'invalid pedal', 'missing brake', 'missing live brake'])(
    'withholds throttle instructions with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.2 as number | undefined };
        if (kind === 'missing throttle') lap.target_features[5] = 'missing throttle';
        if (kind === 'invalid throttle') lap.y[65][5] = NaN;
        if (kind === 'missing pedal') sample.Physics_gas = undefined;
        if (kind === 'invalid pedal') sample.Physics_gas = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
        if (kind === 'missing live brake') sample.Physics_brake = undefined;
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        expect(['throttle', 'moreThrottle', 'lessThrottle', 'releaseThrottle'].filter((key) => key in guidance)).toEqual([]);
    });

it.each(['missing speed', 'invalid speed', 'flat speed'])('falls back to reference throttle with %s', (kind) => {
    const lap = referenceLap();
    addSpeed(lap);
    if (kind === 'missing speed') lap.target_features[6] = 'missing speed';
    if (kind === 'invalid speed') lap.y[65][6] = NaN;
    if (kind === 'flat speed') lap.y.forEach((row) => { row[6] = 102; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.4 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_gas: 1 }, detection(lap, sample), lap, 'acc', 1000).lessThrottle).toBeUndefined();
});

it.each([false, true])('uses reference acceleration across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[99][5] = lap.y[0][5] = 0.8;
    lap.y[99][6] = 96;
    lap.y[0][6] = 100;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const sample = { ...driving(0.995), Physics_speed_kmh: 98, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toEqual({ ready: true, lateralOffset: 0, braking: false, pedalZone: 'throttle', speedAtOrBelowReference: true });
    expect(readReferenceGuidance({ ...sample, Physics_speed_kmh: 90 }, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
});

it('withholds throttle corrections away from the line without overspeed and outward motion', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 1 }, view = detection(lap, sample);
    view.position!.referenceDelta[0] = 3;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).lessThrottle).toBeUndefined();
    const guidance = readReferenceGuidance({ ...sample, Physics_gas: 0 }, view, lap, 'acc', 1000);
    expect(guidance.throttle).toBeUndefined();
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([
    [98, 0, 'brake'], [106, 0.8, 'moreBrake'], [92, 0.8, 'easeBrake'], [80, 0.8, 'easeBrake'],
    [98, 0.8, 'easeBrake'], [106, 1, undefined], [80, 0, undefined],
] as const)('selects one brake instruction at %s km/h with pedal %s: %s', (speed, pedal, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.255), Physics_speed_kmh: speed, Physics_brake: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, speed === 92 ? 4 : 1);
    const matched = ['brake', 'moreBrake', 'easeBrake', 'releaseBrake'].filter((key) => key in guidance);
    expect(matched).toEqual(cue ? [cue] : []);
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([0.2, 0.8, 1])('corrects speed and deceleration independently of reference brake pressure %s', (pressure) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[4]) row[4] = pressure; });
    for (const [speed, deceleration, pedal, cue] of [[106, 1, 0.9, 'moreBrake'], [92, 4, 0.2, 'easeBrake']] as const) {
        const sample = { ...driving(0.255), Physics_speed_kmh: speed, Physics_brake: pedal };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, deceleration);
        expect(['brake', 'moreBrake', 'easeBrake', 'releaseBrake'].filter((key) => key in guidance)).toEqual([cue]);
    }
});

it.each([100, 99.999, 94, 60, 100.001, 104])('eases only at or below the reference speed: %s km/h', (speed) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: speed, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000);
    expect(guidance.easeBrake).toBe(speed <= 100 ? 2500 : undefined);
    if (speed <= 100) expect(guidance.releaseBrake).toBeUndefined();
});

it.each([undefined, NaN, -1, 0, 1, 10])('does not require live deceleration to ease the brake: %s', (deceleration) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, deceleration).easeBrake).toBe(2500);
});

it.each([
    ['steady deceleration', [100, 96, 92, 88, 84, 80], true],
    ['locally flat', [100, 100, 95, 90, 85, 80], true],
    ['locally accelerating', [100, 110, 95, 90, 85, 80], true],
    ['flat overall', [100, 90, 80, 90, 100, 100], false],
    ['accelerating overall', [100, 90, 80, 90, 110, 110], false],
    ['speed rises before the interpolated exit', [100, 96, 92, 88, 90, 110], false],
    ['speed drops before the interpolated exit', [100, 110, 110, 110, 110, 90], true],
] as const)('checks the whole remaining brake zone when %s', (_name, speeds, expected) => {
    const lap = referenceLap();
    addSpeed(lap);
    speeds.forEach((speed, i) => { lap.y[25 + i][6] = speed; });
    const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake).toBe(expected ? 2500 : undefined);
});

it('stops the remaining-speed check at the current brake zone exit', () => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => {
        if (i >= 25 && i <= 30) row[6] = 100;
        if (i >= 35 && i <= 40) { row[4] = 0.8; row[6] = 80 - (i - 35) * 4; }
    });
    const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake).toBeUndefined();
});

it.each([false, true])('eases through a brake zone crossing start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => {
        row[4] = i >= 98 || i <= 2 ? 0.8 : 0;
        row[6] = i >= 98 ? 120 - (i - 98) * 10 : i <= 3 ? 100 - i * 10 : 120;
    });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const position of [0.98, 0.99, 0, 0.01]) {
        const speed = lap.y[lap.x.indexOf(position)][6];
        const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8 };
        expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake).toBe(2500);
    }
});

it.each([0, 0.099, 0.1, 1, undefined, NaN])('requires an applied brake to ease: %s', (pedal) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: pedal };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake)
        .toBe(pedal !== undefined && pedal >= 0.1 ? 2500 : undefined);
});

it.each(['missing speed', 'invalid speed', 'missing brake', 'invalid brake', 'no zone exit', 'outside zone'])(
    'withholds Ease the brake with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[28][6] = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing';
        if (kind === 'invalid brake') lap.y[28][4] = NaN;
        if (kind === 'no zone exit') lap.y.forEach((row) => { row[4] = 0.8; });
        if (kind === 'outside zone') lap.y[25][4] = 0;
        const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
        expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake).toBeUndefined();
    });

it('continues easing through the active trail-braking zone below the entry threshold', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.299), Physics_speed_kmh: 80, Physics_brake: 0.2 };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000).easeBrake).toBeUndefined();
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, undefined, undefined, 'brake').easeBrake).toBe(2500);
});

it.each([
    [106, 1, 'moreBrake'], [92, 4, 'easeBrake'],
    [103.5, 1, 'moreBrake'], [92.5, 4, 'easeBrake'],
    [106, 4, undefined], [92, 1, 'easeBrake'], [98, 1, 'easeBrake'], [98, 4, 'easeBrake'],
    [102, 1, undefined], [94, 4, 'easeBrake'], [106, 3, undefined], [92, 3, 'easeBrake'],
    [106, undefined, undefined], [92, NaN, 'easeBrake'], [106, Infinity, undefined],
] as const)('compares speed %s and live deceleration %s with the interpolated profile: %s', (speed, deceleration, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.255), Physics_speed_kmh: speed, Physics_brake: 0.5 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, deceleration);
    expect(['moreBrake', 'easeBrake'].filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
});

it.each([0.2, 1.2])('compares live deceleration with the local speed slope (%s km/h per metre)', (slope) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => { row[6] = i < 20 || i > 30 ? 100 : 98 - (i * 10 - 255) * slope; });
    const sample = { ...driving(0.255), Physics_speed_kmh: Math.hypot(98, 5 * slope) + 6, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, 4);
    expect(guidance.moreBrake).toBe(slope === 1.2 ? 2500 : undefined);
});

it.each([5, 50])('scales the 10%% deceleration threshold with reference distance scale %s', (distanceScale) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { row[0] *= distanceScale; row[1] *= distanceScale; });
    // 30 -> 20 m/s over 50 or 500 m: reference deceleration is 5 or 0.5 m/s².
    lap.y[25][6] = 108;
    lap.y[26][6] = 72;
    const referenceDeceleration = 25 / distanceScale;
    for (const [speed, ratio, cue] of [
        [114, 0.89, 'moreBrake'], [114, 0.9, 'moreBrake'], [114, 0.91, undefined],
        [102, 1.09, 'easeBrake'], [102, 1.1, 'easeBrake'], [102, 1.11, 'easeBrake'],
    ] as const) {
        const sample = { ...driving(0.25), Physics_speed_kmh: speed, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, referenceDeceleration * ratio);
        expect({ speed, ratio, cues: ['moreBrake', 'easeBrake'].filter((key) => key in guidance) })
            .toEqual({ speed, ratio, cues: cue ? [cue] : [] });
    }
});

it.each([0, 4])('uses brake demand without percentage corrections when reference speed is flat or increasing by %s km/h', (speedIncrease) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { row[6] = 100; });
    lap.y[26][6] += speedIncrease;
    for (const [speed, pedal, deceleration, cue] of [
        [106, 0.8, 0, undefined], [106, 0.8, -1, undefined],
        [94, 0.8, 0, 'releaseBrake'], [94, 0.8, 1, 'releaseBrake'],
        [106, 0, 0, 'brake'],
    ] as const) {
        const sample = { ...driving(0.25), Physics_speed_kmh: speed, Physics_brake: pedal };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, deceleration);
        expect(guidance.pedalZone).toBe('brake');
        expect(['brake', 'moreBrake', 'easeBrake', 'releaseBrake'].filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    }
});

it.each([0, 25, 50])('accepts matching throttle on the active speed profile at %s%% caution', (caution) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, caution)!;
    const sample = { ...driving(0.65), Physics_speed_kmh: local.y[local.x.indexOf(0.65)][6], Physics_gas: 0.8 };
    const engine = new PhraseEngine();
    engine.receiveMap(null, 0, lap);
    engine.setDrivingAggression(caution, 0);
    engine.receiveVision(detection(lap, sample), 1000);
    const snapshot = engine.receiveTelemetry(frame(sample), 1000);
    expect(snapshot.state.path).toEqual(['root', 'Throttle zone']);
    expect(snapshot.events).toEqual([]);
});

it.each([25, 50])('keeps cornering cues identical when the adjusted reference is supplied directly (%s%% caution)', (caution) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, caution)!;
    // Supplying the adjusted lap directly must behave just like selecting that caution in Live Phrases.
    for (const position of [0.21, 0.25, 0.29, 0.3, 0.41, 0.61, 0.65, 0.7]) {
        const targetSpeed = local.y[local.x.indexOf(position)][6];
        for (const error of [-5.1, 0, 6]) {
            for (const pedal of [0, 0.8]) {
                const sample = { ...driving(position), Physics_speed_kmh: targetSpeed + error,
                    Physics_brake: position < 0.4 ? pedal : 0, Physics_gas: position >= 0.4 ? pedal : 0 };
                const snapshots = [lap, local].map((reference, index) => {
                    const engine = new PhraseEngine();
                    engine.receiveMap(null, 0, reference);
                    if (!index) engine.setDrivingAggression(caution, 0);
                    engine.receiveVision(detection(reference, sample), 1000);
                    engine.receiveTelemetry(frame(sample), 1000);
                    engine.receiveVision(detection(reference, sample, 1300), 1300);
                    return engine.receiveTelemetry(frame(sample), 1300);
                });
                expect({ position, error, pedal, path: snapshots[0].state.path, events: snapshots[0].events })
                    .toEqual({ position, error, pedal, path: snapshots[1].state.path, events: snapshots[1].events });
            }
        }
    }
});

it('includes early braking demand in the stretched reference brake zone', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const cautious = copyPhraseReference(lap, 50)!;
    const early = { ...driving(0.19), Physics_speed_kmh: 116 };
    expect(readReferenceGuidance(early, detection(lap, early), cautious, 'acc', 1000, lap).brake).toBe(2500);
    expect(readReferenceGuidance(early, detection(lap, early), cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    const missingPedal = { ...early, Physics_brake: undefined };
    expect(readReferenceGuidance(missingPedal, detection(lap, missingPedal), cautious, 'acc', 1000, lap))
        .toEqual({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake', speedAtOrBelowReference: false });
    const slowed = { ...driving(0.3), Physics_speed_kmh: 40, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(slowed, detection(lap, slowed), cautious, 'acc', 1000, lap);
    expect(guidance.releaseBrake).toBe(2500);
    expect(guidance.moreBrake).toBeUndefined();
    expect(lap.y[19][4]).toBe(0);
    expect(lap.y[29][4]).toBe(0.8);
});

it.each([0.201, 0.205, 0.209, 0.255, 0.299])('preserves the local deceleration cap between samples at %s', (position) => {
    const lap = referenceLap();
    addSpeed(lap);
    // Reference: 40 -> 30 m/s over 100 m, giving 3.5 m/s² at full brake.
    lap.y.forEach((row, i) => {
        row[4] = i >= 20 && i <= 30 ? 1 : 0;
        row[6] = i < 20 || i >= 70 ? 144 : i < 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i <= 60 ? 108 : 108 + (i - 60) * 3.6;
    });
    const cautious = copyPhraseReference(lap, 25)!;
    // At 25% caution the target rate is 2.625 m/s², ending at 22.5 m/s.
    const speed = Math.sqrt(22.5 ** 2 + 2 * 2.625 * (300 - position * 1000)) * 3.6;
    for (const [speedError, deceleration, cue] of [
        [6, 2.3, 'moreBrake'], [6, 2.4, undefined], [6, 2.7, undefined],
        [-6, 2.8, 'easeBrake'], [-6, 2.9, 'easeBrake'],
    ] as const) {
        const sample = { ...driving(position), Physics_speed_kmh: speed + speedError, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap, deceleration);
        expect(['moreBrake', 'easeBrake'].filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    }
});

it('aligns braking responses when added samples precede the first original normalized position', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const shifted = { ...lap, x: lap.x.map((p) => p + 0.005), y: lap.y.map((row) => [...row]) };
    const cautious = copyPhraseReference(lap, 48)!, shiftedCautious = copyPhraseReference(shifted, 48)!;
    expect(shiftedCautious.x[0]).toBeLessThan(shifted.x[0]);
    for (const position of [0.245, 0.25, 0.255, 0.61, 0.65]) {
        const index = cautious.x.findIndex((p) => p >= position);
        const speed = cautious.y[index][6];
        for (let pedal = 0; pedal <= 1; pedal += 0.01) {
            const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: pedal, Physics_gas: pedal };
            const shiftedSample = { ...sample, Graphics_normalized_car_position: position + 0.005 };
            expect(readReferenceGuidance(shiftedSample, detection(shifted, shiftedSample), shiftedCautious, 'acc', 1000, shifted))
                .toEqual(readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap));
        }
    }
});

it.each([false, true])('uses reference deceleration across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[99][4] = lap.y[0][4] = 0.8;
    lap.y[99][6] = 100;
    lap.y[0][6] = 96;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const sample = { ...driving(0.995), Physics_speed_kmh: 98, Physics_brake: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toEqual({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake', speedAtOrBelowReference: true });
    sample.Physics_speed_kmh = 106;
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
});

it.each(['missing brake', 'invalid brake', 'missing pedal', 'invalid pedal', 'missing speed', 'flat speed'])(
    'handles %s without assuming missing data is a released pedal or inventing deceleration', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        const sample = { ...driving(0.255), Physics_speed_kmh: 106, Physics_brake: 0.8 as number | undefined };
        if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
        if (kind === 'invalid brake') lap.y[25][4] = NaN;
        if (kind === 'missing pedal') sample.Physics_brake = undefined;
        if (kind === 'invalid pedal') sample.Physics_brake = NaN;
        if (kind === 'missing speed') lap.target_features[6] = 'missing speed';
        if (kind === 'flat speed') lap.y.forEach((row) => { row[6] = 98; });
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        expect(['brake', 'moreBrake', 'easeBrake', 'releaseBrake'].filter((key) => key in guidance)).toEqual([]);
    });

it('reads speed targets at normalized position regardless of the position-model forward delta', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 108 }, view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_brake: 0.8 }, view, lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_brake: 0.8 }, view, lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    for (const position of [0.59, 0.65]) {
        const fast = { ...driving(position), Physics_speed_kmh: 120 };
        const cues = readReferenceGuidance(fast, detection(lap, fast), lap, 'acc', 1000);
        expect(cues.throttle).toBeUndefined();
        expect(cues.moreThrottle).toBeUndefined();
    }
});

it('exposes the signed position-model offset independently of steering input', () => {
    const lap = referenceLap();
    lap.y.forEach((row) => { row[3] *= -1; });
    const sample = driving(0.45), view = detection(lap, sample);
    view.position!.referenceDelta[0] = -2;
    expect(readReferenceGuidance({ ...sample, Physics_steer_angle: 0.4 }, view, lap, 'acc', 1000).lateralOffset).toBe(-2);
    expect(readReferenceGuidance({ ...sample, Physics_steer_angle: -0.4 }, view, lap, 'acc', 1000).lateralOffset).toBe(-2);
});

it('ignores the position-model forward delta for brake demand and uses lateral error to suppress acceleration', () => {
    const lap = referenceLap(), sample = driving(0.21), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = 10;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = -5;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    const exit = driving(0.7), exitView = detection(lap, exit);
    exitView.position!.referenceDelta[0] = 3;
    expect(readReferenceGuidance(exit, exitView, lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it('interpolates brake demand across start/finish without a prepare-to-brake cue', () => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[4] = i === 0 ? 0.8 : 0; });
    const sample = driving(0.995), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000)).not.toHaveProperty('brakeSoon');
    const before = driving(0.98);
    expect(readReferenceGuidance(before, detection(lap, before), lap, 'acc', 1000).brake).toBeUndefined();
    expect(readReferenceGuidance({ ...sample, Physics_speed_kmh: 10 }, view, lap, 'acc', 1000)).toEqual({});
});

it.each(['wrong car', 'wrong track', 'wrong game', 'bad lap', 'missing position', 'invalid position', 'out-of-range position'])(
    'rejects %s reference or telemetry inputs', (kind) => {
        const lap = referenceLap();
        let sample = driving(0.19);
        const view = detection(lap, sample);
        if (kind === 'wrong car') sample = { ...sample, Static_car_model: 'other' };
        if (kind === 'wrong track') sample = { ...sample, Static_track: 'other' };
        if (kind === 'bad lap') lap.y[2][0] = NaN;
        if (kind === 'missing position') sample = { ...sample, Graphics_normalized_car_position: undefined };
        if (kind === 'invalid position') sample = { ...sample, Graphics_normalized_car_position: NaN };
        if (kind === 'out-of-range position') sample = { ...sample, Graphics_normalized_car_position: 1.1 };
        expect(readReferenceGuidance(sample, view, lap, kind === 'wrong game' ? 'iracing' : 'acc', 1000)).toEqual({});
    });

it.each(['stale', 'future', 'wrong lap', 'jump', 'invalid delta', 'remote offset', 'no position', 'no vision'])(
    'keeps normalized reference guidance but withholds lateral cues with %s', (kind) => {
        const lap = referenceLap();
        lap.y[42][3] = 0.5;
        for (const position of [0.25, 0.401, 0.7]) {
            const sample = driving(position), view = detection(lap, sample);
            if (kind === 'stale') view.capturedAt = 499;
            if (kind === 'future') view.capturedAt = 1001;
            if (kind === 'wrong lap') view.position!.referenceLap = referenceLap();
            if (kind === 'jump') view.position!.normalizedPosition = 0.1;
            if (kind === 'invalid delta') view.position!.referenceDelta[0] = NaN;
            if (kind === 'remote offset') view.position!.referenceDelta[0] = 14;
            if (kind === 'no position') view.position = undefined;
            const guidance = readReferenceGuidance(sample, kind === 'no vision' ? null : view, lap, 'acc', 1000,
                lap, undefined, undefined, undefined, 1);
            expect(guidance.ready).toBe(true);
            expect(guidance.lateralOffset).toBeUndefined();
            expect(guidance.pedalZone).toBe(position === 0.25 ? 'brake' : position === 0.7 ? 'throttle' : 'none');
            expect(guidance.brake).toBe(position === 0.25 ? 2500 : undefined);
            expect(guidance.steering).toBe(position === 0.401 ? 2500 : undefined);
            expect(guidance.throttle).toBeUndefined();
            expect(guidance.moreThrottle).toBeUndefined();
            expect(guidance.lessThrottle).toBeUndefined();
        }
    });

it('does not treat a missing reference channel as zero or advise throttle during braking', () => {
    const lap = referenceLap();
    lap.target_features[3] = 'missing steering';
    const sample = driving(0.45);
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).steering).toBeUndefined();
    const braking = { ...driving(0.7), Physics_brake: 0.5 };
    expect(readReferenceGuidance(braking, detection(lap, braking), lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it.each([20, 180])('follows the sum toward a negative corner independently of live steering at %s km/h', (speed) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[3] = i < 41 ? 0.4 : -Math.min(1, 0.4 + (i - 41) * 0.05); });
    for (const [position, phase] of [[0.401, speed === 20 ? undefined : 'prepare'], [0.405, 'prepare'], [0.407, 'turn']] as const) {
        for (const liveSteering of [-0.4, 0, 0.4, undefined]) {
            const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_steer_angle: liveSteering };
            const result = steeringGuidance(sample, lap, -1);
            expect(result.steering).toBe(phase ? 2500 : undefined);
            expect(result.steeringPhase).toBe(phase);
        }
    }
});

it('suppresses throttle for reference braking within 1.8 seconds, beyond the steering window', () => {
    const lap = referenceLap();
    lap.y.forEach((row) => { row[5] = 0.8; });
    const sample = driving(0.19), view = detection(lap, sample);
    const cues = readReferenceGuidance(sample, view, lap, 'acc', 1000);
    expect(cues.brake).toBeUndefined();
    expect(cues.moreThrottle).toBeUndefined();
});

it('reports reference braking at normalized position independently of advance cues and driver brake input', () => {
    const lap = referenceLap();
    for (const [position, braking] of [[0.19, false], [0.201, false], [0.202, true], [0.25, true]] as const) {
        const sample = { ...driving(position), Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        // Correction demand is separate from reference zone membership.
        expect(guidance.braking).toBe(true);
        expect(guidance.brake).toBeUndefined();
        expect(guidance.releaseBrake !== undefined).toBe(!braking);
        expect(guidance.moreBrake).toBeUndefined();
    }
    const sample = driving(0.21), view = detection(lap, sample);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).braking).toBe(true);
    const missingBrake = referenceLap();
    missingBrake.target_features[4] = 'missing brake';
    expect(readReferenceGuidance(sample, detection(missingBrake, sample), missingBrake, 'acc', 1000).braking).toBeUndefined();
});

function setup(configure?: (lap: TopLapReferenceEntry) => void, map: CircuitMapDto | null = null) {
    const engine = new PhraseEngine(), lap = referenceLap();
    configure?.(lap);
    engine.receiveMap(map, 0, lap);
    const update = (position: number, now: number, patch: StandardTelemetrySample = {}) => {
        const sample = { ...driving(position), ...patch };
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    return { engine, lap, update };
}

it('enters and exits reference brake zones using telemetry without any position-model captures', () => {
    const { engine } = setup();
    const at = (position: number, now: number) => engine.receiveTelemetry(frame(driving(position)), now);
    expect(at(0.19, 1000).state.path).toEqual(['root']);
    expect(at(0.202, 1100).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(at(0.299, 1200).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.2999, 1250).state.path).toEqual(['root']);
    const throttle = at(0.7, 1300);
    expect(throttle.state.path).toEqual(['root', 'Throttle zone']);
    expect(throttle.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(throttle.closures.find(({ id }) => id === 'reference-throttle')).toMatchObject({ status: 'Missing input' });
    const expired = engine.evaluate(2801);
    expect(expired.telemetryReady).toBe(false);
    expect(expired.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Missing input' });
    expect(expired.referenceSteeringPhase).toBeUndefined();
});

it.each([0, 0.8])('keeps brake active down to 1%% reference pressure with live brake %s', (pedal) => {
    const { engine } = setup((lap) => {
        lap.y[26][4] = lap.y[27][4] = 0.05;
        lap.y[28][4] = lap.y[29][4] = 0.01;
        lap.y[30][4] = 0.009;
    });
    const at = (position: number, now: number, brake = pedal) => engine.receiveTelemetry(
        frame({ ...driving(position), Physics_brake: brake, Physics_gas: 1 }), now);
    // Light pressure cannot enter the zone, but it keeps an entered zone active.
    expect(at(0.265, 900).state.path).toEqual(['root']);
    expect(at(0.25, 1000, 0).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(at(0.265, 1100).state.path).toEqual(pedal
        ? ['root', 'Brake zone', 'Release the brake', 'say phrase'] : ['root', 'Brake zone']);
    expect(engine.evaluate(2700).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.285, 3000).state.path).toContain('Brake zone');
    // Crossing back above 10% before exiting must not rearm Brake.
    expect(at(0.25, 3100, 0).events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
    expect(at(0.285, 3200).state.path).toContain('Brake zone');
    const exited = at(0.3, 3300);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events.map(({ sentence }) => sentence)).toEqual(pedal ? ['Brake.', 'Release the brake.'] : ['Brake.']);
    expect(at(0.265, 4300).state.path).toEqual(['root']);
    expect(at(0.25, 4400, 0).events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(2);
});

it('gives immediate brake corrections without position-model captures', () => {
    const { engine } = setup(addSpeed);
    for (const now of [900, 1000, 1100, 1300]) {
        engine.receiveVision(null, now);
        const result = engine.receiveTelemetry(frame({ ...driving(0.255), Physics_brake: 0.8,
            Physics_speed_kmh: 106 - (now - 1000) / 1000 * 3.6 }), now);
        expect(result.events.map(({ sentence, timestamp }) => ({ sentence, timestamp })))
            .toEqual(now >= 1000 ? [{ sentence: 'More brake.', timestamp: 1000 }] : []);
    }
});

function throttleCornerSetup(direction: 'left' | 'right' = 'right', game: 'acc' | 'iracing' = 'acc') {
    const engine = new PhraseEngine(), lap = referenceLap();
    addSpeed(lap);
    lap.game = game;
    const sign = (direction === 'right' ? 1 : -1) * (game === 'iracing' ? -1 : 1);
    lap.y.forEach((row, i) => {
        const angle = i / 100 * 2 * Math.PI;
        row[0] = 100 * Math.cos(angle);
        row[1] = game === 'acc' ? sign * 100 * Math.sin(angle) : 0;
        row[2] = game === 'acc' ? 0 : sign * 100 * Math.sin(angle);
        row[3] = 0; // Corner direction must come from geometry, not steering.
        row[4] = 0;
        row[5] = 0.4;
    });
    engine.receiveMap(null, 0, lap);
    const update = (lateral: number, now: number, patch: StandardTelemetrySample = {}, capturedAt = now) => {
        const sample = { ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 0.4, ...patch };
        const view = detection(lap, sample, capturedAt), event = frame(sample);
        view.position!.referenceDelta[0] = lateral;
        view.telemetry!.game = event.update.game = game;
        engine.receiveVision(view, now);
        return engine.receiveTelemetry(event, now);
    };
    return { engine, lap, update };
}

it.each([-20, 0, 20])('uses normalized reference speed and model lateral movement for Less throttle with forward offset %s', (forward) => {
    const { lap } = throttleCornerSetup();
    const sample = { ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 0.4 };
    const view = detection(lap, sample, 1100);
    view.position!.referenceDelta = [5, forward, 0];
    const previous = { capturedAt: 1000, lateralOffset: 4 };
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBe(2600);
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBeUndefined();
    view.capturedAt = 500;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBeUndefined();
});

it.each([
    ['acc', 'left', -1], ['acc', 'right', 1], ['iracing', 'left', -1], ['iracing', 'right', 1],
] as const)('requires at least 5 units of outward line error in a %s %s corner', (game, direction, sign) => {
    for (const offset of [4.99, 5, 5.01]) {
        const { engine, update } = throttleCornerSetup(direction, game);
        expect(update(4 * sign, 1000).events).toEqual([]);
        const result = update(offset * sign, 1100);
        expect(result.events.map(({ sentence }) => sentence)).toEqual(offset >= 5 ? ['Less throttle.'] : []);
        expect(result.state.path).toEqual(offset >= 5
            ? ['root', 'Throttle zone', 'Less throttle', 'say phrase']
            : ['root', 'Throttle zone']);
        expect(engine.evaluate(1101).state.path).toEqual(['root', 'Throttle zone']);
    }
});

it.each([
    ['matching speed', 4, 5, 102, 1, 0],
    ['slower than reference', 4, 5, 94, 1, 0],
    ['speed within tolerance', 4, 5, 106.9, 0.4, 0],
    ['holding an outside line', 5, 5, 110, 0.4, 0],
    ['returning from outside', 6, 5, 110, 0.4, 0],
    ['moving inward', -4, -5, 110, 0.4, 0],
    ['returning from inside', -6, -5, 110, 0.4, 0],
    ['no applied throttle', 4, 5, 110, 0, 0],
    ['brake applied', 4, 5, 110, 0.4, 0.3],
] as const)('withholds Less throttle when %s', (_name, before, after, speed, throttle, brake) => {
    const { update } = throttleCornerSetup();
    const patch = { Physics_speed_kmh: speed, Physics_gas: throttle, Physics_brake: brake };
    update(before, 1000, patch);
    expect(update(after, 1100, patch).events.some(({ sentence }) => sentence === 'Less throttle.')).toBe(false);
});

it.each(['gap', 'expired capture', 'repeated capture', 'out-of-order capture', 'pause', 'stream reset', 'reference change'])(
    'requires fresh outward movement after %s', (kind) => {
        const { engine, lap, update } = throttleCornerSetup();
        update(4, 1000);
        if (kind === 'pause') engine.receiveTelemetry({ ...frame(driving(0.655)), telemetryStatus: 1 }, 1050);
        if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1050);
        if (kind === 'reference change') engine.receiveMap(null, 1050, { ...lap });
        const now = kind === 'gap' || kind === 'expired capture' ? 1601 : 1100;
        const capturedAt = kind === 'expired capture' || kind === 'repeated capture' ? 1000
            : kind === 'out-of-order capture' ? 999 : now;
        expect(update(5, now, {}, capturedAt).events).toEqual([]);
    });

it('uses the interpolated local speed threshold even when the applied throttle is below reference demand', () => {
    const { update } = throttleCornerSetup();
    const patch = { Physics_speed_kmh: 107.1, Physics_gas: 0.15 };
    update(4, 1000, patch);
    expect(update(5, 1100, patch).events.map(({ sentence }) => sentence)).toEqual(['Less throttle.']);
});

it('compares outward overspeed with the cautious local reference instead of the source lap', () => {
    const { engine, update } = throttleCornerSetup();
    engine.setDrivingAggression(50, 0);
    const patch = { Physics_speed_kmh: 90 }; // Local target is 84; the unadjusted source target is 102.
    update(4, 1000, patch);
    expect(update(5, 1100, patch).events.map(({ sentence }) => sentence)).toEqual(['Less throttle.']);
});

it.each(['straight', 'missing speed', 'invalid speed', 'missing throttle', 'brake zone'])(
    'withholds outward-drift throttle correction with %s', (kind) => {
        const { lap } = throttleCornerSetup();
        if (kind === 'straight') lap.y.forEach((row, i) => { row[0] = i * 10; row[1] = 0; });
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[65][6] = NaN;
        if (kind === 'missing throttle') lap.target_features[5] = 'missing';
        if (kind === 'brake zone') lap.y.forEach((row) => { row[4] = 0.8; });
        const sample = { ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 1 };
        const view = detection(lap, sample, 1100);
        view.position!.referenceDelta[0] = 5;
        const guidance = readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined,
            { capturedAt: 1000, lateralOffset: 4 });
        expect(guidance.lessThrottle).toBeUndefined();
    });

it.each([false, true])('uses local curvature across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const { lap } = throttleCornerSetup();
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const position of [0, 0.005, 0.995, 1]) {
        const sample = { ...driving(position), Physics_speed_kmh: 130, Physics_gas: 0.4 };
        const view = detection(lap, sample, 1100);
        view.position!.referenceDelta[0] = 5;
        expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined,
            { capturedAt: 1000, lateralOffset: 4 }).lessThrottle).toBe(2600);
    }
});

function steeringMap(sign = 1, game: 'acc' | 'iracing' = 'acc'): CircuitMapDto {
    const map = circuitMap();
    map.game = game;
    map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point,
        normalized_position: (point.normalized_position + 0.15) % 1,
        y: game === 'acc' ? point.y * sign : 0, z: game === 'iracing' ? point.y * sign : 0,
    })).sort((a, b) => a.normalized_position - b.normalized_position);
    map.centerline_tags = map.centerline_tags!.map((tag) => ({ ...tag,
        start_position: (tag.start_position + 0.15) % 1, end_position: (tag.end_position + 0.15) % 1,
    }));
    return map;
}

function brakingTurn(lap: TopLapReferenceEntry) {
    addSpeed(lap);
    lap.y.forEach((row, i) => {
        row[3] = i >= 25 && i <= 28 ? (i - 25) * 0.02 : i >= 29 && i <= 40 ? 0.4 + (i - 29) * 0.04 : 0;
        // Steering starts at 250 m; 3 m/s² braking eases to 1 m/s² from 260 m before turn-in near 281 m.
        row[6] = Math.sqrt(100 ** 2 / 3.6 ** 2 - 6 * Math.max(0, Math.min(60, i * 10 - 200))
            - 2 * Math.max(0, Math.min(40, i * 10 - 260))) * 3.6;
    });
}

it.each([
    [0.255, 'prepare', 0, 0], [0.255, 'prepare', 0, -1],
    [0.285, 'turn', 0, 0], [0.285, 'turn', 0, -1],
    [0.255, 'prepare', 50, 0], [0.255, 'prepare', 50, -1],
    [0.285, 'turn', 50, 0], [0.285, 'turn', 50, -1],
] as const)('gates Steering at %s in phase %s using %s%% caution and speed offset %s', (position, phase, caution, offset) => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    engine.setDrivingAggression(caution, 0);
    const adjusted = copyPhraseReference(lap, caution)!;
    const before = adjusted.x.findIndex((point) => point === (position === 0.255 ? 0.25 : 0.28));
    // These positions are midway between reference samples on a decelerating interval.
    const target = Math.sqrt((adjusted.y[before][6] ** 2 + adjusted.y[before + 1][6] ** 2) / 2);
    const at = (speed: number, now: number) => engine.receiveTelemetry(frame({
        ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
    }), now);
    for (const [excess, now] of [[1, 1000], [0.000001, 1100]]) {
        const blocked = at(target + excess, now);
        expect(blocked.referenceSteeringPhase).toBe(phase);
        expect(blocked.state.path).toEqual(['root', 'Brake zone']);
        expect(blocked.events).toEqual([]);
        expect(blocked.closures[0].conditions.find((condition) => 'input' in condition && condition.input === 'speedAtOrBelowReference'))
            .toMatchObject({ conditionFit: false, inputMissing: false });
    }
    const entered = at(target + offset, 1200);
    expect(entered.state.path).toContain('Steering');
    expect(entered.events.map(({ sentence }) => sentence)).toEqual(phase === 'turn' ? ['Turn'] : []);
    expect(entered.closures[0].conditions.find((condition) => 'input' in condition && condition.input === 'speedAtOrBelowReference'))
        .toMatchObject({ conditionFit: true, inputMissing: false });
});

it.each(['missing', 'NaN', 'infinite', 'negative'])('withholds Steering entry with %s reference speed', (kind) => {
    const { update } = setup((lap) => {
        brakingTurn(lap);
        if (kind === 'missing') lap.target_features[6] = 'missing';
        else lap.y[25][6] = kind === 'NaN' ? NaN : kind === 'infinite' ? Infinity : -1;
    }, steeringMap());
    const snapshot = update(0.25, 1000, { Physics_brake: 0.8 });
    expect(snapshot.state.path).toEqual(['root', 'Brake zone']);
    expect(snapshot.events).toEqual([]);
    expect(snapshot.closures.find(({ id }) => id === 'reference-steering')!.conditions
        .find((condition) => 'input' in condition && condition.input === 'speedAtOrBelowReference'))
        .toMatchObject({ conditionFit: false, inputMissing: true });
});

it('checks reference speed only on Steering entry and keeps Turn available after speed rises', () => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = new PhraseEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    const entered = engine.receiveTelemetry(frame({ ...driving(0.25), Physics_speed_kmh: lap.y[25][6] }), 1000);
    expect(entered.state.path).toEqual(['root', 'Brake zone', 'Steering']);
    const turned = engine.receiveTelemetry(frame({ ...driving(0.285), Physics_speed_kmh: lap.y[28][6] + 10 }), 1100);
    expect(turned.state.path).toEqual(['root', 'Brake zone', 'Steering', 'say phrase']);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Turn']);
});

it('enters Steering silently, speaks Turn once, and stays entered until the brake zone ends', () => {
    const { engine, lap, update } = setup(brakingTurn, steeringMap());
    const at = (position: number, now: number, speed: number) => update(position, now, {
        Physics_speed_kmh: speed - 1, Physics_brake: 0.8, Physics_steer_angle: -0.5,
    });
    const approaching = at(0.22, 900, lap.y[22][6]);
    expect(approaching.state.path).toEqual(['root', 'Brake zone', 'Ease the brake', 'say phrase']);
    expect(approaching.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.']);
    const prepared = at(0.25, 1000, lap.y[25][6]);
    expect(prepared.state.path).toEqual(['root', 'Brake zone', 'Steering']);
    expect(prepared.events).toEqual(approaching.events);
    const eased = at(0.26, 2100, lap.y[26][6]);
    expect(eased.events).toEqual(approaching.events);
    expect(eased.state.path).toContain('Steering');
    expect(at(0.26, 2300, lap.y[26][6]).events).toEqual(approaching.events);
    const turned = at(0.283, 2500, lap.y[28][6]);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.', 'Turn']);
    expect(turned.events[1].steeringPhase).toBe('turn');
    expect(turned.state.path).toEqual(['root', 'Brake zone', 'Steering', 'say phrase']);
    expect(at(0.285, 5500, lap.y[28][6]).events).toHaveLength(2);
    expect(engine.evaluate(5501).state.path).toEqual(['root', 'Brake zone', 'Steering', 'say phrase']);
    expect(at(0.299, 5600, lap.y[29][6]).state.path).toContain('Steering');
    const exited = at(0.2999, 5700, lap.y[29][6]);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toHaveLength(2);
    at(0.283, 8000, lap.y[28][6]);
    expect(at(0.283, 8100, lap.y[28][6]).events.filter(({ sentence }) => sentence === 'Turn')).toHaveLength(2);
});

it.each([[0.25, 'prepare', 1200], [0.283, 'turn', 1100]] as const)(
    'does not delay Steering after brake corrections at %s in phase %s', (position, phase, turnAt) => {
        const { lap, update } = setup(brakingTurn, steeringMap());
        const brake = update(0.24, 1000, { Physics_speed_kmh: lap.y[24][6] }).events;
        expect(brake.map(({ sentence }) => sentence)).toEqual(['Brake.']);
        expect(update(0.24, 1050, { Physics_speed_kmh: lap.y[24][6], Physics_brake: 0.8 }).events
            .map(({ sentence }) => sentence)).toEqual(['Brake.', 'Ease the brake.']);

        const entered = update(position, 1100, {
            Physics_speed_kmh: lap.y[Math.floor(position * 100)][6] - 1, Physics_brake: 0.8,
        });
        expect(entered.referenceSteeringPhase).toBe(phase);
        expect(entered.state.path).toEqual(phase === 'prepare'
            ? ['root', 'Brake zone', 'Steering'] : ['root', 'Brake zone', 'Steering', 'say phrase']);
        expect(entered.events).toHaveLength(phase === 'prepare' ? 2 : 3);

        const turning = { Physics_speed_kmh: lap.y[28][6], Physics_brake: 0.8 };
        const turned = update(0.283, 1200, turning);
        expect(turned.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
            { sentence: 'Brake.', timestamp: 1000 }, { sentence: 'Ease the brake.', timestamp: 1050 },
            { sentence: 'Turn', timestamp: turnAt },
        ]);
        expect(turned.state.path).toEqual(['root', 'Brake zone', 'Steering', 'say phrase']);
        expect(update(0.283, 1300, turning).events).toEqual(turned.events);
    });

it('enters Steering when a window containing a reversal sums toward the corner', () => {
    const { update } = setup((lap) => {
        lap.target_features.push('expert_optimal_speed');
        lap.y.forEach((row) => row.push(81));
        [0.1, 0.2, 0.15].forEach((value, i) => { lap.y[25 + i][3] = value; });
    }, steeringMap());
    const snapshot = update(0.25, 1000, { Physics_brake: 0.8, Physics_speed_kmh: 81 });
    expect(snapshot.state.path).toContain('Brake zone');
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Turn']);
});

it('withholds Turn after preparation when the sum changes against the corner direction', () => {
    const { update } = setup((lap) => {
        lap.target_features.push('expert_optimal_speed');
        lap.y.forEach((row) => row.push(36));
        [0.02, 0.04, 0.06, 0.12, -1].forEach((value, i) => { lap.y[25 + i][3] = value; });
    }, steeringMap());
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toContain('Steering');
    const interrupted = update(0.275, 2100, { Physics_brake: 0.8 });
    expect(interrupted.referenceSteeringPhase).toBeUndefined();
    expect(interrupted.events).toEqual([]);
    expect(update(0.31, 2200).state.path).toEqual(['root']);
});

it.each([20, 36, 180])('prepares and turns with a matching steering sum at %s km/h', (speed) => {
    const lap = referenceLap();
    brakingTurn(lap);
    for (const [position, phase] of [[0.25, 'prepare'], [0.26, 'prepare'], [0.283, 'turn']] as const) {
        const sample = { ...driving(position), Physics_speed_kmh: speed };
        expect(steeringGuidance(sample, lap))
            .toMatchObject({ steering: 2500, steeringPhase: phase });
    }
});

it('does not enter Steering for easing deceleration before reference steering starts changing', () => {
    const { lap, update } = setup((lap) => {
        brakingTurn(lap);
        lap.y.forEach((row, i) => { row[3] = i >= 29 && i <= 35 ? 0.4 : 0; });
    }, steeringMap());
    const snapshot = update(0.26, 1000, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.3 });
    expect(snapshot.state.path).toEqual(['root', 'Brake zone', 'Ease the brake', 'say phrase']);
    expect(snapshot.referenceSteeringPhase).toBeUndefined();
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.']);
});

it('eases the brake at matching speed before continuing to reference turn-in', () => {
    const { lap, update } = setup(brakingTurn, steeringMap());
    expect(update(0.26, 1000, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.3 }).events
        .map(({ sentence }) => sentence)).toEqual(['Ease the brake.']);
    const prepared = update(0.27, 2100, { Physics_speed_kmh: lap.y[27][6], Physics_brake: 0.8 });
    expect(prepared.state.path).toEqual(['root', 'Brake zone', 'Steering']);
    expect(prepared.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.']);
    expect(update(0.283, 3200, { Physics_speed_kmh: lap.y[28][6], Physics_brake: 0.8 }).events
        .map(({ sentence }) => sentence)).toEqual(['Ease the brake.', 'Turn']);
});

it('continues nested steering speech with expired position-model data and exits at the normalized brake boundary', () => {
    const { engine, lap, update } = setup((lap) => {
        brakingTurn(lap);
        lap.y[34][5] = 0.8; // Throttle pickup inside the mapped corner (0.25–0.35).
    }, steeringMap());
    update(0.24, 900, { Physics_speed_kmh: lap.y[24][6], Physics_brake: 0.8 });
    update(0.25, 1000, { Physics_speed_kmh: lap.y[25][6], Physics_brake: 0.8 });
    const stale = engine.receiveTelemetry(frame({ ...driving(0.26), Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.8 }), 2100);
    expect(stale.state.path).toContain('Steering');
    expect(stale.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.']);
    const turned = engine.receiveTelemetry(frame({ ...driving(0.283), Physics_speed_kmh: lap.y[28][6], Physics_brake: 0.8 }), 2150);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Ease the brake.', 'Turn']);
    expect(turned.state.path).toContain('Steering');
    const exited = engine.receiveTelemetry(frame(driving(0.31)), 2200);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toHaveLength(2);
    expect(update(0.34, 3300).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
});

it.each(['missing speed', 'invalid speed', 'constant deceleration', 'increasing deceleration', 'missing steering'])(
    'prepares for turn-in independently of reference deceleration with %s', (kind) => {
        const lap = referenceLap();
        brakingTurn(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[26][6] = NaN;
        if (kind === 'missing steering') lap.target_features[3] = 'missing';
        if (kind === 'constant deceleration' || kind === 'increasing deceleration') {
            lap.y[27][6] = Math.sqrt(lap.y[26][6] ** 2 / 3.6 ** 2 - 2 * (kind === 'constant deceleration' ? 3 : 4) * 10) * 3.6;
        }
        const sample = { ...driving(0.26), Physics_speed_kmh: 80 };
        expect(steeringGuidance(sample, lap).steeringPhase)
            .toBe(kind === 'missing steering' ? undefined : 'prepare');
    });

it.each([false, true])('detects preparation and turn-in across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[98][6] = 100;
    lap.y[99][6] = 90;
    lap.y[0][6] = 88;
    lap.y[99][4] = lap.y[0][4] = 0.8;
    lap.y[0][3] = 0.02;
    lap.y[1][3] = 0.4;
    lap.y[2][3] = 0.5;
    lap.y[3][3] = 0.6;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const [position, phase] of [[0.995, 'prepare'], [0.003, 'turn']] as const) {
        const sample = driving(position);
        expect(steeringGuidance(sample, lap).steeringPhase).toBe(phase);
    }
});

it.each([false, true])('allows a matching sum containing a reversal across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    lap.y[99][3] = 0.1;
    lap.y[0][3] = 0.2;
    lap.y[1][3] = 0.1;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    expect(steeringGuidance(driving(0.995), lap).steeringPhase).toBe('turn');
});

it.each([
    [0.201, 'none'], [0.202, 'brake'], [0.298, 'brake'], [0.299, 'none'],
    [0.601, 'none'], [0.602, 'throttle'], [0.908, 'throttle'], [0.909, 'none'],
] as const)('uses interpolated pedal boundaries at %s for the %s zone', (position, zone) => {
    const lap = referenceLap();
    for (const pedal of [0, 0.8, 1]) {
        const sample = { ...driving(position), Physics_brake: pedal, Physics_gas: pedal };
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe(zone);
    }
});

it.each([['brake', 4], ['throttle', 5]] as const)('keeps the reference %s zone across start/finish', (zone, channel) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[4] = 0; row[5] = 0; row[channel] = i >= 98 || i <= 1 ? 0.8 : 0; });
    for (const position of [0.985, 0.999, 0, 0.005, 0.015]) {
        const sample = driving(position);
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe(zone);
    }
    const sample = driving(0.019);
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe('none');
});

it('reads zone membership from normalized position in the private aggression-adjusted reference', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = driving(0.21), view = detection(lap, sample);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).pedalZone).toBe('brake');
    const cautious = copyPhraseReference(lap, 50)!;
    expect(readReferenceGuidance(sample, view, cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    for (const [position, zone] of [[0.983, 'none'], [0.985, 'brake'], [0.201, 'brake'], [0.295, 'brake'], [0.297, 'none']] as const) {
        const braking = { ...driving(position), Physics_speed_kmh: 160 };
        expect(readReferenceGuidance(braking, detection(lap, braking), cautious, 'acc', 1000, lap).pedalZone).toBe(zone);
    }
    for (const speed of [36, 84, 160]) {
        const throttle = { ...driving(0.61), Physics_speed_kmh: speed };
        const prediction = detection(lap, throttle);
        expect(readReferenceGuidance(throttle, prediction, lap, 'acc', 1000).pedalZone).toBe('throttle');
        expect(readReferenceGuidance(throttle, prediction, cautious, 'acc', 1000, lap).pedalZone).toBe('throttle');
    }
});

it.each([[0, 0.20125, 0.29875], [25, 0.12225, 0.29775], [50, 0.984, 0.296]])(
    'uses stretched reference brake boundaries at %s%% caution independently of live speed and pedals', (caution, start, end) => {
        const lap = referenceLap();
        addSpeed(lap);
        const cautious = copyPhraseReference(lap, caution)!;
        for (const speed of [36, 120, 160]) {
            for (const pedal of [0, 0.8, undefined]) {
                for (const [position, zone] of [[start - 0.00001, 'none'], [start + 0.00001, 'brake'],
                    [0.25, 'brake'], [end - 0.00001, 'brake'], [end + 0.00001, 'none'], [0.3, 'none']] as const) {
                    const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: pedal, Physics_gas: pedal };
                    const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap);
                    expect(guidance).toMatchObject(zone === 'brake' ? { pedalZone: zone, braking: true } : { pedalZone: zone });
                }
            }
        }
        // Position-model forward error must not shift the local brake boundary.
        const sample = { ...driving(0.202), Physics_speed_kmh: 120 }, view = detection(lap, sample);
        view.position!.referenceDelta[1] = 5;
        expect(readReferenceGuidance(sample, view, cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    });

it.each([0, 0.099, 0.1, 0.101])('uses only local brake pressure and response at pressure %s', (pressure) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, 50)!;
    local.y.forEach((row) => { row[4] = pressure; });
    const sample = { ...driving(0.19), Physics_speed_kmh: 160 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), local, 'acc', 1000, lap);
    // A valid original response cannot replace an absent response in the active lap.
    expect(guidance.brake).toBe(pressure >= 0.1 ? 2500 : undefined);
    expect(guidance.pedalZone).toBe(pressure >= 0.1 ? 'brake' : 'none');
    expect(lap.y[19][4]).toBe(0);
});

it.each(['missing speed', 'invalid speed', 'missing brake', 'invalid brake', 'no brake response'])(
    'does not invent an earlier brake zone with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[0][6] = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing';
        if (kind === 'invalid brake') lap.y[0][4] = NaN;
        if (kind === 'no brake response') lap.y.forEach((row) => { row[4] = 0; });
        const cautious = copyPhraseReference(lap, 50)!;
        const sample = { ...driving(0.19), Physics_speed_kmh: 160 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap);
        expect(guidance.pedalZone).toBe(kind.includes('brake') && kind !== 'no brake response' ? undefined : 'none');
        expect(guidance.brake).toBeUndefined();
    });

it('stays quiet in the throttle zone when overspeed calls for releasing throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.655, 1000, { Physics_speed_kmh: 102, Physics_gas: 0.8 }).state.path)
        .toEqual(['root', 'Throttle zone']);
    const correction = update(0.655, 2200, { Physics_speed_kmh: 120, Physics_gas: 0.8 });
    expect(correction.events).toEqual([]);
    expect(correction.state.path).toEqual(['root', 'Throttle zone']);
    expect(correction.closures.find(({ id }) => id === 'reference-brake')!.conditions.some((condition) => !condition.conditionFit)).toBe(true);
    expect(engine.evaluate(2201).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.655, 3400, { Physics_speed_kmh: 120, Physics_gas: 0 }).state.path)
        .toEqual(['root', 'Throttle zone']);
    expect(update(0.95, 3500, { Physics_speed_kmh: 120 }).state.path).toEqual(['root', 'Throttle zone']);
});

it('keeps throttle past the reference boundary until corner exit, but exits brake at its reference boundary', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.95 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.908, 1000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const afterReference = update(0.909, 1100);
    expect(afterReference.state.path).toEqual(['root', 'Throttle zone']);
    expect(afterReference.events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    const throttle = afterReference.root.children
        .find(({ id }) => id === 'reference-throttle-zone')!;
    expect(throttle.children[0].conditions).toEqual([
        expect.objectContaining({ description: 'The driver has left the corner', conditionFit: false }),
    ]);
    expect(update(0.951, 1150).state.path).toEqual(['root']);
    const braking = update(0.298, 1200);
    expect(braking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(braking.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Throttle.', timestamp: 1000 }, { sentence: 'Brake.', timestamp: 1200 },
    ]);
    expect(update(0.299, 1300).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.2999, 1400).state.path).toEqual(['root']);
});

it.each(['straight', 'long straight', 'untagged'])('ends throttle at the mapped corner exit before %s, even during speech cooldown', (label) => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 },
        ...(label === 'untagged' ? [] : [{ id: 'after', tags: [label], start_position: 0.7, end_position: 0.9 }]),
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.699, 1000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const exited = update(0.701, 1100);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    expect(exited.closures.find(({ id }) => id === 'reference-throttle')!.conditions[1].conditionFit).toBe(false);
    expect(update(0.8, 4000).state.path).toEqual(['root']);
    expect(engine.evaluate(4001).events).toHaveLength(1);
});

it('exits throttle at the mapped corner end with fresh telemetry even when reference location expires', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    update(0.699, 1000, { Physics_gas: 0.8 });
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Throttle zone']);
    expect(engine.receiveTelemetry(frame(driving(0.701)), 1600).state.path).toEqual(['root']);
});

it('keeps the brake zone before mapped corner entry when reference location expires', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Brake zone']);
});

it('caps a throttle zone at a mapped corner exit across start/finish', () => {
    const { engine, lap, update } = setup((reference) => {
        reference.y.forEach((row) => { row[4] = 0; row[5] = 0.8; });
    });
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['fast corner'], start_position: 0.98, end_position: 0.01 }];
    engine.receiveMap(map, 0, lap);
    [0.985, 0.999, 0.001, 0.009].forEach((position, index) => {
        expect(update(position, 1000 + index * 100, { Physics_gas: 0.8 }).state.path)
            .toEqual(['root', 'Throttle zone']);
    });
    expect(update(0.011, 1400, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
});

it('returns from the brake zone to movement guidance when throttle pickup is beyond the mapped corner', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.5 },
        { id: 'after', tags: ['straight'], start_position: 0.5, end_position: 0.9 },
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toContain('Brake zone');
    expect(update(0.4, 2000).state.path).toEqual(['root']);
    expect(update(0.7, 3000).state.path).toEqual(['root']);
    const sample = driving(0.701), view = detection(lap, sample, 4000);
    view.position!.referenceDelta[0] = -4;
    engine.receiveVision(view, 4000);
    const movement = engine.receiveTelemetry(frame(sample), 4000);
    expect(movement.state.path).toContain('Turn to reference line');
    expect(movement.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it.each(['no tags', 'wrong track', 'wrong game'])('waits for corner-exit evidence when the map has %s', (kind) => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 }];
    if (kind === 'no tags') map.centerline_segments = [];
    if (kind === 'wrong track') map.source_track_key = 'other';
    if (kind === 'wrong game') map.game = 'iracing';
    engine.receiveMap(map, 0, lap);
    expect(update(0.8, 1000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.95, 1100, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.25, 1200, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
});

it('keeps zone ownership through missing captures and switches only after fresh boundary evidence', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.9 }];
    engine.receiveMap(map, 0, lap);
    update(0.7, 1000);
    const stale = engine.evaluate(1501);
    expect(stale.state.path).toEqual(['root', 'Throttle zone']);
    expect(stale.events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    expect(update(0.7, 2000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.25, 2100, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
});

it('keeps a throttle closure selected across start/finish until the corner exit', () => {
    const { engine, lap, update } = setup((lap) => {
        lap.y.forEach((row, i) => { row[4] = 0; row[5] = i >= 98 || i <= 1 ? 0.8 : 0; });
    });
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.98, end_position: 0.03 }];
    engine.receiveMap(map, 0, lap);
    [0.985, 0.999, 0.001, 0.015].forEach((position, index) => {
        expect(update(position, 1000 + index * 100, { Physics_gas: position === 0.015 ? 0.4 : 0.8 }).state.path)
            .toEqual(['root', 'Throttle zone']);
    });
    expect(update(0.019, 1400, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.031, 1500, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
});

it('gives overlapping interpolated pedal zones to the brake closure', () => {
    const { update } = setup((lap) => lap.y.forEach((row) => { row[5] = 0.8; }));
    const result = update(0.25, 1000, { Physics_brake: 0.8 });
    expect(result.state.path).toEqual(['root', 'Brake zone']);
    expect(result.events).toEqual([]);
});

it('transitions through all four brake phrases as speed, remaining profile and applied pressure change', () => {
    const { engine, update } = setup((lap) => { addSpeed(lap); lap.y[29][6] = 80; });
    const expected: string[] = [];
    ([[0.255, 98, 0, 0, 'Brake.'], [0.255, 106, 0.8, 1, 'More brake.'],
        [0.255, 98, 0.8, 0, 'Ease the brake.'], [0.295, 80, 0.8, 0, 'Release the brake.']] as const)
        .forEach(([position, speed, pedal, deceleration, sentence], index) => {
            const now = 1000 + index * 2000;
            if (sentence === 'More brake.') update(position, now - 100, { Physics_speed_kmh: speed + deceleration * 3.6 * 0.1, Physics_brake: pedal });
            const snapshot = update(position, now, { Physics_speed_kmh: speed, Physics_brake: pedal });
            expected.push(sentence);
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.events[index].expiresAt).toBe(now + 1500);
            expect(snapshot.state.path).toContain('Brake zone');
            expect(engine.evaluate(now + 1).state.path).toEqual(['root', 'Brake zone']);
        });
});

it.each([['More brake', 106, 1]] as const)(
    'says %s on the first qualifying pair of consecutive telemetry frames without a hold', (name, speed, deceleration) => {
        const { engine, update } = setup(addSpeed);
        const patch = { Physics_speed_kmh: speed, Physics_brake: 0.8 };
        expect(update(0.255, 1000, patch).events).toEqual([]);
        expect(engine.evaluate(1015).events).toEqual([]);
        const result = update(0.255, 1016, { ...patch, Physics_speed_kmh: speed - deceleration * 3.6 * 0.016 });
        expect(result.events).toEqual([
            expect.objectContaining({ sentence: `${name}.`, timestamp: 1016 }),
        ]);
        expect(result.state.path).toEqual(['root', 'Brake zone', name, 'say phrase']);
    });

it.each([0, 25, 50])('says Ease the brake on the first matching frame of the active reference at %s%% caution', (caution) => {
    const { engine, lap, update } = setup(addSpeed);
    engine.setDrivingAggression(caution, 0);
    const local = copyPhraseReference(lap, caution)!;
    const speed = local.y[local.x.indexOf(0.25)][6];
    const result = update(0.25, 1000, { Physics_speed_kmh: speed, Physics_brake: 0.8 });
    expect(result.events).toEqual([
        expect.objectContaining({ ruleId: 'reference-easeBrake', sentence: 'Ease the brake.', timestamp: 1000, expiresAt: 2500 }),
    ]);
    expect(result.state.path).toEqual(['root', 'Brake zone', 'Ease the brake', 'say phrase']);
    expect(engine.evaluate(1001).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.25, 3500, { Physics_speed_kmh: speed, Physics_brake: 0.8 }).events).toEqual(result.events);
});

it.each(['gap', 'pause', 'stream reset', 'missing speed', 'invalid speed', 'repeated time', 'backward time', 'session change', 'stream change'])(
    'requires a new pair of speed samples after %s before correcting brake pressure', (kind) => {
        const { engine, lap, update } = setup(addSpeed);
        const patch = { Physics_speed_kmh: 106, Physics_brake: 0.8 };
        update(0.255, 1000, patch);
        const active = update(0.255, 1100, { ...patch, Physics_speed_kmh: 105.64 });
        expect(active.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[2].conditionFit).toBe(true);
        if (kind === 'pause') engine.receiveTelemetry({ ...frame({ ...driving(0.255), ...patch }), telemetryStatus: 1 }, 1200);
        if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1200);
        if (kind === 'missing speed' || kind === 'invalid speed') {
            update(0.255, 1200, { ...patch, Physics_speed_kmh: kind === 'missing speed' ? undefined : NaN });
        }
        const now = kind === 'gap' ? 2701 : kind === 'repeated time' ? 1100 : kind === 'backward time' ? 1099 : 1300;
        const sample = { ...driving(0.255), ...patch, Physics_speed_kmh: 105.28 };
        const event = frame(sample), view = detection(lap, sample, now);
        if (kind === 'session change') event.sessionGeneration = view.telemetry!.sessionGeneration = 1;
        if (kind === 'stream change') event.streamGeneration = view.telemetry!.streamGeneration = 1;
        engine.receiveVision(view, now);
        const result = engine.receiveTelemetry(event, now);
        const conditions = result.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions;
        expect(conditions[1].conditionFit).toBe(true);
        expect(conditions[2].conditionFit).toBe(false);
        expect(result.events).toEqual(kind === 'stream reset' ? [] : active.events);
    });

it('withholds pedal corrections outside reference pedal zones', () => {
    const { update } = setup(addSpeed);
    const result = update(0.45, 1000, { Physics_speed_kmh: 80, Physics_brake: 0.4 });
    expect(result.events).toEqual([]);
    expect(result.state.path).toEqual(['root']);
});

it('enters the throttle zone to apply and add throttle without a preceding braking zone', () => {
    const { engine, update } = setup(addSpeed);
    const expected: string[] = [];
    ([[0.655, 102, 0, 'Throttle.'], [0.655, 94, 0.8, 'More throttle.']] as const)
        .forEach(([position, speed, pedal, sentence], index) => {
            const now = 1000 + index * 1200;
            expected.push(sentence);
            const snapshot = update(position, now, { Physics_speed_kmh: speed, Physics_gas: pedal });
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.state.path).toEqual(['root', 'Throttle zone', sentence.slice(0, -1), 'say phrase']);
            expect(engine.evaluate(now + 1).state.path).toEqual(['root', 'Throttle zone']);
        });
});

it('gives only brake corrections in the brake zone, even when the driver applies throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.255, 1000, { Physics_speed_kmh: 98, Physics_gas: 0.8 }).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    engine.evaluate(1001);
    const snapshot = update(0.255, 2200, { Physics_speed_kmh: 98, Physics_gas: 0.8, Physics_brake: 0.8 });
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Brake.', 'Ease the brake.']);
    expect(snapshot.state.path).toEqual(['root', 'Brake zone', 'Ease the brake', 'say phrase']);
});

it('stays in the throttle zone after speech and while the driver matches the reference', () => {
    const { engine, update } = setup();
    update(0.7, 1000);
    expect(engine.evaluate(1001).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 1250).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 2250).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 2500, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(engine.evaluate(2501).events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
});

it('emits short, expiring cues without opponents, calibration or circuit tags and avoids repeats', () => {
    const { engine, update } = setup();
    const first = update(0.25, 1000);
    expect(first.visionReady).toBe(false);
    expect(first.events).toEqual([{ id: 1, ruleId: 'reference-brake', sentence: 'Brake.', timestamp: 1000, expiresAt: 2500 }]);
    expect(update(0.251, 1100).events).toHaveLength(1);
    expect(update(0.25, 1500).events).toHaveLength(1);
    expect(update(0.25, 2000).events.map((event) => event.sentence)).toEqual(['Brake.']);
    expect(update(0.25, 3000).events).toHaveLength(1);
    expect(engine.evaluate(3501).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Not matched', missing: [] });
    expect(engine.evaluate(4501).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Missing input' });
    update(0.3, 4600);
    expect(update(0.25, 5000).events).toHaveLength(2);
});

it('does not delay a different brake prompt after Brake', () => {
    const { update } = setup();
    expect(update(0.25, 1000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const released = update(0.299, 1100, { Physics_brake: 0.8 });
    expect(released.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Brake.', timestamp: 1000 }, { sentence: 'Release the brake.', timestamp: 1100 },
    ]);
});

it('speaks Brake only once per brake zone despite pedal changes and expired inputs', () => {
    const { engine, lap, update } = setup();
    const first = update(0.25, 1000).events[0];
    expect(first.sentence).toBe('Brake.');
    update(0.25, 1500, { Physics_brake: 0.8 });
    update(0.25, 2000, { Physics_brake: 0.8 });
    expect(update(0.25, 3100).events).toEqual([first]);
    expect(engine.evaluate(5000).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.25, 6000).events).toEqual([first]);
    engine.receiveMap(circuitMap(), 6100, lap);
    expect(update(0.25, 6200).events).toEqual([first]);
});

it('rearms Brake after leaving the zone even with a short gap before re-entry', () => {
    const { update } = setup();
    update(0.25, 1000);
    expect(update(0.25, 3000).events).toHaveLength(1);
    expect(update(0.4, 3100).state.path).toEqual(['root']);
    const reentered = update(0.25, 3200);
    expect(reentered.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(reentered.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Brake.', timestamp: 1000 },
        { sentence: 'Brake.', timestamp: 3200 },
    ]);
});

it.each([0.19, 0.39, 0.59])('stays at root outside reference pedal zones at %s', (position) => {
    const { update } = setup();
    const snapshot = update(position, 1000);
    expect(snapshot.events).toEqual([]);
    expect(snapshot.state.path).toEqual(['root']);
    for (const zone of ['brake', 'throttle']) {
        expect(snapshot.root.children.find(({ id }) => id === `reference-${zone}-zone`)!.conditions).toEqual([
            expect.objectContaining({ description: `Inside the interpolated reference ${zone} zone`, conditionFit: false, inputMissing: false }),
        ]);
    }
});

it.each([[0, 'Throttle', 'Throttle.'], [0.4, 'More throttle', 'More throttle.']] as const)(
    'returns to root between brake and throttle zones with pedal %s (%s)', (pedal, name, sentence) => {
        const { engine, lap, update } = setup();
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'straight', tags: ['straight'], start_position: 0, end_position: 0.99 },
            { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.8 },
        ];
        engine.receiveMap(map, 0, lap);
        const entered = update(0.25, 1000, { Physics_brake: 0.8 });
        expect(entered.events).toEqual([]);
        expect(entered.state.path).toEqual(['root', 'Brake zone']);
        expect(entered.root.children.find(({ id }) => id === 'reference-brake-zone')!.conditions[0].conditionFit).toBe(true);
        expect(update(0.3, 2000).state.path).toEqual(['root']);
        expect(update(0.39, 3000).state.path).toEqual(['root']);
        expect(engine.evaluate(3001).state.path).toEqual(['root']);
        // The corner takes precedence over an overlapping straight tag.
        const sample = driving(0.45), view = detection(lap, sample, 4000);
        view.position!.referenceDelta[0] = -4;
        engine.receiveVision(view, 4000);
        expect(engine.receiveTelemetry(frame(sample), 4000).state.path).toEqual(['root']);
        const acceleration = update(0.7, 5000, { Physics_gas: pedal });
        expect(acceleration.state.path).toEqual(['root', 'Throttle zone', name, 'say phrase']);
        expect(acceleration.events.map(({ sentence }) => sentence)).toEqual([sentence]);
        expect(engine.evaluate(5001).state.path).toEqual(['root', 'Throttle zone']);
        expect(update(0.7, 6000, { Physics_gas: pedal }).events).toHaveLength(1);
        expect(update(0.25, 7000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    });

it('waits until braking is needed inside the brake zone and returns to the same scope', () => {
    const { engine, update } = setup();
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.19, 2000).events).toEqual([]);
    const nextBraking = update(0.25, 3000);
    expect(nextBraking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(nextBraking.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(engine.evaluate(3001).state.path).toEqual(['root', 'Brake zone']);
});

it('exits the previous throttle phrase before entering the next pedal correction', () => {
    const { update } = setup();
    update(0.25, 1000, { Physics_brake: 0.8 });
    expect(update(0.7, 2000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const next = update(0.7, 3000, { Physics_gas: 0.4 });
    expect(next.state.path).toEqual(['root', 'Throttle zone', 'More throttle', 'say phrase']);
    expect(next.events.map(({ sentence }) => sentence)).toEqual(['Throttle.', 'More throttle.']);
});

it('withholds reference guidance while paused and resumes from telemetry after reference changes', () => {
    const { engine, lap, update } = setup();
    update(0.19, 1000);
    engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1100);
    expect(engine.receiveTelemetry(frame(driving(0.19)), 1200).events).toEqual([]);
    const sample = driving(0.19);
    const view = detection(lap, sample, 1300);
    view.telemetry!.streamGeneration = 9;
    engine.receiveVision(view, 1300);
    expect(engine.receiveTelemetry(frame(sample), 1300).events).toEqual([]);
    update(0.19, 2000);
    expect(engine.receiveTelemetry({ ...frame(sample), telemetryStatus: 1 }, 2100).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Missing input' });
    engine.receiveMap(null, 2200, referenceLap());
    expect(engine.receiveTelemetry(frame(driving(0.25)), 2200).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it.each([0, 0.8])('preempts a waiting overtaking closure at reference braking with driver brake %s', (brake) => {
    const { engine, lap } = setup();
    const map = circuitMap();
    engine.receiveMap(map, 0, lap);
    const sample = { ...driving(0.05), Graphics_player_car_id: 1, Graphics_normalized_positions: { 1: 0.05, 2: 0.06 } };
    engine.receiveVision(trafficVision(0), 0);
    const waiting = engine.receiveTelemetry(frame(sample), 0);
    expect(waiting.state.path).toContain('One Tight Slow Corner');
    const braking = { ...driving(0.25), Physics_brake: brake };
    engine.receiveVision(detection(lap, braking, 1000), 1000);
    const result = engine.receiveTelemetry(frame(braking), 1000);
    expect(result.state.path).toContain('Brake zone');
    expect(result.events.map(({ sentence }) => sentence)).toEqual(brake ? [] : ['Brake.']);
});

function movementSetup(map: CircuitMapDto | null = circuitMap(), configure?: (lap: TopLapReferenceEntry) => void) {
    const engine = new PhraseEngine(), lap = referenceLap();
    // Steering and throttle channels are optional; reference braking must be known.
    lap.target_features[3] = 'missing steering';
    lap.target_features[5] = 'missing throttle';
    configure?.(lap);
    engine.receiveMap(map, 0, lap);
    const update = (lateral: number, now: number, position = 0.45, patch: StandardTelemetrySample = {}) => {
        const sample = { ...driving(position), Physics_steer_angle: undefined, ...patch };
        const view = detection(lap, sample, now);
        view.position!.referenceDelta[0] = lateral;
        engine.receiveVision(view, now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    return { engine, lap, update };
}

it.each(['straight', 'long straight'])('allows movements only inside the Live Map %s bounds', (label) => {
    const map = circuitMap();
    map.centerline_segments = [{ id: 'straight', tags: [label], start_position: 0.4, end_position: 0.8 }];
    expect(movementSetup(map).update(-4, 1000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
    expect(movementSetup(map).update(-4, 1000, 0.35).events).toEqual([]);
});

it.each(['no map', 'no tags', 'wrong track', 'wrong game', 'corner', 'missing brake', 'invalid brake', 'braking'])(
    'withholds movement with %s even when the position model requests it', (kind) => {
        const map = circuitMap();
        if (kind === 'no tags') map.centerline_segments = [];
        if (kind === 'wrong track') map.source_track_key = 'other';
        if (kind === 'wrong game') map.game = 'iracing';
        if (kind === 'corner') map.centerline_tags!.push({ id: 'overlap', label: 'corner', start_position: 0.4, end_position: 0.5 });
        const { update } = movementSetup(kind === 'no map' ? null : map, (lap) => {
            if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
            if (kind === 'invalid brake') lap.y[0][4] = NaN;
            if (kind === 'braking') lap.y.forEach((row) => { row[4] = 0.8; });
        });
        const snapshot = update(-4, 1000, 0.45, { Physics_brake: kind === 'braking' ? 0.8 : 0 });
        expect(snapshot.events).toEqual([]);
        expect(snapshot.state.path).not.toContain('Turn to reference line');
    });

it('interrupts movement at reference brake onset and resumes movement after the brake zone ends', () => {
    const { engine, update } = movementSetup(circuitMap(), (lap) => {
        lap.y.forEach((row, index) => { row[4] = index >= 46 && index < 50 ? 0.8 : 0; });
    });
    expect(update(-4, 1000).state.path).toContain('Turn to reference line');
    // Braking is imminent, but the reference brake zone has not started yet.
    for (let now = 1100; now <= 1400; now += 100) {
        expect(update(-4, now, 0.451).state.path).toContain('Turn to reference line');
    }
    expect(engine.evaluate(1400).state.path).toEqual(['root', 'Turn to reference line']);
    const braking = update(-4, 1500, 0.452, { Physics_brake: 0.8 });
    expect(braking.state.path).toEqual(['root', 'Brake zone']);
    expect(braking.events).toHaveLength(1);
    expect(braking.closures.find(({ id }) => id === 'reference-move-left-small')!.conditions.every((condition) => condition.conditionFit)).toBe(false);
    for (let now = 1750; now <= 3500; now += 250) update(-0.1, now, 0.47, { Physics_brake: 0.8 });
    expect(engine.evaluate(3500).events).toHaveLength(1);
    const exited = update(-0.1, 3750, 0.55);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toHaveLength(1);
    const afterBraking = update(4, 4000, 0.55);
    expect(afterBraking.state.path).toEqual(['root', 'Turn to reference line', 'Move to the right', 'Small movement', 'say phrase']);
    expect(afterBraking.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.', 'Move to the right.']);
});

it('exits a selected movement speech action at brake onset inside its speech cooldown', () => {
    const { update } = movementSetup(circuitMap(), (lap) => {
        lap.y.forEach((row, index) => { row[4] = index >= 46 && index < 50 ? 0.8 : 0; });
    });
    expect(update(-4, 1000, 0.451).state.path).toEqual([
        'root', 'Turn to reference line', 'Move to the left', 'Small movement', 'say phrase',
    ]);
    const braking = update(-4, 1100, 0.4513, { Physics_brake: 0.8 });
    expect(braking.state.path).toEqual(['root', 'Brake zone']);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it.each(['left straight', 'map removed', 'vision expired'])('exits a nested movement closure and clears feedback when %s', (kind) => {
    const { engine, lap, update } = movementSetup();
    expect(update(-4, 1000).state.path).toContain('Small movement');
    const exited = kind === 'left straight' ? update(-4, 1250, 0.85)
        : kind === 'map removed' ? engine.receiveMap(null, 1250, lap) : engine.evaluate(1501);
    expect(exited.state.path).toEqual(['root']);
    engine.receiveMap(circuitMap(), 1750, lap);
    expect(update(-0.1, 2000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it.each([
    [-3, 'left', 'Small movement', 'Move to the left.'],
    [-6.99, 'left', 'Small movement', 'Move to the left.'],
    [-7, 'left', 'Medium movement', 'Further to the left.'],
    [-13, 'left', 'Medium movement', 'Further to the left.'],
    [3, 'right', 'Small movement', 'Move to the right.'],
    [6.99, 'right', 'Small movement', 'Move to the right.'],
    [7, 'right', 'Medium movement', 'Further to the right.'],
    [8.01, 'right', 'Medium movement', 'Further to the right.'],
    [13, 'right', 'Medium movement', 'Further to the right.'],
] as const)('selects movement from signed offset %s, without a steering channel', (offset, direction, level, sentence) => {
    const { update } = movementSetup();
    const snapshot = update(offset, 1000);
    expect(snapshot.events).toEqual([expect.objectContaining({ sentence, timestamp: 1000, expiresAt: 2500 })]);
    expect(snapshot.state.path).toEqual(['root', 'Turn to reference line', `Move to the ${direction}`, level, 'say phrase']);
});

it.each([0, -0.35, 0.35, -0.6, 0.6, -1.5, 1.5, -2.99, 2.99, -13.01, 13.01, NaN])('withholds movement for a near-line or invalid offset %s', (offset) => {
    expect(movementSetup().update(offset, 1000).events).toEqual([]);
});

it.each([-1, 1])('uses alternative wordings for movement, progress and completion on side %s', (sign) => {
    const random = jest.spyOn(Math, 'random').mockReturnValue(0.5);
    const { update } = movementSetup();
    const direction = sign < 0 ? 'left' : 'right';
    update(sign * 4, 1000);
    for (let now = 1250; now <= 3000; now += 250) update(sign * (4 - (now - 1000) / 10000), now);
    update(sign * 1.4, 3250);
    update(sign, 3500);
    update(sign * 0.6, 3750);
    const snapshot = update(sign * 0.3, 4000);
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual([`Shift to the ${direction}.`, 'A bit more.', "That's enough."]);
    expect(snapshot.events.map(({ ruleId }) => ruleId)).toEqual([
        `reference-move-${direction}-small`, `reference-move-${direction}-more`, `reference-move-${direction}-enough`,
    ]);
    expect(random).toHaveBeenCalledTimes(3);
});

it.each([-1, 1])('nudges slow progress and confirms enough movement on side %s', (sign) => {
    const { engine, update } = movementSetup();
    const direction = sign < 0 ? 'left' : 'right';
    update(sign * 4, 1000);
    for (let now = 1250; now <= 3000; now += 250) update(sign * (4 - (now - 1000) / 10000), now);
    expect(engine.evaluate(3000).events.map(({ sentence }) => sentence)).toEqual([`Move to the ${direction}.`, 'A little bit more.']);
    update(sign * 1.4, 3250);
    update(sign, 3500);
    update(sign * 0.6, 3750);
    const enough = update(sign * 0.3, 4000);
    expect(enough.events.map(({ sentence }) => sentence)).toEqual([`Move to the ${direction}.`, 'A little bit more.', "OK, that's enough."]);
    expect(enough.state.path).toContain('Enough movement');
    for (let now = 4250; now <= 7000; now += 250) update(sign * 0.2, now);
    expect(engine.evaluate(7000).events).toHaveLength(3);
});

it.each([0, 1.25, -0.1])('does not request more for stopped, adequate or wrong-way progress (%s units/s)', (rate) => {
    const { engine, update } = movementSetup();
    update(4, 1000);
    for (let now = 1250; now <= 3250; now += 250) update(4 - rate * (now - 1000) / 1000, now);
    expect(engine.evaluate(3250).events.map(({ sentence }) => sentence)).toEqual(['Move to the right.']);
});

it('does not delay completion after a movement prompt', () => {
    const { update } = movementSetup();
    update(-4, 1000);
    const completed = update(-0.3, 1100);
    expect(completed.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Move to the left.', timestamp: 1000 }, { sentence: "OK, that's enough.", timestamp: 1100 },
    ]);
});

it('confirms crossing the line before issuing an opposite-side correction', () => {
    const { update } = movementSetup();
    update(-4, 1000);
    for (let now = 1250; now <= 1750; now += 250) update(-4, now);
    expect(update(0.8, 2000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.', "OK, that's enough."]);
    for (let now = 2250; now <= 4000; now += 250) {
        expect(update(0.8, now).events).toHaveLength(2);
    }
    expect(update(3, 4250).events.map(({ sentence }) => sentence)).toEqual([
        'Move to the left.', "OK, that's enough.", 'Move to the right.',
    ]);
});

it.each(['capture gap', 'invalid position', 'pause', 'stream reset', 'reference change'])('clears movement feedback after %s', (kind) => {
    const { engine, update } = movementSetup();
    update(-4, 1000);
    if (kind === 'invalid position') update(NaN, 1250);
    if (kind === 'pause') engine.receiveTelemetry({ ...frame(driving(0.45)), telemetryStatus: 1 }, 1250);
    if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1250);
    if (kind === 'reference change') engine.receiveMap(null, 1250, referenceLap());
    const snapshot = update(0.1, kind === 'capture gap' ? 1600 : 1500);
    expect(snapshot.events.some(({ sentence }) => sentence === "OK, that's enough.")).toBe(false);
    expect(snapshot.events.some(({ sentence }) => sentence === 'A little bit more.')).toBe(false);
});

it('does not treat repeated or out-of-order captures or timer evaluations as movement', () => {
    const { engine, lap, update } = movementSetup();
    update(-4, 1000);
    const sample = driving(0.45);
    const view = detection(lap, sample, 1000);
    view.position!.referenceDelta[0] = 0;
    engine.receiveVision(view, 1250);
    engine.receiveTelemetry(frame(sample), 1250);
    engine.receiveVision({ ...view, capturedAt: 900 }, 1400);
    engine.receiveTelemetry(frame(sample), 1400);
    expect(engine.evaluate(1500).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
    expect(engine.evaluate(1501).closures.find(({ id }) => id === 'reference-move-left-small')?.status).toBe('Missing input');
});

it('keeps brake priority and does not confirm a movement that was never requested', () => {
    const { engine, lap } = setup();
    const sample = driving(0.25), view = detection(lap, sample, 1000);
    view.position!.referenceDelta[0] = -4;
    engine.receiveVision(view, 1000);
    expect(engine.receiveTelemetry(frame(sample), 1000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const near = detection(lap, sample, 1250);
    near.position!.referenceDelta[0] = -0.1;
    engine.receiveVision(near, 1250);
    engine.receiveTelemetry(frame(sample), 1250);
    expect(engine.evaluate(1250).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it('uses the private stretched brake copy for cues and restores the baseline when caution is reset', () => {
    const { engine, lap, update } = setup(addSpeed);
    const speed = { Physics_speed_kmh: 122 };
    expect(update(0.19, 1000, speed).events).toEqual([]);
    engine.setDrivingAggression(50, 1100);
    const beforeBrake = update(0.19, 1200, speed);
    expect(beforeBrake.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(beforeBrake.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(update(0.21, 1250, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(lap.y[19][4]).toBe(0);
    expect(lap.y[19][6]).toBe(120);
    engine.setDrivingAggression(25, 1300);
    expect(update(0.20, 1400, speed).state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(update(0.21, 1450, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    // Restoring Reference must use the original local baseline, not the stretched copy or a mutated source.
    lap.y.forEach((row) => { row[4] = 0; row[6] = 200; });
    engine.setDrivingAggression(0, 1500);
    expect(update(0.19, 1600, speed).state.path).toEqual(['root']);
    expect(update(0.25, 1650, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const view = detection(referenceLap(), driving(0.19), 1700);
    engine.receiveVision(view, 1700);
    expect(engine.receiveTelemetry(frame(driving(0.19)), 1700).closures.find(({ id }) => id === 'reference-brake'))
        .toMatchObject({ status: 'Not matched', missing: [] });
});

it.each([[0, 0.201, 0.202, 0.2998, 0.2999], [25, 0.122, 0.123, 0.2997, 0.2998], [50, 0.983, 0.985, 0.2995, 0.2997]])(
    'enters and exits the stretched brake zone at %s%% caution, speaking Brake once per entry', (caution, before, entry, last, exit) => {
        const { engine, lap, update } = setup(addSpeed);
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'approach', tags: ['straight'], start_position: 0.9, end_position: 0.2 },
            { id: 'turn', tags: ['corner'], start_position: 0.2, end_position: 0.7 },
        ];
        engine.receiveMap(map, 0, lap);
        const speed = { Physics_speed_kmh: 120 };
        expect(update(before, 1000, speed).state.path).toEqual(['root']);
        engine.setDrivingAggression(caution, 1100);
        const approach = update(before, 1200, speed);
        expect(approach.state.path).toEqual(['root']);
        expect(approach.events).toEqual([]);
        // Normalized telemetry alone can enter the earlier zone and speak.
        const entered = engine.receiveTelemetry(frame({ ...driving(entry), ...speed }), 1300);
        expect(entered.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        expect(entered.events.map(({ sentence, timestamp }) => ({ sentence, timestamp })))
            .toEqual([{ sentence: 'Brake.', timestamp: 1300 }]);
        expect(engine.evaluate(1301).state.path).toEqual(['root', 'Brake zone']);
        for (const [position, now] of [[caution === 50 ? 0.99 : 0.21, 2000], [0.25, 3400], [last, 4600]]) {
            const snapshot = update(position, now, speed);
            expect(snapshot.state.path).toEqual(['root', 'Brake zone']);
            expect(snapshot.events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
        }
        expect(update(exit, 4700, speed).state.path).toEqual(['root']);
        const next = update(entry, 6000, speed);
        expect(next.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        expect(next.events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(2);
        expect(lap.y[19][4]).toBe(0);
        expect(lap.y[19][6]).toBe(120);
    });

it('uses the lower cautious speed for throttle cues without delaying the acceleration zone', () => {
    const { engine, lap, update } = setup(addSpeed);
    expect(update(0.61, 1000, { Physics_speed_kmh: 84 }).events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    engine.setDrivingAggression(50, 1100);
    expect(update(0.61, 1200, { Physics_speed_kmh: 84 }).events).toEqual([]);
    expect(update(0.61, 1300, { Physics_speed_kmh: 48 }).events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    expect(lap.y[61][5]).toBe(0.8);
    engine.setDrivingAggression(0, 1400);
    expect(update(0.61, 1500, { Physics_speed_kmh: 84 }).events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
});

it('uses copied centerline tags and the lower speed target to end movement guidance', () => {
    const map = circuitMap();
    const { engine, update } = movementSetup(map, (lap) => {
        lap.y.forEach((row, index) => { row[4] = index >= 46 && index < 50 ? 0.8 : 0; });
        lap.target_features.push('expert_optimal_speed');
        lap.y.forEach((row, i) => row.push(i >= 46 && i < 50 ? 80 : 120));
    });
    map.centerline_tags!.find(({ id }) => id === 'straight')!.label = 'corner';
    expect(update(-4, 1000, 0.449, { Physics_speed_kmh: 120 }).state.path).toContain('Turn to reference line');
    engine.setDrivingAggression(50, 1100);
    const cautious = update(-4, 1200, 0.449, { Physics_speed_kmh: 120, Physics_brake: 0.5 });
    expect(cautious.state.path).toEqual(['root', 'Brake zone', 'More brake', 'say phrase']);
    expect(cautious.events).toEqual([
        expect.objectContaining({ sentence: 'More brake.', timestamp: 1200 }),
    ]);
});

it('uses a replacement reference before new position captures arrive and clears guidance when the reference is removed', () => {
    const { engine, lap, update } = setup();
    engine.setDrivingAggression(50, 0);
    update(0.17, 1000);
    const replacement = referenceLap();
    replacement.y.forEach((row) => { row[4] = 0; });
    engine.receiveMap(circuitMap(), 1100, replacement);
    const stale = update(0.17, 1200);
    expect(stale.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Not matched', missing: [] });
    const sample = driving(0.17);
    engine.receiveVision(detection(replacement, sample, 1300), 1300);
    const fresh = engine.receiveTelemetry(frame(sample), 1300);
    expect(fresh.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Not matched' });
    expect(fresh.referenceLoaded).toBe(true);
    expect(engine.receiveMap(null, 1400).referenceLoaded).toBe(false);
    engine.setDrivingAggression(0, 1450);
    engine.receiveVision(detection(lap, sample, 1500), 1500);
    const removed = engine.receiveTelemetry(frame(sample), 1500);
    expect(removed.mapReady).toBe(false);
    expect(removed.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Missing input' });
    expect(removed.events).toEqual([]);
});
