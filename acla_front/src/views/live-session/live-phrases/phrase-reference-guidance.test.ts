import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { readReferenceGuidance } from './phrase-reference-guidance';
import { copyPhraseReference } from './phrase-driving-reference';
import { PhraseEngine } from './phrase-engine';
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

it.each([[0.39, 'steerSoon', 2100]] as const)(
    'anticipates %s with %s before its interpolated action point', (position, cue, deadline) => {
        const lap = referenceLap(), sample = driving(position);
        const result = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        expect(result.ready).toBe(true);
        expect(result[cue]).toBeCloseTo(deadline);
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
    expect(readReferenceGuidance(brake, detection(lap, brake), lap, 'acc', 1000).moreBrake).toBe(2500);
    expect(readReferenceGuidance(throttle, detection(lap, throttle), lap, 'acc', 1000).moreThrottle).toBe(2500);
    for (const [sample, speed] of [[brake, 98], [throttle, 102]] as const) {
        const matching = { ...sample, Physics_speed_kmh: speed };
        const cues = readReferenceGuidance(matching, detection(lap, matching), lap, 'acc', 1000);
        expect(cues.moreBrake).toBeUndefined();
        expect(cues.moreThrottle).toBeUndefined();
    }
    expect(readReferenceGuidance({ ...brake, Physics_brake: 1 }, detection(lap, brake), lap, 'acc', 1000).moreBrake).toBeUndefined();
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
    [102, 0, 'throttle'], [94, 0.8, 'moreThrottle'], [102, 1, 'lessThrottle'], [120, 0.8, undefined],
    [102, 0.8, undefined], [94, 1, undefined], [120, 0, undefined],
] as const)('selects one throttle instruction at %s km/h with pedal %s: %s', (speed, pedal, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: speed, Physics_gas: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(['throttle', 'moreThrottle', 'lessThrottle', 'releaseThrottle'].filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    if (cue) expect(guidance[cue]).toBe(2500);
});

it.each([0.4, 0.8])('uses reference throttle %s to estimate demand at matching speed', (pedal) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[5]) row[5] = pedal; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.6 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance[pedal === 0.4 ? 'lessThrottle' : 'moreThrottle']).toBe(2500);
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
        .toEqual({ ready: true, lateralOffset: 0, braking: false, pedalZone: 'throttle' });
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
    expect(readReferenceGuidance({ ...sample, Physics_gas: 1 }, detection(lap, sample), lap, 'acc', 1000).lessThrottle).toBe(2500);
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
        .toEqual({ ready: true, lateralOffset: 0, braking: false, pedalZone: 'throttle' });
    expect(readReferenceGuidance({ ...sample, Physics_speed_kmh: 90 }, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
});

it('allows easing throttle away from the line while withholding acceleration', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 1 }, view = detection(lap, sample);
    view.position!.referenceDelta[0] = 3;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).lessThrottle).toBe(2500);
    const guidance = readReferenceGuidance({ ...sample, Physics_gas: 0 }, view, lap, 'acc', 1000);
    expect(guidance.throttle).toBeUndefined();
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([
    [98, 0, 'brake'], [106, 0.8, 'moreBrake'], [92, 0.8, 'lessBrake'], [80, 0.8, 'releaseBrake'],
    [98, 0.8, undefined], [106, 1, undefined], [80, 0, undefined],
] as const)('selects one brake instruction at %s km/h with pedal %s: %s', (speed, pedal, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.255), Physics_speed_kmh: speed, Physics_brake: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    const matched = ['brake', 'moreBrake', 'lessBrake', 'releaseBrake'].filter((key) => key in guidance);
    expect(matched).toEqual(cue ? [cue] : []);
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([0.4, 0.8])('uses reference brake pressure %s to estimate demand at matching speed', (pressure) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[4]) row[4] = pressure; });
    const sample = { ...driving(0.255), Physics_speed_kmh: Math.sqrt((100 ** 2 + 96 ** 2) / 2), Physics_brake: 0.6 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance[pressure === 0.4 ? 'lessBrake' : 'moreBrake']).toBe(2500);
});

it.each([
    [0.31, undefined], [0.69, undefined],
    [0.3, 'moreBrake'], [0.7, 'lessBrake'],
    [0.29, 'moreBrake'], [0.71, 'lessBrake'],
] as const)('requires a 20-point brake difference before correcting pedal %s', (pedal, cue) => {
    const lap = referenceLap();
    lap.y.forEach((row) => { if (row[4]) row[4] = 0.5; });
    const sample = { ...driving(0.255), Physics_brake: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(['moreBrake', 'lessBrake'].filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
});

it.each([0.2, 1.2])('scales speed correction using reference deceleration (%s km/h per metre)', (slope) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => { row[6] = i < 20 || i > 30 ? 100 : 98 - (i * 10 - 255) * slope; });
    const sample = { ...driving(0.255), Physics_speed_kmh: Math.hypot(98, 5 * slope) + 5, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance.moreBrake).toBe(slope === 0.2 ? 2500 : undefined);
});

it('starts braking earlier and releases for the cautious speed profile using the original braking response', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const cautious = copyPhraseReference(lap, 50)!;
    const early = { ...driving(0.19), Physics_speed_kmh: 116 };
    expect(readReferenceGuidance(early, detection(lap, early), cautious, 'acc', 1000, lap, lap).brake).toBe(2500);
    const missingPedal = { ...early, Physics_brake: undefined };
    expect(readReferenceGuidance(missingPedal, detection(lap, missingPedal), cautious, 'acc', 1000, lap, lap))
        .toEqual({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake' });
    const slowed = { ...driving(0.3), Physics_speed_kmh: 40, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(slowed, detection(lap, slowed), cautious, 'acc', 1000, lap, lap);
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
    for (const [pedal, moreBrake] of [[0.5499, 2500], [0.5501, undefined]] as const) {
        const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: pedal };
        expect(readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap, lap).moreBrake).toBe(moreBrake);
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
            expect(readReferenceGuidance(shiftedSample, detection(shifted, shiftedSample), shiftedCautious, 'acc', 1000, shifted, shifted))
                .toEqual(readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap, lap));
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
        .toEqual({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake' });
    sample.Physics_speed_kmh = 106;
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).moreBrake).toBe(2500);
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
        expect(['brake', 'moreBrake', 'lessBrake', 'releaseBrake'].filter((key) => key in guidance)).toEqual([]);
    });

it('aligns speed targets with the position-model forward delta and withholds acceleration above target', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 108 }, view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    // With the correct pressure, the forward alignment removes the speed correction.
    expect(readReferenceGuidance({ ...sample, Physics_brake: 0.8 }, view, lap, 'acc', 1000).moreBrake).toBeUndefined();
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

it('uses the position-model forward delta to align brake demand and lateral error to suppress acceleration', () => {
    const lap = referenceLap(), sample = driving(0.21), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = 10;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBeUndefined();
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

it.each(['stale', 'future', 'wrong lap', 'wrong car', 'wrong track', 'wrong game', 'jump', 'invalid delta', 'no position', 'bad lap'])(
    'rejects %s inputs', (kind) => {
        const lap = referenceLap();
        let sample = driving(0.19);
        const view = detection(lap, sample);
        if (kind === 'stale') view.capturedAt = 499;
        if (kind === 'future') view.capturedAt = 1001;
        if (kind === 'wrong lap') view.position!.referenceLap = referenceLap();
        if (kind === 'wrong car') sample = { ...sample, Static_car_model: 'other' };
        if (kind === 'wrong track') sample = { ...sample, Static_track: 'other' };
        if (kind === 'jump') view.position!.normalizedPosition = 0.1;
        if (kind === 'invalid delta') view.position!.referenceDelta[0] = NaN;
        if (kind === 'no position') view.position = undefined;
        if (kind === 'bad lap') lap.y[2][0] = NaN;
        expect(readReferenceGuidance(sample, view, lap, kind === 'wrong game' ? 'iracing' : 'acc', 1000)).toEqual({});
    });

it('does not treat a missing reference channel as zero or advise throttle during braking', () => {
    const lap = referenceLap();
    lap.target_features[3] = 'missing steering';
    const sample = driving(0.45);
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).steerSoon).toBeUndefined();
    const braking = { ...driving(0.7), Physics_brake: 0.5 };
    expect(readReferenceGuidance(braking, detection(lap, braking), lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it('anticipates reference steering reversals independently of live steering', () => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[3] = i < 41 ? 0.4 : -0.4; });
    const sample = { ...driving(0.395), Physics_steer_angle: 0.4 }, view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).steerSoon).toBeCloseTo(2000);
    expect(readReferenceGuidance({ ...sample, Physics_steer_angle: -0.4 }, view, lap, 'acc', 1000).steerSoon).toBeCloseTo(2000);
});

it('suppresses throttle when reference braking is imminent', () => {
    const lap = referenceLap();
    lap.y.forEach((row) => { row[5] = 0.8; });
    const sample = driving(0.199), view = detection(lap, sample);
    const cues = readReferenceGuidance(sample, view, lap, 'acc', 1000);
    expect(cues.brake).toBeUndefined();
    expect(cues.moreThrottle).toBeUndefined();
});

it('reports current aligned reference braking independently of advance cues and driver brake input', () => {
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
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).braking).toBe(false);
    const missingBrake = referenceLap();
    missingBrake.target_features[4] = 'missing brake';
    expect(readReferenceGuidance(sample, detection(missingBrake, sample), missingBrake, 'acc', 1000).braking).toBeUndefined();
});

function setup(configure?: (lap: TopLapReferenceEntry) => void) {
    const engine = new PhraseEngine(), lap = referenceLap();
    configure?.(lap);
    engine.receiveMap(null, 0, lap);
    const update = (position: number, now: number, patch: StandardTelemetrySample = {}) => {
        const sample = { ...driving(position), ...patch };
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    return { engine, lap, update };
}

function brakingTurn(lap: TopLapReferenceEntry) {
    addSpeed(lap);
    lap.y.forEach((row, i) => {
        row[3] = i >= 29 && i <= 35 ? 0.4 : 0;
        // Constant 3 m/s² braking, easing to 1 m/s² from 260 m before turn-in at 282 m.
        row[6] = Math.sqrt(100 ** 2 / 3.6 ** 2 - 6 * Math.max(0, Math.min(60, i * 10 - 200))
            - 2 * Math.max(0, Math.min(40, i * 10 - 260))) * 3.6;
    });
}

it('enters Steer soon silently and speaks only the easing and turn-in phrases, once each', () => {
    const { engine, lap, update } = setup(brakingTurn);
    const at = (position: number, now: number, speed: number) => update(position, now, {
        Physics_speed_kmh: speed, Physics_brake: 0.8, Physics_steer_angle: -0.5,
    });
    const prepared = at(0.25, 1000, lap.y[25][6]);
    expect(prepared.state.path).toEqual(['root', 'Cornering', 'Brake zone', 'Steer soon']);
    expect(prepared.events).toEqual([]);
    const eased = at(0.26, 2100, lap.y[26][6]);
    expect(eased.events.map(({ sentence }) => sentence)).toEqual(['turn, slowly lift up the brake']);
    expect(eased.state.path).toContain('Steer soon');
    expect(at(0.26, 2300, lap.y[26][6]).events).toHaveLength(1);
    // Turn waits for the shared speech window, even if its reference threshold is already crossed.
    expect(at(0.283, 2500, lap.y[28][6]).events).toHaveLength(1);
    const turned = at(0.283, 3200, lap.y[28][6]);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['turn, slowly lift up the brake', 'Turn']);
    expect(turned.events.map(({ steeringPhase }) => steeringPhase)).toEqual(['easeBrake', 'turn']);
    expect(turned.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(at(0.285, 5500, lap.y[28][6]).events).toHaveLength(2);
    expect(engine.evaluate(5501).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(update(0.31, 6000).state.path).toEqual(['root', 'Cornering']);
});

it('stays quiet after brake release while waiting for reference turn-in', () => {
    const { lap, update } = setup(brakingTurn);
    expect(update(0.26, 1000, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.3 }).events
        .map(({ sentence }) => sentence)).toEqual(['turn, slowly lift up the brake']);
    // Deceleration is now constant, so the release condition has cleared before turn-in.
    expect(update(0.27, 2100, { Physics_speed_kmh: lap.y[27][6], Physics_brake: 0.8 }).events).toHaveLength(1);
    expect(update(0.283, 3200, { Physics_speed_kmh: lap.y[28][6], Physics_brake: 0.8 }).events
        .map(({ sentence }) => sentence)).toEqual(['turn, slowly lift up the brake', 'Turn']);
});

it('withholds nested steering speech with expired position data and exits at the brake-zone boundary', () => {
    const { engine, lap, update } = setup(brakingTurn);
    update(0.25, 1000, { Physics_speed_kmh: lap.y[25][6], Physics_brake: 0.8 });
    const stale = engine.receiveTelemetry(frame({ ...driving(0.26), Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.8 }), 2100);
    expect(stale.state.path).toContain('Steer soon');
    expect(stale.events).toEqual([]);
    const exited = update(0.31, 2200);
    expect(exited.state.path).toEqual(['root', 'Cornering']);
    expect(exited.events).toHaveLength(0);
    expect(update(0.7, 3300).state.path).toEqual(['root', 'Cornering', 'Throttle zone', 'Throttle', 'say phrase']);
});

it.each(['missing speed', 'invalid speed', 'constant deceleration', 'increasing deceleration', 'missing steering'])(
    'does not give the easing prompt with %s in the reference', (kind) => {
        const lap = referenceLap();
        brakingTurn(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[26][6] = NaN;
        if (kind === 'missing steering') lap.target_features[3] = 'missing';
        if (kind === 'constant deceleration' || kind === 'increasing deceleration') {
            lap.y[27][6] = Math.sqrt(lap.y[26][6] ** 2 / 3.6 ** 2 - 2 * (kind === 'constant deceleration' ? 3 : 4) * 10) * 3.6;
        }
        const sample = { ...driving(0.26), Physics_speed_kmh: 80 };
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).steeringPhase).not.toBe('easeBrake');
    });

it.each([false, true])('detects easing and turn-in across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[98][6] = 100;
    lap.y[99][6] = 90;
    lap.y[0][6] = 88;
    lap.y[99][4] = lap.y[0][4] = 0.8;
    lap.y[1][3] = 0.4;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const [position, phase] of [[0.995, 'easeBrake'], [0.003, 'turn']] as const) {
        const sample = driving(position);
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).steeringPhase).toBe(phase);
    }
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

it('aligns zone membership with the position model and the private aggression-adjusted reference', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = driving(0.21), view = detection(lap, sample);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).pedalZone).toBe('none');
    const cautious = copyPhraseReference(lap, 50)!;
    expect(readReferenceGuidance(sample, view, cautious, 'acc', 1000, lap, lap).pedalZone).toBe('brake');
    for (const speed of [36, 84, 160]) {
        const throttle = { ...driving(0.61), Physics_speed_kmh: speed };
        const prediction = detection(lap, throttle);
        expect(readReferenceGuidance(throttle, prediction, lap, 'acc', 1000).pedalZone).toBe('throttle');
        expect(readReferenceGuidance(throttle, prediction, cautious, 'acc', 1000, lap, lap).pedalZone).toBe('throttle');
    }
});

it('stays quiet in the throttle zone when overspeed calls for releasing throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.655, 1000, { Physics_speed_kmh: 102, Physics_gas: 0.8 }).state.path)
        .toEqual(['root', 'Cornering', 'Throttle zone']);
    const correction = update(0.655, 2200, { Physics_speed_kmh: 120, Physics_gas: 0.8 });
    expect(correction.events).toEqual([]);
    expect(correction.state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(correction.closures.find(({ id }) => id === 'reference-brake')!.conditions.some((condition) => !condition.conditionFit)).toBe(true);
    expect(engine.evaluate(2201).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.655, 3400, { Physics_speed_kmh: 120, Physics_gas: 0 }).state.path)
        .toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.95, 3500, { Physics_speed_kmh: 120 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
});

it('keeps throttle past the reference boundary until corner exit, but exits brake at its reference boundary', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.95 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.908, 1000).state.path).toEqual(['root', 'Cornering', 'Throttle zone', 'Throttle', 'say phrase']);
    const afterReference = update(0.909, 1100);
    expect(afterReference.state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(afterReference.events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    const throttle = afterReference.root.children.find(({ id }) => id === 'cornering')!.children
        .find(({ id }) => id === 'reference-throttle-zone')!;
    expect(throttle.children[0].conditions).toEqual([
        expect.objectContaining({ description: 'The driver has left the corner', conditionFit: false }),
    ]);
    expect(update(0.951, 1150).state.path).toEqual(['root']);
    expect(update(0.298, 1200).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(update(0.299, 1300).state.path).toEqual(['root', 'Cornering']);
});

it.each(['straight', 'long straight', 'untagged'])('ends throttle at the mapped corner exit before %s, even during speech cooldown', (label) => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 },
        ...(label === 'untagged' ? [] : [{ id: 'after', tags: [label], start_position: 0.7, end_position: 0.9 }]),
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.699, 1000).state.path).toEqual(['root', 'Cornering', 'Throttle zone', 'Throttle', 'say phrase']);
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
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(engine.receiveTelemetry(frame(driving(0.701)), 1600).state.path).toEqual(['root']);
});

it('keeps the brake zone before mapped corner entry when reference location expires', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
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
            .toEqual(['root', 'Cornering', 'Throttle zone']);
    });
    expect(update(0.011, 1400, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
});

it('returns from Cornering to movement guidance when throttle pickup is beyond the mapped corner', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.5 },
        { id: 'after', tags: ['straight'], start_position: 0.5, end_position: 0.9 },
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toContain('Brake zone');
    expect(update(0.4, 2000).state.path).toEqual(['root', 'Cornering']);
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
    expect(update(0.8, 1000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.95, 1100, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.25, 1200, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
});

it('keeps zone ownership through missing captures and switches only after fresh boundary evidence', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.9 }];
    engine.receiveMap(map, 0, lap);
    update(0.7, 1000);
    const stale = engine.evaluate(1501);
    expect(stale.state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(stale.events.map(({ sentence }) => sentence)).toEqual(['Throttle.']);
    expect(update(0.7, 2000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.25, 2100, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
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
            .toEqual(['root', 'Cornering', 'Throttle zone']);
    });
    expect(update(0.019, 1400, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.031, 1500, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
});

it('gives overlapping interpolated pedal zones to the brake closure', () => {
    const { update } = setup((lap) => lap.y.forEach((row) => { row[5] = 0.8; }));
    const result = update(0.25, 1000, { Physics_brake: 0.8 });
    expect(result.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(result.events).toEqual([]);
});

it('transitions through all four brake phrases as speed and applied pressure change', () => {
    const { engine, update } = setup(addSpeed);
    const expected: string[] = [];
    ([[98, 0, 'Brake.'], [106, 0.8, 'More brake.'], [92, 0.8, 'Less brake.'], [80, 0.8, 'Release the brake.']] as const)
        .forEach(([speed, pedal, sentence], index) => {
            const now = 1000 + index * 1600;
            const correction = sentence === 'More brake.' || sentence === 'Less brake.';
            let snapshot = update(0.255, now, { Physics_speed_kmh: speed, Physics_brake: pedal });
            if (correction) {
                expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
                snapshot = update(0.255, now + 300, { Physics_speed_kmh: speed, Physics_brake: pedal });
            }
            expected.push(sentence);
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.events[index].expiresAt).toBe(now + (correction ? 300 : 0) + 1500);
            expect(snapshot.state.path).toContain('Cornering');
            expect(snapshot.state.path).toContain('Brake zone');
            expect(engine.evaluate(now + (correction ? 300 : 0) + 1).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
        });
});

it.each([[106, 'More brake.'], [92, 'Less brake.']] as const)(
    'confirms %s km/h brake corrections for 300 ms before saying %s', (speed, sentence) => {
        const { update } = setup(addSpeed);
        const patch = { Physics_speed_kmh: speed, Physics_brake: 0.8 };
        expect(update(0.255, 1000, patch).events).toEqual([]);
        expect(update(0.255, 1299, patch).events).toEqual([]);
        expect(update(0.255, 1300, patch).events).toEqual([
            expect.objectContaining({ sentence, timestamp: 1300 }),
        ]);
    });

it.each([[106, 'More brake.'], [92, 'Less brake.']] as const)(
    'restarts confirmation when a %s km/h brake correction clears', (speed, sentence) => {
        const { update } = setup(addSpeed);
        const patch = { Physics_speed_kmh: speed, Physics_brake: 0.8 };
        expect(update(0.255, 1000, patch).events).toEqual([]);
        expect(update(0.255, 1200, { ...patch, Physics_speed_kmh: 98 }).events).toEqual([]);
        expect(update(0.255, 1300, patch).events).toEqual([]);
        expect(update(0.255, 1599, patch).events).toEqual([]);
        expect(update(0.255, 1600, patch).events).toEqual([
            expect.objectContaining({ sentence, timestamp: 1600 }),
        ]);
    });

it('withholds pedal corrections outside reference pedal zones', () => {
    const { update } = setup(addSpeed);
    const result = update(0.45, 1000, { Physics_speed_kmh: 80, Physics_brake: 0.4 });
    expect(result.events).toEqual([]);
    expect(result.state.path).toEqual(['root']);
});

it('enters Cornering for all three throttle corrections without a preceding braking zone', () => {
    const { engine, update } = setup(addSpeed);
    const expected: string[] = [];
    ([[0.655, 102, 0, 'Throttle.'], [0.655, 94, 0.8, 'More throttle.'],
        [0.655, 102, 1, 'Less throttle.']] as const)
        .forEach(([position, speed, pedal, sentence], index) => {
            const now = 1000 + index * 1200;
            expected.push(sentence);
            const snapshot = update(position, now, { Physics_speed_kmh: speed, Physics_gas: pedal });
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.state.path).toEqual(['root', 'Cornering', 'Throttle zone', sentence.slice(0, -1), 'say phrase']);
            expect(engine.evaluate(now + 1).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
        });
});

it('gives only brake corrections in the brake zone, even when the driver applies throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.255, 1000, { Physics_speed_kmh: 98, Physics_gas: 0.8 }).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    engine.evaluate(1001);
    const snapshot = update(0.255, 2200, { Physics_speed_kmh: 98, Physics_gas: 0.8, Physics_brake: 0.8 });
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(snapshot.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
});

it('stays in the throttle zone after speech and while the driver matches the reference', () => {
    const { engine, update } = setup();
    update(0.7, 1000);
    expect(engine.evaluate(1001).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.7, 1250).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.7, 2250).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
    expect(update(0.7, 2500, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
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
        .toMatchObject({ status: 'Missing input', missing: ['Interpolated reference pedal zone', 'Reference-line guidance'] });
    update(0.3, 4000);
    expect(update(0.25, 5000).events).toHaveLength(2);
});

it.each([0.19, 0.39, 0.59])('does not enter Cornering outside reference pedal zones at %s', (position) => {
    const { update } = setup();
    const snapshot = update(position, 1000);
    expect(snapshot.events).toEqual([]);
    expect(snapshot.state.path).toEqual(['root']);
    expect(snapshot.root.children.find(({ id }) => id === 'cornering')!.conditions).toEqual([
        expect.objectContaining({ description: 'Inside an interpolated reference brake or throttle zone', conditionFit: false, inputMissing: false }),
    ]);
});

it.each([[0, 'Throttle', 'Throttle.'], [0.4, 'More throttle', 'More throttle.']] as const)(
    'stays in Cornering through steering and the throttle zone with pedal %s (%s)', (pedal, name, sentence) => {
        const { engine, lap, update } = setup();
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'straight', tags: ['straight'], start_position: 0, end_position: 0.99 },
            { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.8 },
        ];
        engine.receiveMap(map, 0, lap);
        const entered = update(0.25, 1000, { Physics_brake: 0.8 });
        expect(entered.events).toEqual([]);
        expect(entered.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
        expect(entered.root.children.find(({ id }) => id === 'cornering')!.conditions[0].conditionFit).toBe(true);
        expect(update(0.3, 2000).state.path).toEqual(['root', 'Cornering']);
        expect(update(0.39, 3000).state.path).toEqual(['root', 'Cornering']);
        expect(engine.evaluate(3001).state.path).toEqual(['root', 'Cornering']);
        // The corner takes precedence over an overlapping straight tag.
        const sample = driving(0.45), view = detection(lap, sample, 4000);
        view.position!.referenceDelta[0] = -4;
        engine.receiveVision(view, 4000);
        expect(engine.receiveTelemetry(frame(sample), 4000).state.path).toEqual(['root', 'Cornering']);
        const acceleration = update(0.7, 5000, { Physics_gas: pedal });
        expect(acceleration.state.path).toEqual(['root', 'Cornering', 'Throttle zone', name, 'say phrase']);
        expect(acceleration.events.map(({ sentence }) => sentence)).toEqual([sentence]);
        expect(engine.evaluate(5001).state.path).toEqual(['root', 'Cornering', 'Throttle zone']);
        expect(update(0.7, 6000, { Physics_gas: pedal }).events).toHaveLength(1);
        expect(update(0.25, 7000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    });

it('waits until braking is needed inside Cornering and returns to the same scope', () => {
    const { engine, update } = setup();
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(update(0.19, 2000).events).toEqual([]);
    const nextBraking = update(0.25, 3000);
    expect(nextBraking.state.path).toEqual(['root', 'Cornering', 'Brake zone', 'Brake', 'say phrase']);
    expect(nextBraking.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(engine.evaluate(3001).state.path).toEqual(['root', 'Cornering', 'Brake zone']);
});

it('exits the previous throttle phrase before entering the next pedal correction', () => {
    const { update } = setup();
    update(0.25, 1000, { Physics_brake: 0.8 });
    expect(update(0.7, 2000).state.path).toEqual(['root', 'Cornering', 'Throttle zone', 'Throttle', 'say phrase']);
    const next = update(0.7, 3000, { Physics_gas: 0.4 });
    expect(next.state.path).toEqual(['root', 'Cornering', 'Throttle zone', 'More throttle', 'say phrase']);
    expect(next.events.map(({ sentence }) => sentence)).toEqual(['Throttle.', 'More throttle.']);
});

it('rejects retained vision after stream resets, pause and reference changes', () => {
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
    expect(engine.receiveTelemetry(frame(sample), 2200).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Missing input', missing: ['Interpolated reference pedal zone', 'Reference-line guidance'] });
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
    expect(result.state.path).toContain('Cornering');
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

it('stays in movement guidance between cues and exits immediately at reference brake onset even when driver braking matches', () => {
    const { engine, update } = movementSetup(circuitMap(), (lap) => {
        lap.y.forEach((row, index) => { row[4] = index >= 46 && index < 50 ? 0.8 : 0; });
    });
    expect(update(-4, 1000).state.path).toContain('Turn to reference line');
    // The 0.6–1.8 s advance cue window has passed; the reference is not braking yet.
    for (let now = 1100; now <= 1400; now += 100) {
        expect(update(-4, now, 0.451).state.path).toContain('Turn to reference line');
    }
    expect(engine.evaluate(1400).state.path).toEqual(['root', 'Turn to reference line']);
    const braking = update(-4, 1500, 0.452, { Physics_brake: 0.8 });
    expect(braking.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(braking.events).toHaveLength(1);
    expect(braking.closures.find(({ id }) => id === 'reference-move-left-small')!.conditions.every((condition) => condition.conditionFit)).toBe(false);
    for (let now = 1750; now <= 3500; now += 250) update(-0.1, now, 0.47, { Physics_brake: 0.8 });
    expect(engine.evaluate(3500).events).toHaveLength(1);
    expect(update(-0.1, 3750, 0.55).events).toHaveLength(1);
    const afterBraking = update(4, 4000, 0.55);
    expect(afterBraking.state.path).toEqual(['root', 'Cornering']);
    expect(afterBraking.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it('exits a selected movement speech action at brake onset inside its speech cooldown', () => {
    const { update } = movementSetup(circuitMap(), (lap) => {
        lap.y.forEach((row, index) => { row[4] = index >= 46 && index < 50 ? 0.8 : 0; });
    });
    expect(update(-4, 1000, 0.451).state.path).toEqual([
        'root', 'Turn to reference line', 'Move to the left', 'Small movement', 'say phrase',
    ]);
    const braking = update(-4, 1100, 0.4513, { Physics_brake: 0.8 });
    expect(braking.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
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

it('uses the private speed copy for cues while validating location against the Live Map source', () => {
    const { engine, lap, update } = setup(addSpeed);
    const speed = { Physics_speed_kmh: 122 };
    expect(update(0.19, 1000, speed).events).toEqual([]);
    engine.setDrivingAggression(50, 1100);
    expect(update(0.19, 1200, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(lap.y[19][4]).toBe(0);
    expect(lap.y[19][6]).toBe(120);
    engine.setDrivingAggression(25, 1300);
    expect(update(0.20, 1400, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    // Restoring Reference must use the original local baseline, not the stretched copy or a mutated source.
    lap.y.forEach((row) => { row[4] = 0; row[6] = 200; });
    engine.setDrivingAggression(0, 1500);
    expect(update(0.19, 1600, speed).state.path).toEqual(['root']);
    expect(update(0.25, 1650, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const view = detection(referenceLap(), driving(0.19), 1700);
    engine.receiveVision(view, 1700);
    expect(engine.receiveTelemetry(frame(driving(0.19)), 1700).closures.find(({ id }) => id === 'reference-brake'))
        .toMatchObject({ status: 'Missing input' });
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
    expect(cautious.state.path).toEqual(['root', 'Cornering', 'Brake zone']);
    expect(cautious.events).toEqual([]);
});

it('refreshes copies from new Live Map data and rejects location from a replaced or removed source', () => {
    const { engine, lap, update } = setup();
    engine.setDrivingAggression(50, 0);
    update(0.17, 1000);
    const replacement = referenceLap();
    replacement.y.forEach((row) => { row[4] = 0; });
    engine.receiveMap(circuitMap(), 1100, replacement);
    const stale = update(0.17, 1200);
    expect(stale.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Missing input' });
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
