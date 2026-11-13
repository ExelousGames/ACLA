import { buildReferenceCarPerformance, carDecelerationAt, type CarPerformanceSample } from './phrase-car-performance';
import { readReferenceGuidance, referenceCarPerformance } from './phrase-reference-guidance';
import { steeringCornerFixture } from './test-fixtures';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';

const observation = (patch: Partial<CarPerformanceSample> = {}): CarPerformanceSample => ({
    distance: 0, speed: 20, curvature: 0.01, elevation: 0, grade: 0,
    deceleration: 6, brake: 0.8, throttle: 0, coasting: false, ...patch,
});

it('reserves more grip for a tighter radius or higher speed instead of reusing straight-line braking', () => {
    const model = buildReferenceCarPerformance([observation()]);
    const at = (speed: number, curvature: number) => carDecelerationAt(model, { speed, curvature, grade: 0 });
    expect(at(20, 0.01).availableBrakeDeceleration).toBeCloseTo(6);
    expect(at(20, 0).availableBrakeDeceleration).toBeGreaterThan(6);
    expect(at(20, 0.015).availableBrakeDeceleration).toBeLessThan(6);
    expect(at(22, 0.01).availableBrakeDeceleration).toBeLessThan(6);
    expect(at(20, 0.03).availableBrakeDeceleration).toBe(0);
    expect(at(60, 0.01).availableBrakeDeceleration).toBeUndefined();
});

it('separates coasting and gravity from brake performance at different heights', () => {
    const model = buildReferenceCarPerformance([
        observation({ brake: 0, coasting: true, deceleration: 0.5 + 9.80665 * 0.1, grade: 0.1, elevation: 100 }),
        observation({ deceleration: 6.5 }),
    ]);
    const flat = carDecelerationAt(model, { speed: 20, curvature: 0.01, grade: 0 });
    const uphill = carDecelerationAt(model, { speed: 20, curvature: 0.01, grade: 0.1 });
    const downhill = carDecelerationAt(model, { speed: 20, curvature: 0.01, grade: -0.1 });
    expect(flat.coastingDeceleration).toBeCloseTo(0.5);
    expect(flat.availableBrakeDeceleration).toBeCloseTo(6);
    expect(uphill.coastingDeceleration).toBeCloseTo(0.5 + 0.980665);
    expect(downhill.coastingDeceleration).toBeCloseTo(0.5 - 0.980665);
    expect(uphill.possibleDeceleration! - downhill.possibleDeceleration!).toBeCloseTo(2 * 0.980665);
});

it('keeps coasting observations specific to speed and corner load', () => {
    const model = buildReferenceCarPerformance([
        observation({ speed: 20, curvature: 0, brake: 0, coasting: true, deceleration: 0.4 }),
        observation({ speed: 20, curvature: 0.01, brake: 0, coasting: true, deceleration: 1.2 }),
        observation({ speed: 40, curvature: 0, brake: 0, coasting: true, deceleration: 1.5 }),
    ]);
    expect(carDecelerationAt(model, { speed: 20, curvature: 0, grade: 0 }).coastingDeceleration).toBe(0.4);
    expect(carDecelerationAt(model, { speed: 20, curvature: 0.01, grade: 0 }).coastingDeceleration).toBe(1.2);
    expect(carDecelerationAt(model, { speed: 40, curvature: 0, grade: 0 }).coastingDeceleration).toBe(1.5);
    expect(carDecelerationAt(model, { speed: 80, curvature: 0, grade: 0 }).coastingDeceleration).toBe(0);
});

function corner(game: 'acc' | 'iracing' = 'acc', rise = 0, coasting = false) {
    const { reference } = steeringCornerFixture(game);
    reference.y.forEach((row, i) => {
        row[game === 'acc' ? 2 : 1] = i * rise;
        row[4] = coasting ? 0 : 0.5;
        row[6] = Math.sqrt(400 - 20 * Math.max(0, Math.min(5, i - 25))) * 3.6;
    });
    const at = (position: number, speed = reference.y[Math.round(position * 100)][6], brake = 0.5, deceleration = 1) =>
        readReferenceGuidance({ Static_track: 'test', Static_car_model: 'car',
            Graphics_normalized_car_position: position, Physics_speed_kmh: speed, Physics_brake: brake, Physics_gas: 0 },
        null, reference, game, 1000, reference, deceleration, 1, { start: 0.25, end: 0.45 });
    return { reference, at };
}

it('requires Ease brake at matching speed while the corner still needs deceleration, even at a constant rate', () => {
    const { at } = corner();
    const result = at(0.27);
    expect(result.easeBrake).toBe(2500);
    expect(result.releaseBrake).toBeUndefined();
    expect(result.holdBrake).toBeUndefined();
    expect(result.moreBrake).toBeUndefined();
    expect(result.brakeReleaseRate).toBeUndefined();
});

it('does not release merely because speed is below the reference entry speed', () => {
    const { at } = corner();
    expect(at(0.28, 70)).toMatchObject({ easeBrake: 2500 });
    expect(at(0.28, 70).releaseBrake).toBeUndefined();
    expect(at(0.31).releaseBrake).toBe(2500);
    expect(at(0.31).easeBrake).toBeUndefined();
});

it('can release at matching speed when observed coasting already supplies the needed slowdown', () => {
    const { at } = corner('acc', 0, true);
    expect(at(0.27).releaseBrake).toBe(2500);
    expect(at(0.27).easeBrake).toBeUndefined();
    expect(at(0.27, undefined, 0).brake).toBeUndefined();
});

it.each(['acc', 'iracing'] as const)('uses the correct elevation axis for %s on uphill and downhill corners', (game) => {
    const uphill = corner(game, 2);
    expect(uphill.at(0.27).releaseBrake).toBe(2500);
    expect(uphill.at(0.27, undefined, 0).brake).toBeUndefined();
    const downhill = corner(game, -2);
    expect(downhill.at(0.31).easeBrake).toBe(2500);
    expect(downhill.at(0.31).releaseBrake).toBeUndefined();
    const profile = referenceCarPerformance(downhill.reference)!;
    expect(profile.samples[27].grade).toBeCloseTo(-2 / Math.hypot(10, 2));
    expect(profile.samples[27].elevation).toBe(-55);
    expect(Math.abs(profile.samples[27].curvature)).toBeGreaterThan(0);
});

it('withholds full release on a plateau before another slowdown in the same corner', () => {
    const { reference, at } = corner();
    reference.y.forEach((row, i) => { row[6] = i < 30 ? 72 : 60; });
    expect(at(0.27).easeBrake).toBe(2500);
    expect(at(0.27).releaseBrake).toBeUndefined();
});

it.each([0, 0.09, NaN, undefined])('requires a valid applied brake for either release instruction (%s)', (brake) => {
    const { reference } = corner();
    const result = readReferenceGuidance({ Static_track: 'test', Static_car_model: 'car',
        Graphics_normalized_car_position: 0.27, Physics_speed_kmh: 72, Physics_brake: brake },
    null, reference, 'acc', 1000, reference, 1, 1, { start: 0.25, end: 0.45 });
    expect(result.easeBrake).toBeUndefined();
    expect(result.releaseBrake).toBeUndefined();
});

it('captures an independent immutable car profile and rejects missing reference speed', () => {
    const { reference } = corner();
    const model = referenceCarPerformance(reference)!;
    const speed = model.samples[27].speed;
    reference.y[27][6] = 0;
    expect(model.samples[27].speed).toBe(speed);
    expect(Object.isFrozen(model.samples[27])).toBe(true);
    const missing = { ...reference, target_features: reference.target_features.slice(0, 6) };
    expect(referenceCarPerformance(missing)!.samples).toEqual([]);
});

it('retains coasting capability across source edits and aggression changes, then refreshes it on reference replacement', () => {
    const { reference } = corner('acc', 0, true), { map } = steeringCornerFixture();
    const phrases = PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'releaseBrake' || referenceCue === 'easeBrake');
    const engine = new PhraseEngine(phrases);
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances(phrases).map((instance) => [instance.cacheKey, 0])));
    engine.receiveMap(map, 0, reference);
    // This edit must not retroactively turn captured coasting into brake performance.
    reference.y.forEach((row) => { row[4] = 0.5; });
    engine.setDrivingAggression(50, 100);
    const sample = { Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: 0.27,
        Physics_speed_kmh: reference.y[27][6], Physics_brake: 0.5, Physics_gas: 0 };
    const frame: LiveTelemetryFrameEvent = { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
        sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
    const first = engine.receiveTelemetry(frame, 1000).events[0];
    expect(first.sentence).toBe('Off brake.');
    engine.speechFinished(first.id, 1100);
    engine.receiveMap(map, 2000, { ...reference, y: reference.y.map((row) => [...row]) });
    expect(engine.receiveTelemetry(frame, 3000).events.slice(-1)[0].sentence).toBe('Ease brake');
});
