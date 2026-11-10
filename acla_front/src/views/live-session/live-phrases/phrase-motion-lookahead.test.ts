import type { StandardTelemetrySample } from '../live-session-types';
import { Action, Closure } from './closure';
import { PhraseEngine, type PhraseContext } from './phrase-engine';
import { PhraseLookahead } from './phrase-lookahead';
import { readReferenceGuidance, referencePositionAhead, type ReferenceGuidance } from './phrase-reference-guidance';
import { compareSteeringDirection } from './phrase-steering-direction';
import { steeringCornerFixture } from './test-fixtures';

const motion = (t: number): StandardTelemetrySample => ({
    Static_track: 'test', Static_car_model: 'car',
    Graphics_normalized_car_position: 0.1 + 0.01 * t + 0.004 * t ** 2,
    Physics_speed_kmh: 36 + 28.8 * t + 4 * t ** 2,
    Physics_steer_angle: 0.1 + 0.2 * t + 0.1 * t ** 2,
    Physics_brake: 0.1 + 0.2 * t + 0.1 * t ** 2, Physics_gas: 0.8 - 0.3 * t - 0.1 * t ** 2,
    Physics_heading: 0.1 * t ** 2, Physics_pitch: 0, Physics_roll: 0,
    Physics_velocity_x: 10 + 2 * t + 3 * t ** 2,
    Physics_velocity_y: 1 - 2 * t + t ** 2, Physics_velocity_z: -(t ** 2),
    Physics_local_velocity_x: t ** 2, Physics_local_velocity_y: 10 + 2 * t ** 2, Physics_local_velocity_z: 0,
    Graphics_player_car_id: 42, Graphics_car_id: [7, 42],
    Graphics_car_coordinates: [{ x: 1000, y: 0, z: 0 }, { x: 100 + 10 * t + 2 * t ** 2, y: t ** 2, z: -(t ** 2) }],
});

const frame = (sample: StandardTelemetrySample) => ({
    type: 'frame' as const, sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame' as const, game: 'acc' as const, sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});

it('defaults to a fixed three-second forecast and clamps longer requests for every projected channel', () => {
    const predictor = new PhraseLookahead();
    const sample = { Physics_speed_kmh: 72, Physics_brake: 0.6, Physics_gas: 0.2,
        Physics_steer_angle: 0.1, Graphics_normalized_car_position: 0.2 };
    const next = { ...sample, Physics_speed_kmh: 68.4, Physics_brake: 0.5,
        Physics_gas: 0.25, Physics_steer_angle: 0.15, Graphics_normalized_car_position: 0.22 };
    predictor.update(sample, 0, false);
    predictor.update(next, 1000, true);
    const projected = predictor.project(next, null);
    expect(projected.Physics_speed_kmh).toBeCloseTo(57.6);
    expect(projected.Physics_brake).toBeCloseTo(0.2);
    expect(projected.Physics_gas).toBeCloseTo(0.4);
    expect(projected.Physics_steer_angle).toBeCloseTo(0.3);
    expect(projected.Graphics_normalized_car_position).toBeCloseTo(0.28);
    expect(predictor.project(next, null, 9000)).toEqual(predictor.project(next, null));
    expect(predictor.projectDeceleration(next, 9000)).toBe(predictor.projectDeceleration(next));
    expect(predictor.projectBrakeReleasing(next, 9000)).toBe(predictor.projectBrakeReleasing(next));
});

it('enters a closure for a brief match inside its preparation window and only then measures its child', () => {
    const measure = jest.fn(() => true), run = jest.fn();
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure('brief match', '', ({ intendedTelemetry }) => (intendedTelemetry.Physics_steer_angle ?? 0) >= 0.059
            && (intendedTelemetry.Physics_steer_angle ?? 0) <= 0.071,
        [new Action('child', '', run, measure)]),
    ]));
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0 }), 1000);
    expect(measure).not.toHaveBeenCalled();
    const entered = engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0.01 }), 1100);
    expect(entered.state.path).toEqual(['root', 'brief match', 'child']);
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls[0][0].intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.06);
});

it('seeds the preview with the first valid rate without curvature or telemetry mutation', () => {
    const predictor = new PhraseLookahead();
    const times = [0.13, 0.4];
    const samples = times.map(motion);
    const before = JSON.stringify(samples);
    times.forEach((at, i) => predictor.update(samples[i], at * 1000, i > 0));
    const current = samples[samples.length - 1], projected = predictor.project(current, null, 500);
    // The curved measurements keep their latest secant rate for the next 500 ms.
    const expected = {
        Graphics_normalized_car_position: 0.1107, Physics_speed_kmh: 63.62,
        Physics_steer_angle: 0.3225, Physics_heading: 0.0425, Physics_brake: 0.3225, Physics_gas: 0.4875,
        Physics_velocity_x: 13.075, Physics_velocity_y: -0.375, Physics_velocity_z: -0.425,
        Physics_local_velocity_x: 0.425, Physics_local_velocity_y: 10.85, Physics_local_velocity_z: 0,
        Graphics_car_coordinates: [{ x: 1000, y: 0, z: 0 }, { x: 109.85, y: 0.425, z: -0.425 }],
    };
    for (const field of ['Graphics_normalized_car_position', 'Physics_speed_kmh', 'Physics_steer_angle', 'Physics_heading',
        'Physics_brake', 'Physics_gas',
        'Physics_velocity_x', 'Physics_velocity_y', 'Physics_velocity_z',
        'Physics_local_velocity_x', 'Physics_local_velocity_y', 'Physics_local_velocity_z'] as const) {
        expect(projected[field]).toBeCloseTo(expected[field]!, 9);
    }
    for (const axis of ['x', 'y', 'z'] as const) {
        expect(projected.Graphics_car_coordinates![1][axis]).toBeCloseTo(expected.Graphics_car_coordinates![1][axis], 9);
    }
    expect(projected.Graphics_car_coordinates![0]).toEqual(current.Graphics_car_coordinates![0]);
    expect(JSON.stringify(samples)).toBe(before);
});

it.each(['Physics_steer_angle', 'Physics_brake', 'Physics_gas', 'Physics_speed_kmh', 'Physics_heading',
    'Physics_velocity_x', 'Physics_local_velocity_y', 'Graphics_normalized_car_position', 'x', 'y', 'z'] as const)(
    'reduces oscillating %s prediction error while tracking a steady trend', (field) => {
        const predictor = new PhraseLookahead(), smoothed: number[] = [], raw: number[] = [];
        let previous = 0;
        for (let i = 0; i <= 30; i++) {
            const t = i / 10, current = 0.4 + 0.02 * t + (i % 2 === 0 ? 0.002 : -0.002);
            const coordinate = field === 'x' || field === 'y' || field === 'z';
            const sample: StandardTelemetrySample = coordinate
                ? { Graphics_player_car_id: 42, Graphics_car_id: [42], Graphics_car_coordinates: [{ x: 0, y: 0, z: 0, [field]: current }] }
                : { Physics_speed_kmh: 36, [field]: current };
            predictor.update(sample, i * 100, i > 0);
            const projected = predictor.project(sample, null, 500);
            if (i >= 10) {
                const ideal = 0.4 + 0.02 * (t + 0.5);
                const value = coordinate ? projected.Graphics_car_coordinates![0][field] : projected[field]!;
                smoothed.push(value - ideal);
                raw.push(current + (current - previous) / 0.1 * 0.5 - ideal);
            }
            previous = current;
        }
        const range = (values: number[]) => Math.max(...values) - Math.min(...values);
        expect(range(smoothed)).toBeLessThan(range(raw) * 0.5);
        expect(Math.abs(smoothed.reduce((sum, value) => sum + value, 0) / smoothed.length)).toBeLessThan(0.001);
    });

it('settles a sustained rate change within 500 ms regardless of frame cadence or repeated reads', () => {
    const run = (step: number, read: boolean) => {
        const predictor = new PhraseLookahead();
        const sampleAt = (at: number) => ({ Physics_steer_angle: 0.2 + Math.max(0, at - 100) / 1000 * 0.2 });
        [0, 100, 200].forEach((at) => predictor.update(sampleAt(at), at, at > 0));
        for (let at = 200 + step; at <= 600; at += step) {
            predictor.update(sampleAt(at), at, true);
            if (read) {
                const projected = predictor.project(sampleAt(at), null, 500);
                for (let i = 0; i < 5; i++) expect(predictor.project(sampleAt(at), null, 500)).toEqual(projected);
            }
        }
        return predictor.project(sampleAt(600), null, 500).Physics_steer_angle!;
    };
    const result = run(100, false);
    expect(result).toBeGreaterThan(0.395);
    expect(result).toBeLessThan(0.4);
    expect(run(10, true)).toBeCloseTo(result, 9);
    expect(run(25, false)).toBeCloseTo(result, 9);
});

it('shares the smoothed brake trend with release detection and forgets it after invalid data', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (brake: number | undefined) => ({ Physics_brake: brake });
    predictor.update(sampleAt(0.6), 0, false);
    predictor.update(sampleAt(0.58), 100, true);
    // A brief upward tick must not immediately reverse the established release trend.
    predictor.update(sampleAt(0.585), 200, true);
    expect(predictor.project(sampleAt(0.585), null, 500).Physics_brake).toBeLessThan(0.585);
    expect(predictor.projectBrakeReleasing(sampleAt(0.585))).toBe(true);
    predictor.update(sampleAt(undefined), 300, true);
    expect(predictor.projectBrakeReleasing(sampleAt(undefined))).toBeUndefined();
    predictor.update(sampleAt(0.6), 400, true);
    expect(predictor.project(sampleAt(0.6), null, 500).Physics_brake).toBe(0.6);
    predictor.update(sampleAt(0.6), 500, true);
    expect(predictor.project(sampleAt(0.6), null, 500).Physics_brake).toBe(0.6);
    expect(predictor.projectBrakeReleasing(sampleAt(0.6))).toBe(false);
});

it('waits for 100 ms of history and uses the eligible anchor between stored samples', () => {
    const predictor = new PhraseLookahead();
    predictor.update(motion(0), 0, false);
    predictor.update(motion(0.05), 50, true);
    expect(predictor.project(motion(0.05), null, 500).Physics_steer_angle).toBe(motion(0.05).Physics_steer_angle);
    predictor.update(motion(0.1), 100, true);
    expect(predictor.project(motion(0.1), null, 500).Physics_steer_angle).toBeCloseTo(0.226, 9);
    [0.2, 0.4, 0.45].forEach((t) => predictor.update(motion(t), t * 1000, true));
    expect(predictor.project(motion(0.45), null, 500).Physics_steer_angle).toBeCloseTo(0.337155818, 9);
});

it('tracks the player by ID when coordinate slots change', () => {
    const predictor = new PhraseLookahead();
    [0, 0.2, 0.4].forEach((t, i) => {
        const sample = motion(t);
        if (i === 1) { sample.Graphics_car_id!.reverse(); sample.Graphics_car_coordinates!.reverse(); }
        predictor.update(sample, t * 1000, i > 0);
    });
    expect(predictor.project(motion(0.4), null, 500).Graphics_car_coordinates![1].x).toBeCloseTo(109.814561145, 9);
});

it.each([0, 2, -2, -6])('smooths deceleration without projecting curvature for a %s m/s³ change', (change) => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({ ...motion(t), Physics_speed_kmh: (30 - 4 * t - change * t ** 2 / 2) * 3.6 });
    [0, 0.13, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
    const deceleration = predictor.projectDeceleration(sampleAt(0.4));
    if (change === 0) expect(deceleration).toBeCloseTo(4, 9);
    else {
        const rates = [4 + change * 0.065, 4 + change * 0.265];
        expect(deceleration).toBeGreaterThan(Math.min(...rates));
        expect(deceleration).toBeLessThan(Math.max(...rates));
    }
    const speed = predictor.project(sampleAt(0.4), null, 500).Physics_speed_kmh!;
    expect(deceleration).toBeCloseTo((sampleAt(0.4).Physics_speed_kmh - speed) / 3.6 / 0.5, 9);
});

it.each([undefined, NaN, Infinity, -1])('withholds predicted deceleration with insufficient or invalid speed history: %s', (speed) => {
    const predictor = new PhraseLookahead();
    predictor.update(motion(0), 0, false);
    expect(predictor.projectDeceleration(motion(0))).toBeUndefined();
    predictor.update({ ...motion(0.2), Physics_speed_kmh: speed }, 200, true);
    predictor.update(motion(0.4), 400, true);
    expect(predictor.projectDeceleration(motion(0.4))).toBeUndefined();
    predictor.reset();
    expect(predictor.projectDeceleration(motion(0.4))).toBeUndefined();
});

it.each([
    ['still applying', 0.4, -0.8, false], ['still releasing', -0.4, 0.8, true],
    ['holding', 0, 0, false], ['slow release', -0.005, 0, false], ['release', -0.02, 0, true],
] as const)('detects %s from the first brake derivative with uneven samples', (_name, rate, acceleration, releasing) => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({ ...motion(t), Physics_brake: 0.5 + rate * t + acceleration * t ** 2 / 2 });
    [0, 0.13, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
    expect(predictor.projectBrakeReleasing(sampleAt(0.4), 500)).toBe(releasing);
    predictor.reset();
    expect(predictor.projectBrakeReleasing(sampleAt(0.4), 500)).toBeUndefined();
});

it.each([undefined, NaN, Infinity, -0.1, 1.1])('withholds predicted brake release with insufficient or invalid pressure history: %s', (pressure) => {
    const predictor = new PhraseLookahead();
    predictor.update(motion(0), 0, false);
    expect(predictor.projectBrakeReleasing(motion(0))).toBeUndefined();
    predictor.update({ ...motion(0.2), Physics_brake: pressure }, 200, true);
    predictor.update(motion(0.4), 400, true);
    expect(predictor.projectBrakeReleasing(motion(0.4))).toBeUndefined();
    predictor.update(motion(0.6), 600, true);
    predictor.update(motion(0.8), 800, true);
    expect(predictor.projectBrakeReleasing(motion(0.8))).toBe(false);
    expect(predictor.projectBrakeReleasing({ ...motion(0.8), Physics_brake: pressure })).toBeUndefined();
});

it('unwraps normalized position and heading before taking derivatives', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({ ...motion(t),
        Graphics_normalized_car_position: (0.999 + 0.01 * t) % 1,
        Physics_heading: ((Math.PI - 0.01 + 0.1 * t + Math.PI) % (2 * Math.PI)) - Math.PI,
    });
    [0, 0.2, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
    const projected = predictor.project(sampleAt(0.4), null, 500);
    expect(projected.Graphics_normalized_car_position).toBeCloseTo(0.008, 9);
    expect(Math.sin(projected.Physics_heading!)).toBeCloseTo(Math.sin(Math.PI + 0.08), 9);
});

it('bounds speed and steering and keeps position fixed when live speed is zero', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({ ...motion(t),
        Graphics_normalized_car_position: 0.1 + 0.01 * t - 0.01 * t ** 2,
        Physics_speed_kmh: 36 - 72 * t, Physics_steer_angle: 0.4 + t + t ** 2,
    });
    [0, 0.2, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
    const projected = predictor.project(sampleAt(0.4), null, 500);
    expect(projected.Graphics_normalized_car_position).toBeCloseTo(0.104927194, 9);
    expect(projected.Physics_speed_kmh).toBe(0);
    expect(predictor.projectDeceleration(sampleAt(0.4))).toBe(0);
    expect(projected.Physics_steer_angle).toBe(1);
    const stopped = { ...sampleAt(0.4), Physics_speed_kmh: 0 };
    expect(predictor.project(stopped, null, 500).Graphics_normalized_car_position).toBe(stopped.Graphics_normalized_car_position);
});

it('falls back per channel when history is insufficient or a measurement is missing', () => {
    const { reference } = steeringCornerFixture(), predictor = new PhraseLookahead();
    predictor.update(motion(0), 0, false);
    const initial = predictor.project(motion(0), reference, 500);
    expect(initial.Graphics_normalized_car_position).toBeCloseTo(0.105, 9);
    expect(initial.Physics_steer_angle).toBe(0.1);
    predictor.update({ ...motion(0.2), Physics_velocity_x: undefined }, 200, true);
    predictor.update(motion(0.4), 400, true);
    const projected = predictor.project(motion(0.4), reference, 500);
    expect(projected.Physics_velocity_x).toBe(motion(0.4).Physics_velocity_x);
    expect(projected.Physics_velocity_y).toBeCloseTo(-0.392719428, 9);
});

it.each(['Physics_brake', 'Physics_gas'] as const)('bounds intended %s during application and release', (field) => {
    for (const releasing of [false, true]) {
        const predictor = new PhraseLookahead();
        const sampleAt = (t: number) => ({ ...motion(t), [field]: releasing ? 0.8 - t : 0.2 + t });
        [0, 0.2, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
        expect(predictor.project(sampleAt(0.4), null, 500)[field]).toBe(releasing ? 0 : 1);
        if (field === 'Physics_brake') expect(predictor.projectBrakeReleasing(sampleAt(0.4), 500)).toBe(false);
    }
});

it.each(['Physics_brake', 'Physics_gas'] as const)('preserves measured %s until its history is valid', (field) => {
    for (const invalid of [undefined, NaN, Infinity, -0.1, 1.1]) {
        const predictor = new PhraseLookahead(), sample = motion(0.4);
        predictor.update(motion(0), 0, false);
        expect(predictor.project(motion(0), null, 500)[field]).toBe(motion(0)[field]);
        predictor.update({ ...motion(0.2), [field]: invalid }, 200, true);
        predictor.update(sample, 400, true);
        expect(predictor.project(sample, null, 500)[field]).toBe(sample[field]);
        expect(predictor.project(sample, null, 500).Physics_steer_angle).toBeCloseTo(0.320728057, 9);
        predictor.update({ ...motion(0.6), [field]: invalid }, 600, true);
        expect(predictor.project({ ...motion(0.6), [field]: invalid }, null, 500)[field]).toBe(invalid);
    }
});

it('uses predicted local velocity but preserves the captured orientation of the lateral error', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({ ...motion(t), Physics_heading: 0, Physics_local_velocity_x: 10 * t ** 2,
        Physics_local_velocity_y: 10, Physics_local_velocity_z: 0 });
    [0, 0.2, 0.4].forEach((t, i) => predictor.update(sampleAt(t), t * 1000, i > 0));
    const sample = sampleAt(0.4), trajectory = { heading: Math.PI / 2, remainingDistance: 100 };
    expect(compareSteeringDirection(trajectory, [0, 0, 0], sample, 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection(trajectory, [0, 0, 0], sample, 'acc', 1, predictor.project(sample, null, 500))).toBe('right');
    // The newer heading must rotate motion only; the old lateral error still points along world -X.
    expect(compareSteeringDirection({ heading: Math.PI / 2, remainingDistance: 10 }, [4, 0, 0], sample, 'acc', 1,
        { ...sample, Physics_heading: Math.PI / 2, Physics_local_velocity_x: 10, Physics_local_velocity_y: 4 })).toBe('aligned');
});

it('does not anticipate a speed reversal from curvature in engine reference comparisons', () => {
    const { reference } = steeringCornerFixture();
    reference.y.forEach((row) => { row[4] = 0.8; });
    let guidance: ReferenceGuidance = {};
    let intendedSpeed: number | undefined;
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure('inspect', '', (context) => {
            guidance = context.guidance; intendedSpeed = context.intendedTelemetry.Physics_speed_kmh; return false;
        }, []),
    ]));
    engine.receiveMap(null, 0, reference);
    const sampleAt = (t: number) => ({ ...motion(t), Physics_brake: 0, Physics_speed_kmh: 36 + 24 * (t - 0.4) ** 2 });
    [0, 0.2, 0.4].forEach((t) => engine.receiveTelemetry(frame(sampleAt(t)), 1000 + t * 1000));
    // Measured speed is 36 km/h; the smoothed downward rate still predicts slowing.
    expect(readReferenceGuidance(sampleAt(0.4), null, reference, 'acc', 1400).brake).toBeUndefined();
    expect(intendedSpeed).toBeCloseTo(28.669467474, 9);
    expect(guidance.brake).toBeUndefined();
});

it.each([[3.2, 'Hold brake pressure.'], [4, undefined]] as const)(
    'compares predicted deceleration with the %s m/s² target at the lookahead position', (required, sentence) => {
        const { reference } = steeringCornerFixture(), engine = new PhraseEngine();
        const sampleAt = (t: number) => ({ ...motion(t), Graphics_normalized_car_position: 0.1 + 0.03 * t,
            Physics_speed_kmh: (30 - 4 * t + t ** 2) * 3.6, Physics_brake: 0.8, Physics_gas: 0 });
        // At t=0.6, the latest rate predicts 133 m, 26.36 m/s and 3.2 m/s² deceleration.
        reference.y.forEach((row, i) => {
            row[4] = i >= 5 && i < 23 ? 0.8 : 0;
            row[6] = i <= 12 ? 108 + (12 - i) * 4
                : Math.sqrt(Math.max(1, 26.36 ** 2 + 2 * required * (133 - i * 10))) * 3.6;
        });
        engine.setDrivingAggression(0, 0);
        engine.receiveMap(null, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(0.2)), 1200).events).toEqual([]);
        const result = engine.receiveTelemetry(frame(sampleAt(0.6)), 1600);
        expect(result.events).toEqual(sentence ? [expect.objectContaining({ sentence, timestamp: 1600 })] : []);
        // A predicted deficit alone cannot request More brake.
        const intended = { ...sampleAt(0.6), Physics_speed_kmh: 26.36 * 3.6, Graphics_normalized_car_position: 0.133 };
        const measured = readReferenceGuidance(sampleAt(0.6), null, reference, 'acc', 1600, reference, 1.8,
            undefined, undefined, undefined, false, 0.133, intended);
        expect(measured.moreBrake).toBeUndefined();
        expect(engine.evaluate(1700).events).toEqual(result.events);
    });

it.each([1, -1])(
    'withholds removed release cues when the measured pedal trend is %s', (direction) => {
        const { reference } = steeringCornerFixture(), engine = new PhraseEngine();
        const sampleAt = (t: number) => ({ ...motion(t), Graphics_normalized_car_position: 0.1 + 0.03 * t,
            Physics_speed_kmh: (30 - 4 * t + t ** 2) * 3.6,
            Physics_brake: 0.5 + direction * (0.4 * t - 0.25 * t ** 2), Physics_gas: 0 });
        reference.y.forEach((row, i) => {
            row[4] = i >= 5 && i < 23 ? 0.8 : 0;
            row[6] = i <= 12 ? 108 + (12 - i) * 4
                : Math.sqrt(Math.max(1, 26.36 ** 2 + 2 * 4 * (133 - i * 10))) * 3.6;
        });
        engine.setDrivingAggression(0, 0);
        engine.receiveMap(null, 0, reference);
        expect(engine.receiveTelemetry(frame(sampleAt(0.2)), 1200).events).toEqual([]);
        // Neither pedal trend should produce a removed release cue.
        const result = engine.receiveTelemetry(frame(sampleAt(0.6)), 1600);
        expect(result.events).toEqual([]);
        expect(engine.evaluate(1700).events).toEqual(result.events);
    });

it.each(['frozen', 'teleport'] as const)('discards %s position history before projecting motion', (change) => {
    const { reference } = steeringCornerFixture(), predictor = new PhraseLookahead();
    [0, 0.2, 0.4].forEach((t, i) => predictor.update(motion(t), t * 1000, i > 0, reference));
    const sample = { ...motion(0.45), Graphics_normalized_car_position: change === 'frozen'
        ? motion(0.4).Graphics_normalized_car_position : 0.8 };
    predictor.update(sample, change === 'frozen' ? 500 : 450, true, reference);
    const projected = predictor.project(sample, reference, 500);
    expect(projected.Physics_steer_angle).toBe(sample.Physics_steer_angle);
    expect(projected.Graphics_normalized_car_position).toBe(referencePositionAhead(sample, reference, 500));
    expect(predictor.projectDeceleration(sample)).toBeUndefined();
    expect(predictor.projectBrakeReleasing(sample)).toBeUndefined();
});

it.each(['Physics_steer_angle', 'Physics_brake', 'Physics_gas'] as const)(
    'supplies intended %s to closure entry and exit, without advancing on timer evaluations', (field) => {
    const observed: StandardTelemetrySample[] = [];
    const sampleAt = (t: number) => ({ ...motion(t), [field]: 0.1 + 0.2 * t + 0.1 * t ** 2 });
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure('prediction', '', ({ intendedTelemetry }) => {
            observed.push(intendedTelemetry);
            return (intendedTelemetry[field] ?? 0) >= 0.44;
        }, [Action.exitToRoot(({ intendedTelemetry }) => (intendedTelemetry[field] ?? 0) >= 0.56)]),
    ]));
    engine.receiveTelemetry(frame(sampleAt(0)), 1000);
    engine.receiveTelemetry(frame(sampleAt(0.2)), 1200);
    expect(engine.receiveTelemetry(frame(sampleAt(0.4)), 1400).state.path).toEqual(['root', 'prediction']);
    expect(observed[observed.length - 1].Physics_speed_kmh).toBeCloseTo(78.93824458, 8);
    const beforeTimer = observed[observed.length - 1];
    expect(engine.evaluate(1450).state.path).toEqual(['root', 'prediction']);
    expect(observed[observed.length - 1]).toEqual(beforeTimer);
    expect(engine.receiveTelemetry(frame(sampleAt(0.6)), 1600).state.path).toEqual(['root', 'prediction']);
    // Smoothing delays this crossing slightly, but it still precedes the measured control.
    expect(sampleAt(0.65)[field]).toBeLessThan(0.4);
    expect(engine.receiveTelemetry(frame(sampleAt(0.65)), 1650).state.path).toEqual(['root']);
});

it.each(['session', 'stream', 'track', 'car', 'pause', 'gap', 'clock'] as const)(
    'discards derivative history after a %s discontinuity', (change) => {
        let intended: StandardTelemetrySample = {};
        const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
            new Closure('inspect', '', (context) => { intended = context.intendedTelemetry; return false; }, []),
        ]));
        [0, 0.2, 0.4].forEach((t) => engine.receiveTelemetry(frame(motion(t)), 1000 + t * 1000));
        expect(intended.Physics_steer_angle).toBeCloseTo(0.445456114, 9);
        const next = frame(motion(0.6));
        if (change === 'session') next.sessionGeneration++;
        if (change === 'stream') next.streamGeneration++;
        if (change === 'track') next.sample = { ...next.sample, Static_track: 'other' };
        if (change === 'car') next.sample = { ...next.sample, Static_car_model: 'other' };
        if (change === 'pause') engine.receiveTelemetry({ ...next, telemetryStatus: 1 }, 1500);
        engine.receiveTelemetry(next, change === 'gap' ? 3000 : change === 'clock' ? 1400 : 1600);
        expect(intended.Physics_steer_angle).toBe(next.sample.Physics_steer_angle);
        expect(intended.Physics_brake).toBe(next.sample.Physics_brake);
        expect(intended.Physics_gas).toBe(next.sample.Physics_gas);
    });
