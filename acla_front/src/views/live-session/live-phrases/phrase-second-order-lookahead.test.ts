import type { StandardTelemetrySample } from '../live-session-types';
import { PhraseLookahead } from './phrase-lookahead';

const motion = (t: number): StandardTelemetrySample => ({
    Physics_speed_kmh: 72 + 3.6 * t + 1.8 * t ** 2,
    Physics_steer_angle: 0.1 + 0.05 * t + 0.025 * t ** 2,
    Physics_brake: 0.7 + 0.1 * t - 0.1 * t ** 2,
    Physics_gas: 0.2 + 0.04 * t + 0.02 * t ** 2,
    Graphics_normalized_car_position: 0.2 + 0.02 * t + 0.005 * t ** 2,
    Physics_heading: 0.1 + 0.2 * t + 0.05 * t ** 2,
    Physics_pitch: -0.1 - 0.15 * t + 0.025 * t ** 2,
    Physics_roll: 0.2 + 0.03 * t + 0.01 * t ** 2,
    Physics_velocity_x: 12 + t + 2 * t ** 2,
    Physics_velocity_y: 2 - 0.5 * t - 0.1 * t ** 2,
    Physics_velocity_z: -2 + 0.3 * t + 0.2 * t ** 2,
    Physics_local_velocity_x: 0.1 + 0.2 * t + 0.4 * t ** 2,
    Physics_local_velocity_y: 20 + 3 * t + 0.2 * t ** 2,
    Physics_local_velocity_z: 2 - 0.4 * t + 0.3 * t ** 2,
    Graphics_player_car_id: 42, Graphics_car_id: [7, 42],
    Graphics_car_coordinates: [{ x: 1000, y: 0, z: 0 },
        { x: 10 + 2 * t + 3 * t ** 2, y: 1 - t + 0.5 * t ** 2, z: 20 - 3 * t - 0.2 * t ** 2 }],
});

it('follows quadratic telemetry through uneven and sub-anchor updates without mutating samples', () => {
    const predictor = new PhraseLookahead(), times = [0, 0.13, 0.4, 0.45, 0.6];
    const samples = times.map(motion), before = JSON.stringify(samples);
    times.forEach((at, index) => {
        predictor.update(samples[index], at * 1000, index > 0);
        if (index < 2) return;
        const projected = predictor.project(samples[index], null, 500), expected = motion(at + 0.5);
        for (const field of ['Physics_speed_kmh', 'Physics_steer_angle', 'Physics_brake', 'Physics_gas',
            'Graphics_normalized_car_position', 'Physics_heading', 'Physics_pitch', 'Physics_roll',
            'Physics_velocity_x', 'Physics_velocity_y', 'Physics_velocity_z',
            'Physics_local_velocity_x', 'Physics_local_velocity_y', 'Physics_local_velocity_z'] as const) {
            expect(projected[field]).toBeCloseTo(expected[field]!, 8);
        }
        for (const axis of ['x', 'y', 'z'] as const) {
            expect(projected.Graphics_car_coordinates![1][axis]).toBeCloseTo(expected.Graphics_car_coordinates![1][axis], 8);
        }
        expect(projected.Graphics_car_coordinates![0]).toEqual(samples[index].Graphics_car_coordinates![0]);
        expect(predictor.project(samples[index], null, 0)).toEqual(samples[index]);
    });
    expect(JSON.stringify(samples)).toBe(before);
});

it('preserves curvature across lap and orientation wrapping', () => {
    const predictor = new PhraseLookahead();
    const wrap = (value: number, period: number) => ((value % period) + period) % period;
    const sampleAt = (t: number) => ({ ...motion(t),
        Graphics_normalized_car_position: wrap(0.999 + 0.01 * t + 0.005 * t ** 2, 1),
        Physics_heading: wrap(Math.PI - 0.015 + 0.1 * t + 0.05 * t ** 2 + Math.PI, 2 * Math.PI) - Math.PI,
        Physics_pitch: wrap(-Math.PI + 0.02 - 0.15 * t + 0.025 * t ** 2 + Math.PI, 2 * Math.PI) - Math.PI,
        Physics_roll: wrap(Math.PI - 0.005 + 0.03 * t + 0.01 * t ** 2 + Math.PI, 2 * Math.PI) - Math.PI,
    });
    [0, 0.13, 0.4].forEach((at, index) => predictor.update(sampleAt(at), at * 1000, index > 0));
    const projected = predictor.project(sampleAt(0.4), null, 500), expected = sampleAt(0.9);
    expect(projected.Graphics_normalized_car_position).toBeCloseTo(expected.Graphics_normalized_car_position, 8);
    for (const field of ['Physics_heading', 'Physics_pitch', 'Physics_roll'] as const) {
        expect(Math.sin(projected[field]!)).toBeCloseTo(Math.sin(expected[field]), 8);
        expect(Math.cos(projected[field]!)).toBeCloseTo(Math.cos(expected[field]), 8);
    }
});

it('uses each retained horizon for curved speed, deceleration and pedal release', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({
        Physics_speed_kmh: (10 - 3 * t + t ** 2) * 3.6,
        Physics_brake: 0.6 + 0.08 * t - 0.1 * t ** 2,
    });
    [0, 0.13, 0.4].forEach((at, index) => predictor.update(sampleAt(at), at * 1000, index > 0));
    const sample = sampleAt(0.4), frames = predictor.getFrames(sample, null);
    expect(frames).toHaveLength(31);
    [...frames, predictor.getFrame(sample, null, 333)].forEach((frame) => {
        const at = 0.4 + frame.offsetMs / 1000, expected = sampleAt(at);
        expect(frame.intendedTelemetry.Physics_speed_kmh).toBeCloseTo(expected.Physics_speed_kmh, 8);
        expect(frame.intendedTelemetry.Physics_brake).toBeCloseTo(Math.max(0, expected.Physics_brake), 8);
        expect(frame.deceleration).toBeCloseTo(3 - 2 * at, 8);
        expect(frame.brakeReleasing).toBe(expected.Physics_brake > 0 && 0.2 * at - 0.08 > 0.01);
        expect(predictor.getFrame(sample, null, frame.offsetMs)).toBe(frame);
    });
    expect(frames[0].brakeReleasing).toBe(false);
    expect(frames[5].brakeReleasing).toBe(true);
    expect(frames[30].brakeReleasing).toBe(false);
    expect(frames[0].deceleration).toBeGreaterThan(0);
    expect(frames[30].deceleration).toBeLessThan(0);
    expect(predictor.getFrame(sample, null, 9000)).toBe(frames[30]);
    expect(predictor.getFrames(sample, null)).toBe(frames);
});

it('uses a secant until three valid samples establish curvature', () => {
    const predictor = new PhraseLookahead(), first = motion(0.13), second = motion(0.4);
    predictor.update(first, 130, false);
    expect(predictor.projectDeceleration(first)).toBeUndefined();
    expect(predictor.projectBrakeReleasing(first)).toBeUndefined();
    predictor.update(second, 400, true);
    const projected = predictor.project(second, null, 500);
    for (const field of ['Physics_speed_kmh', 'Physics_steer_angle', 'Physics_brake', 'Physics_gas'] as const) {
        expect(projected[field]).toBeCloseTo(second[field]! + (second[field]! - first[field]!) / 0.27 * 0.5, 8);
    }
    expect(predictor.projectDeceleration(second, 500))
        .toBeCloseTo(-(second.Physics_speed_kmh! - first.Physics_speed_kmh!) / 0.27 / 3.6, 8);
});

it('rebuilds curvature per channel after invalid measurements', () => {
    const predictor = new PhraseLookahead();
    const sampleAt = (t: number) => ({
        Physics_speed_kmh: 36 - 7.2 * t + 3.6 * t ** 2,
        Physics_brake: 0.6 - 0.1 * t + 0.05 * t ** 2,
    });
    [0, 0.1, 0.2].forEach((at, index) => predictor.update(sampleAt(at), at * 1000, index > 0));
    predictor.update({ Physics_speed_kmh: NaN, Physics_brake: undefined }, 300, true);
    predictor.update(sampleAt(0.4), 400, true);
    expect(predictor.project(sampleAt(0.4), null, 500).Physics_speed_kmh).toBe(sampleAt(0.4).Physics_speed_kmh);
    expect(predictor.project(sampleAt(0.4), null, 500).Physics_brake).toBe(sampleAt(0.4).Physics_brake);
    expect(predictor.projectDeceleration(sampleAt(0.4), 500)).toBeUndefined();
    expect(predictor.projectBrakeReleasing(sampleAt(0.4), 500)).toBeUndefined();
    predictor.update(sampleAt(0.5), 500, true);
    const linear = predictor.project(sampleAt(0.5), null, 500);
    for (const field of ['Physics_speed_kmh', 'Physics_brake'] as const) {
        expect(linear[field]).toBeCloseTo(sampleAt(0.5)[field] + (sampleAt(0.5)[field] - sampleAt(0.4)[field]) / 0.1 * 0.5, 8);
    }
    predictor.update(sampleAt(0.6), 600, true);
    const quadratic = predictor.project(sampleAt(0.6), null, 500);
    expect(quadratic.Physics_speed_kmh).toBeCloseTo(sampleAt(1.1).Physics_speed_kmh, 8);
    expect(quadratic.Physics_brake).toBeCloseTo(sampleAt(1.1).Physics_brake, 8);
    expect(predictor.projectDeceleration(sampleAt(0.6), 500)).toBeCloseTo(-0.2, 8);
    expect(predictor.projectBrakeReleasing(sampleAt(0.6), 500)).toBe(false);
});

it.each(['reset', 'discontinuous'] as const)('forgets both derivatives after %s', (kind) => {
    const predictor = new PhraseLookahead();
    [0, 0.13, 0.4].forEach((at, index) => predictor.update(motion(at), at * 1000, index > 0));
    if (kind === 'reset') predictor.reset();
    const sample = motion(0.6);
    predictor.update(sample, 600, kind === 'reset');
    const projected = predictor.project(sample, null, 500);
    expect(projected.Physics_speed_kmh).toBe(sample.Physics_speed_kmh);
    expect(projected.Physics_brake).toBe(sample.Physics_brake);
    expect(projected.Physics_steer_angle).toBe(sample.Physics_steer_angle);
    expect(predictor.projectDeceleration(sample)).toBeUndefined();
    expect(predictor.projectBrakeReleasing(sample)).toBeUndefined();
});

it.each([
    ['Physics_brake', 0, 0.2], ['Physics_brake', 1, 0.8],
    ['Physics_gas', 0, 0.2], ['Physics_gas', 1, 0.8],
    ['Physics_steer_angle', -1, -0.8], ['Physics_steer_angle', 1, 0.8],
    ['Physics_speed_kmh', 0, 7.2],
] as const)('keeps %s at a repeated physical boundary of %s', (field, boundary, earlier) => {
    const predictor = new PhraseLookahead();
    predictor.update({ [field]: earlier }, 0, false);
    predictor.update({ [field]: boundary }, 200, true);
    const sample = { [field]: boundary };
    predictor.update(sample, 1100, true);
    predictor.getFrames(sample, null).forEach((frame) => {
        expect(frame.intendedTelemetry[field]).toBe(boundary);
        if (field === 'Physics_brake') expect(frame.brakeReleasing).toBe(false);
        if (field === 'Physics_speed_kmh') expect(frame.deceleration).toBe(0);
    });
});
