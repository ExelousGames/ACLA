import { drivingPlansAgree } from './driving-plan-agreement';
import { NEUTRAL } from './driving-observation';
import type { PolicyDecision } from './driving-policy';

const plan = (points: number[][]): PolicyDecision['plan'] => points.map(([timeSeconds, throttle = 0, brake = 0, steering = 0, shift = 0]) => ({
    timeSeconds, means: [Math.atanh(throttle), Math.atanh(brake), Math.atanh(steering)], latent: [0, 0, 0],
    action: { ...NEUTRAL }, shift: 0, shifts: { up: true, down: true },
    shiftProbabilities: [0, 1, 2].map((choice) => Number(choice === shift)),
}));

it('aligns the old remaining plan with the fresh execution timeline', () => {
    const previous = plan([[0, 0, 0.8], [1, 0], [2, 0.8]]);
    const next = plan([[0, 0], [1, 0.8], [2, 0.8]]);
    expect(drivingPlansAgree(previous, 1, next, 0, 0.05)).toBe(true);
    expect(drivingPlansAgree(previous, 0, next, 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 1.5, next, 0.5, 0.05)).toBe(true);
});

it('checks both interpolation grids and the distant tail, not just the next action', () => {
    const previous = plan([[0], [1], [2]]);
    expect(drivingPlansAgree(previous, 0, plan([[0], [0.5, 0, 0, 0.2], [2]]), 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 0, plan([[0], [1], [2, 0.2]]), 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 0, plan([[0], [0.5, 0.05, 0.05, -0.05], [2]]), 0, 0.05)).toBe(true);
});

it('compares throttle and brake separately even when their signed difference matches', () => {
    const previous = plan([[0, 0.5], [1, 0.5]]);
    const next = plan([[0, 0.8, 0.3], [1, 0.8, 0.3]]);
    expect(drivingPlansAgree(previous, 0, next, 0, 0.05)).toBe(false);
});

it('compares released pedal intent after clamping and before interpolation', () => {
    const previous = plan([[0], [1, 0.8, 0.6]]);
    previous[0].means = [-1, -2, 0];
    const next = plan([[0, 0.4, 0.3], [0.5, 0.8, 0.6], [1, 0.8, 0.6]]);
    expect(drivingPlansAgree(previous, 0.5, next, 0, 0.05)).toBe(true);
});

it('requires the same categorical intent and ignores independent exploration draws', () => {
    const previous = plan([[0], [1, 0, 0, 0, 1], [2]]);
    const next = plan([[0], [1, 0, 0, 0, 1], [2]]);
    next[0].action.steering = 0.8;
    next[1].shift = 2;
    expect(drivingPlansAgree(previous, 0, next, 0, 0.05)).toBe(true);
    next[1].shiftProbabilities = [0, 0, 1];
    expect(drivingPlansAgree(previous, 0, next, 0, 0.05)).toBe(false);
});

it('rejects exhausted plans and fresh predictions that leave an unchecked tail', () => {
    const previous = plan([[0], [1], [2]]);
    expect(drivingPlansAgree(previous, 2, previous, 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 2.1, previous, 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 0, previous, 0.1, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 0.5, plan([[0], [1]]), 0, 0.05)).toBe(false);
    expect(drivingPlansAgree([], 0, previous, 0, 0.05)).toBe(false);
});

it.each([NaN, Infinity, -1])('rejects invalid elapsed time %s', (elapsed) => {
    const previous = plan([[0], [1]]);
    expect(drivingPlansAgree(previous, elapsed, previous, 0, 0.05)).toBe(false);
    expect(drivingPlansAgree(previous, 0, previous, elapsed, 0.05)).toBe(false);
});
