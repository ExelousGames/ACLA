import { drivingPlanControlsDiffer } from './driving-plan-agreement';
import type { PolicyDecision, ShiftChoice } from './driving-policy';

type PlanPoint = [timeSeconds: number, pedal?: number, steering?: number, shift?: ShiftChoice];
const plan = (points: PlanPoint[]): PolicyDecision['plan'] => points.map(([timeSeconds, pedal = 0, steering = 0, shift = 0]) => ({
    timeSeconds, means: [Math.atanh(pedal), Math.atanh(steering)], latent: [Math.atanh(pedal), Math.atanh(steering)],
    action: { pedal, steering, shiftUp: shift === 1, shiftDown: shift === 2 },
    shift, shifts: { up: true, down: true }, shiftProbabilities: [1, 0, 0],
}));

it('ignores steps already executed and aligns the remaining steps with the interpolated fresh plan', () => {
    const current = plan([[0, -0.9], [0.5, -0.9], [1, 0.2], [2, 0.4]]);
    const next = plan([[0], [1, 0.4], [2, 0.4], [3, -1]]);
    expect(drivingPlanControlsDiffer(current, 0.5, next)).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(true);
});

it.each([[0.399, 0.399, false], [0.4, 0, true], [0, -0.4, true], [0.6, 0, true]] as const)
('requires at least 0.40 on either axis independently (pedal=%s, steering=%s)', (pedal, steering, differs) => {
    const current = plan([[0], [1], [2], [3]]);
    const next = plan([[0, pedal, steering], [1, pedal, steering], [2, pedal, steering], [3, pedal, steering]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(differs);
});

it('averages absolute differences across the whole remaining plan without cancelling opposite signs', () => {
    const current = plan([[0], [1], [2], [3], [4]]);
    expect(drivingPlanControlsDiffer(current, 0, plan([[0], [1], [2], [3], [4, 0, 0.9]]))).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0, plan([[0], [1], [2], [3, 0, -0.8], [4, 0, 0.8]]))).toBe(true);
});

it('counts a throttle-to-brake reversal as a change on the signed pedal', () => {
    const current = plan([[0, 0.25], [1, 0.25], [2, 0.25]]);
    const next = plan([[0, -0.15], [1, -0.15], [2, -0.15]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(true);
});

it('includes exactly 40% despite rounding across a dense plan', () => {
    const times = Array.from({ length: 64 }, (_, step) => step / 10);
    expect(drivingPlanControlsDiffer(plan(times.map((time) => [time])), 0,
        plan(times.map((time) => [time, 0.4])))).toBe(true);
});

it.each([[0, 1], [1, 0], [1, 2]] as const)
('refreshes on a later sampled gear choice change from %s to %s even with identical shift probabilities', (before, after) => {
    const current = plan([[0], [1], [2], [3, 0, 0, before]]);
    const next = plan([[0], [1], [2, 0, 0, after], [3]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(true);
});

it('keeps matching future shifts aligned even when their step indices differ', () => {
    const current = plan([[0, 0, 0, 2], [0.1, 0, 0, 2], [0.3, 0, 0, 1], [0.4]]);
    const next = plan([[0], [0.1], [0.2, 0, 0, 1], [0.3], [0.4]]);
    expect(drivingPlanControlsDiffer(current, 0.1, next)).toBe(false);
});

it.each([[0.5, 1.5], [1.5, 0.5]] as const)('keeps the plan when the same pending shift moves from %s to %s seconds', (before, after) => {
    const current = plan([[0], [before, 0, 0, 1], [2]]);
    const next = plan([[0], [after, 0, 0, 1], [2]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(false);
});

it('catches fresh shifts at time zero and between outgoing steps', () => {
    const current = plan([[0], [1], [2], [3]]);
    expect(drivingPlanControlsDiffer(current, 1, plan([[0, 0, 0, 1], [1], [2], [3]]))).toBe(true);
    expect(drivingPlanControlsDiffer(current, 1, plan([[0], [0.5, 0, 0, 2], [1], [2], [3]]))).toBe(true);
});

it('does not refresh just to repeat an already issued shift but accepts a different immediate shift', () => {
    const current = plan([[0, 0, 0, 1], [0.1], [0.2]]);
    expect(drivingPlanControlsDiffer(current, 0.04, current)).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0.04, plan([[0, 0, 0, 2], [0.1], [0.2]]))).toBe(true);
});

it('keeps the plan when a pending shift moves to time zero', () => {
    const current = plan([[0], [0.1, 0, 0, 1], [0.2]]);
    const next = plan([[0, 0, 0, 1], [0.1], [0.2]]);
    expect(drivingPlanControlsDiffer(current, 0.04, next)).toBe(false);
});

it('compares a due shift until the controller acknowledges its step', () => {
    const current = plan([[0], [0.1, 0, 0, 1], [0.3]]);
    const next = plan([[0], [0.1, 0, 0, 1], [0.3]]);
    expect(drivingPlanControlsDiffer(current, 0.105, next, 0)).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0.105, next, 1)).toBe(true);
});

it('does not match multiple outgoing presses to a single fresh press', () => {
    const current = plan([[0], [1], [1.5, 0, 0, 1], [2, 0, 0, 1], [3]]);
    const next = plan([[0, 0, 0, 1], [2], [3]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(true);
});

it('refreshes when the order of pending shift actions changes', () => {
    const current = plan([[0], [1, 0, 0, 1], [2, 0, 0, 2]]);
    const next = plan([[0], [1, 0, 0, 2], [2, 0, 0, 1]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(true);
});

it('keeps the plan when the same shift moves past the end of the outgoing plan', () => {
    const current = plan([[0], [1.5, 0, 0, 1], [2]]);
    const next = plan([[0], [1], [2, 0, 0, 1]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(false);
});

it('ignores a fresh shift beyond the comparable horizon and probability changes with the same choice', () => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0], [1], [2, 0, 0, 2]]);
    next[1].shiftProbabilities = [0.4, 0.3, 0.3];
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(false);
});

it.each([[0.399, false], [0.4, true]] as const)
('applies the same control threshold to sampled exploration (%s)', (sample, differs) => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0], [1], [2]]);
    next.forEach((point) => {
        point.action = { pedal: -sample, steering: sample };
        point.latent = [Math.atanh(-sample), Math.atanh(sample)];
    });
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(differs);
});

it.each([2, 2.1, -1, NaN, Infinity])('cannot trigger an early refresh without a valid remaining plan (%s seconds)', (elapsed) => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0, 0.9], [1, 0.9], [2, 0.9]]);
    expect(drivingPlanControlsDiffer(current, elapsed, next)).toBe(false);
});
