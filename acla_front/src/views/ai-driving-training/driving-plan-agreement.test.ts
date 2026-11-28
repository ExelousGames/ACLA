import { drivingPlanControlsDiffer } from './driving-plan-agreement';
import { DRIVING_GEAR_COUNT, DRIVING_MAX_GEAR, DRIVING_MIN_GEAR } from './driving-model-config';
import type { PolicyDecision } from './driving-policy';

type PlanPoint = [timeSeconds: number, pedal?: number, steering?: number, gear?: number];
const plan = (points: PlanPoint[]): PolicyDecision['plan'] => points.map(([timeSeconds, pedal = 0, steering = 0, gear = 3]) => ({
    timeSeconds, means: [Math.atanh(pedal), Math.atanh(steering)], latent: [Math.atanh(pedal), Math.atanh(steering)],
    action: { pedal, steering, gear },
    gears: { max: DRIVING_MAX_GEAR }, gearProbabilities: Array.from({ length: DRIVING_GEAR_COUNT }, () => 1 / DRIVING_GEAR_COUNT),
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

it.each([[3, 4], [4, 3], [4, 2], [0, -1], [-1, 0]] as const)
('refreshes when a future target changes from gear %s to %s even with identical probabilities', (before, after) => {
    const current = plan([[0], [1], [2], [3, 0, 0, before]]);
    const next = plan([[0], [1], [2, 0, 0, after], [3, 0, 0, after]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(true);
});

it('keeps matching future targets aligned even when their step indices differ', () => {
    const current = plan([[0], [0.1], [0.3, 0, 0, 4], [0.4, 0, 0, 4]]);
    const next = plan([[0], [0.1], [0.2, 0, 0, 4], [0.3, 0, 0, 4], [0.4, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 0.1, next)).toBe(false);
});

it.each([[0.5, 1.5], [1.5, 0.5]] as const)('keeps the plan when the same pending target moves from %s to %s seconds', (before, after) => {
    const current = plan([[0], [before, 0, 0, 4], [2, 0, 0, 4]]);
    const next = plan([[0], [after, 0, 0, 4], [2, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(false);
});

it('catches new gear targets at time zero and between outgoing steps', () => {
    const current = plan([[0], [1], [2], [3]]);
    expect(drivingPlanControlsDiffer(current, 1, plan([[0, 0, 0, 1], [1], [2], [3]]))).toBe(true);
    expect(drivingPlanControlsDiffer(current, 1, plan([[0], [0.5, 0, 0, 2], [1], [2], [3]]))).toBe(true);
});

it('keeps a repeated target and refreshes for a different immediate target', () => {
    const current = plan([[0, 0, 0, 4], [0.1, 0, 0, 4], [0.2, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 0.04, current)).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0.04, plan([[0, 0, 0, 2], [0.1, 0, 0, 2], [0.2, 0, 0, 2]]))).toBe(true);
});

it('keeps the plan when a pending target moves to time zero', () => {
    const current = plan([[0], [0.1, 0, 0, 4], [0.2, 0, 0, 4]]);
    const next = plan([[0, 0, 0, 4], [0.1, 0, 0, 4], [0.2, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 0.04, next)).toBe(false);
});

it('compares a due target until the controller applies its step', () => {
    const current = plan([[0], [0.1, 0, 0, 4], [0.3, 0, 0, 4]]);
    const next = plan([[0], [0.1, 0, 0, 4], [0.3, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 0.105, next, 0)).toBe(false);
    expect(drivingPlanControlsDiffer(current, 0.105, next, 1)).toBe(true);
});

it('matches repeated outgoing targets to one fresh change to that gear', () => {
    const current = plan([[0], [1], [1.5, 0, 0, 4], [2, 0, 0, 4], [3, 0, 0, 4]]);
    const next = plan([[0, 0, 0, 4], [2, 0, 0, 4], [3, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(false);
});

it('refreshes when the order of pending target gears changes', () => {
    const current = plan([[0], [1, 0, 0, 1], [2, 0, 0, 2]]);
    const next = plan([[0], [1, 0, 0, 2], [2, 0, 0, 1]]);
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(true);
});

it('keeps the plan when the same gear target moves past the end of the outgoing plan', () => {
    const current = plan([[0], [1.5, 0, 0, 4], [2, 0, 0, 4]]);
    const next = plan([[0], [1], [2, 0, 0, 4]]);
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(false);
});

it('ignores a fresh target beyond the comparable horizon and probability changes with the same target', () => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0], [1], [2, 0, 0, 2]]);
    next[1].gearProbabilities = Array.from({ length: DRIVING_GEAR_COUNT }, (_, index) => Number(index + DRIVING_MIN_GEAR === 3));
    expect(drivingPlanControlsDiffer(current, 1, next)).toBe(false);
});

it.each([[0.399, false], [0.4, true]] as const)
('applies the same control threshold to sampled exploration (%s)', (sample, differs) => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0], [1], [2]]);
    next.forEach((point) => {
        point.action = { ...point.action, pedal: -sample, steering: sample };
        point.latent = [Math.atanh(-sample), Math.atanh(sample)];
    });
    expect(drivingPlanControlsDiffer(current, 0, next)).toBe(differs);
});

it.each([2, 2.1, -1, NaN, Infinity])('cannot trigger an early refresh without a valid remaining plan (%s seconds)', (elapsed) => {
    const current = plan([[0], [1], [2]]);
    const next = plan([[0, 0.9], [1, 0.9], [2, 0.9]]);
    expect(drivingPlanControlsDiffer(current, elapsed, next)).toBe(false);
});
