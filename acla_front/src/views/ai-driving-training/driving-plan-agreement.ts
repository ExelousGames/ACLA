import type { PolicyDecision } from './driving-policy';

type Plan = PolicyDecision['plan'];
const TIME_EPSILON = 1e-9;

function intentAt(plan: Plan, seconds: number) {
    let step = 0;
    while (step + 1 < plan.length && plan[step + 1].timeSeconds <= seconds + TIME_EPSILON) step++;
    const point = plan[step], next = plan[step + 1];
    const weight = next ? (seconds - point.timeSeconds) / (next.timeSeconds - point.timeSeconds) : 0;
    const axes = point.means.map((mean, axis) => {
        const value = Math.tanh(mean);
        const nextValue = next ? Math.tanh(next.means[axis]) : value;
        return value + (nextValue - value) * weight;
    });
    const shift = point.shiftProbabilities.indexOf(Math.max(...point.shiftProbabilities));
    return { axes, shift };
}

/** Compare policy intent at matching future times, without comparing independent
 * exploration draws. Retaining a plan also retains its sampled actions and credit
 * owner. A fresh inference must cover the entire remaining plan. */
export function drivingPlansAgree(previous: Plan, previousElapsedSeconds: number,
    next: Plan, nextElapsedSeconds: number, tolerance: number): boolean {
    if (!previous.length || !next.length || ![previousElapsedSeconds, nextElapsedSeconds, tolerance]
        .every((value) => Number.isFinite(value) && value >= 0)) return false;
    const remaining = previous[previous.length - 1].timeSeconds - previousElapsedSeconds;
    const nextRemaining = next[next.length - 1].timeSeconds - nextElapsedSeconds;
    if (remaining <= TIME_EPSILON || nextRemaining + TIME_EPSILON < remaining) return false;
    // Linear analog differences reach their maxima at an endpoint or a knot of
    // either plan. Including both grids also checks each categorical interval.
    const times = [0, remaining,
        ...previous.map(({ timeSeconds }) => timeSeconds - previousElapsedSeconds),
        ...next.map(({ timeSeconds }) => timeSeconds - nextElapsedSeconds)]
        .filter((time) => time >= 0 && time <= remaining);
    return times.every((time) => {
        const before = intentAt(previous, previousElapsedSeconds + time);
        const after = intentAt(next, nextElapsedSeconds + time);
        return before.shift === after.shift && before.axes.every((value, axis) =>
            Math.abs(value - after.axes[axis]) <= tolerance + TIME_EPSILON);
    });
}
