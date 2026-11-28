import type { PolicyDecision } from './driving-policy';

type Plan = PolicyDecision['plan'];
const EPSILON = 1e-9;
const CONTROL_AXES = ['pedal', 'steering'] as const;

export const DRIVING_PLAN_REFRESH_SECONDS = 4;

function gearChanges(plan: Plan, previous: number | undefined): Plan {
    return plan.filter(({ action }) => {
        if (action.gear === previous) return false;
        previous = action.gear;
        return true;
    });
}

/** Compare the sampled controls over the plans' shared future, keeping the outgoing clock. */
export function drivingPlanControlsDiffer(current: Plan, elapsedSeconds: number, next: Plan,
    tolerance: number, lastAppliedStep?: number): boolean {
    if (!current.length || !next.length || !Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) return false;
    const remaining = current[current.length - 1].timeSeconds - elapsedSeconds;
    if (remaining > next[next.length - 1].timeSeconds + EPSILON) return false;

    let activeStep = 0;
    while (activeStep + 1 < current.length && current[activeStep + 1].timeSeconds <= elapsedSeconds + EPSILON) activeStep++;
    const appliedStep = Math.min(lastAppliedStep ?? activeStep, activeStep);
    const appliedGear = current[appliedStep]?.action.gear;
    // Match changes in target gear in order. Repeated targets and fresh-only
    // changes beyond the comparison window do not change the remaining plan.
    const pending = gearChanges(current.slice(Math.max(activeStep, appliedStep + 1)), appliedGear);
    const fresh = gearChanges(next, appliedGear);
    if (pending.some(({ action }, i) => action.gear !== fresh[i]?.action.gear)
        || (fresh[pending.length]?.timeSeconds ?? Infinity) <= Math.max(0, remaining) + EPSILON) return true;

    const differences = [0, 0]; // Signed pedal and steering, in normalized control units.
    let count = 0, nextStep = 0;
    for (let step = activeStep + 1; step < current.length; step++) {
        const point = current[step];
        const future = point.timeSeconds - elapsedSeconds;
        while (nextStep + 1 < next.length && next[nextStep + 1].timeSeconds <= future + EPSILON) nextStep++;
        const from = next[nextStep], to = next[nextStep + 1];
        const weight = to ? Math.max(0, future - from.timeSeconds) / (to.timeSeconds - from.timeSeconds) : 0;
        CONTROL_AXES.forEach((axis, i) => {
            const start = from.action[axis];
            const fresh = to ? start + (to.action[axis] - start) * weight : start;
            differences[i] += Math.abs(point.action[axis] - fresh);
        });
        count++;
    }
    return count > 0 && differences.some((total) => total / count >= tolerance * (1 - EPSILON));
}
