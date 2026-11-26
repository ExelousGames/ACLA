import type { PolicyDecision } from './driving-policy';

type Plan = PolicyDecision['plan'];
const EPSILON = 1e-9;
const CONTROL_AXES = ['pedal', 'steering'] as const;

export const DRIVING_PLAN_REFRESH_DIFFERENCE = 0.4;
export const DRIVING_PLAN_REFRESH_SECONDS = 4;

function gearActionsDiffer(pendingPlan: Plan, activeShift: Plan[number]['shift'], next: Plan, remainingSeconds: number): boolean {
    const pending = pendingPlan.filter(({ shift }) => shift !== 0);
    const fresh = next.filter(({ shift }) => shift !== 0);
    const matches = (skip: number) => {
        const extra = fresh[pending.length + skip];
        // Match gear actions in order regardless of timing. A fresh-only tail
        // outside the comparison window does not add a comparable action.
        return pending.every(({ shift }, i) => shift === fresh[i + skip]?.shift)
            && (!extra || extra.timeSeconds > Math.max(0, remainingSeconds) + EPSILON);
    };
    // An immediate repeat of the active step's shift is not a reason to re-press it.
    return !matches(0) && !(activeShift !== 0 && next[0].shift === activeShift && matches(1));
}

/** Compare the sampled controls over the plans' shared future, keeping the outgoing clock. */
export function drivingPlanControlsDiffer(current: Plan, elapsedSeconds: number, next: Plan, lastAppliedStep?: number): boolean {
    if (!current.length || !next.length || !Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) return false;
    const remaining = current[current.length - 1].timeSeconds - elapsedSeconds;
    if (remaining > next[next.length - 1].timeSeconds + EPSILON) return false;

    let activeStep = 0;
    while (activeStep + 1 < current.length && current[activeStep + 1].timeSeconds <= elapsedSeconds + EPSILON) activeStep++;
    // A shift in the active step is still pending until its controller write is acknowledged.
    // Earlier skipped steps can no longer run and are not part of the outgoing sequence.
    const activeStepApplied = (lastAppliedStep ?? activeStep) >= activeStep;
    if (gearActionsDiffer(current.slice(activeStepApplied ? activeStep + 1 : activeStep),
        activeStepApplied ? current[activeStep].shift : 0, next, remaining)) return true;

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
    return count > 0 && differences.some((total) => total / count >= DRIVING_PLAN_REFRESH_DIFFERENCE - EPSILON);
}
