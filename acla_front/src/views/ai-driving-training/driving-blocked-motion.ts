import type { DrivingTelemetrySample } from './driving-observation';

const THROTTLE_GRACE_MS = 350;
const RAMP_MS = 450;
const MAX_SAMPLE_GAP_MS = 250;
const PENALTY_PER_SECOND = 2;
const BUMP_WINDOW_MS = 2000;
const MIN_REPEAT_MS = 150;
const MAX_BUMP_HOLD_MS = 750;
const MIN_LATERAL_SWING_MPS = 0.25;
const MIN_FORWARD_SWING_MPS = 0.5;
const MIN_WALL_STEERING = 0.04;
const MIN_TURN_RATE_RAD_S = 0.05;
const CONSISTENT_TURN_FRACTION = 0.75;

type BlockedResponse = 'throttle' | -1 | 1 | undefined;

interface MotionSample {
    at: number;
    lateral: number;
    forward: number;
    yawRate: number;
    input: BlockedResponse;
    response: BlockedResponse;
}

function hasRepeatedBumps(samples: MotionSample[], start: number, axis: 'lateral' | 'forward', minSwing: number) {
    let low = samples[start][axis], high = low;
    let direction: 'rise' | 'fall' | undefined;
    const lastSwingIndex: { rise?: number; fall?: number } = {};
    let repeatStart: number | undefined;
    const now = samples[samples.length - 1].at;
    for (let i = start + 1; i < samples.length; i++) {
        const value = samples[i][axis];
        low = Math.min(low, value);
        high = Math.max(high, value);
        if (direction !== 'rise' && value - low >= minSwing) {
            direction = 'rise';
            high = value;
        } else if (direction !== 'fall' && high - value >= minSwing) {
            direction = 'fall';
            low = value;
        } else continue;
        // The return from one jolt is an opposite swing. Look for another swing
        // in the original direction, allowing time for a distinct rebound.
        const at = samples[i].at, previous = lastSwingIndex[direction];
        if (previous !== undefined) {
            const period = at - samples[previous].at;
            if (period < MIN_REPEAT_MS) continue;
            const holdMs = Math.min(MAX_BUMP_HOLD_MS, Math.max(MAX_SAMPLE_GAP_MS, period));
            if (now - at <= holdMs) repeatStart = previous;
        }
        lastSwingIndex[direction] = i;
    }
    if (repeatStart === undefined) return false;
    // Judge the turn over the rebounds themselves. A straight approach to a bumpy
    // corner can otherwise dilute the corner's yaw over the longer search window.
    let positiveYawMs = 0, negativeYawMs = 0;
    for (let i = repeatStart; i < samples.length - 1; i++) {
        const elapsed = samples[i + 1].at - samples[i].at;
        if (samples[i].yawRate >= MIN_TURN_RATE_RAD_S) positiveYawMs += elapsed;
        if (samples[i].yawRate <= -MIN_TURN_RATE_RAD_S) negativeYawMs += elapsed;
    }
    return Math.max(positiveYawMs, negativeYawMs) < CONSISTENT_TURN_FRACTION * (now - samples[repeatStart].at);
}

function isRidingWall(samples: MotionSample[]) {
    let start = samples.length - 1;
    const current = samples[start];
    while (start > 0) {
        const previous = samples[start - 1];
        const elapsed = samples[start].at - previous.at;
        if (previous.input !== current.input || elapsed > MAX_SAMPLE_GAP_MS || current.at - previous.at > BUMP_WINDOW_MS) break;
        start--;
    }
    return hasRepeatedBumps(samples, start, 'lateral', MIN_LATERAL_SWING_MPS)
        || hasRepeatedBumps(samples, start, 'forward', MIN_FORWARD_SWING_MPS);
}

// Area under the penalty ramp, expressed in milliseconds at full strength.
function integratedStrength(elapsedMs: number) {
    const ramp = Math.min(Math.max(0, elapsedMs), RAMP_MS);
    return ramp ** 2 / (2 * RAMP_MS) + Math.max(0, elapsedMs - RAMP_MS);
}

/** Keep the sustained response history on the telemetry clock. */
export class DrivingBlockedMotionMonitor {
    private samples: MotionSample[] = [];

    reset() { this.samples = []; }

    observe({ measuredAt: at, context, controls, localVelocity, yawRate }: DrivingTelemetrySample) {
        const last = this.samples[this.samples.length - 1];
        if (last && at <= last.at) return;
        const forward = localVelocity[context.game === 'iracing' ? 2 : 1];
        let input: BlockedResponse;
        if (controls.throttle > 0.3 && Math.abs(forward) < 0.5) input = 'throttle';
        else if (forward > 0 && Math.abs(controls.steering) >= MIN_WALL_STEERING) input = controls.steering < 0 ? -1 : 1;
        const current: MotionSample = { at, lateral: localVelocity[0], forward, yawRate, input, response: undefined };
        this.samples.push(current);
        if (this.samples.length > 512) this.samples.shift();
        if (input === 'throttle' || (input && isRidingWall(this.samples))) current.response = input;
    }

    penaltyBetween(start: number, end: number) {
        let rampStart: number | undefined;
        let weightedMs = 0;
        for (let i = 1; i < this.samples.length; i++) {
            const from = this.samples[i - 1], to = this.samples[i];
            if (to.at > end) break;
            if (!from.response || from.response !== to.response || to.at - from.at > MAX_SAMPLE_GAP_MS) {
                rampStart = undefined;
                continue;
            }
            rampStart ??= from.at + (from.response === 'throttle' ? THROTTLE_GRACE_MS : 0);
            const overlapStart = Math.max(start, from.at);
            if (to.at > overlapStart) {
                weightedMs += integratedStrength(to.at - rampStart) - integratedStrength(overlapStart - rampStart);
            }
        }
        return weightedMs > 0 ? -PENALTY_PER_SECOND * weightedMs / 1000 : 0;
    }
}
