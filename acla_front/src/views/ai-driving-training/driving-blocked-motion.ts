import type { DrivingTelemetrySample } from './driving-observation';

const RAMP_MS = 450;
const VELOCITY_AVERAGE_MS = 75;
const AVERAGE_SETTLE_MS = 2 * VELOCITY_AVERAGE_MS;
const MAX_SAMPLE_GAP_MS = 250;
const PENALTY_PER_SECOND = 2;
const BUMP_WINDOW_MS = 2000;
const MIN_REPEAT_MS = 150;
const MAX_BUMP_HOLD_MS = 750;
const MIN_SWING_MPS = { vertical: 0.5, horizontal: 0.25 };
const MIN_THROTTLE = 0.3;
const MIN_STEERING = 0.04;
const STATIONARY_SPEED_MPS = 0.5;
const MIN_STATIONARY_PEDAL = 0.05;
const STATIONARY_GRACE_MS = 350;
const MIN_TURN_RATE_RAD_S = 0.05;
const CONSISTENT_TURN_FRACTION = 0.75;

const BLOCKING_AXES = ['vertical', 'horizontal'] as const;
type BlockingAxis = typeof BLOCKING_AXES[number];
type ByAxis<T> = Record<BlockingAxis, T>;
export type BlockedMotionPenalties = ByAxis<number>;

interface MotionSample {
    at: number;
    velocity: ByAxis<number>;
    yawRate: number;
    input: ByAxis<number>;
    inputSince: ByAxis<number>;
    blockedSince: ByAxis<number | undefined>;
    forwardSpeed: number;
    stationarySince: number | undefined;
}

function hasRepeatedBumps(samples: MotionSample[], start: number, axis: BlockingAxis) {
    const minSwing = MIN_SWING_MPS[axis];
    let low = samples[start].velocity[axis], high = low;
    let direction: 'rise' | 'fall' | undefined;
    const lastSwingIndex: { rise?: number; fall?: number } = {};
    let repeatStart: number | undefined;
    const now = samples[samples.length - 1].at;
    for (let i = start + 1; i < samples.length; i++) {
        const value = samples[i].velocity[axis];
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

function isBlocked(samples: MotionSample[], axis: BlockingAxis) {
    let start = samples.length - 1;
    const current = samples[start];
    // Initial settling from the seed sample is not evidence of a rebound.
    const evidenceStart = Math.max(current.inputSince[axis] + AVERAGE_SETTLE_MS, current.at - BUMP_WINDOW_MS);
    while (start > 0 && samples[start - 1].at >= evidenceStart) start--;
    return hasRepeatedBumps(samples, start, axis);
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
        const velocity = { vertical: localVelocity[context.game === 'iracing' ? 2 : 1], horizontal: localVelocity[0] };
        const input = {
            vertical: Number(controls.throttle > MIN_THROTTLE),
            horizontal: Math.abs(controls.steering) >= MIN_STEERING ? Math.sign(controls.steering) : 0,
        };
        const inputSince = { vertical: at, horizontal: at };
        const previous = last && at - last.at <= MAX_SAMPLE_GAP_MS ? last : undefined;
        // Weight by elapsed time so the average behaves similarly at different sample rates.
        const weight = previous ? 1 - Math.exp(-(at - previous.at) / VELOCITY_AVERAGE_MS) : 1;
        // Average the magnitude so forward and backward movement cannot cancel into a standstill.
        const measuredForwardSpeed = Math.abs(velocity.vertical);
        const forwardSpeed = previous ? previous.forwardSpeed + weight * (measuredForwardSpeed - previous.forwardSpeed) : measuredForwardSpeed;
        const pedalHeld = Math.max(controls.throttle, controls.brake) > MIN_STATIONARY_PEDAL;
        const stationarySince = pedalHeld && forwardSpeed < STATIONARY_SPEED_MPS ? previous?.stationarySince ?? at : undefined;
        const sustainedStop = stationarySince !== undefined && at - stationarySince >= STATIONARY_GRACE_MS;
        if (previous) {
            for (const axis of BLOCKING_AXES) {
                if (previous.input[axis] === input[axis]) {
                    velocity[axis] = previous.velocity[axis] + weight * (velocity[axis] - previous.velocity[axis]);
                    inputSince[axis] = previous.inputSince[axis];
                }
            }
        }
        const current: MotionSample = { at, velocity, yawRate, input, inputSince,
            blockedSince: { vertical: undefined, horizontal: undefined }, forwardSpeed, stationarySince };
        this.samples.push(current);
        if (this.samples.length > 512) this.samples.shift();
        for (const axis of BLOCKING_AXES) {
            if ((axis === 'vertical' && sustainedStop) || (input[axis] && isBlocked(this.samples, axis))) {
                current.blockedSince[axis] = previous?.blockedSince[axis] ?? at;
            }
        }
    }

    penaltyBetween(start: number, end: number): BlockedMotionPenalties {
        const penalty = { vertical: 0, horizontal: 0 };
        for (let i = 1; i < this.samples.length; i++) {
            const from = this.samples[i - 1], to = this.samples[i];
            if (to.at > end) break;
            const overlapStart = Math.max(start, from.at);
            if (to.at <= overlapStart) continue;
            for (const axis of BLOCKING_AXES) {
                const rampStart = from.blockedSince[axis];
                if (rampStart === undefined || rampStart !== to.blockedSince[axis]) continue;
                const weightedMs = integratedStrength(to.at - rampStart) - integratedStrength(overlapStart - rampStart);
                penalty[axis] -= PENALTY_PER_SECOND * weightedMs / 1000;
            }
        }
        return penalty;
    }
}
