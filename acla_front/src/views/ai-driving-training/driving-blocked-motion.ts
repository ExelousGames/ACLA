import type { DrivingTelemetrySample } from './driving-observation';

const GRACE_MS = 350;
const RAMP_MS = 450;
const MAX_SAMPLE_GAP_MS = 250;
const PENALTY_PER_SECOND = 2;

type BlockedResponse = 'throttle' | -1 | 1 | undefined;

function blockedResponse({ context, controls, localVelocity, gForce }: DrivingTelemetrySample): BlockedResponse {
    const forward = localVelocity[context.game === 'iracing' ? 2 : 1];
    if (controls.throttle > 0.3 && Math.abs(forward) < 0.5) return 'throttle';

    // The adapters expose normalized steering. Use a conservative curvature estimate
    // and cap the expected lateral load in fast corners.
    const steering = Math.abs(controls.steering);
    const expectedG = Math.min(1, forward ** 2 * 0.06 * steering / 9.80665);
    if (forward > 0 && steering >= 0.08 && expectedG >= 0.1 && Math.abs(gForce[0]) < 0.4 * expectedG) {
        return controls.steering < 0 ? -1 : 1;
    }
    return undefined;
}

// Area under the penalty ramp, expressed in milliseconds at full strength.
function integratedStrength(blockedMs: number) {
    const afterGrace = Math.max(0, blockedMs - GRACE_MS);
    const ramp = Math.min(afterGrace, RAMP_MS);
    return ramp ** 2 / (2 * RAMP_MS) + Math.max(0, afterGrace - RAMP_MS);
}

/** Keep the sustained response history on the telemetry clock. */
export class DrivingBlockedMotionMonitor {
    private samples: { at: number; response: BlockedResponse }[] = [];

    reset() { this.samples = []; }

    observe(sample: DrivingTelemetrySample) {
        const at = sample.measuredAt;
        const last = this.samples[this.samples.length - 1];
        if (last && at <= last.at) return;
        this.samples.push({ at, response: blockedResponse(sample) });
        if (this.samples.length > 512) this.samples.shift();
    }

    penaltyBetween(start: number, end: number) {
        let blockedSince: number | undefined;
        let weightedMs = 0;
        for (let i = 1; i < this.samples.length; i++) {
            const from = this.samples[i - 1], to = this.samples[i];
            if (to.at > end) break;
            if (!from.response || from.response !== to.response || to.at - from.at > MAX_SAMPLE_GAP_MS) {
                blockedSince = undefined;
                continue;
            }
            blockedSince ??= from.at;
            const overlapStart = Math.max(start, from.at);
            if (to.at > overlapStart) {
                weightedMs += integratedStrength(to.at - blockedSince) - integratedStrength(overlapStart - blockedSince);
            }
        }
        return weightedMs > 0 ? -PENALTY_PER_SECOND * weightedMs / 1000 : 0;
    }
}
