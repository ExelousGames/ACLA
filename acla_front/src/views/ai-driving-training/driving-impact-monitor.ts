import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { DrivingObservation } from './driving-observation';
import { detectDrivingImpact, DrivingImpact, ImpactSample, MIN_IMPACT_SAMPLE_MS, MAX_IMPACT_SAMPLE_MS } from './driving-impact';

/** Keep short telemetry intervals independent of camera/GPU processing. Reward
 * queries use capture-time boundaries, so newer contacts cannot leak into a frame. */
export class DrivingImpactMonitor {
    private session?: string;
    private samples: ImpactSample[] = [];

    observe(live: LiveTelemetrySnapshot) {
        const telemetry = live.currentTelemetry;
        const current: ImpactSample = { measuredAt: live.receivedAt!,
            velocity: [telemetry.Physics_local_velocity_x!, live.game === 'iracing'
                ? telemetry.Physics_local_velocity_z! : telemetry.Physics_local_velocity_y!],
            steering: telemetry.Physics_steer_angle!, brake: telemetry.Physics_brake! };
        if (live.telemetryStatus !== ACC_STATUS.ACC_LIVE || !live.game
            || ![current.measuredAt, ...current.velocity, current.steering, current.brake].every(Number.isFinite)
            || Math.abs(current.steering) > 1 || current.brake < 0 || current.brake > 1) {
            this.session = undefined; this.samples = [];
            return;
        }
        const session = JSON.stringify([live.sessionGeneration, live.streamGeneration, live.game,
            telemetry.Static_car_model, telemetry.Static_track]);
        const previous = this.samples[this.samples.length - 1];
        if (session !== this.session || (previous && current.measuredAt < previous.measuredAt)) {
            this.session = session; this.samples = [];
        } else if (previous && current.measuredAt === previous.measuredAt) {
            return;
        }
        this.samples.push(current);
        // More than eight seconds at 60 Hz, with bounded memory.
        if (this.samples.length > 512) this.samples.shift();
    }

    between(previous: DrivingObservation, current: DrivingObservation): DrivingImpact | undefined {
        if (previous.session !== this.session || current.session !== this.session) return undefined;
        const samples = this.samples.filter((sample) => sample.measuredAt >= previous.measuredAt && sample.measuredAt <= current.measuredAt);
        if (samples.length < 2) return undefined;
        const unavailable: DrivingImpact = { available: false, detected: false, confidence: 0, severity: 0, deltaV: 0 };
        let strongest = unavailable;
        let incomplete = samples[0].measuredAt > previous.measuredAt || samples[samples.length - 1].measuredAt < current.measuredAt;
        for (let end = 1; end < samples.length; end++) {
            if (samples[end].measuredAt - samples[end - 1].measuredAt > MAX_IMPACT_SAMPLE_MS) incomplete = true;
            let brake = samples[end].brake, minSteering = samples[end].steering, maxSteering = minSteering;
            // Check short and longer intervals: a small impact can span several
            // high-rate samples, while a brief bounce can vanish between images.
            for (let start = end - 1; start >= 0; start--) {
                const elapsed = samples[end].measuredAt - samples[start].measuredAt;
                if (elapsed > MAX_IMPACT_SAMPLE_MS) break;
                brake = Math.max(brake, samples[start].brake);
                minSteering = Math.min(minSteering, samples[start].steering);
                maxSteering = Math.max(maxSteering, samples[start].steering);
                if (elapsed < MIN_IMPACT_SAMPLE_MS) continue;
                // Preserve controls anywhere inside the interval, including a
                // brake press or steering correction released before its end.
                const impact = detectDrivingImpact({ ...samples[start], brake, steering: minSteering },
                    { ...samples[end], brake, steering: maxSteering });
                if ((!strongest.available && impact.available) || (impact.detected && !strongest.detected)
                    || (impact.detected === strongest.detected
                        && impact.confidence * impact.severity > strongest.confidence * strongest.severity)) strongest = impact;
            }
        }
        // A missing interval cannot be reported as a clean, impact-free interval.
        return !strongest.detected && incomplete ? unavailable : strongest;
    }
}
