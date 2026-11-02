export interface ImpactSample {
    measuredAt: number;
    /** Car-local [right, forward] velocity in m/s; exclude the vertical axis. */
    velocity: [number, number];
    /** Measured steering angle normalized by the adapter to [-1, 1]. */
    steering: number;
}

export interface DrivingImpact {
    available: boolean;
    detected: boolean;
    /** Heuristic evidence score in [0, 1], not a calibrated probability. */
    confidence: number;
    /** Normalized excess velocity change: 1 corresponds to at least 15 m/s. */
    severity: number;
    /** Estimated velocity impulse beyond the ordinary-motion allowance, in m/s. */
    deltaV: number;
}

export const IMPACT_CONFIDENCE_THRESHOLD = 0.6;
export const IMPACT_PENALTY_WEIGHT = 5;
const unit = (value: number) => Math.max(0, Math.min(1, value));
const MIN_SAMPLE_SECONDS = 0.02;
const MAX_SAMPLE_SECONDS = 0.5;
const VELOCITY_NOISE = 0.35; // m/s; reject small quantization/noise impulses.
const SPEED_ACCELERATION_ALLOWANCE = 2.5 * 9.81; // m/s²; conservative braking/acceleration envelope.

/** Infer possible planar impacts from two consecutive samples. A collision cannot be
 * proved from these signals alone: spins, resets and extreme braking can look similar.
 * Call only within a continuous session, after the training transition boundary checks.
 */
export function detectDrivingImpact(previous: ImpactSample, current: ImpactSample): DrivingImpact {
    const empty: DrivingImpact = { available: false, detected: false, confidence: 0, severity: 0, deltaV: 0 };
    const seconds = (current.measuredAt - previous.measuredAt) / 1000;
    if (![previous.measuredAt, current.measuredAt, ...previous.velocity, ...current.velocity,
        previous.steering, current.steering].every(Number.isFinite)
        || Math.abs(previous.steering) > 1 || Math.abs(current.steering) > 1
        || seconds < MIN_SAMPLE_SECONDS || seconds > MAX_SAMPLE_SECONDS) return empty;

    const [rightBefore, forwardBefore] = previous.velocity;
    const [rightAfter, forwardAfter] = current.velocity;
    const speedBefore = Math.hypot(rightBefore, forwardBefore);
    const speedAfter = Math.hypot(rightAfter, forwardAfter);
    // Magnitude changes also detect shunts that accelerate the car. Ordinary turning
    // preserves speed and therefore cannot contribute to this part of the evidence.
    const speedImpulse = Math.max(0, Math.abs(speedAfter - speedBefore)
        - SPEED_ACCELERATION_ALLOWANCE * seconds - VELOCITY_NOISE);

    // Local velocity direction measures sideslip, not world heading. Allow more slip
    // change during steering and countersteering; do not assume a steering sign, road
    // wheel angle or wheelbase shared by both games. Vertical bumps are excluded.
    const directionChange = Math.abs(Math.atan2(rightAfter * forwardBefore - forwardAfter * rightBefore,
        rightAfter * rightBefore + forwardAfter * forwardBefore));
    const steering = Math.max(Math.abs(previous.steering), Math.abs(current.steering));
    const steeringChange = Math.abs(current.steering - previous.steering);
    const allowedDirectionChange = (0.35 + 1.5 * steering) * seconds + 0.5 * steeringChange;
    const unexplainedDirection = Math.max(0, directionChange - allowedDirectionChange);
    const directionImpulse = Math.max(0, 2 * Math.min(speedBefore, speedAfter)
        * Math.sin(unexplainedDirection / 2) - VELOCITY_NOISE);
    const deltaV = Math.hypot(speedImpulse, directionImpulse);

    // Require both meaningful impulse and abruptness. Sparse samples carry less
    // confidence; long gaps are unavailable instead of being reported as no impact.
    const sampleQuality = 1 - Math.max(0, seconds - 0.2);
    const confidence = unit((deltaV - 0.75) / 2) * unit((deltaV / seconds - 5) / 20) * sampleQuality;
    return { available: true, detected: confidence >= IMPACT_CONFIDENCE_THRESHOLD,
        confidence, severity: unit(deltaV / 15), deltaV };
}
