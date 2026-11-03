export interface ImpactSample {
    measuredAt: number;
    /** Car-local [right, forward] velocity in m/s; exclude the vertical axis. */
    velocity: [number, number];
    /** Measured steering angle normalized by the adapter to [-1, 1]. */
    steering: number;
    /** Measured brake pedal normalized to [0, 1]. */
    brake: number;
}

export interface DrivingImpact {
    available: boolean;
    detected: boolean;
    /** Heuristic evidence score in [0, 1], not a calibrated probability. */
    confidence: number;
    /** Normalized excess velocity change: 1 corresponds to at least 10 m/s. */
    severity: number;
    /** Estimated velocity impulse beyond the ordinary-motion allowance, in m/s. */
    deltaV: number;
}

export const IMPACT_CONFIDENCE_THRESHOLD = 0.5;
export const IMPACT_PENALTY_WEIGHT = 5;
const unit = (value: number) => Math.max(0, Math.min(1, value));
export const MIN_IMPACT_SAMPLE_MS = 20;
export const MAX_IMPACT_SAMPLE_MS = 500;
const VELOCITY_NOISE = 0.35; // m/s; reject small quantization/noise impulses.
const SPEED_ACCELERATION_ALLOWANCE = 2.5 * 9.81; // m/s²; conservative braking/acceleration envelope.
const COAST_DECELERATION_ALLOWANCE = 3; // m/s²; allow engine braking and rolling resistance.

function excessImpulse(speedChange: number, speedAllowance: number, directionImpulse: number, directionAllowance: number) {
    const relativeImpulse = Math.hypot(speedChange / speedAllowance,
        directionAllowance > 0 ? directionImpulse / directionAllowance : 0);
    return Math.max(0, Math.hypot(speedChange, directionImpulse)
        * (1 - 1 / Math.max(1, relativeImpulse)) - VELOCITY_NOISE);
}

/** Infer possible planar impacts from two consecutive samples. A collision cannot be
 * proved from these signals alone: spins, resets and extreme braking can look similar.
 * Call only within a continuous session, after the training transition boundary checks.
 */
export function detectDrivingImpact(previous: ImpactSample, current: ImpactSample): DrivingImpact {
    const empty: DrivingImpact = { available: false, detected: false, confidence: 0, severity: 0, deltaV: 0 };
    const seconds = (current.measuredAt - previous.measuredAt) / 1000;
    if (![previous.measuredAt, current.measuredAt, ...previous.velocity, ...current.velocity,
        previous.steering, current.steering, previous.brake, current.brake].every(Number.isFinite)
        || Math.abs(previous.steering) > 1 || Math.abs(current.steering) > 1
        || previous.brake < 0 || previous.brake > 1 || current.brake < 0 || current.brake > 1
        || seconds * 1000 < MIN_IMPACT_SAMPLE_MS || seconds * 1000 > MAX_IMPACT_SAMPLE_MS) return empty;

    const [rightBefore, forwardBefore] = previous.velocity;
    const [rightAfter, forwardAfter] = current.velocity;
    const speedBefore = Math.hypot(rightBefore, forwardBefore);
    const speedAfter = Math.hypot(rightAfter, forwardAfter);
    // Magnitude changes also detect shunts that accelerate the car. Ordinary turning
    // preserves speed and therefore cannot contribute to this part of the evidence.
    const speedChange = Math.abs(speedAfter - speedBefore);
    const speedAllowance = SPEED_ACCELERATION_ALLOWANCE * seconds;

    // Local velocity direction measures sideslip, not world heading. Allow more slip
    // change during steering and countersteering; do not assume a steering sign, road
    // wheel angle or wheelbase shared by both games. Vertical bumps are excluded.
    const directionChange = Math.abs(Math.atan2(rightAfter * forwardBefore - forwardAfter * rightBefore,
        rightAfter * rightBefore + forwardAfter * forwardBefore));
    const steering = Math.max(Math.abs(previous.steering), Math.abs(current.steering));
    const steeringChange = Math.abs(current.steering - previous.steering);
    const allowedDirectionChange = (0.35 + 1.5 * steering) * seconds + 0.5 * steeringChange;
    // Together with speedChange, this chord recovers the full planar velocity
    // change, including deflection during a slowdown. Using the smaller speed
    // would discard part of a diagonal impact.
    const directionScale = 2 * Math.sqrt(speedBefore * speedAfter);
    const directionImpulse = directionScale * Math.sin(directionChange / 2);
    const directionAllowance = directionScale * Math.sin(Math.min(Math.PI, allowedDirectionChange) / 2);
    // Share the ordinary-motion budget across speed and direction as an ellipse.
    // Independent allowances hide diagonal jolts that use most of both budgets.
    // Remove the permitted fraction of the combined impulse, then noise once.
    const motionImpulse = excessImpulse(speedChange, speedAllowance, directionImpulse, directionAllowance);

    // A released brake cannot explain emergency-braking-level losses at any speed.
    // While steering is also neutral, combine slowdown with unexplained deflection
    // so a glancing bounce is not hidden by keeping much of the car's speed.
    // Acceleration contributes no extra evidence to this more sensitive path.
    const coastImpulse = Math.max(previous.brake, current.brake) < 0.05
        ? excessImpulse(Math.max(0, speedBefore - speedAfter), COAST_DECELERATION_ALLOWANCE * seconds,
            steering < 0.05 ? directionImpulse : 0, directionAllowance) : 0;
    const deltaV = Math.max(motionImpulse, coastImpulse);

    // Score smaller jolts after the ordinary-motion/noise allowances, while still
    // requiring both excess impulse and abruptness. Sparse samples carry less
    // confidence; long gaps are unavailable instead of being reported as no impact.
    const sampleQuality = 1 - Math.max(0, seconds - 0.2);
    const motionConfidence = unit((motionImpulse - 0.25) / 1.25) * unit((motionImpulse / seconds - 3) / 10);
    // Coasting already removes an allowance proportional to elapsed time; a
    // separate high acceleration gate would hide bounces and stops in sparse samples.
    const coastConfidence = unit((coastImpulse - 0.1) / 0.5);
    const confidence = Math.max(motionConfidence, coastConfidence) * sampleQuality;
    return { available: true, detected: confidence >= IMPACT_CONFIDENCE_THRESHOLD,
        confidence, severity: unit(deltaV / 10), deltaV };
}
