import type { StandardTelemetrySample } from '../live-session-types';

export type ReferenceSteeringDirection = 'tooWide' | 'tooTight' | 'aligned';
export interface SteeringTrajectory { heading: number; remainingDistance: number }
type Vector = readonly [number, number, number];
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const ANGLE_TOLERANCE = 6 * Math.PI / 180;
const LATERAL_TOLERANCE = 3; // Position-model coordinate units, not metres.

/** Convert the position model's right/forward/up offset into the reference's world axes. */
function worldOffset([right, forward, up]: Vector, heading: number, pitch: number, roll: number, game: string): Vector {
    const ch = Math.cos(heading), sh = Math.sin(heading), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    if (game === 'acc') {
        // Inverse of position_preparation.local_targets: Rz(heading) Rx(pitch) Ry(roll) [-right, forward, up].
        const x = -cr * right + sr * up, z = sr * right + cr * up;
        const y = cp * forward - sp * z;
        return [ch * x - sh * y, sh * x + ch * y, sp * forward + cp * z];
    }
    // iRacing adapter: native body [forward, left, up], yaw=-heading, roll=-roll;
    // native world axes map to standard [-Y, Z, X].
    const y = -cr * right + sr * up, z = sr * right + cr * up;
    const x = cp * forward + sp * z;
    return [sh * x - ch * y, -sp * forward + cp * z, ch * x + sh * y];
}

/** Compare local motion with the tangent of the reference curve shifted from the driver to Steering's end. */
export function compareSteeringDirection(trajectory: SteeringTrajectory, referenceError: Vector,
    sample: StandardTelemetrySample, game: string, cornerTurnSign?: -1 | 1): ReferenceSteeringDirection | undefined {
    if ((game !== 'acc' && game !== 'iracing') || cornerTurnSign === undefined
        || !finite(trajectory.heading) || !finite(trajectory.remainingDistance) || trajectory.remainingDistance <= 1e-9
        || !referenceError.every(finite)) return undefined;
    const heading = sample.Physics_heading, pitch = sample.Physics_pitch, roll = sample.Physics_roll;
    const localX = sample.Physics_local_velocity_x, localY = sample.Physics_local_velocity_y, localZ = sample.Physics_local_velocity_z;
    if (!finite(heading) || !finite(pitch) || !finite(roll) || !finite(localX) || !finite(localY) || !finite(localZ)) return undefined;
    const lateral = Math.abs(referenceError[0]) < LATERAL_TOLERANCE ? 0 : referenceError[0];
    const offset = worldOffset([lateral, 0, 0], heading, pitch, roll, game);
    // ACC's telemetry body X is opposite the position model's right axis; iRacing's forward is Z.
    const velocity = worldOffset(game === 'acc' ? [-localX, localY, localZ] : [localX, localZ, localY], heading, pitch, roll, game);
    const ground = game === 'acc' ? 1 : 2;
    // For remaining reference curve R(s), 0 <= s <= L, shift its start sideways and keep its end fixed:
    // P(s) = R(s) - offset * (1 - s/L). Its current tangent is R'(0) + offset/L.
    // This retains the reference's bends instead of aiming across the corner at an endpoint chord.
    const dx = Math.cos(trajectory.heading) + offset[0] / trajectory.remainingDistance;
    const dy = Math.sin(trajectory.heading) + offset[ground] / trajectory.remainingDistance;
    const vx = velocity[0], vy = velocity[ground];
    const desiredLength = Math.hypot(dx, dy), velocityLength = Math.hypot(vx, vy);
    if (!finite(desiredLength) || !finite(velocityLength) || desiredLength <= 1e-9 || velocityLength <= 1e-9) return undefined;
    const ux = vx / velocityLength, uy = vy / velocityLength, tx = dx / desiredLength, ty = dy / desiredLength;
    const dot = ux * tx + uy * ty;
    // An opposing tangent cannot provide an unambiguous corner correction.
    if (dot <= 0) return undefined;
    const error = Math.atan2(ux * ty - uy * tx, dot) * cornerTurnSign;
    return Math.abs(error) <= ANGLE_TOLERANCE ? 'aligned' : error > 0 ? 'tooWide' : 'tooTight';
}
