import type { StandardTelemetrySample } from '../live-session-types';

export type ReferenceSteeringDirection = 'tooWide' | 'tooTight' | 'aligned';
type Vector = readonly [number, number, number];
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const ANGLE_TOLERANCE = 3 * Math.PI / 180;

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

/** Both vectors use the same capture. Magnitudes and speed targets do not affect the angular comparison. */
export function compareSteeringDirection(referenceDisplacement: Vector, referenceError: Vector,
    sample: StandardTelemetrySample, game: string, cornerTurnSign?: -1 | 1): ReferenceSteeringDirection | undefined {
    const heading = sample.Physics_heading, pitch = sample.Physics_pitch, roll = sample.Physics_roll;
    const vx = sample.Physics_velocity_x, vy = game === 'acc' ? sample.Physics_velocity_y : sample.Physics_velocity_z;
    if ((game !== 'acc' && game !== 'iracing') || cornerTurnSign === undefined
        || !finite(heading) || !finite(pitch) || !finite(roll) || !finite(vx) || !finite(vy)
        || !referenceDisplacement.every(finite) || !referenceError.every(finite)) return undefined;
    const offset = worldOffset(referenceError, heading, pitch, roll, game);
    // start = referenceStart - error; end stays on the reference.
    // end - start = (referenceEnd - referenceStart) + error.
    const ground = game === 'acc' ? 1 : 2;
    const dx = referenceDisplacement[0] + offset[0], dy = referenceDisplacement[ground] + offset[ground];
    const desiredLength = Math.hypot(dx, dy), velocityLength = Math.hypot(vx, vy);
    if (!finite(desiredLength) || !finite(velocityLength) || desiredLength <= 1e-9 || velocityLength <= 1e-9) return undefined;
    const ux = vx / velocityLength, uy = vy / velocityLength, tx = dx / desiredLength, ty = dy / desiredLength;
    const dot = ux * tx + uy * ty;
    // A target behind the driver's motion cannot provide an unambiguous corner correction.
    if (dot <= 0) return undefined;
    const error = Math.atan2(ux * ty - uy * tx, dot) * cornerTurnSign;
    return Math.abs(error) <= ANGLE_TOLERANCE ? 'aligned' : error > 0 ? 'tooWide' : 'tooTight';
}
