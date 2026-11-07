import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { selectLiveMapReference } from '../live-map-reference';
import type { ReferenceMovementCue } from './phrase-reference-movement';
import { compareSteeringDirection, type ReferenceSteeringDirection } from './phrase-steering-direction';

export const REFERENCE_POSITION_MAX_AGE_MS = 500;
export const REFERENCE_CUE_COOLDOWN_MS = 2000;
export type ReferenceCue = 'brake' | 'moreBrake' | 'releaseBrake' | 'releaseBrakeSlower' | 'releaseBrakeFaster' | 'holdBrake' | 'steering' | 'throttle' | 'moreThrottle' | 'lessThrottle' | ReferenceMovementCue;
export type ReferencePedalZone = 'brake' | 'throttle' | 'none';
export type ReferenceSteeringPhase = 'prepare' | 'turn';
export interface ReferencePositionSample { capturedAt: number; lateralOffset: number }
export type ReferenceGuidance = Partial<Record<ReferenceCue, number>> & {
    ready?: boolean; lateralOffset?: number; braking?: boolean; pedalZone?: ReferencePedalZone;
    /** Brake-zone exit uses the trajectory range from its start, independently of Steering eligibility. */
    brakeZoneExit?: boolean;
    steeringPhase?: ReferenceSteeringPhase;
    steeringDirection?: ReferenceSteeringDirection;
    /** Current trajectory turn after exit-tail trimming; null means outside a turn, undefined means unavailable. */
    steeringRange?: { start: number; end: number; turnSign: -1 | 1 } | null;
};
type Control = 'steering' | 'brake' | 'throttle';
type Channel = Control | 'speed';
interface Point {
    position: number; distance: number; steering?: number; brake?: number; throttle?: number; speed?: number;
    x: number; ground: number; xyz: [number, number, number];
    deceleration?: number;
    heading?: number;
}
interface Reference {
    points: Point[]; length: number; brakeEvents: number[]; pedalZones: PedalZone[];
    steeringChanges: { start: number; end: number; turnSign: -1 | 1 }[];
    directionChanges: { start: number; end: number; turnSign: -1 | 1 }[];
    brakeResponses: { distance: number; decelerationPerBrake: number }[];
    throttleResponses: { distance: number; accelerationPerThrottle: number }[];
}
interface PedalZone { pedal: 'brake' | 'throttle'; start: number; length: number; seconds?: number }
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
// Preserve in-range distances exactly so a sample boundary cannot round back into the previous interval.
const wrap = (v: number, length: number) => v >= 0 && v < length ? v : ((v % length) + length) % length;
const controls: Control[] = ['steering', 'brake', 'throttle'];
const channels: Channel[] = [...controls, 'speed'];
const thresholds = { steering: 0.08, brake: 0.1, throttle: 0.15 };
const MIN_PEDAL_ZONE_SECONDS = 0.5;
const SPEED_TOLERANCE_KMH = 5;
const DECELERATION_TOLERANCE_RATIO = 0.1;
const THROTTLE_TOLERANCE = 0.15;
const SPEED_CORRECTION_SECONDS = 1.5;
const ANGLE_EPSILON = 1e-9;
const TRAJECTORY_END_PEAK_RATIO = 0.1;
const MAX_TRAJECTORY_END_CURVATURE = Math.PI / 180 / 10; // 1 degree per 10 m.
const angleDifference = (from: number, to: number) => Math.atan2(Math.sin(to - from), Math.cos(to - from));
const STEERING_DIRECTION_LOOKAHEAD = 10;
const BRAKE_LOOKAHEAD_SECONDS = 1.8;
const cache = new WeakMap<TopLapReferenceEntry, Reference | null>();

function prepare(lap: TopLapReferenceEntry): Reference | null {
    if (cache.has(lap)) return cache.get(lap)!;
    const build = (): Reference | null => {
        const xyz = ['x', 'y', 'z'].map((axis) => lap.target_features.indexOf(`expert_optimal_player_pos_${axis}`));
        if (xyz.some((i) => i < 0) || lap.x.length < 2 || lap.x.length !== lap.y.length
            || !lap.x.every((p, i) => finite(p) && p >= 0 && p <= 1 && (!i || p > lap.x[i - 1]))) return null;
        const positions = lap.y.map((row) => xyz.map((i) => row[i]));
        if (!positions.every((p) => p.every(finite))) return null;
        const indices = channels.map((channel) => lap.target_features.indexOf(`expert_optimal_${channel}`));
        const valid = indices.map((index, i) => index >= 0 && lap.y.every((row) => finite(row[index])
            && row[index] >= (channels[i] === 'steering' ? -1 : 0) && (channels[i] === 'speed' || row[index] <= 1)));
        const distance = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]));
        let cumulative = 0;
        const points = lap.x.map((position, i) => {
            if (i) cumulative += distance(positions[i - 1], positions[i]);
            const point: Point = { position, distance: cumulative, x: positions[i][0], ground: positions[i][lap.game === 'acc' ? 1 : 2],
                xyz: positions[i] as [number, number, number] };
            channels.forEach((channel, j) => { if (valid[j]) point[channel] = lap.y[i][indices[j]]; });
            return point;
        });
        const length = cumulative + distance(positions[positions.length - 1], positions[0]);
        if (!finite(length) || length <= 0) return null;
        const brakeEvents: number[] = [];
        const steeringChanges: Reference['steeringChanges'] = [];
        const boundaries: { pedal: ReferencePedalZone; index: number; endingPedal?: 'brake' | 'throttle' }[] = [];
        const brakeResponses: Reference['brakeResponses'] = [];
        const throttleResponses: Reference['throttleResponses'] = [];
        points.forEach((point, i) => {
            const next = points[(i + 1) % points.length];
            const end = i + 1 === points.length ? length : next.distance;
            const previous = points[(i + points.length - 1) % points.length];
            // Pedal release or full throttle ends only its own zone, including at a duplicated lap endpoint.
            if (point.brake === 0 && previous.brake !== undefined && previous.brake > 0) {
                boundaries.push({ pedal: 'none', index: i, endingPedal: 'brake' });
            }
            if ((point.throttle === 1 && previous.throttle !== 1)
                || (point.throttle === 0 && previous.throttle !== undefined && previous.throttle > 0)) {
                boundaries.push({ pedal: 'none', index: i, endingPedal: 'throttle' });
            }
            if (end <= point.distance) return;
            if (point.steering !== undefined && next.steering !== undefined && next.steering !== point.steering) {
                const turnSign = next.steering > point.steering ? 1 : -1;
                const previous = steeringChanges[steeringChanges.length - 1];
                if (previous?.turnSign === turnSign && previous.end === point.distance) previous.end = end;
                else steeringChanges.push({ start: point.distance, end, turnSign });
            }
            if (point.speed !== undefined && next.speed !== undefined) {
                point.deceleration = (point.speed ** 2 - next.speed ** 2) / (2 * 3.6 ** 2 * (end - point.distance));
            }
            if (point.deceleration !== undefined && point.brake !== undefined && next.brake !== undefined) {
                const brake = (point.brake + next.brake) / 2;
                if (brake >= thresholds.brake && point.deceleration >= 0.1) {
                    brakeResponses.push({ distance: (point.distance + end) / 2, decelerationPerBrake: point.deceleration / brake });
                }
            }
            if (point.deceleration !== undefined && point.throttle !== undefined && next.throttle !== undefined) {
                const throttle = (point.throttle + next.throttle) / 2;
                const acceleration = -point.deceleration;
                if (throttle >= thresholds.throttle && acceleration >= 0.1) {
                    throttleResponses.push({ distance: (point.distance + end) / 2, accelerationPerThrottle: acceleration / throttle });
                }
            }
            // The zero sample is the beginning of the interpolated pedal rise.
            if (point.brake === 0 && next.brake !== undefined && next.brake > 0) {
                brakeEvents.push(point.distance);
                boundaries.push({ pedal: 'brake', index: i });
            } else if (point.throttle === 0 && next.throttle !== undefined && next.throttle > 0) {
                boundaries.push({ pedal: 'throttle', index: i });
            }
        });
        const pedalZones: PedalZone[] = [];
        if (points[0].brake !== undefined && points[0].throttle !== undefined) {
            const starts: typeof boundaries = [];
            let active: ReferencePedalZone = points[0].brake > 0 ? 'brake'
                : points[0].throttle > 0 && points[0].throttle < 1 ? 'throttle' : 'none';
            // First establish the cyclic state; then retain only boundaries that change it.
            // Reapplying the same pedal does not restart an unfinished zone.
            for (let pass = 0; pass < 2; pass++) boundaries.forEach((boundary) => {
                if (boundary.pedal === active || (boundary.endingPedal && active !== boundary.endingPedal)) return;
                active = boundary.pedal;
                if (pass === 1) starts.push(boundary);
            });
            if (!starts.length && active !== 'none') {
                starts.push({ pedal: active, index: boundaries.find(({ pedal }) => pedal === active)?.index ?? 0 });
            }
            starts.forEach(({ pedal, index }, i) => {
                if (pedal === 'none') return;
                const nextIndex = starts[(i + 1) % starts.length].index;
                const count = (nextIndex - index + points.length) % points.length || points.length;
                let span = 0, seconds: number | undefined = 0;
                for (let offset = 0; offset < count; offset++) {
                    const at = (index + offset) % points.length, a = points[at], b = points[(at + 1) % points.length];
                    const distance = (at + 1 === points.length ? length : b.distance) - a.distance;
                    span += distance;
                    if (distance <= 0) continue;
                    if (a.speed === undefined || b.speed === undefined) seconds = undefined;
                    else if (seconds !== undefined) {
                        const startSpeed = a.speed / 3.6, endSpeed = b.speed / 3.6;
                        // Match targetsAt: linear speed during acceleration, linear v² during braking.
                        seconds += endSpeed > startSpeed
                            ? distance * Math.log(endSpeed / startSpeed) / (endSpeed - startSpeed)
                            : 2 * distance / (startSpeed + endSpeed);
                    }
                }
                pedalZones.push({ pedal, start: points[index].distance, length: span, seconds });
            });
        }
        // Join an uninterrupted change across start/finish, including a duplicated endpoint.
        const first = steeringChanges[0], last = steeringChanges[steeringChanges.length - 1];
        if (first && last !== first && first.start === 0 && last.end === length && first.turnSign === last.turnSign) {
            last.end = length + first.end;
            steeringChanges.shift();
        }
        const reference: Reference = { points, length, brakeEvents, pedalZones, steeringChanges,
            directionChanges: [], brakeResponses, throttleResponses };
        prepareTrajectoryDirections(reference);
        return reference;
    };
    const result = build();
    cache.set(lap, result);
    return result;
}

function bracket(reference: Reference, value: number, key: 'position' | 'distance') {
    const { points, length } = reference, period = key === 'position' ? 1 : length;
    let low = 0, high = points.length;
    while (low < high) {
        const mid = (low + high) >>> 1;
        if (points[mid][key] <= value) low = mid + 1;
        else high = mid;
    }
    const a = points[(low + points.length - 1) % points.length], b = points[low % points.length];
    const start = a[key] - (low === 0 ? period : 0), end = b[key] + (low === points.length ? period : 0);
    return { a, b, fraction: end > start ? (value - start) / (end - start) : 0, index: low };
}
function distanceAt(reference: Reference, position: number): number {
    const { a, b, fraction, index } = bracket(reference, position, 'position');
    const start = a.distance - (index === 0 ? reference.length : 0);
    const end = b.distance + (index === reference.points.length ? reference.length : 0);
    return wrap(start + (end - start) * fraction, reference.length);
}
function targetsAt(reference: Reference, distance: number): Partial<Record<Channel, number>> {
    const { a, b, fraction } = bracket(reference, wrap(distance, reference.length), 'distance');
    const targets: Partial<Record<Channel, number>> = {};
    channels.forEach((channel) => {
        if (a[channel] !== undefined && b[channel] !== undefined) {
            // Linear v² keeps the interval's braking rate constant between samples.
            targets[channel] = channel === 'speed' && b.speed! < a.speed!
                ? Math.sqrt(a.speed! ** 2 + (b.speed! ** 2 - a.speed! ** 2) * fraction)
                : a[channel]! + (b[channel]! - a[channel]!) * fraction;
        }
    });
    return targets;
}

/** Estimate horizontal tangents over 10 m so inserted reference samples do not create false plateaus. */
function prepareTrajectoryDirections(reference: Reference): void {
    const { points, length } = reference;
    const ranges: (Reference['directionChanges'][number] & { changes: { end: number; curvature: number }[] })[] = [];
    points.forEach((point) => {
        const positions = [-1, 1].map((sign) => {
            const { a, b, fraction } = bracket(reference, wrap(point.distance + sign * Math.min(5, length / 4), length), 'distance');
            return [a.x + (b.x - a.x) * fraction, a.ground + (b.ground - a.ground) * fraction];
        });
        const dx = positions[1][0] - positions[0][0], dy = positions[1][1] - positions[0][1];
        if (Math.hypot(dx, dy) > 1e-9) point.heading = Math.atan2(dy, dx);
    });
    points.forEach((point, i) => {
        const next = points[(i + 1) % points.length];
        const end = i + 1 === points.length ? length : next.distance;
        if (end <= point.distance || point.heading === undefined || next.heading === undefined) return;
        const delta = angleDifference(point.heading, next.heading);
        if (Math.abs(delta) <= ANGLE_EPSILON) return;
        const turnSign = delta > 0 ? 1 : -1;
        const change = { end, curvature: Math.abs(delta) / (end - point.distance) };
        const previous = ranges[ranges.length - 1];
        if (previous?.turnSign === turnSign && previous.end === point.distance) {
            previous.end = end;
            previous.changes.push(change);
        } else ranges.push({ start: point.distance, end, turnSign, changes: [change] });
    });
    const first = ranges[0], last = ranges[ranges.length - 1];
    if (first && last !== first && first.start === 0 && last.end === length && first.turnSign === last.turnSign) {
        last.end = length + first.end;
        last.changes = last.changes.concat(first.changes.map((change) => ({ ...change, end: length + change.end })));
        ranges.shift();
    }
    reference.directionChanges = ranges.map(({ start, turnSign, changes }) => {
        // Trim only the trailing weak curvature, retaining the original entry and any easing before a later apex.
        // A relative threshold preserves shallow bends; the cap preserves meaningful exits after sharp turns.
        const peak = changes.reduce((value, change) => Math.max(value, change.curvature), 0);
        const threshold = Math.min(MAX_TRAJECTORY_END_CURVATURE, peak * TRAJECTORY_END_PEAK_RATIO);
        const end = changes.reduce((value, change) => change.curvature >= threshold ? change.end : value, start);
        return { start, end, turnSign };
    });
}

function headingAt(reference: Reference, distance: number): number | undefined {
    const { a, b, fraction } = bracket(reference, wrap(distance, reference.length), 'distance');
    return a.heading === undefined || b.heading === undefined ? undefined
        : a.heading + angleDifference(a.heading, b.heading) * fraction;
}

function steeringDisplacementAt(reference: Reference, distance: number): [number, number, number] {
    const positions = [0, STEERING_DIRECTION_LOOKAHEAD].map((offset) => {
        const { a, b, fraction } = bracket(reference, wrap(distance + offset, reference.length), 'distance');
        return a.xyz.map((value, axis) => value + (b.xyz[axis] - value) * fraction);
    });
    return positions[1].map((value, axis) => value - positions[0][axis]) as [number, number, number];
}

/** Every reference steering segment from the current position to the throttle endpoint must be flat. */
function remainingThrottleZoneIsSteady(reference: Reference, distance: number, zone: PedalZone): boolean {
    if (reference.points[0].steering === undefined) return false;
    const remaining = zone.length - wrap(distance - zone.start, reference.length);
    return !reference.steeringChanges.some(({ start, end }) =>
        wrap(start - distance, reference.length) < remaining
        || wrap(distance - start, reference.length) < end - start);
}

/** The continuous trajectory turn is eligible from its start when its remaining range overlaps the corner. */
function trajectoryTurnOverlapsCorner(reference: Reference, distance: number,
    cornerTurnSign?: -1 | 1, cornerBounds?: { start: number; end: number }): boolean {
    if (cornerTurnSign === undefined || !cornerBounds) return false;
    const change = reference.directionChanges.find(({ start, end, turnSign }) => turnSign === cornerTurnSign
        && wrap(distance - start, reference.length) < end - start);
    if (!change) return false;
    const cornerStart = distanceAt(reference, cornerBounds.start);
    const cornerLength = cornerBounds.end - cornerBounds.start === 1 ? reference.length
        : wrap(distanceAt(reference, cornerBounds.end) - cornerStart, reference.length);
    const remaining = change.end - change.start - wrap(distance - change.start, reference.length);
    return cornerLength > 0 && (wrap(cornerStart - distance, reference.length) < remaining
        || wrap(distance - cornerStart, reference.length) < cornerLength);
}

function steeringPhaseAt(reference: Reference, distance: number, steering: number | undefined,
    cornerTurnSign?: -1 | 1, cornerBounds?: { start: number; end: number }): ReferenceSteeringPhase | undefined {
    if (cornerTurnSign === undefined || !trajectoryTurnOverlapsCorner(reference, distance, cornerTurnSign, cornerBounds)) return undefined;
    return steering !== undefined && steering * cornerTurnSign >= thresholds.steering ? 'turn' : 'prepare';
}

function speedSlope(reference: Reference, distance: number): number {
    const { a, b, index, fraction } = bracket(reference, wrap(distance, reference.length), 'distance');
    const span = b.distance + (index === reference.points.length ? reference.length : 0)
        - a.distance + (index === 0 ? reference.length : 0);
    if (span <= 0 || a.speed === undefined || b.speed === undefined) return 0;
    if (b.speed < a.speed) {
        const speed = Math.sqrt(a.speed ** 2 + (b.speed ** 2 - a.speed ** 2) * fraction) / 3.6;
        return speed > 0 ? (b.speed ** 2 - a.speed ** 2) / (2 * 3.6 ** 2 * span * speed) : 0;
    }
    return (b.speed - a.speed) / (3.6 * span);
}

/** Signed horizontal curvature from reference points 10 m either side of the current normalized position. */
function curvatureAt(reference: Reference, distance: number): number {
    const points = [-10, 0, 10].map((offset) => {
        const { a, b, fraction } = bracket(reference, wrap(distance + offset, reference.length), 'distance');
        return { x: a.x + (b.x - a.x) * fraction, y: a.ground + (b.ground - a.ground) * fraction };
    });
    const [a, b, c] = points;
    const ux = b.x - a.x, uy = b.y - a.y, vx = c.x - b.x, vy = c.y - b.y;
    const span = Math.hypot(ux, uy) * Math.hypot(vx, vy) * Math.hypot(c.x - a.x, c.y - a.y);
    return span > 0 ? 2 * (ux * vy - uy * vx) / span : 0;
}

function nearestResponse<T extends { distance: number }>(responses: readonly T[], distance: number, length: number): T | undefined {
    const distanceTo = (at: number) => Math.abs(wrap(at - distance + length / 2, length) - length / 2);
    return responses.reduce<T | undefined>((nearest, item) =>
        !nearest || distanceTo(item.distance) < distanceTo(nearest.distance) ? item : nearest, undefined);
}

/** Follow the local speed profile while closing the speed error before turn-in. */
function brakingDecelerationTarget(reference: Reference, distance: number, targetSpeed: number, speed: number): number {
    const profileDeceleration = Math.max(0, -speed / 3.6 * speedSlope(reference, distance));
    return Math.max(0, profileDeceleration + (speed - targetSpeed) / (3.6 * SPEED_CORRECTION_SECONDS));
}

/** Estimate brake demand using only the active phrase reference's speed and pedal response. */
function brakeTarget(reference: Reference, distance: number,
    target: Partial<Record<Channel, number>>, speed: number): number | undefined {
    if (target.brake === undefined) return undefined;
    if (target.speed === undefined) return target.brake;
    const deceleration = brakingDecelerationTarget(reference, distance, target.speed, speed);
    if (deceleration === 0) return 0;
    const response = nearestResponse(reference.brakeResponses, distance, reference.length);
    // Without a measured reference response, keep pedal comparisons; never invent braking strength.
    if (!response) return target.speed - speed >= SPEED_TOLERANCE_KMH ? 0 : target.brake;
    return Math.max(0, Math.min(1, deceleration / response.decelerationPerBrake));
}

/** Estimate throttle demand using only the active phrase reference's speed and pedal response. */
function throttleTarget(reference: Reference, distance: number,
    target: Partial<Record<Channel, number>>, speed: number): number | undefined {
    if (target.throttle === undefined) return undefined;
    if (target.speed === undefined) return target.throttle;
    const response = nearestResponse(reference.throttleResponses, distance, reference.length);
    if (!response) return speed - target.speed >= SPEED_TOLERANCE_KMH ? 0 : target.throttle;
    const slope = speedSlope(reference, distance);
    const referenceAcceleration = Math.max(0, target.speed / 3.6 * slope);
    // During acceleration, align the response to the interpolated pedal, including throttle ramps.
    const accelerationPerThrottle = referenceAcceleration >= 0.1 && target.throttle >= thresholds.throttle
        ? referenceAcceleration / target.throttle : response.accelerationPerThrottle;
    const speedError = target.speed - speed;
    const correction = Math.abs(speedError) >= SPEED_TOLERANCE_KMH ? speedError / (3.6 * SPEED_CORRECTION_SECONDS) : 0;
    const acceleration = Math.max(0, speed / 3.6 * slope) + correction;
    // Preserve the reference's maintenance throttle on flat-speed sections; adjust only the acceleration demand.
    return Math.max(0, Math.min(1, target.throttle + (acceleration - referenceAcceleration) / accelerationPerThrottle));
}

/** All targets and responses come from the active phrase lap. locationReference is only a prediction identity check.
 * Values are speech deadlines, not commands to the car. Missing data never means zero input. */
export function readReferenceGuidance(sample: StandardTelemetrySample,
    vision: Pick<TrackVisionDetection, 'capturedAt' | 'position' | 'telemetry'> | null,
    lap: TopLapReferenceEntry | null, game: string | undefined, now: number,
    locationReference: TopLapReferenceEntry | null = lap,
    liveDeceleration?: number, previousPosition?: ReferencePositionSample,
    cornerTurnSign?: -1 | 1, cornerBounds?: { start: number; end: number }, brakeReleasing = false): ReferenceGuidance {
    const prediction = vision?.position, position = sample.Graphics_normalized_car_position, speed = sample.Physics_speed_kmh;
    if (!lap || !finite(position) || position < 0 || position > 1 || !finite(speed) || speed < 20
        || !selectLiveMapReference([lap], game ?? null, sample.Static_track, sample.Static_car_model, sample.Graphics_track_grip_status)) return {};
    const reference = prepare(lap);
    if (!reference) return {};
    // Reference channels and cue timing follow live normalized progress in the private lap.
    const distance = distanceAt(reference, position);
    const metresPerSecond = speed / 3.6;
    // Position-model validation only gates lateral checks, never reference lookup.
    let lateral: number | undefined;
    if (vision && prediction && locationReference && prediction.referenceLap === locationReference
        && now >= vision.capturedAt && now - vision.capturedAt <= REFERENCE_POSITION_MAX_AGE_MS
        && prediction.coordinateFrame === 'car_local' && prediction.referenceDelta.every(finite)
        && prediction.referenceDelta.length === 3 && finite(prediction.normalizedPosition)
        && prediction.normalizedPosition >= 0 && prediction.normalizedPosition <= 1) {
        const captureDistance = distanceAt(reference, prediction.normalizedPosition);
        const moved = wrap(distance - captureDistance + reference.length / 2, reference.length) - reference.length / 2;
        if (moved >= -2 && moved <= metresPerSecond * ((now - vision.capturedAt) / 1000 + 0.5) + 5
            && Math.abs(prediction.referenceDelta[0]) <= 13 && Math.abs(prediction.referenceDelta[1]) <= 30) {
            lateral = prediction.referenceDelta[0];
        }
    }
    const target = targetsAt(reference, distance);
    const actual = { brake: sample.Physics_brake, throttle: sample.Physics_gas };
    const valid = (control: keyof typeof actual) => finite(actual[control]) && actual[control]! >= 0 && actual[control]! <= 1;
    const tooFast = target.speed !== undefined && speed - target.speed >= SPEED_TOLERANCE_KMH;
    const desiredBrake = brakeTarget(reference, distance, target, speed);
    const zone = reference.pedalZones.find((zone) => wrap(distance - zone.start, reference.length) < zone.length);
    // Filter the complete reference interval; never wait 500 ms after entering it.
    const pedalZone = target.brake === undefined || target.throttle === undefined ? undefined
        : zone && (zone.seconds ?? zone.length / metresPerSecond) > MIN_PEDAL_ZONE_SECONDS
            && (zone.pedal === 'brake' || remainingThrottleZoneIsSteady(reference, distance, zone)) ? zone.pedal : 'none';
    const brakeZone = pedalZone === undefined ? undefined : pedalZone === 'brake';
    const result: ReferenceGuidance = { ready: true, lateralOffset: lateral,
        pedalZone,
        braking: target.brake === undefined ? undefined : (brakeZone ?? target.brake > 0) || tooFast
            || (desiredBrake !== undefined && desiredBrake >= thresholds.brake) };
    result.steeringRange = headingAt(reference, distance) === undefined ? undefined
        : reference.directionChanges.find(({ start, end }) => wrap(distance - start, reference.length) < end - start) ?? null;
    result.brakeZoneExit = result.steeringRange === undefined
        || cornerTurnSign === undefined || !cornerBounds ? undefined
        : trajectoryTurnOverlapsCorner(reference, distance, cornerTurnSign, cornerBounds);
    if (lateral !== undefined && prediction && vision?.telemetry && reference.length > STEERING_DIRECTION_LOOKAHEAD) {
        // Use the capture's progress, orientation and velocity together; never rotate an old error with a newer heading.
        const capturedDistance = distanceAt(reference, prediction.normalizedPosition);
        result.steeringDirection = compareSteeringDirection(
            steeringDisplacementAt(reference, capturedDistance),
            prediction.referenceDelta, vision.telemetry.currentTelemetry, lap.game, cornerTurnSign);
    }
    if (valid('brake')) {
        const pedal = actual.brake!;
        let cue: ReferenceCue | undefined;
        if (brakeZone && pedal >= thresholds.brake && target.speed !== undefined) {
            const requiredDeceleration = brakingDecelerationTarget(reference, distance, target.speed, speed);
            if (requiredDeceleration === 0) cue = 'releaseBrake';
            else if (finite(liveDeceleration)) {
                const tolerance = Math.max(0.1, requiredDeceleration * DECELERATION_TOLERANCE_RATIO);
                if (liveDeceleration < requiredDeceleration - tolerance) {
                    // Slow an actual pedal release; otherwise the driver needs to add pressure.
                    if (brakeReleasing) cue = 'releaseBrakeSlower';
                    else if (pedal < 1) cue = 'moreBrake';
                } else if (liveDeceleration > requiredDeceleration + tolerance) cue = 'releaseBrakeFaster';
                else cue = 'holdBrake';
            }
        }
        if (!cue && desiredBrake !== undefined) {
            if (desiredBrake >= thresholds.brake && pedal < thresholds.brake) cue = 'brake';
            else if (target.speed === undefined && desiredBrake < thresholds.brake && pedal >= thresholds.brake) cue = 'releaseBrake';
        }
        if (cue) {
            result[cue] = now + 1500;
            result.braking = true;
        }
    }
    const brakeAhead = reference.brakeEvents.some((event) => wrap(event - distance, reference.length) <= metresPerSecond * BRAKE_LOOKAHEAD_SECONDS);
    const phase = steeringPhaseAt(reference, distance, target.steering, cornerTurnSign, cornerBounds);
    if (phase) {
        result.steeringPhase = phase;
        result.steering = now + 1500;
    }
    const braking = result.braking !== false || brakeAhead;
    const desiredThrottle = throttleTarget(reference, distance, target, speed);
    if (valid('throttle') && valid('brake') && target.brake !== undefined && desiredThrottle !== undefined) {
        const pedal = actual.throttle!;
        let cue: ReferenceCue | undefined;
        const elapsed = previousPosition && vision ? vision.capturedAt - previousPosition.capturedAt : 0;
        if (tooFast && result.pedalZone === 'throttle' && pedal >= thresholds.throttle && actual.brake! < thresholds.brake
            && lateral !== undefined && previousPosition && elapsed > 0 && elapsed <= REFERENCE_POSITION_MAX_AGE_MS) {
            const curvature = curvatureAt(reference, distance) * (lap.game === 'iracing' ? -1 : 1);
            const direction = Math.sign(curvature);
            // Delta points from the driver to the reference: on the outside it points into the bend.
            const outwardRate = (lateral - previousPosition.lateralOffset) * direction / (elapsed / 1000);
            if (Math.abs(curvature) >= 0.0015 && lateral * direction >= 5 && outwardRate > 0.05) cue = 'lessThrottle';
        }
        if (!cue && actual.brake! < thresholds.brake && lateral !== undefined && Math.abs(lateral) <= 2) {
            if (!braking && desiredThrottle >= thresholds.throttle && pedal < thresholds.throttle) cue = 'throttle';
            else if (result.pedalZone === 'throttle' && !tooFast && pedal >= thresholds.throttle
                && desiredThrottle - pedal >= THROTTLE_TOLERANCE) cue = 'moreThrottle';
        }
        if (cue) result[cue] = now + 1500;
    }
    return result;
}
