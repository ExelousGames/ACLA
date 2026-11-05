import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { selectLiveMapReference } from '../live-map-reference';
import type { ReferenceMovementCue } from './phrase-reference-movement';

export const REFERENCE_POSITION_MAX_AGE_MS = 500;
export const REFERENCE_CUE_COOLDOWN_MS = 2000;
export type ReferenceCue = 'brake' | 'moreBrake' | 'releaseBrake' | 'easeBrake' | 'steering' | 'throttle' | 'moreThrottle' | 'lessThrottle' | ReferenceMovementCue;
export type ReferencePedalZone = 'brake' | 'throttle' | 'none';
export type ReferenceSteeringPhase = 'prepare' | 'turn';
export interface ReferencePositionSample { capturedAt: number; lateralOffset: number }
export type ReferenceGuidance = Partial<Record<ReferenceCue, number>> & {
    ready?: boolean; lateralOffset?: number; braking?: boolean; pedalZone?: ReferencePedalZone;
    speedAtOrBelowReference?: boolean;
    steeringPhase?: ReferenceSteeringPhase;
};
type Control = 'steering' | 'brake' | 'throttle';
type Channel = Control | 'speed';
interface Point {
    position: number; distance: number; steering?: number; brake?: number; throttle?: number; speed?: number;
    x: number; ground: number;
    deceleration?: number;
}
interface Reference {
    points: Point[]; length: number; brakeEvents: number[]; brakeZoneEnds: number[];
    brakeResponses: { distance: number; decelerationPerBrake: number }[];
    throttleResponses: { distance: number; accelerationPerThrottle: number }[];
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const wrap = (v: number, length: number) => ((v % length) + length) % length;
const controls: Control[] = ['steering', 'brake', 'throttle'];
const channels: Channel[] = [...controls, 'speed'];
const thresholds = { steering: 0.08, brake: 0.1, throttle: 0.15 };
const BRAKE_ZONE_EXIT_THRESHOLD = 0.01;
const SPEED_TOLERANCE_KMH = 5;
// Ignore only rounding noise from reference interpolation, not a driving-speed tolerance.
const SPEED_COMPARISON_EPSILON = 1e-7;
const DECELERATION_TOLERANCE_RATIO = 0.1;
const THROTTLE_TOLERANCE = 0.15;
const SPEED_CORRECTION_SECONDS = 1.5;
const STEERING_LOOKAHEAD_SECONDS = 0.8;
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
            const point: Point = { position, distance: cumulative, x: positions[i][0], ground: positions[i][lap.game === 'acc' ? 1 : 2] };
            channels.forEach((channel, j) => { if (valid[j]) point[channel] = lap.y[i][indices[j]]; });
            return point;
        });
        const length = cumulative + distance(positions[positions.length - 1], positions[0]);
        if (!finite(length) || length <= 0) return null;
        const brakeEvents: number[] = [];
        const brakeZoneEnds: number[] = [];
        const brakeResponses: Reference['brakeResponses'] = [];
        const throttleResponses: Reference['throttleResponses'] = [];
        points.forEach((point, i) => {
            const next = points[(i + 1) % points.length];
            const end = i + 1 === points.length ? length : next.distance;
            if (end <= point.distance) return;
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
            if (point.brake !== undefined && next.brake !== undefined
                && point.brake < thresholds.brake && next.brake >= thresholds.brake) {
                const fraction = (thresholds.brake - point.brake) / (next.brake - point.brake);
                brakeEvents.push(point.distance + fraction * (end - point.distance));
            }
            if (point.brake !== undefined && next.brake !== undefined
                && point.brake >= BRAKE_ZONE_EXIT_THRESHOLD && next.brake < BRAKE_ZONE_EXIT_THRESHOLD) {
                const fraction = (BRAKE_ZONE_EXIT_THRESHOLD - point.brake) / (next.brake - point.brake);
                brakeZoneEnds.push(point.distance + fraction * (end - point.distance));
            }
        });
        return { points, length, brakeEvents, brakeZoneEnds, brakeResponses, throttleResponses };
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

/** Net speed change to this zone's interpolated 1% brake exit, including across start/finish. */
function remainingBrakeZoneDecelerates(reference: Reference, distance: number, speed: number): boolean {
    const remaining = reference.brakeZoneEnds.reduce((nearest, end) =>
        Math.min(nearest, wrap(end - distance, reference.length)), Infinity);
    if (!finite(remaining) || remaining === 0) return false;
    const exitSpeed = targetsAt(reference, distance + remaining).speed;
    return exitSpeed !== undefined && exitSpeed < speed - SPEED_COMPARISON_EPSILON;
}

/** Sum signed steering at the window boundaries and every reference sample between them. */
function steeringPhaseAhead(reference: Reference, distance: number, lookahead: number,
    steering: number, cornerTurnSign?: -1 | 1): ReferenceSteeringPhase | undefined {
    if (cornerTurnSign === undefined || lookahead >= reference.length) return undefined;
    const { index } = bracket(reference, distance, 'distance');
    const end = distance + lookahead;
    let sum = steering, previousDistance = distance;
    for (let i = index; i <= index + reference.points.length; i++) {
        const next = reference.points[i % reference.points.length];
        const nextDistance = next.distance + Math.floor(i / reference.points.length) * reference.length;
        // Count a duplicated start/finish sample only once.
        if (nextDistance > previousDistance) {
            if (nextDistance >= end) {
                sum += targetsAt(reference, end).steering!;
                if (sum * cornerTurnSign <= 0) return undefined;
                return steering * cornerTurnSign >= thresholds.steering ? 'turn' : 'prepare';
            }
            sum += next.steering!;
        }
        previousDistance = nextDistance;
    }
    return undefined;
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

/** Estimate brake demand using only the active phrase reference's speed and pedal response. */
function brakeTarget(reference: Reference, distance: number,
    target: Partial<Record<Channel, number>>, speed: number): number | undefined {
    if (target.brake === undefined) return undefined;
    if (target.speed === undefined) return target.brake;
    const response = nearestResponse(reference.brakeResponses, distance, reference.length);
    // Without a measured reference response, keep pedal comparisons; never invent braking strength.
    if (!response) return target.speed - speed >= SPEED_TOLERANCE_KMH ? 0 : target.brake;
    const slope = speedSlope(reference, distance);
    const speedError = speed - target.speed;
    const correction = Math.abs(speedError) >= SPEED_TOLERANCE_KMH ? speedError / (3.6 * SPEED_CORRECTION_SECONDS) : 0;
    const deceleration = Math.max(0, -speed / 3.6 * slope) + correction;
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
    vision: Pick<TrackVisionDetection, 'capturedAt' | 'position'> | null,
    lap: TopLapReferenceEntry | null, game: string | undefined, now: number,
    locationReference: TopLapReferenceEntry | null = lap,
    liveDeceleration?: number, previousPosition?: ReferencePositionSample,
    activePedalZone?: ReferencePedalZone, cornerTurnSign?: -1 | 1): ReferenceGuidance {
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
    // Enter at 10% reference brake and stay through trail braking until pressure falls below 1%.
    const brakeThreshold = activePedalZone === 'brake' ? BRAKE_ZONE_EXIT_THRESHOLD : thresholds.brake;
    const brakeZone = target.brake === undefined ? undefined : target.brake >= brakeThreshold;
    const result: ReferenceGuidance = { ready: true, lateralOffset: lateral,
        speedAtOrBelowReference: target.speed === undefined ? undefined : speed <= target.speed + SPEED_COMPARISON_EPSILON,
        pedalZone: brakeZone === undefined ? undefined
            : brakeZone ? 'brake'
                : target.throttle !== undefined && target.throttle >= thresholds.throttle ? 'throttle' : 'none',
        braking: brakeZone === undefined ? undefined : brakeZone || tooFast
            || (desiredBrake !== undefined && desiredBrake >= thresholds.brake) };
    if (valid('brake')) {
        const pedal = actual.brake!;
        let cue: ReferenceCue | undefined;
        if (pedal >= thresholds.brake && target.speed !== undefined && finite(liveDeceleration)) {
            // Compare motion with the local speed profile, independent of reference brake pressure.
            const referenceDeceleration = Math.max(0, -target.speed / 3.6 * speedSlope(reference, distance));
            // A percentage comparison needs a positive reference deceleration.
            if (referenceDeceleration > 0) {
                if (tooFast && liveDeceleration <= referenceDeceleration * (1 - DECELERATION_TOLERANCE_RATIO) && pedal < 1) cue = 'moreBrake';
            }
        }
        if (brakeZone && pedal >= thresholds.brake && target.speed !== undefined && speed <= target.speed + SPEED_COMPARISON_EPSILON
            && remainingBrakeZoneDecelerates(reference, distance, target.speed)) cue = 'easeBrake';
        if (!cue && desiredBrake !== undefined) {
            if (desiredBrake >= thresholds.brake && pedal < thresholds.brake) cue = 'brake';
            else if (desiredBrake < thresholds.brake && pedal >= thresholds.brake) cue = 'releaseBrake';
        }
        if (cue) {
            result[cue] = now + 1500;
            result.braking = true;
        }
    }
    const brakeAhead = reference.brakeEvents.some((event) => wrap(event - distance, reference.length) <= metresPerSecond * BRAKE_LOOKAHEAD_SECONDS);
    if (target.steering !== undefined) {
        const phase = steeringPhaseAhead(reference, distance, metresPerSecond * STEERING_LOOKAHEAD_SECONDS, target.steering, cornerTurnSign);
        if (phase) {
            result.steeringPhase = phase;
            result.steering = now + 1500;
        }
    }
    const braking = result.braking !== false || brakeAhead;
    const desiredThrottle = throttleTarget(reference, distance, target, speed);
    if (valid('throttle') && valid('brake') && target.brake !== undefined && desiredThrottle !== undefined) {
        const pedal = actual.throttle!;
        const demand = braking || actual.brake! >= thresholds.brake ? 0 : desiredThrottle;
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
        if (!cue && !braking && actual.brake! < thresholds.brake && lateral !== undefined && Math.abs(lateral) <= 2) {
            if (demand >= thresholds.throttle && pedal < thresholds.throttle) cue = 'throttle';
            else if (demand - pedal >= THROTTLE_TOLERANCE) cue = 'moreThrottle';
        }
        if (cue) result[cue] = now + 1500;
    }
    return result;
}
