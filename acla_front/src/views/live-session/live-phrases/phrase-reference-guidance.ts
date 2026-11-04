import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { selectLiveMapReference } from '../live-map-reference';
import type { ReferenceMovementCue } from './phrase-reference-movement';

export const REFERENCE_POSITION_MAX_AGE_MS = 500;
export const REFERENCE_CUE_COOLDOWN_MS = 2000;
export type ReferenceCue = 'brake' | 'moreBrake' | 'releaseBrake' | 'lessBrake' | 'steerSoon' | 'throttle' | 'moreThrottle' | 'lessThrottle' | ReferenceMovementCue;
export type ReferencePedalZone = 'brake' | 'throttle' | 'none';
export type ReferenceSteeringPhase = 'prepare' | 'easeBrake' | 'turn';
export type ReferenceGuidance = Partial<Record<ReferenceCue, number>> & {
    ready?: boolean; lateralOffset?: number; braking?: boolean; pedalZone?: ReferencePedalZone;
    steeringPhase?: ReferenceSteeringPhase;
};
type Control = 'steering' | 'brake' | 'throttle';
type Channel = Control | 'speed';
interface Point {
    position: number; distance: number; steering?: number; brake?: number; throttle?: number; speed?: number;
    deceleration?: number; decelerationEasing?: boolean;
}
interface Reference {
    points: Point[]; length: number; events: { distance: number; control: Control }[];
    brakeResponses: { distance: number; decelerationPerBrake: number }[];
    throttleResponses: { distance: number; accelerationPerThrottle: number }[];
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const wrap = (v: number, length: number) => ((v % length) + length) % length;
const controls: Control[] = ['steering', 'brake', 'throttle'];
const channels: Channel[] = [...controls, 'speed'];
const thresholds = { steering: 0.08, brake: 0.1, throttle: 0.15 };
const SPEED_TOLERANCE_KMH = 5;
const BRAKE_TOLERANCE = 0.2;
const THROTTLE_TOLERANCE = 0.15;
const SPEED_CORRECTION_SECONDS = 1.5;
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
            const point: Point = { position, distance: cumulative };
            channels.forEach((channel, j) => { if (valid[j]) point[channel] = lap.y[i][indices[j]]; });
            return point;
        });
        const length = cumulative + distance(positions[positions.length - 1], positions[0]);
        if (!finite(length) || length <= 0) return null;
        const events: Reference['events'] = [];
        const brakeResponses: Reference['brakeResponses'] = [];
        const throttleResponses: Reference['throttleResponses'] = [];
        points.forEach((point, i) => {
            const next = points[(i + 1) % points.length];
            const end = i + 1 === points.length ? length : next.distance;
            if (end <= point.distance) return;
            if (point.speed !== undefined && next.speed !== undefined) {
                point.deceleration = (point.speed ** 2 - next.speed ** 2) / (2 * 3.6 ** 2 * (end - point.distance));
            }
            if (point.speed !== undefined && next.speed !== undefined && point.brake !== undefined && next.brake !== undefined) {
                const brake = (point.brake + next.brake) / 2;
                const deceleration = (point.speed ** 2 - next.speed ** 2) / (2 * 3.6 ** 2 * (end - point.distance));
                if (brake >= thresholds.brake && deceleration >= 0.1) {
                    brakeResponses.push({ distance: (point.distance + end) / 2, decelerationPerBrake: deceleration / brake });
                }
            }
            if (point.speed !== undefined && next.speed !== undefined && point.throttle !== undefined && next.throttle !== undefined) {
                const throttle = (point.throttle + next.throttle) / 2;
                const acceleration = (next.speed ** 2 - point.speed ** 2) / (2 * 3.6 ** 2 * (end - point.distance));
                if (throttle >= thresholds.throttle && acceleration >= 0.1) {
                    throttleResponses.push({ distance: (point.distance + end) / 2, accelerationPerThrottle: acceleration / throttle });
                }
            }
            (['steering', 'brake'] as const).forEach((control) => {
                const a = point[control], b = next[control], threshold = thresholds[control];
                if (a === undefined || b === undefined) return;
                // Locate threshold crossings within the interpolated trace, including a steering reversal.
                const target = control === 'steering' ? Math.sign(b) * threshold : threshold;
                if (Math.abs(b) >= threshold && (Math.abs(a) < threshold || (control === 'steering' && a * b < 0))) {
                    const fraction = (target - a) / (b - a);
                    events.push({ distance: point.distance + fraction * (end - point.distance), control });
                }
            });
        });
        // Compare adjacent physical intervals, skipping a duplicated start/finish endpoint.
        const intervals = points.filter((point) => point.deceleration !== undefined);
        intervals.forEach((point, i) => {
            const previous = intervals[(i + intervals.length - 1) % intervals.length].deceleration!;
            point.decelerationEasing = previous >= 0.1 && point.deceleration! >= 0
                && previous - point.deceleration! >= 0.1;
        });
        return { points, length, events, brakeResponses, throttleResponses };
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

/** Estimate pedal demand from the original lap's braking response and the adjusted speed slope. */
function brakeTarget(reference: Reference, baseline: Reference | null, distance: number,
    target: Partial<Record<Channel, number>>, speed: number): number | undefined {
    if (target.brake === undefined) return undefined;
    if (target.speed === undefined) return target.brake;
    const responses = baseline?.brakeResponses ?? [];
    // Added samples can move the adjusted lap's first point across normalized zero.
    const originalDistance = baseline ? distance + distanceAt(baseline, reference.points[0].position) : distance;
    const distanceTo = (at: number) => Math.abs(wrap(at - originalDistance + reference.length / 2, reference.length) - reference.length / 2);
    const response = responses.reduce<Reference['brakeResponses'][number] | undefined>((nearest, item) =>
        !nearest || distanceTo(item.distance) < distanceTo(nearest.distance) ? item : nearest, undefined);
    // Without a measured reference response, keep pedal comparisons; never invent braking strength.
    if (!response) return target.speed - speed >= SPEED_TOLERANCE_KMH ? 0 : target.brake;
    const slope = speedSlope(reference, distance);
    const speedError = speed - target.speed;
    const correction = Math.abs(speedError) >= SPEED_TOLERANCE_KMH ? speedError / (3.6 * SPEED_CORRECTION_SECONDS) : 0;
    const deceleration = Math.max(0, -speed / 3.6 * slope) + correction;
    return Math.max(0, Math.min(1, deceleration / response.decelerationPerBrake));
}

/** Adjust reference throttle for the acceleration needed by the aggression-adjusted speed profile. */
function throttleTarget(reference: Reference, baseline: Reference | null, distance: number,
    target: Partial<Record<Channel, number>>, speed: number): number | undefined {
    if (target.throttle === undefined) return undefined;
    if (target.speed === undefined) return target.throttle;
    const originalDistance = baseline ? distance + distanceAt(baseline, reference.points[0].position) : distance;
    const distanceTo = (at: number) => Math.abs(wrap(at - originalDistance + reference.length / 2, reference.length) - reference.length / 2);
    const response = baseline?.throttleResponses.reduce<Reference['throttleResponses'][number] | undefined>((nearest, item) =>
        !nearest || distanceTo(item.distance) < distanceTo(nearest.distance) ? item : nearest, undefined);
    if (!baseline || !response) return speed - target.speed >= SPEED_TOLERANCE_KMH ? 0 : target.throttle;
    const original = targetsAt(baseline, originalDistance);
    const originalAcceleration = Math.max(0, original.speed! / 3.6 * speedSlope(baseline, originalDistance));
    // During acceleration, align the response to the interpolated pedal, including throttle ramps.
    const accelerationPerThrottle = originalAcceleration >= 0.1 && original.throttle! >= thresholds.throttle
        ? originalAcceleration / original.throttle! : response.accelerationPerThrottle;
    const speedError = target.speed - speed;
    const correction = Math.abs(speedError) >= SPEED_TOLERANCE_KMH ? speedError / (3.6 * SPEED_CORRECTION_SECONDS) : 0;
    const acceleration = Math.max(0, speed / 3.6 * speedSlope(reference, distance)) + correction;
    // Preserve the reference's maintenance throttle on flat-speed sections; adjust only the acceleration demand.
    return Math.max(0, Math.min(1, target.throttle + (acceleration - originalAcceleration) / accelerationPerThrottle));
}

/** Values are speech deadlines, not commands to the car. Missing data never means zero input. */
export function readReferenceGuidance(sample: StandardTelemetrySample,
    vision: Pick<TrackVisionDetection, 'capturedAt' | 'position'> | null,
    lap: TopLapReferenceEntry | null, game: string | undefined, now: number,
    locationReference: TopLapReferenceEntry | null = lap,
    responseReference: TopLapReferenceEntry | null = lap): ReferenceGuidance {
    const prediction = vision?.position, position = sample.Graphics_normalized_car_position, speed = sample.Physics_speed_kmh;
    if (!vision || !prediction || !lap || !locationReference || prediction.referenceLap !== locationReference
        || now < vision.capturedAt || now - vision.capturedAt > REFERENCE_POSITION_MAX_AGE_MS
        || prediction.coordinateFrame !== 'car_local' || !prediction.referenceDelta.every(finite)
        || prediction.referenceDelta.length !== 3 || !finite(prediction.normalizedPosition)
        || prediction.normalizedPosition < 0 || prediction.normalizedPosition > 1
        || !finite(position) || position < 0 || position > 1 || !finite(speed) || speed < 20
        || !selectLiveMapReference([lap], game ?? null, sample.Static_track, sample.Static_car_model, sample.Graphics_track_grip_status)) return {};
    const reference = prepare(lap);
    if (!reference) return {};
    const currentDistance = distanceAt(reference, position), captureDistance = distanceAt(reference, prediction.normalizedPosition);
    const moved = wrap(currentDistance - captureDistance + reference.length / 2, reference.length) - reference.length / 2;
    const metresPerSecond = speed / 3.6;
    if (moved < -2 || moved > metresPerSecond * ((now - vision.capturedAt) / 1000 + 0.5) + 5) return {};
    const [lateral, forward] = prediction.referenceDelta;
    // A remote reference point is not a reliable timing anchor for following this line.
    if (Math.abs(lateral) > 13 || Math.abs(forward) > 30) return {};
    const distance = wrap(currentDistance - forward, reference.length);
    const target = targetsAt(reference, distance);
    const actual = { steering: sample.Physics_steer_angle, brake: sample.Physics_brake, throttle: sample.Physics_gas };
    const valid = (control: Control) => finite(actual[control]) && actual[control]! >= (control === 'steering' ? -1 : 0) && actual[control]! <= 1;
    const tooFast = target.speed !== undefined && speed - target.speed >= SPEED_TOLERANCE_KMH;
    const baseline = responseReference ? prepare(responseReference) : null;
    const desiredBrake = brakeTarget(reference, baseline, distance, target, speed);
    // Zone membership follows the local reference, never the live driver's speed or pedals.
    // Use reference-speed demand so Driving aggression can advance braking for the lower corner speed.
    const referenceBrake = brakeTarget(reference, baseline, distance, target, target.speed ?? 0);
    const referenceThrottle = throttleTarget(reference, baseline, distance, target, target.speed ?? 0);
    const result: ReferenceGuidance = { ready: true, lateralOffset: lateral,
        pedalZone: target.brake === undefined ? undefined
            : target.brake >= thresholds.brake || (referenceBrake !== undefined && referenceBrake >= thresholds.brake) ? 'brake'
                : referenceThrottle !== undefined && referenceThrottle >= thresholds.throttle ? 'throttle' : 'none',
        braking: target.brake === undefined ? undefined : target.brake >= thresholds.brake || tooFast
            || (desiredBrake !== undefined && desiredBrake >= thresholds.brake) };
    if (valid('brake') && desiredBrake !== undefined) {
        const pedal = actual.brake!;
        let cue: ReferenceCue | undefined;
        if (desiredBrake >= thresholds.brake && pedal < thresholds.brake) cue = 'brake';
        else if (desiredBrake < thresholds.brake && pedal >= thresholds.brake) cue = 'releaseBrake';
        // Include exact 20-point differences despite floating-point subtraction (e.g. 1 - 0.8).
        else if (desiredBrake - pedal >= BRAKE_TOLERANCE - Number.EPSILON) cue = 'moreBrake';
        else if (pedal - desiredBrake >= BRAKE_TOLERANCE - Number.EPSILON) cue = 'lessBrake';
        if (cue) {
            result[cue] = now + 1500;
            result.braking = true;
        }
    }
    const upcoming: Partial<Record<Control, number>> = {};
    let brakeAhead = false;
    reference.events.forEach((event) => {
        const eta = wrap(event.distance - distance, reference.length) / metresPerSecond;
        if (event.control === 'brake' && eta <= 1.8) brakeAhead = true;
        if (eta <= 1.8 && (upcoming[event.control] === undefined || eta < upcoming[event.control]!)) upcoming[event.control] = eta;
    });
    if (target.steering !== undefined) {
        const easing = bracket(reference, distance, 'distance').a.decelerationEasing;
        const future = upcoming.steering === undefined ? undefined
            : targetsAt(reference, distance + metresPerSecond * (upcoming.steering + 0.1)).steering;
        if (Math.abs(target.steering) >= thresholds.steering && (future === undefined || target.steering * future > 0)) result.steeringPhase = 'turn';
        else if (upcoming.steering !== undefined && easing) result.steeringPhase = 'easeBrake';
        else if (upcoming.steering !== undefined && upcoming.steering >= 0.6) result.steeringPhase = 'prepare';
        if (result.steeringPhase) result.steerSoon = result.steeringPhase === 'prepare'
            ? now + upcoming.steering! * 1000 - 100 : now + 1500;
    }
    const braking = result.braking !== false || brakeAhead;
    const desiredThrottle = throttleTarget(reference, baseline, distance, target, speed);
    if (valid('throttle') && valid('brake') && target.brake !== undefined && desiredThrottle !== undefined) {
        const pedal = actual.throttle!;
        const demand = braking || actual.brake! >= thresholds.brake ? 0 : desiredThrottle;
        let cue: ReferenceCue | undefined;
        if (demand >= thresholds.throttle && pedal - demand >= THROTTLE_TOLERANCE) cue = 'lessThrottle';
        else if (!braking && actual.brake! < thresholds.brake && Math.abs(lateral) <= 2) {
            if (demand >= thresholds.throttle && pedal < thresholds.throttle) cue = 'throttle';
            else if (demand - pedal >= THROTTLE_TOLERANCE) cue = 'moreThrottle';
        }
        if (cue) result[cue] = now + 1500;
    }
    return result;
}
