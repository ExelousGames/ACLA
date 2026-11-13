import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { selectLiveMapReference } from '../live-map-reference';
import type { ReferenceMovementCue } from './phrase-reference-movement';
import { compareSteeringDirection, type ReferenceSteeringDirection, type SteeringTrajectory } from './phrase-steering-direction';
import { buildReferenceCarPerformance, carDecelerationAt, type ReferenceCarPerformance } from './phrase-car-performance';

export const REFERENCE_POSITION_MAX_AGE_MS = 500;
export const REFERENCE_CUE_COOLDOWN_MS = 2000;
export const REFERENCE_LOOKAHEAD_MS = 3000;
export type ReferenceCue = 'brake' | 'moreBrake' | 'releaseBrake' | 'easeBrake' | 'holdBrake' | 'steering' | 'cornerExit' | 'throttle' | ReferenceMovementCue;
export type ReferencePedalZone = 'brake' | 'throttle' | 'none';
export type ReferenceSteeringPhase = 'prepare' | 'turn';
export interface ReferenceCornerBounds { start: number; end: number }
export type ReferenceGuidance = Partial<Record<ReferenceCue, number>> & {
    ready?: boolean; lateralOffset?: number; braking?: boolean; pedalZone?: ReferencePedalZone;
    currentPedalZone?: ReferencePedalZone;
    /** Nominal brake zone or a required extension based on the available deceleration. */
    brakeRequired?: boolean;
    /** Time until braking must start, and braking time to the limiting upcoming speed target. */
    brakeInMs?: number;
    brakeDurationMs?: number;
    /** Positive easing of reference deceleration (m/s³), while the driver is braking. */
    brakeReleaseRate?: number;
    /** Remaining reference travel time through this easing episode, including lower steady braking. */
    brakeReleaseDurationMs?: number;
    /** Exit below the speed at the upcoming or active direction-change start, independently of Steering eligibility. */
    brakeZoneExit?: boolean;
    /** Reference speed-profile acceleration at the evaluation position, in m/s². */
    acceleration?: number;
    steeringPhase?: ReferenceSteeringPhase;
    /** A loaded reference arc or its final steering release, not a geometric apex estimate. */
    cornerPhase?: 'hold' | 'unwind';
    steeringDirection?: ReferenceSteeringDirection;
    /** Current turn limited to centerline corner bounds; null means outside a turn, undefined means unavailable. */
    steeringRange?: { start: number; end: number; turnSign: -1 | 1 } | null;
    /** Mapped corner bounds in reference distance, only during its first matching turning arc. */
    turnInCorner?: { start: number; end: number };
    /** Measured progress along the reference, independent of the evaluation position. */
    currentReferencePosition?: { distance: number; length: number };
};
type Control = 'steering' | 'brake' | 'throttle';
type Channel = Control | 'speed';
interface Point {
    position: number; distance: number; steering?: number; brake?: number; throttle?: number; speed?: number;
    x: number; ground: number; elevation: number;
    curvature?: number; grade?: number;
    deceleration?: number;
    brakeReleaseRate?: number;
    heading?: number;
}
interface Reference {
    points: Point[]; length: number; brakeEvents: number[]; pedalZones: PedalZone[];
    directionChanges: { start: number; end: number; turnSign: -1 | 1 }[];
    brakeResponses: { distance: number; decelerationPerBrake: number }[];
    possibleBrakingDeceleration?: number;
    carPerformance: ReferenceCarPerformance;
    throttleResponses: { distance: number; accelerationPerThrottle: number }[];
    cornerProfiles: Map<string, CornerSteeringProfile | null>;
}
interface CornerSteeringProfile { points: { distance: number; steering: number }[]; peak: number; peakEnd: number }
interface PedalZone {
    pedal: 'brake' | 'throttle'; start: number; length: number; seconds?: number;
    brakeTargets?: { offset: number; speed: number }[];
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
// Preserve in-range distances exactly so a sample boundary cannot round back into the previous interval.
const wrap = (v: number, length: number) => v >= 0 && v < length ? v : ((v % length) + length) % length;
const controls: Control[] = ['steering', 'brake', 'throttle'];
const channels: Channel[] = [...controls, 'speed'];
const thresholds = { steering: 0.08, brake: 0.1, throttle: 0.15 };
const MIN_PEDAL_ZONE_SECONDS = 0.5;
const SPEED_TOLERANCE_KMH = 5;
const DECELERATION_TOLERANCE_RATIO = 0.1;
const SPEED_CORRECTION_SECONDS = 1.5;
const MIN_TRAJECTORY_CURVATURE = Math.PI / 180 / 10; // Confirms a turn and trims its tail; does not locate its onset.
const CURVATURE_EPSILON = 1e-12;
const angleDifference = (from: number, to: number) => Math.atan2(Math.sin(to - from), Math.cos(to - from));
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
                elevation: positions[i][lap.game === 'acc' ? 2 : 1] };
            channels.forEach((channel, j) => { if (valid[j]) point[channel] = lap.y[i][indices[j]]; });
            return point;
        });
        const length = cumulative + distance(positions[positions.length - 1], positions[0]);
        if (!finite(length) || length <= 0) return null;
        const brakeEvents: number[] = [];
        const boundaries: { pedal: ReferencePedalZone; index: number; endingPedal?: 'brake' | 'throttle' }[] = [];
        const brakeResponses: Reference['brakeResponses'] = [];
        let possibleBrakingDeceleration: number | undefined;
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
            if (point.speed !== undefined && next.speed !== undefined) {
                point.deceleration = (point.speed ** 2 - next.speed ** 2) / (2 * 3.6 ** 2 * (end - point.distance));
            }
            if (point.deceleration !== undefined && point.brake !== undefined && next.brake !== undefined) {
                const brake = (point.brake + next.brake) / 2;
                if (brake >= thresholds.brake && point.deceleration >= 0.1) {
                    brakeResponses.push({ distance: (point.distance + end) / 2, decelerationPerBrake: point.deceleration / brake });
                    possibleBrakingDeceleration = Math.max(possibleBrakingDeceleration ?? 0, point.deceleration);
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
        points.forEach((point, i) => {
            let previous = points[(i + points.length - 1) % points.length];
            // A duplicated start/finish sample has no interval of its own.
            for (let offset = 2; offset < points.length && wrap(point.distance - previous.distance, length) === 0; offset++) {
                previous = points[(i + points.length - offset) % points.length];
            }
            const next = points[(i + 1) % points.length];
            if (point.deceleration === undefined || previous.deceleration === undefined
                || point.deceleration <= 0 || previous.deceleration <= point.deceleration
                || !point.speed || !previous.speed || !next.speed) return;
            const before = wrap(point.distance - previous.distance, length);
            const after = wrap(next.distance - point.distance, length);
            // Deceleration belongs to interval midpoints. Use reference travel time, not live speed.
            const seconds = before / ((previous.speed + point.speed) / 3.6)
                + after / ((point.speed + next.speed) / 3.6);
            if (before <= 0 || after <= 0 || !finite(seconds) || seconds <= 0) return;
            const rate = (previous.deceleration - point.deceleration) / seconds;
            if (finite(rate) && rate > 0.01) point.brakeReleaseRate = rate;
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
                const startSpeed = points[index].speed;
                const brakeTargets = pedal === 'brake' && startSpeed !== undefined ? [{ offset: 0, speed: startSpeed }] : undefined;
                for (let offset = 0; offset < count; offset++) {
                    const at = (index + offset) % points.length, a = points[at], b = points[(at + 1) % points.length];
                    const distance = (at + 1 === points.length ? length : b.distance) - a.distance;
                    span += distance;
                    if (distance <= 0) continue;
                    if (a.speed !== undefined && b.speed !== undefined && b.speed < a.speed) {
                        brakeTargets?.push({ offset: span, speed: b.speed });
                    }
                    if (a.speed === undefined || b.speed === undefined) seconds = undefined;
                    else if (seconds !== undefined) {
                        const startSpeed = a.speed / 3.6, endSpeed = b.speed / 3.6;
                        // Match targetsAt: linear speed during acceleration, linear v² during braking.
                        seconds += endSpeed > startSpeed
                            ? distance * Math.log(endSpeed / startSpeed) / (endSpeed - startSpeed)
                            : 2 * distance / (startSpeed + endSpeed);
                    }
                }
                pedalZones.push({ pedal, start: points[index].distance, length: span, seconds, brakeTargets });
            });
        }
        const reference: Reference = { points, length, brakeEvents, pedalZones,
            directionChanges: [], brakeResponses, possibleBrakingDeceleration, throttleResponses, cornerProfiles: new Map(),
            carPerformance: buildReferenceCarPerformance([]) };
        prepareTrajectoryDirections(reference);
        reference.carPerformance = buildReferenceCarPerformance(points.flatMap((point, i) => {
            const next = points[(i + 1) % points.length];
            const span = (i + 1 === points.length ? length : next.distance) - point.distance;
            if (span <= 0) return [];
            point.grade = (next.elevation - point.elevation) / span;
            const horizontal = Math.hypot(next.x - point.x, next.ground - point.ground);
            point.curvature = horizontal > 1e-9 && point.heading !== undefined && next.heading !== undefined
                ? angleDifference(point.heading, next.heading) / horizontal : 0;
            if (point.speed === undefined || next.speed === undefined || point.deceleration === undefined
                || point.brake === undefined || next.brake === undefined || point.throttle === undefined || next.throttle === undefined) return [];
            return [{ distance: (point.distance + span / 2), speed: Math.sqrt((point.speed ** 2 + next.speed ** 2) / 2) / 3.6,
                curvature: point.curvature, elevation: (point.elevation + next.elevation) / 2, grade: point.grade,
                deceleration: point.deceleration, brake: (point.brake + next.brake) / 2, throttle: (point.throttle + next.throttle) / 2,
                coasting: point.brake <= 0.01 && next.brake <= 0.01 && point.throttle <= 0.01 && next.throttle <= 0.01 }];
        }));
        return reference;
    };
    const result = build();
    cache.set(lap, result);
    return result;
}

/** Capture the strongest observed braking rate once, before rebuilding the driving reference. */
export function referenceBrakingDeceleration(lap: TopLapReferenceEntry | null): number | undefined {
    return lap ? prepare(lap)?.possibleBrakingDeceleration : undefined;
}

/** Retain speed, turn radius, elevation, slope and pedal response from the original car/lap. */
export function referenceCarPerformance(lap: TopLapReferenceEntry | null): ReferenceCarPerformance | undefined {
    return lap ? prepare(lap)?.carPerformance : undefined;
}

/** Limit each continuous turn to the corner, excluding entry positioning and exit adjustments. */
function cornerDirectionChanges(reference: Reference, corners?: readonly ReferenceCornerBounds[]): Reference['directionChanges'] {
    if (!corners) return reference.directionChanges;
    const { length } = reference;
    return corners.flatMap((corner) => {
        if (![corner.start, corner.end].every((value) => finite(value) && value >= 0 && value <= 1)) return [];
        const start = distanceAt(reference, corner.start);
        const span = corner.end - corner.start === 1 ? length : wrap(distanceAt(reference, corner.end) - start, length);
        const end = start + span;
        const clipped = reference.directionChanges.flatMap((range) => [-1, 0, 1].flatMap((offset) => {
            const head = Math.max(start, range.start + offset * length);
            const tail = Math.min(end, range.end + offset * length);
            return tail > head ? [{ start: head, end: tail, turnSign: range.turnSign }] : [];
        })).sort((a, b) => a.start - b.start);
        const joined: Reference['directionChanges'] = [];
        clipped.forEach((range) => {
            const previous = joined[joined.length - 1];
            if (previous?.turnSign === range.turnSign && previous.end === range.start) previous.end = range.end;
            else joined.push(range);
        });
        return joined.map((range) => ({ ...range, start: wrap(range.start, length),
            end: wrap(range.start, length) + range.end - range.start }));
    });
}

/** Raw-coordinate-distance ranges shared by braking adjustment and steering guidance. */
export function referenceDirectionChanges(lap: TopLapReferenceEntry, corners?: readonly ReferenceCornerBounds[]): readonly { start: number; end: number }[] {
    const reference = prepare(lap);
    return reference ? cornerDirectionChanges(reference, corners) : [];
}

/** Map intervals use the same speed profile and cyclic direction ranges as guidance. */
export function referenceRangeSegments(lap: TopLapReferenceEntry | null, corners?: readonly ReferenceCornerBounds[]) {
    const reference = lap && prepare(lap);
    if (!reference) return null;
    const { points, length } = reference;
    const directionChanges = cornerDirectionChanges(reference, corners);
    const boundaries = directionChanges.flatMap(({ start, end }) => [start, wrap(end, length)]);
    return {
        speedAvailable: points[0].speed !== undefined,
        segments: points.flatMap((point, i) => {
            const next = points[(i + 1) % points.length];
            const end = i + 1 === points.length ? length : next.distance;
            if (end <= point.distance) return [];
            const cuts = [point.distance, ...Array.from(new Set(boundaries.filter((at) => at > point.distance && at < end)))
                .sort((a, b) => a - b), end];
            const at = (distance: number) => {
                const fraction = (distance - point.distance) / (end - point.distance);
                return { x: point.x + (next.x - point.x) * fraction, y: point.ground + (next.ground - point.ground) * fraction };
            };
            return cuts.slice(1).map((tail, index) => ({ start: at(cuts[index]), end: at(tail),
                deceleration: point.deceleration,
                changesDirection: directionChanges.some((range) =>
                    wrap((cuts[index] + tail) / 2 - range.start, length) < range.end - range.start) }));
        }),
    };
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

/** Signed travel between telemetry samples, unwrapped over the nearest lap seam. */
export function referenceTravelDistance(lap: TopLapReferenceEntry | null, from: number, to: number): number | undefined {
    const reference = lap && prepare(lap);
    if (!reference || !finite(from) || !finite(to) || from < 0 || from > 1 || to < 0 || to > 1) return undefined;
    return wrap(distanceAt(reference, to) - distanceAt(reference, from) + reference.length / 2, reference.length) - reference.length / 2;
}

/** Shared decision position for the reference-guide tree; measured telemetry stays untouched. */
export function referencePositionAhead(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null, lookaheadMs = REFERENCE_LOOKAHEAD_MS): number | undefined {
    const position = sample.Graphics_normalized_car_position, speed = sample.Physics_speed_kmh;
    if (!lap || !finite(position) || position < 0 || position > 1 || !finite(speed) || speed < 0) return undefined;
    const reference = prepare(lap);
    if (!reference) return undefined;
    if (speed === 0) return position;
    const distance = wrap(distanceAt(reference, position) + speed / 3.6 * Math.max(0, Math.min(REFERENCE_LOOKAHEAD_MS, lookaheadMs)) / 1000, reference.length);
    const { a, b, fraction, index } = bracket(reference, distance, 'distance');
    const start = a.position - (index === 0 ? 1 : 0), end = b.position + (index === reference.points.length ? 1 : 0);
    return wrap(start + (end - start) * fraction, 1);
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

function brakeReleaseDurationMs(reference: Reference, distance: number): number | undefined {
    const { points, length } = reference;
    const { a, fraction, index } = bracket(reference, distance, 'distance');
    let previousDeceleration = a.deceleration!, seconds = 0;
    for (let offset = 0; offset < points.length; offset++) {
        const at = (index - 1 + points.length + offset) % points.length;
        const point = points[at], next = points[(at + 1) % points.length];
        const span = (at + 1 === points.length ? length : next.distance) - point.distance;
        // A duplicated start/finish sample has no interval or deceleration of its own.
        if (span <= 0) continue;
        if (point.deceleration === undefined || point.speed === undefined || next.speed === undefined) return undefined;
        if (point.deceleration <= 0 || point.deceleration > previousDeceleration + 1e-9) break;
        const remaining = offset === 0 ? 1 - fraction : 1;
        // Match targetsAt: the initial partial braking interval has linear squared speed.
        const startSpeed = Math.sqrt(next.speed ** 2 + (point.speed ** 2 - next.speed ** 2) * remaining) / 3.6;
        const elapsed = 2 * span * remaining / (startSpeed + next.speed / 3.6);
        if (!finite(elapsed) || elapsed < 0) return undefined;
        seconds += elapsed;
        previousDeceleration = point.deceleration;
    }
    return seconds > 0 ? seconds * 1000 : undefined;
}

/** Estimate horizontal tangents over 10 raw coordinate units so inserted reference samples do not create false plateaus. */
function prepareTrajectoryDirections(reference: Reference): void {
    const { points, length } = reference;
    const ranges: Reference['directionChanges'] = [];
    points.forEach((point) => {
        const positions = [-1, 1].map((sign) => {
            const { a, b, fraction } = bracket(reference, wrap(point.distance + sign * Math.min(5, length / 4), length), 'distance');
            return [a.x + (b.x - a.x) * fraction, a.ground + (b.ground - a.ground) * fraction];
        });
        const dx = positions[1][0] - positions[0][0], dy = positions[1][1] - positions[0][1];
        if (Math.hypot(dx, dy) > 1e-9) point.heading = Math.atan2(dy, dx);
    });
    const segments = points.flatMap((point, i) => {
        const next = points[(i + 1) % points.length];
        const end = i + 1 === points.length ? length : next.distance;
        if (end <= point.distance) return [];
        const delta = point.heading === undefined || next.heading === undefined ? 0 : angleDifference(point.heading, next.heading);
        const curvature = delta / (end - point.distance);
        const turnSign: -1 | 0 | 1 = Math.abs(curvature) <= CURVATURE_EPSILON ? 0 : curvature > 0 ? 1 : -1;
        const dx = next.x - point.x, dy = next.ground - point.ground;
        const heading = Math.hypot(dx, dy) > 1e-9 ? Math.atan2(dy, dx) : undefined;
        return [{ start: point.distance, end, curvature, turnSign, heading }];
    });
    // Begin at a direction boundary so an onset across start/finish is processed only once.
    const boundary = segments.findIndex((segment, i) => segment.turnSign !== segments[(i + segments.length - 1) % segments.length].turnSign);
    const origin = Math.max(0, boundary);
    let runSign = 0, onset: number | undefined, confirmed = false;
    for (let step = 0; step < segments.length; step++) {
        const i = (origin + step) % segments.length, segment = segments[i];
        const offset = i < origin ? length : 0, distance = segment.start + offset, end = segment.end + offset;
        const { turnSign } = segment;
        if (!step || turnSign !== runSign) {
            runSign = turnSign;
            onset = boundary < 0 ? distance : undefined;
            confirmed = false;
        }
        if (!turnSign) continue;
        const incoming = segments[(i + segments.length - 1) % segments.length].heading;
        // Use the incoming/outgoing geometry at the vertex, not the centered tangent's anticipation.
        if (onset === undefined && incoming !== undefined && segment.heading !== undefined
            && angleDifference(incoming, segment.heading) * turnSign > CURVATURE_EPSILON) onset = distance;
        if (onset === undefined || Math.abs(segment.curvature) < MIN_TRAJECTORY_CURVATURE - CURVATURE_EPSILON) continue;
        const previous = ranges[ranges.length - 1];
        if (previous?.turnSign === turnSign && previous.end === distance) {
            previous.end = end;
        } else ranges.push({ start: confirmed ? distance : onset, end, turnSign });
        // A later strong section must not absorb the preceding turn's excluded tail.
        confirmed = true;
    }
    reference.directionChanges = ranges.map(({ start, end, turnSign }) => ({
        start: wrap(start, length), end: wrap(start, length) + end - start, turnSign,
    }));
}

function headingAt(reference: Reference, distance: number): number | undefined {
    const { a, b, fraction } = bracket(reference, wrap(distance, reference.length), 'distance');
    return a.heading === undefined || b.heading === undefined ? undefined
        : a.heading + angleDifference(a.heading, b.heading) * fraction;
}

function steeringTrajectoryAt(reference: Reference, ranges: Reference['directionChanges'], distance: number, cornerTurnSign?: -1 | 1): SteeringTrajectory | undefined {
    const range = ranges.find(({ start, end, turnSign }) => turnSign === cornerTurnSign
        && wrap(distance - start, reference.length) < end - start);
    const heading = headingAt(reference, distance);
    if (!range || heading === undefined) return undefined;
    const remainingDistance = range.end - range.start - wrap(distance - range.start, reference.length);
    return { heading, remainingDistance };
}

/** Infer the loaded arc and final unwind from steering, never from map fractions or acceleration. */
function cornerSteeringPhase(reference: Reference, range: NonNullable<ReferenceGuidance['steeringRange']>,
    distance: number): ReferenceGuidance['cornerPhase'] {
    const key = `${range.start}:${range.end}:${range.turnSign}`;
    if (!reference.cornerProfiles.has(key)) {
        const span = range.end - range.start;
        const points = [
            { distance: 0, steering: targetsAt(reference, range.start).steering },
            ...reference.points.map((point) => ({ distance: wrap(point.distance - range.start, reference.length), steering: point.steering }))
                .filter((point) => point.distance > 0 && point.distance < span).sort((a, b) => a.distance - b.distance),
            { distance: span, steering: targetsAt(reference, range.end).steering },
        ];
        let profile: CornerSteeringProfile | null = null;
        if (points.every((point) => point.steering !== undefined && point.steering * range.turnSign >= -0.01)) {
            const signed = points.map((point) => ({ distance: point.distance, steering: point.steering! * range.turnSign }));
            const peak = Math.max(...signed.map((point) => point.steering));
            if (peak >= thresholds.steering) {
                // The last near-maximum sample prevents an early release between equal apex loads.
                const peakEnd = signed.filter((point) => point.steering >= peak - 0.01).slice(-1)[0].distance;
                profile = { points: signed, peak, peakEnd };
            }
        }
        reference.cornerProfiles.set(key, profile);
    }
    const profile = reference.cornerProfiles.get(key);
    if (!profile) return undefined;
    const at = wrap(distance - range.start, reference.length), { points, peak, peakEnd } = profile;
    const amountAt = (position: number) => {
        let index = 1;
        while (index < points.length - 1 && points[index].distance < position) index++;
        const a = points[index - 1], b = points[index];
        return a.steering + (b.steering - a.steering) * (position - a.distance) / (b.distance - a.distance);
    };
    const amount = amountAt(at);
    const before = amountAt(Math.max(0, at - 5)), after = amountAt(Math.min(range.end - range.start, at + 5));
    if (amount >= thresholds.steering && amount >= peak - 0.02
        && at >= 5 && range.end - range.start - at >= 5
        && Math.abs(before - amount) <= 0.01 && Math.abs(after - amount) <= 0.01) return 'hold';
    if (at <= peakEnd || amount < 0.04 || peak - amount < 0.02 || amount - after < 0.005) return undefined;
    // Reject a temporary release before a later, lower peak as well as equal/higher peaks.
    let minimum = amount;
    for (const point of points) {
        if (point.distance <= at) continue;
        if (point.steering > minimum + 0.02) return undefined;
        minimum = Math.min(minimum, point.steering);
    }
    return 'unwind';
}

function pedalZoneAt(reference: Reference, distance: number, speed: number): ReferencePedalZone | undefined {
    const target = targetsAt(reference, distance);
    if (target.brake === undefined || target.throttle === undefined) return undefined;
    const zone = reference.pedalZones.find((zone) => wrap(distance - zone.start, reference.length) < zone.length);
    // Filter the complete reference interval; never wait 500 ms after entering it.
    return zone && (zone.seconds ?? zone.length / (speed / 3.6)) > MIN_PEDAL_ZONE_SECONDS ? zone.pedal : 'none';
}

function brakeTiming(reference: Reference, distance: number, speed: number,
    deceleration: number | undefined): Pick<ReferenceGuidance, 'brakeInMs' | 'brakeDurationMs'> {
    if (!finite(deceleration) || deceleration <= 0 || speed <= 0) return {};
    let nextZone: PedalZone | undefined, approach = Infinity;
    for (const zone of reference.pedalZones) {
        if (zone.pedal !== 'brake' || zone.seconds === undefined || zone.seconds <= MIN_PEDAL_ZONE_SECONDS) continue;
        const progress = wrap(distance - zone.start, reference.length);
        const remaining = progress < zone.length ? -progress : wrap(zone.start - distance, reference.length);
        if (remaining < approach) { nextZone = zone; approach = remaining; }
    }
    // Do not borrow a later corner's targets or extrapolate a zone with no measured speed loss.
    if (!nextZone?.brakeTargets || nextZone.brakeTargets.length < 2) return {};
    const velocity = speed / 3.6;
    let brakeInMs = Infinity, brakeDurationMs: number | undefined;
    for (const target of nextZone.brakeTargets) {
        const remaining = approach + target.offset, targetVelocity = target.speed / 3.6;
        if (remaining < 0 || velocity <= targetVelocity) continue;
        const seconds = (velocity - targetVelocity) / deceleration;
        const brakingDistance = (velocity + targetVelocity) * seconds / 2;
        const untilBrake = (remaining - brakingDistance) / velocity * 1000;
        // Equal onsets belong to the same constant-rate episode; retain its full braking time.
        if (untilBrake < brakeInMs - 1e-6
            || (Math.abs(untilBrake - brakeInMs) <= 1e-6 && seconds * 1000 > (brakeDurationMs ?? 0))) {
            brakeInMs = untilBrake;
            brakeDurationMs = seconds * 1000;
        }
    }
    return brakeDurationMs === undefined ? {} : { brakeInMs: Math.max(0, brakeInMs), brakeDurationMs };
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

function carMotionAt(reference: Reference, distance: number, speed: number) {
    const { a } = bracket(reference, wrap(distance, reference.length), 'distance');
    return { speed: speed / 3.6, curvature: a.curvature ?? 0, grade: a.grade ?? 0 };
}

/** Additional brake work needed after allowing for coasting along the remaining corner.
 * Check every speed target, including a second slowdown after a brief plateau. A
 * matching entry speed does not justify release while a lower target still needs braking.
 */
function remainingBrakeDemand(reference: Reference, performance: ReferenceCarPerformance, distance: number,
    speed: number, targetSpeed: number, range?: { start: number; end: number } | null, brakeDue = false): number {
    const motion = carDecelerationAt(performance, carMotionAt(reference, distance, speed));
    const signedDemand = Math.max(0, -speed / 3.6 * speedSlope(reference, distance))
        + (speed - targetSpeed) / (3.6 * SPEED_CORRECTION_SECONDS);
    let demand = Math.max(0, signedDemand - motion.coastingDeceleration);
    let zone = reference.pedalZones.find((zone) => zone.pedal === 'brake'
        && wrap(distance - zone.start, reference.length) < zone.length);
    if (!zone) zone = reference.pedalZones.filter((zone) => zone.pedal === 'brake'
        && (brakeDue || wrap(zone.start - distance, reference.length) <= speed / 3.6 * BRAKE_LOOKAHEAD_SECONDS))
        .sort((a, b) => wrap(a.start - distance, reference.length) - wrap(b.start - distance, reference.length))[0];
    const remaining = range ? wrap(range.end - distance, reference.length) || (range.end - range.start >= reference.length ? reference.length : 0)
        : zone ? wrap(zone.start + zone.length - distance, reference.length) || (zone.length === reference.length ? reference.length : 0) : 0;
    if (!remaining) return demand;
    let energy = (speed / 3.6) ** 2, travelled = 0;
    const { index } = bracket(reference, distance, 'distance');
    for (let offset = 0; offset < reference.points.length && travelled < remaining; offset++) {
        const point = reference.points[(index + offset) % reference.points.length];
        const ahead = wrap(point.distance - distance, reference.length);
        if (ahead <= travelled) continue; // Duplicated lap endpoint.
        const end = Math.min(ahead, remaining), span = end - travelled;
        const coast = carDecelerationAt(performance, carMotionAt(reference, distance + travelled, Math.sqrt(energy) * 3.6)).coastingDeceleration;
        energy = Math.max(0, energy - 2 * coast * span);
        const futureSpeed = end === ahead ? point.speed : targetsAt(reference, distance + end).speed;
        if (futureSpeed === undefined) return Infinity; // Missing data cannot authorize full release.
        demand = Math.max(demand, (energy - (futureSpeed / 3.6) ** 2) / (2 * end));
        travelled = end;
    }
    return demand;
}

/** Estimate brake demand using only the active phrase reference's speed and pedal response. */
function brakeTarget(reference: Reference, distance: number,
    target: Partial<Record<Channel, number>>, speed: number, performance: ReferenceCarPerformance): number | undefined {
    if (target.brake === undefined) return undefined;
    if (target.speed === undefined) return target.brake;
    if (target.speed - speed >= SPEED_TOLERANCE_KMH) return 0;
    const capability = carDecelerationAt(performance, carMotionAt(reference, distance, speed));
    const deceleration = Math.max(0, brakingDecelerationTarget(reference, distance, target.speed, speed) - capability.coastingDeceleration);
    if (deceleration === 0) return 0;
    if (capability.decelerationPerBrake !== undefined) return Math.min(1, deceleration / capability.decelerationPerBrake);
    const response = nearestResponse(reference.brakeResponses, distance, reference.length);
    // Without a measured reference response, keep pedal comparisons; never invent braking strength.
    if (!response) return target.brake;
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

/** Targets come from the active phrase lap; possible braking strength is captured from the baseline at load.
 * locationReference is only a prediction identity check.
 * Values are speech deadlines, not commands to the car. Missing data never means zero input. */
export function readReferenceGuidance(sample: StandardTelemetrySample,
    vision: Pick<TrackVisionDetection, 'capturedAt' | 'position' | 'telemetry'> | null,
    lap: TopLapReferenceEntry | null, game: string | undefined, now: number,
    locationReference: TopLapReferenceEntry | null = lap,
    intendedDeceleration?: number,
    cornerTurnSign?: -1 | 1, cornerBounds?: { start: number; end: number }, brakeReleasing = false,
    evaluationPosition = sample.Graphics_normalized_car_position,
    intendedTelemetry?: StandardTelemetrySample,
    currentBraking = { deceleration: intendedTelemetry ? undefined : intendedDeceleration, releasing: brakeReleasing },
    measure = { brake: true, throttle: true, steering: true },
    possibleBrakingDeceleration = referenceBrakingDeceleration(lap),
    carPerformance = referenceCarPerformance(lap)): ReferenceGuidance {
    const prediction = vision?.position, position = sample.Graphics_normalized_car_position;
    const intended = intendedTelemetry ?? sample;
    const speed = intended.Physics_speed_kmh, liveSpeed = sample.Physics_speed_kmh;
    if (!lap || !finite(position) || position < 0 || position > 1 || !finite(speed) || speed < 0
        || !finite(liveSpeed) || liveSpeed < 0
        || !finite(evaluationPosition) || evaluationPosition < 0 || evaluationPosition > 1
        || !selectLiveMapReference([lap], game ?? null, sample.Static_track, sample.Static_car_model, sample.Graphics_track_grip_status)) return {};
    const reference = prepare(lap);
    if (!reference) return {};
    const performance = carPerformance ?? reference.carPerformance;
    // All reference conditions share the decision position; capture validation uses actual progress.
    const liveDistance = distanceAt(reference, position);
    const distance = distanceAt(reference, evaluationPosition);
    const metresPerSecond = speed / 3.6;
    // Position-model validation only gates lateral checks, never reference lookup.
    let lateral: number | undefined;
    if (vision && prediction && locationReference && prediction.referenceLap === locationReference
        && now >= vision.capturedAt && now - vision.capturedAt <= REFERENCE_POSITION_MAX_AGE_MS
        && prediction.coordinateFrame === 'car_local' && prediction.referenceDelta.every(finite)
        && prediction.referenceDelta.length === 3 && finite(prediction.normalizedPosition)
        && prediction.normalizedPosition >= 0 && prediction.normalizedPosition <= 1) {
        const captureDistance = distanceAt(reference, prediction.normalizedPosition);
        const moved = wrap(liveDistance - captureDistance + reference.length / 2, reference.length) - reference.length / 2;
        if (moved >= -2 && moved <= liveSpeed / 3.6 * ((now - vision.capturedAt) / 1000 + 0.5) + 5
            && Math.abs(prediction.referenceDelta[0]) <= 13 && Math.abs(prediction.referenceDelta[1]) <= 30) {
            lateral = prediction.referenceDelta[0];
        }
    }
    const target = targetsAt(reference, distance);
    const actual = { brake: intended.Physics_brake, throttle: intended.Physics_gas };
    const valid = (control: keyof typeof actual) => finite(actual[control]) && actual[control]! >= 0 && actual[control]! <= 1;
    const tooFast = target.speed !== undefined && speed - target.speed >= SPEED_TOLERANCE_KMH;
    const tooSlow = target.speed !== undefined && target.speed - speed >= SPEED_TOLERANCE_KMH;
    const desiredBrake = brakeTarget(reference, distance, target, speed, performance);
    const pedalZone = pedalZoneAt(reference, distance, speed);
    const brakeZone = pedalZone === undefined ? undefined : pedalZone === 'brake';
    const capability = carDecelerationAt(performance, carMotionAt(reference, distance, speed));
    const inTurn = reference.directionChanges.some(({ start, end }) => wrap(distance - start, reference.length) < end - start);
    const timing = measure.brake ? brakeTiming(reference, distance, speed,
        inTurn && brakeZone ? capability.possibleDeceleration ?? possibleBrakingDeceleration : possibleBrakingDeceleration) : {};
    const brakeDue = timing.brakeInMs !== undefined && timing.brakeInMs <= 1e-6;
    const brakeReleaseRate = measure.steering && valid('brake') && actual.brake! >= thresholds.brake
        ? bracket(reference, distance, 'distance').a.brakeReleaseRate : undefined;
    const result: ReferenceGuidance = { ready: true, lateralOffset: lateral,
        acceleration: target.speed === undefined ? undefined : target.speed / 3.6 * speedSlope(reference, distance),
        currentReferencePosition: headingAt(reference, liveDistance) === undefined ? undefined : { distance: liveDistance, length: reference.length },
        pedalZone, currentPedalZone: pedalZoneAt(reference, liveDistance, liveSpeed),
        ...timing, brakeRequired: brakeZone === undefined ? undefined : brakeZone || brakeDue,
        brakeReleaseRate,
        brakeReleaseDurationMs: brakeReleaseRate === undefined ? undefined : brakeReleaseDurationMs(reference, distance),
        braking: target.brake === undefined ? undefined : (brakeZone ?? target.brake > 0) || tooFast || brakeDue
            || (desiredBrake !== undefined && desiredBrake >= thresholds.brake) };
    const directionChanges = cornerDirectionChanges(reference, cornerBounds ? [cornerBounds] : undefined);
    result.steeringRange = headingAt(reference, distance) === undefined ? undefined
        : directionChanges.find(({ start, end }) => wrap(distance - start, reference.length) < end - start) ?? null;
    const directionActive = result.steeringRange !== undefined && cornerTurnSign !== undefined && cornerBounds
        && result.steeringRange?.turnSign === cornerTurnSign;
    if (directionActive && cornerBounds
        && result.steeringRange === directionChanges.find(({ turnSign }) => turnSign === cornerTurnSign)) {
        result.turnInCorner = { start: distanceAt(reference, cornerBounds.start), end: distanceAt(reference, cornerBounds.end) };
    }
    if (directionActive && result.steeringRange) {
        result.cornerPhase = cornerSteeringPhase(reference, result.steeringRange, distance);
        const currentBrake = sample.Physics_brake;
        if (result.cornerPhase === 'unwind' && liveSpeed >= 20 && result.braking === false
            && finite(currentBrake) && currentBrake >= 0 && currentBrake < thresholds.brake) result.cornerExit = now + 1500;
    }
    const cornerEnd = cornerBounds ? distanceAt(reference, cornerBounds.end) : undefined;
    // Before turn-in, use the first matching range ahead in the mapped corner, even after reference braking ends.
    const exitRange = directionActive ? result.steeringRange : cornerEnd !== undefined && directionChanges.find(({ start, turnSign }) =>
        turnSign === cornerTurnSign && wrap(start - distance, reference.length)
            < wrap(cornerEnd - distance, reference.length));
    const turnInSpeed = result.steeringRange !== undefined && cornerTurnSign !== undefined && cornerBounds && exitRange
        ? targetsAt(reference, exitRange.start).speed : undefined;
    result.brakeZoneExit = result.steeringRange === undefined || cornerTurnSign === undefined || !cornerBounds
        ? undefined : !exitRange ? false : turnInSpeed === undefined ? undefined : speed < turnInSpeed;
    if (measure.steering && lateral !== undefined && prediction && vision?.telemetry) {
        // Keep the error's captured orientation; compare the intended motion at the decision position.
        const leadDistance = wrap(distance - liveDistance, reference.length);
        const capturedDistance = wrap(distanceAt(reference, prediction.normalizedPosition) + leadDistance, reference.length);
        const trajectory = steeringTrajectoryAt(reference, directionChanges, intendedTelemetry ? distance : capturedDistance, cornerTurnSign);
        if (trajectory) result.steeringDirection = compareSteeringDirection(trajectory,
            prediction.referenceDelta, vision.telemetry.currentTelemetry, lap.game, cornerTurnSign, intendedTelemetry);
    }
    if (measure.brake && valid('brake')) {
        const pedal = actual.brake!;
        const remainingRange = cornerEnd !== undefined && exitRange ? { start: exitRange.start, end: cornerEnd }
            : result.steeringRange;
        const demand = target.speed === undefined ? undefined
            : remainingBrakeDemand(reference, performance, distance, speed, target.speed, remainingRange, brakeDue);
        let cue: ReferenceCue | undefined;
        if (pedal >= thresholds.brake && target.speed !== undefined) {
            const requiredDeceleration = brakingDecelerationTarget(reference, distance, target.speed, speed);
            if (speed <= target.speed + 1e-9 && demand! <= 1e-9) cue = 'releaseBrake';
            else if (directionActive && demand! > 1e-9) cue = 'easeBrake';
            else if (brakeZone && requiredDeceleration > 0 && !tooSlow && !tooFast && finite(intendedDeceleration)) {
                const tolerance = Math.max(0.1, requiredDeceleration * DECELERATION_TOLERANCE_RATIO);
                if (intendedDeceleration >= requiredDeceleration - tolerance
                    && intendedDeceleration <= requiredDeceleration + tolerance) cue = 'holdBrake';
            }
        }
        // A nominal reference brake zone must not contradict a justified full release.
        if (!cue && pedal < thresholds.brake && (demand === undefined || demand > 1e-9)) {
            if (brakeDue || (desiredBrake !== undefined && desiredBrake >= thresholds.brake)) cue = 'brake';
        }
        if (cue) {
            result[cue] = now + 1500;
            result.braking = true;
            if (cue === 'easeBrake') result.brakeReleaseDurationMs = brakeReleaseDurationMs(reference, distance);
        }
    }
    // More brake compares measured motion with the reference at the current position only.
    // Its closure confirms this strict deficit continuously for 500 ms before speaking.
    // The 1e-9 margin only absorbs floating-point roundoff at equal rates.
    const currentBrake = sample.Physics_brake;
    const referenceDeceleration = bracket(reference, liveDistance, 'distance').a.deceleration;
    if (measure.brake && result.releaseBrake === undefined && result.easeBrake === undefined
        && result.currentPedalZone === 'brake' && finite(currentBrake) && currentBrake >= thresholds.brake && currentBrake < 1
        && !currentBraking.releasing && finite(currentBraking.deceleration) && finite(referenceDeceleration)
        && referenceDeceleration > 0 && currentBraking.deceleration < referenceDeceleration - 1e-9) {
        result.moreBrake = now + 1500;
        result.braking = true;
    }
    const brakeAhead = reference.brakeEvents.some((event) => wrap(event - distance, reference.length) <= metresPerSecond * BRAKE_LOOKAHEAD_SECONDS);
    const phase = directionActive
        ? target.steering !== undefined && target.steering * cornerTurnSign! >= thresholds.steering ? 'turn' : 'prepare'
        : undefined;
    if (phase) {
        result.steeringPhase = phase;
        result.steering = now + 1500;
    }
    const braking = result.braking !== false || brakeAhead;
    const desiredThrottle = measure.throttle ? throttleTarget(reference, distance, target, speed) : undefined;
    if (measure.throttle && !brakeDue && !braking && valid('throttle') && valid('brake')
        && target.brake !== undefined && desiredThrottle !== undefined && desiredThrottle >= thresholds.throttle
        && actual.throttle! < thresholds.throttle && actual.brake! < thresholds.brake) {
        result.throttle = now + 1500;
    }
    return result;
}
