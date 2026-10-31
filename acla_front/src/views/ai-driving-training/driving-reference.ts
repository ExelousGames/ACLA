import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';

interface ReferencePoint {
    readonly position: number;
    readonly distance: number;
    readonly speed: number;
    readonly throttle: number;
    readonly brake: number;
}
interface ReferencePedalEvent {
    readonly distance: number;
    readonly kind: 'acceleration' | 'deceleration';
    readonly shiftLimit: number;
}
export interface DrivingReference {
    readonly points: readonly ReferencePoint[];
    readonly pedalEvents: readonly ReferencePedalEvent[];
    /** Distance along the closed reference polyline, in unchanged coordinate units. */
    readonly length: number;
}
export interface ReferenceSpeedTargets {
    /** Reference and aggression-adjusted speeds in km/h. */
    reference: number;
    target: number;
    radius: number;
}

type ReferenceResult = { reference: DrivingReference; error?: never } | { reference: null; error: string };
const cache = new WeakMap<TopLapReferenceEntry, ReferenceResult>();
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const wrap = (distance: number, length: number) => ((distance % length) + length) % length;
export const normalizedAggression = (aggression: number) => (aggression - 50) / 50;
export const aggressionRadius = (aggression: number, maxRadius = DEFAULT_DRIVING_CONFIG.reward.aggressionMaxRadius) => maxRadius * Math.abs(normalizedAggression(aggression));

/** Copy once per selected lap so pending actions keep an immutable reference snapshot. */
export function prepareDrivingReference(lap: TopLapReferenceEntry | null): DrivingReference | null {
    return inspectDrivingReference(lap).reference;
}

export function inspectDrivingReference(lap: TopLapReferenceEntry | null): ReferenceResult {
    if (!lap) return { reference: null, error: 'Live Map has no reference lap selected. Load a reference lap for the current car and circuit in Live Session.' };
    if (cache.has(lap)) return cache.get(lap)!;
    const build = (): ReferenceResult => {
        const invalid = (reason: string): ReferenceResult => ({ reference: null, error: `Live Map reference lap ${reason}` });
        const indices = ['player_pos_x', 'player_pos_y', 'player_pos_z', 'speed', 'throttle', 'brake']
            .map((name) => lap.target_features.indexOf(`expert_optimal_${name}`));
        const missing = ['raw X coordinates', 'raw Y coordinates', 'raw Z coordinates', 'speed', 'throttle', 'brake']
            .filter((_, i) => indices[i] < 0);
        if (missing.length) return invalid(`is missing ${missing.join(', ')} data. The map can display without all training inputs; load a reference lap containing these channels.`);
        if (lap.x.length < 2 || lap.y.length !== lap.x.length) return invalid('needs at least two samples with matching positions and telemetry rows.');
        const rows = lap.y.map((row) => indices.map((index) => row[index]));
        if (!lap.x.every((position, i) => finite(position) && position >= 0 && position <= 1
            && (i === 0 || position > lap.x[i - 1]))) return invalid('has invalid normalized positions. Positions must be strictly increasing and between 0 and 1.');
        for (let i = 0; i < rows.length; i++) {
            if (!rows[i].slice(0, 3).every(finite)) return invalid(`has invalid raw XYZ coordinates at sample ${i + 1}.`);
            const speed = rows[i][3];
            if (!finite(speed) || speed < 0) return invalid(`has invalid speed at sample ${i + 1} (${speed}). Speed must be a nonnegative value in km/h.`);
            for (const [index, name] of [[4, 'throttle'], [5, 'brake']] as const) {
                const pedal = rows[i][index];
                if (!finite(pedal) || pedal < 0 || pedal > 1) return invalid(`has invalid ${name} at sample ${i + 1}. Pedals must be between 0 and 1.`);
            }
        }
        const distance = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        let cumulative = 0;
        const points = rows.map((row, i) => {
            if (i) cumulative += distance(rows[i - 1], row);
            return Object.freeze({ position: lap.x[i], distance: cumulative, speed: row[3], throttle: row[4], brake: row[5] });
        });
        const closing = distance(rows[rows.length - 1], rows[0]);
        // Recorded start/finish samples can differ even at explicit 0/1 positions.
        // Keep their raw coordinates and include the closing segment in distance windows.
        const length = cumulative + closing;
        return finite(length) && length > 0
            ? { reference: Object.freeze({ points: Object.freeze(points), pedalEvents: pedalEvents(points, length), length }) }
            : invalid('has no usable travel distance in its raw XYZ coordinates.');
    };
    const reference = build();
    cache.set(lap, reference);
    return reference;
}

/** Anchor timing to pedal intent, including lift-off and throttle pickup during trail braking.
 * Brake release alone is not acceleration. Flat pedals retain the preceding phase. */
function pedalEvents(points: readonly ReferencePoint[], length: number): readonly ReferencePedalEvent[] {
    const changes: { distance: number; kind: ReferencePedalEvent['kind'] }[] = [];
    points.forEach((point, i) => {
        const next = points[(i + 1) % points.length];
        if ((i + 1 === points.length ? length : next.distance) <= point.distance) return;
        const throttle = next.throttle - point.throttle, brake = next.brake - point.brake;
        const kind = brake > 1e-6 || throttle < -1e-6 ? 'deceleration' : throttle > 1e-6 ? 'acceleration' : null;
        if (kind) changes.push({ distance: point.distance, kind });
    });
    const events = changes.filter((event, i) => event.kind !== changes[(i + changes.length - 1) % changes.length].kind);
    if (!events.length && changes.length) events.push(changes[0]);
    return Object.freeze(events.map((event, i) => {
        const previous = events[(i + events.length - 1) % events.length];
        const next = events[(i + 1) % events.length];
        const before = event.distance - previous.distance + (i === 0 ? length : 0);
        const after = next.distance - event.distance + (i === events.length - 1 ? length : 0);
        // Keep neighboring events ordered even on short laps or closely spaced corners.
        return Object.freeze({ ...event, shiftLimit: Math.min(before, after) / 4 });
    }));
}

/** Map a driver distance back onto the reference between shifted pedal-event anchors. */
function timedReferenceDistance(reference: DrivingReference, distance: number, aggression: number, radius: number): number {
    const events = reference.pedalEvents;
    if (!events.length) return distance;
    const shifted = events.map((event) => event.distance + Math.sign(aggression - 50)
        * (event.kind === 'deceleration' ? 1 : -1) * Math.min(radius, event.shiftLimit));
    const fromStart = wrap(distance - shifted[0], reference.length) + shifted[0];
    let index = 0;
    while (index + 1 < events.length && shifted[index + 1] <= fromStart) index++;
    const last = index === events.length - 1;
    const end = last ? shifted[0] + reference.length : shifted[index + 1];
    const referenceEnd = last ? events[0].distance + reference.length : events[index + 1].distance;
    const fraction = (fromStart - shifted[index]) / (end - shifted[index]);
    return events[index].distance + fraction * (referenceEnd - events[index].distance);
}

function upperBound(points: readonly ReferencePoint[], value: number, key: 'position' | 'distance'): number {
    let low = 0, high = points.length;
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (points[mid][key] <= value) low = mid + 1;
        else high = mid;
    }
    return low;
}

function interpolate(a: ReferencePoint, b: ReferencePoint, fraction: number): number {
    return a.speed + (b.speed - a.speed) * fraction;
}

export function referenceSpeedTargets(reference: DrivingReference, position: number, aggression: number, maxRadius = DEFAULT_DRIVING_CONFIG.reward.aggressionMaxRadius): ReferenceSpeedTargets {
    if (!finite(position) || position < 0 || position > 1 || !finite(aggression) || aggression < 0 || aggression > 100) {
        throw new Error('Invalid reference position or driving aggression.');
    }
    const { points, length } = reference;
    const index = upperBound(points, position, 'position');
    const previous = points[(index + points.length - 1) % points.length];
    const next = points[index % points.length];
    const start = previous.position - (index === 0 ? 1 : 0);
    const end = next.position + (index === points.length ? 1 : 0);
    const fraction = end > start ? (position - start) / (end - start) : 0;
    const speed = interpolate(previous, next, fraction);
    const startDistance = previous.distance - (index === 0 ? length : 0);
    const endDistance = next.distance + (index === points.length ? length : 0);
    const center = wrap(startDistance + (endDistance - startDistance) * fraction, length);
    const radius = aggressionRadius(aggression, maxRadius);
    if (!radius || !reference.pedalEvents.length) return { reference: speed, target: speed, radius };

    const atDistance = (value: number): number => {
        const d = wrap(value, length);
        const i = upperBound(points, d, 'distance');
        const a = points[(i + points.length - 1) % points.length];
        const b = points[i % points.length];
        const from = a.distance - (i === 0 ? length : 0);
        const to = b.distance + (i === points.length ? length : 0);
        return interpolate(a, b, to > from ? (d - from) / (to - from) : 0);
    };
    const shiftedSpeed = atDistance(timedReferenceDistance(reference, center, aggression, radius));
    // Widen the slower-speed area for caution and narrow it for aggression.
    // Speed bumps inside a pedal phase must never reverse that relationship.
    const target = aggression < 50 ? Math.min(speed, shiftedSpeed) : Math.max(speed, shiftedSpeed);
    return { reference: speed, target, radius };
}
