import type { TopLapReferenceEntry } from 'services/topLapReferenceService';

export interface Pedals { readonly throttle: number; readonly brake: number }
interface ReferencePoint extends Pedals {
    readonly position: number;
    readonly distance: number;
}
export interface DrivingReference {
    readonly points: readonly ReferencePoint[];
    /** Distance along the closed reference polyline, in unchanged coordinate units. */
    readonly length: number;
}
export interface ReferencePedalTargets {
    reference: Pedals;
    target: Pedals;
    radius: number;
}

type ReferenceResult = { reference: DrivingReference; error?: never } | { reference: null; error: string };
const cache = new WeakMap<TopLapReferenceEntry, ReferenceResult>();
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const wrap = (distance: number, length: number) => ((distance % length) + length) % length;
export const normalizedAggression = (aggression: number) => (aggression - 50) / 50;
export const aggressionRadius = (aggression: number) => 20 * Math.abs(normalizedAggression(aggression));

/** Copy once per selected lap so pending actions keep an immutable reference snapshot. */
export function prepareDrivingReference(lap: TopLapReferenceEntry | null): DrivingReference | null {
    return inspectDrivingReference(lap).reference;
}

export function inspectDrivingReference(lap: TopLapReferenceEntry | null): ReferenceResult {
    if (!lap) return { reference: null, error: 'Live Map has no reference lap selected. Load a reference lap for the current car and circuit in Live Session.' };
    if (cache.has(lap)) return cache.get(lap)!;
    const build = (): ReferenceResult => {
        const invalid = (reason: string): ReferenceResult => ({ reference: null, error: `Live Map reference lap ${reason}` });
        const indices = ['player_pos_x', 'player_pos_y', 'player_pos_z', 'throttle', 'brake']
            .map((name) => lap.target_features.indexOf(`expert_optimal_${name}`));
        const missing = ['raw X coordinates', 'raw Y coordinates', 'raw Z coordinates', 'throttle', 'brake']
            .filter((_, i) => indices[i] < 0);
        if (missing.length) return invalid(`is missing ${missing.join(', ')} data. The map can display without all training inputs; load a reference lap containing these channels.`);
        if (lap.x.length < 2 || lap.y.length !== lap.x.length) return invalid('needs at least two samples with matching positions and telemetry rows.');
        const rows = lap.y.map((row) => indices.map((index) => row[index]));
        if (!lap.x.every((position, i) => finite(position) && position >= 0 && position <= 1
            && (i === 0 || position > lap.x[i - 1]))) return invalid('has invalid normalized positions. Positions must be strictly increasing and between 0 and 1.');
        for (let i = 0; i < rows.length; i++) {
            if (!rows[i].slice(0, 3).every(finite)) return invalid(`has invalid raw XYZ coordinates at sample ${i + 1}.`);
            for (const [index, name] of [[3, 'throttle'], [4, 'brake']] as const) {
                const value = rows[i][index];
                if (!finite(value) || value < 0 || value > 1) return invalid(`has invalid ${name} at sample ${i + 1} (${value}). Pedal values must be between 0 and 1.`);
            }
        }
        const distance = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        let cumulative = 0;
        const points = rows.map((row, i) => {
            if (i) cumulative += distance(rows[i - 1], row);
            return Object.freeze({ position: lap.x[i], distance: cumulative, throttle: row[3], brake: row[4] });
        });
        const closing = distance(rows[rows.length - 1], rows[0]);
        // Recorded start/finish samples can differ even at explicit 0/1 positions.
        // Keep their raw coordinates and include the closing segment in distance windows.
        const length = cumulative + closing;
        return finite(length) && length > 0
            ? { reference: Object.freeze({ points: Object.freeze(points), length }) }
            : invalid('has no usable travel distance in its raw XYZ coordinates.');
    };
    const reference = build();
    cache.set(lap, reference);
    return reference;
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

function interpolate(a: Pedals, b: Pedals, fraction: number): Pedals {
    return { throttle: a.throttle + (b.throttle - a.throttle) * fraction,
        brake: a.brake + (b.brake - a.brake) * fraction };
}

export function referencePedalTargets(reference: DrivingReference, position: number, aggression: number): ReferencePedalTargets {
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
    const pedals = interpolate(previous, next, fraction);
    const startDistance = previous.distance - (index === 0 ? length : 0);
    const endDistance = next.distance + (index === points.length ? length : 0);
    const center = wrap(startDistance + (endDistance - startDistance) * fraction, length);
    const radius = aggressionRadius(aggression);
    if (!radius) return { reference: pedals, target: pedals, radius };

    const atDistance = (value: number): Pedals => {
        const d = wrap(value, length);
        const i = upperBound(points, d, 'distance');
        const a = points[(i + points.length - 1) % points.length];
        const b = points[i % points.length];
        const from = a.distance - (i === 0 ? length : 0);
        const to = b.distance + (i === points.length ? length : 0);
        return interpolate(a, b, to > from ? (d - from) / (to - from) : 0);
    };
    // Piecewise-linear extrema occur at vertices or the interpolated window boundaries.
    let throttle = pedals.throttle, brake = pedals.brake;
    const include = (point: Pedals) => {
        throttle = aggression < 50 ? Math.min(throttle, point.throttle) : Math.max(throttle, point.throttle);
        brake = aggression < 50 ? Math.max(brake, point.brake) : Math.min(brake, point.brake);
    };
    include(atDistance(center - radius));
    include(atDistance(center + radius));
    points.forEach((point) => {
        const delta = wrap(point.distance - center, length);
        if (Math.min(delta, length - delta) <= radius) include(point);
    });
    return { reference: pedals, target: { throttle, brake }, radius };
}
