import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { createDrivingSpeedProfile, type DrivingSpeedPoint } from './driving-speed-profile';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';

interface ReferencePoint extends DrivingSpeedPoint {
    readonly throttle: number;
    readonly brake: number;
}
export interface DrivingReference {
    readonly points: readonly ReferencePoint[];
    /** Cached target samples from an owned lap and corner snapshot. */
    readonly targetSpeeds: (aggression: number) => readonly number[];
    /** Distance along the closed reference polyline, in unchanged coordinate units. */
    readonly length: number;
}
export interface ReferenceSpeedTargets {
    /** Reference and aggression-adjusted speeds in km/h. */
    reference: number;
    target: number;
}

type ReferenceResult = { reference: DrivingReference; error?: never } | { reference: null; error: string };
const cache = new WeakMap<TopLapReferenceEntry, WeakMap<object, ReferenceResult>>();
const noMap = {};
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
export const normalizedAggression = (aggression: number) => (aggression - 50) / 50;

/** Copy once per selected lap so pending actions keep an immutable reference snapshot. */
export function prepareDrivingReference(lap: TopLapReferenceEntry | null, map: CircuitMapDto | null = null): DrivingReference | null {
    return inspectDrivingReference(lap, map).reference;
}

export function inspectDrivingReference(lap: TopLapReferenceEntry | null, map: CircuitMapDto | null = null): ReferenceResult {
    if (!lap) return { reference: null, error: 'Live Map has no reference lap selected. Load a reference lap for the current car and circuit in Live Session.' };
    const references = cache.get(lap) ?? new WeakMap<object, ReferenceResult>();
    const cached = references.get(map ?? noMap);
    if (cached) return cached;
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
            return Object.freeze({ position: lap.x[i], distance: cumulative, x: row[0], ground: row[lap.game === 'acc' ? 1 : 2],
                speed: row[3], throttle: row[4], brake: row[5] });
        });
        const closing = distance(rows[rows.length - 1], rows[0]);
        // Recorded start/finish samples can differ even at explicit 0/1 positions.
        // Keep their raw coordinates and include the closing segment in travel distance.
        const length = cumulative + closing;
        if (!finite(length) || length <= 0) return invalid('has no usable travel distance in its raw XYZ coordinates.');
        const targetSpeeds = createDrivingSpeedProfile(points, length, map);
        return { reference: Object.freeze({ points: Object.freeze(points), targetSpeeds, length }) };
    };
    const result = build();
    references.set(map ?? noMap, result);
    cache.set(lap, references);
    return result;
}

function upperBound(points: readonly ReferencePoint[], value: number): number {
    let low = 0, high = points.length;
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (points[mid].position <= value) low = mid + 1;
        else high = mid;
    }
    return low;
}

export function referenceSpeedTargets(reference: DrivingReference, position: number, aggression: number): ReferenceSpeedTargets {
    if (!finite(position) || position < 0 || position > 1 || !finite(aggression) || aggression < 0 || aggression > 100) {
        throw new Error('Invalid reference position or driving aggression.');
    }
    const { points } = reference;
    const index = upperBound(points, position);
    const previousIndex = (index + points.length - 1) % points.length, nextIndex = index % points.length;
    const previous = points[previousIndex], next = points[nextIndex];
    const start = previous.position - (index === 0 ? 1 : 0);
    const end = next.position + (index === points.length ? 1 : 0);
    const fraction = end > start ? (position - start) / (end - start) : 0;
    const speeds = reference.targetSpeeds(aggression);
    return {
        reference: previous.speed + (next.speed - previous.speed) * fraction,
        target: speeds[previousIndex] + (speeds[nextIndex] - speeds[previousIndex]) * fraction,
    };
}
