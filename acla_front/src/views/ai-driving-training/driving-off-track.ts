import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';
import { getLiveMapPosition } from 'views/live-session/live-map-data';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';

export const OFF_TRACK_PENALTY_PER_SECOND = 5;
export const ON_TRACK_REWARD_PER_SECOND = 1;
const wheels = ['front_left', 'front_right', 'rear_left', 'rear_right'] as const;
type Point = { x: number; y: number };
type Strip = { start: number; end: number; polygon: Point[]; a: Point; b: Point };
const stripsByMap = new WeakMap<CircuitMapDto, Strip[]>();
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const cross = (a: Point, b: Point, p: Point) => (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const wrap = (position: number) => ((position % 1) + 1) % 1;

function prepareLine(samples: CircuitMapBinSample[] = []): CircuitMapBinSample[] {
    if (!samples.every((p) => [p.x, p.y, p.z, p.normalized_position].every(finite)
        && p.normalized_position >= 0 && p.normalized_position <= 1)) return [];
    return [...samples].sort((a, b) => a.normalized_position - b.normalized_position)
        .filter((p, i, all) => !i || p.normalized_position !== all[i - 1].normalized_position);
}

// Do not invent a road across missing sections of either recorded boundary.
function covered(line: CircuitMapBinSample[], position: number): boolean {
    if (line.length < 2) return false;
    let next = line.findIndex((p) => p.normalized_position >= position);
    if (next < 0) next = line.length;
    const end = line[next % line.length].normalized_position + (next === line.length ? 1 : 0);
    const start = line[(next + line.length - 1) % line.length].normalized_position - (next === 0 ? 1 : 0);
    return end - start <= 0.02 + 1e-9;
}

function trackStrips(map: CircuitMapDto): Strip[] {
    const cached = stripsByMap.get(map);
    if (cached) return cached;
    const left = prepareLine(map.samples.left_boundary), right = prepareLine(map.samples.right_boundary);
    const positions = Array.from(new Set([...left, ...right].map((p) => p.normalized_position))).sort((a, b) => a - b);
    const strips: Strip[] = [];
    for (let i = 0; i < positions.length; i++) {
        const start = positions[i], end = i + 1 < positions.length ? positions[i + 1] : positions[0] + 1;
        if (end <= start || !covered(left, wrap((start + end) / 2)) || !covered(right, wrap((start + end) / 2))) continue;
        const l0 = getLiveMapPosition(left, start)!, r0 = getLiveMapPosition(right, start)!;
        const l1 = getLiveMapPosition(left, wrap(end))!, r1 = getLiveMapPosition(right, wrap(end))!;
        const polygon = [l0, l1, r1, r0];
        // Crossed or collapsed edges cannot define a reliable track surface.
        const turns = polygon.map((p, j) => cross(p, polygon[(j + 1) % 4], polygon[(j + 2) % 4]));
        if (!turns.every((n) => n > 1e-9) && !turns.every((n) => n < -1e-9)) continue;
        strips.push({ start, end, polygon, a: midpoint(l0, r0), b: midpoint(l1, r1) });
    }
    stripsByMap.set(map, strips);
    return strips;
}

function wheelOffTrack(point: Point, strips: Strip[]): boolean | null {
    let nearest = Infinity, coveredPosition = false;
    for (const { polygon, a, b, start, end } of strips) {
        const sides = polygon.map((p, i) => cross(p, polygon[(i + 1) % 4], point));
        // A tyre touching either boundary is still on track.
        if (sides.every((n) => n >= -1e-9) || sides.every((n) => n <= 1e-9)) return false;
        const dx = b.x - a.x, dy = b.y - a.y, lengthSquared = dx * dx + dy * dy;
        if (lengthSquared <= 1e-9) continue;
        const fraction = ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared;
        const t = Math.max(0, Math.min(1, fraction));
        const distance = (point.x - a.x - t * dx) ** 2 + (point.y - a.y - t * dy) ** 2;
        if (distance < nearest) {
            nearest = distance;
            coveredPosition = (fraction >= 0 && fraction <= 1) || strips.some((other) => fraction < 0
                ? Math.abs(wrap(other.end) - wrap(start)) < 1e-9
                : Math.abs(wrap(other.start) - wrap(end)) < 1e-9);
        }
    }
    // Beyond the end of a partial recording, outside and unmapped are indistinguishable.
    return coveredPosition ? true : null;
}

/** null means unavailable; reward-only data never changes the policy's feature layout. */
export function detectDrivingOffTrack(game: string, sample: StandardTelemetrySample, map: CircuitMapDto | null): boolean | null {
    if (game === 'iracing') {
        const surface = sample.Graphics_player_track_surface;
        // IRSDK TrkLoc: -1 not in world, 0 off track, 1 pit stall, 2 approaching pits, 3 on track.
        return finite(surface) && [0, 1, 2, 3].includes(surface) ? surface === 0 : null;
    }
    const trackKey = (track: string | null | undefined) => getAccTelemetryTrackKey(track) || track?.trim();
    if (game !== 'acc' || !map || map.game !== game || !sample.Static_track
        || trackKey(map.source_track_key || map.circuit_name) !== trackKey(sample.Static_track)) return null;
    const position = sample.Graphics_normalized_car_position;
    if (!finite(position) || position < 0 || position > 1) return null;
    // ACC's adapter has already converted tyre contacts and circuit maps to horizontal X/Y.
    const points = wheels.map((wheel) => ({ x: sample[`Physics_tyre_contact_point_${wheel}_x`],
        y: sample[`Physics_tyre_contact_point_${wheel}_y`] }));
    if (!points.every((p) => finite(p.x) && finite(p.y))) return null;
    const strips = trackStrips(map).filter(({ start, end }) => {
        const distance = Math.abs(wrap((start + end) / 2) - position);
        return Math.min(distance, 1 - distance) <= 0.05;
    });
    if (!strips.some(({ start, end }) => (position >= start && position <= end)
        || (position + 1 >= start && position + 1 <= end))) return null;
    const outside = (points as Point[]).map((point) => wheelOffTrack(point, strips));
    return outside.includes(null) ? null : outside.every((offTrack) => offTrack);
}
