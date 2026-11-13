import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';

export interface DrivingSpeedPoint {
    readonly position: number;
    readonly distance: number;
    readonly x: number;
    readonly ground: number;
    readonly speed: number;
}
interface Turn { start: number; end: number; sign: number }
interface EnergyPoint { distance: number; energy: number }
const wrap = (value: number, length: number) => value >= 0 && value < length ? value : ((value % length) + length) % length;
const angleChange = (from: number, to: number) => Math.atan2(Math.sin(to - from), Math.cos(to - from));
const MIN_CURVATURE = Math.PI / 180 / 10;
const CURVATURE_EPSILON = 1e-12;

function bracket(points: readonly DrivingSpeedPoint[], value: number, key: 'position' | 'distance', period: number) {
    let low = 0, high = points.length;
    while (low < high) {
        const middle = (low + high) >>> 1;
        if (points[middle][key] <= value) low = middle + 1;
        else high = middle;
    }
    const a = points[(low + points.length - 1) % points.length], b = points[low % points.length];
    const start = a[key] - (low === 0 ? period : 0), end = b[key] + (low === points.length ? period : 0);
    return { a, b, fraction: end > start ? (value - start) / (end - start) : 0, index: low };
}

/** Find actual reference turns and clip them to the circuit's tagged corners. */
function trainingTurns(points: readonly DrivingSpeedPoint[], length: number, map: CircuitMapDto | null): Turn[] {
    const corners = getCircuitMapCenterlineSegments(map).filter((segment) => {
        const { start_position: start, end_position: end } = segment;
        const tags = segment.tags.map((tag) => tag.trim().toLowerCase());
        return [start, end].every((p) => Number.isFinite(p) && p >= 0 && p <= 1)
            && (end < start ? end - start + 1 : end - start) > 0
            && tags.some((tag) => ['corner', 'slow corner', 'fast corner'].includes(tag)) && !tags.includes('consecutive corners');
    }).sort((a, b) => a.start_position - b.start_position)
        .filter((corner, i, all) => !all.slice(0, i).some((other) =>
            other.start_position === corner.start_position && other.end_position === corner.end_position));
    if (!corners.length) return [];
    const headings = points.map((point) => {
        const positions = [-1, 1].map((sign) => {
            const { a, b, fraction } = bracket(points, wrap(point.distance + sign * Math.min(5, length / 4), length), 'distance', length);
            return [a.x + (b.x - a.x) * fraction, a.ground + (b.ground - a.ground) * fraction];
        });
        const dx = positions[1][0] - positions[0][0], dy = positions[1][1] - positions[0][1];
        return Math.hypot(dx, dy) > 1e-9 ? Math.atan2(dy, dx) : undefined;
    });
    const segments = points.flatMap((point, i) => {
        const next = points[(i + 1) % points.length], end = i + 1 === points.length ? length : next.distance;
        if (end <= point.distance) return [];
        const heading = headings[i], nextHeading = headings[(i + 1) % points.length];
        const curvature = heading === undefined || nextHeading === undefined ? 0 : angleChange(heading, nextHeading) / (end - point.distance);
        const dx = next.x - point.x, dy = next.ground - point.ground;
        return [{ start: point.distance, end, curvature, sign: Math.abs(curvature) <= CURVATURE_EPSILON ? 0 : Math.sign(curvature),
            heading: Math.hypot(dx, dy) > 1e-9 ? Math.atan2(dy, dx) : undefined }];
    });
    const boundary = segments.findIndex((segment, i) => segment.sign !== segments[(i + segments.length - 1) % segments.length].sign);
    const origin = Math.max(0, boundary), turns: Turn[] = [];
    let sign = 0, onset: number | undefined, confirmed = false;
    for (let step = 0; step < segments.length; step++) {
        const i = (origin + step) % segments.length, segment = segments[i];
        const offset = i < origin ? length : 0, start = segment.start + offset, end = segment.end + offset;
        if (!step || sign !== segment.sign) {
            sign = segment.sign;
            onset = boundary < 0 ? start : undefined;
            confirmed = false;
        }
        if (!sign) continue;
        const incoming = segments[(i + segments.length - 1) % segments.length].heading;
        // Centered tangents confirm curvature; the vertex locates turn-in without anticipating it.
        if (onset === undefined && incoming !== undefined && segment.heading !== undefined
            && angleChange(incoming, segment.heading) * sign > CURVATURE_EPSILON) onset = start;
        if (onset === undefined || Math.abs(segment.curvature) < MIN_CURVATURE - CURVATURE_EPSILON) continue;
        const previous = turns[turns.length - 1];
        if (previous?.sign === sign && previous.end === start) previous.end = end;
        else turns.push({ start: confirmed ? start : onset, end, sign });
        confirmed = true;
    }
    const cyclicTurns = turns.flatMap((turn) => [-1, 0, 1].map((offset) => ({ ...turn,
        start: wrap(turn.start, length) + offset * length, end: wrap(turn.start, length) + turn.end - turn.start + offset * length })));
    const distanceAt = (position: number) => {
        const { a, b, fraction, index } = bracket(points, position, 'position', 1);
        const start = a.distance - (index === 0 ? length : 0), end = b.distance + (index === points.length ? length : 0);
        return wrap(start + (end - start) * fraction, length);
    };
    return corners.flatMap((corner) => {
        const start = distanceAt(corner.start_position);
        const end = start + (corner.end_position - corner.start_position === 1 ? length : wrap(distanceAt(corner.end_position) - start, length));
        const clipped = cyclicTurns.map((turn) => ({ start: Math.max(start, turn.start), end: Math.min(end, turn.end), sign: turn.sign }))
            .filter((turn) => turn.end > turn.start).sort((a, b) => a.start - b.start);
        const joined: Turn[] = [];
        clipped.forEach((turn) => {
            const previous = joined[joined.length - 1];
            if (previous?.sign === turn.sign && previous.end === turn.start) previous.end = turn.end;
            else joined.push(turn);
        });
        return joined.map((turn) => ({ ...turn, start: wrap(turn.start, length), end: wrap(turn.start, length) + turn.end - turn.start }));
    });
}

function energyAt(profile: EnergyPoint[], distance: number): number {
    let low = 0, high = profile.length - 1;
    while (low + 1 < high) {
        const middle = (low + high) >>> 1;
        if (profile[middle].distance <= distance) low = middle;
        else high = middle;
    }
    const a = profile[low], b = profile[high];
    const fraction = Math.max(0, Math.min(1, (distance - a.distance) / (b.distance - a.distance)));
    return a.energy + (b.energy - a.energy) * fraction;
}

/** Extend low-rate braking toward turn-in while retaining braking work and the original speed minimum. */
function cautiousSpeeds(samples: readonly DrivingSpeedPoint[], length: number, turns: Turn[], caution: number): number[] {
    if (!turns.length) return samples.map((point) => point.speed);
    const points: EnergyPoint[] = [];
    [-1, 0, 1].forEach((offset) => samples.forEach((point) => {
        const distance = point.distance + offset * length;
        if (points.length && points[points.length - 1].distance === distance) points.pop();
        points.push({ distance, energy: (point.speed / 3.6) ** 2 });
    }));
    const cyclicTurns = turns.flatMap((turn) => [-1, 0, 1].map((offset) => ({
        start: turn.start + offset * length, end: turn.end + offset * length,
    }))).sort((a, b) => a.start - b.start);
    const zones: { profile: EnergyPoint[]; start: number; end: number }[] = [];
    const rateAt = (i: number) => (points[i].energy - points[i + 1].energy) / (2 * (points[i + 1].distance - points[i].distance));
    for (let start = 1; start + 1 < points.length; start++) {
        if (points[start + 1].energy >= points[start].energy) continue;
        let bottom = start + 1;
        while (bottom + 1 < points.length && points[bottom + 1].energy < points[bottom].energy) bottom++;
        const originalStart = points[start].distance;
        let turn = cyclicTurns.find((range) => range.end > originalStart && range.start < points[bottom].distance);
        // Short coasts or acceleration within one corner remain part of the same braking sequence.
        while (bottom + 1 < points.length) {
            let nextStart = bottom, seconds = 0;
            while (nextStart + 1 < points.length && points[nextStart + 1].energy >= points[nextStart].energy) {
                const a = points[nextStart], b = points[nextStart + 1];
                seconds += 2 * (b.distance - a.distance) / (Math.sqrt(a.energy) + Math.sqrt(b.energy));
                nextStart++;
                if (seconds > 0.5) break;
            }
            if (seconds > 0.5 || nextStart + 1 >= points.length) break;
            let nextBottom = nextStart + 1;
            while (nextBottom + 1 < points.length && points[nextBottom + 1].energy < points[nextBottom].energy) nextBottom++;
            const nextTurn = turn ?? cyclicTurns.find((range) => range.end > originalStart && range.start < points[nextBottom].distance);
            if (!nextTurn || points[nextStart].distance < nextTurn.start - 1e-9 || points[nextStart].distance >= nextTurn.end
                || points[nextBottom].energy >= points[bottom].energy) break;
            bottom = nextBottom;
            turn = nextTurn;
        }
        let exit = bottom;
        while (exit + 1 < points.length && points[exit + 1].energy === points[exit].energy) exit++;
        const end = points[bottom].distance;
        if (end < 0 || end >= length || !turn) { start = exit - 1; continue; }
        const turnStart = turn.start;
        let peak = 0, peakEnd = start + 1;
        for (let i = start; i < bottom; i++) {
            const rate = rateAt(i);
            if (!peak || rate > peak + 1e-9) { peak = rate; peakEnd = i + 1; }
            else if (Math.abs(rate - peak) <= 1e-9) peakEnd = i + 1;
        }
        const floor = rateAt(bottom - 1);
        let floorStart = bottom - 1, peakStart = peakEnd - 1;
        while (floorStart > peakEnd && Math.abs(rateAt(floorStart - 1) - floor) <= 1e-9) floorStart--;
        while (peakStart > start && Math.abs(rateAt(peakStart - 1) - peak) <= 1e-9) peakStart--;
        if (floor >= peak - 1e-9 || points[floorStart].distance <= turnStart + 1e-9) { start = exit - 1; continue; }
        let releaseWork = 0;
        for (let i = peakEnd; i < floorStart; i++) releaseWork += rateAt(i) * (points[i + 1].distance - points[i].distance);
        const build = (amount: number): EnergyPoint[] | null => {
            const advance = amount * (points[floorStart].distance - turnStart);
            const releaseScale = Math.max(0, 1 - advance / (points[floorStart].distance - originalStart));
            const floorScale = 1 + advance / (end - points[floorStart].distance);
            const peakDistance = points[peakEnd].distance - points[peakStart].distance
                + (releaseWork * (1 - releaseScale) - floor * advance) / peak;
            if (peakDistance <= 0) return null;
            const peakScale = peakDistance / (points[peakEnd].distance - points[peakStart].distance);
            const profile: EnergyPoint[] = [{ distance: end, energy: points[bottom].energy }];
            for (let i = bottom - 1; i >= start; i--) {
                const scale = i >= floorStart ? floorScale : i >= peakEnd ? releaseScale : i >= peakStart ? peakScale : 1;
                const distance = (points[i + 1].distance - points[i].distance) * scale;
                if (distance <= 0) continue;
                const next = profile[profile.length - 1];
                profile.push({ distance: next.distance - distance, energy: next.energy + 2 * rateAt(i) * distance });
            }
            return profile.reverse();
        };
        let approach = start;
        while (approach > 0 && points[approach - 1].energy <= points[approach].energy) approach--;
        while (approach < start && points[approach + 1].energy === points[approach].energy) approach++;
        let profile = build(caution);
        if (!profile || energyAt(profile, points[approach].distance) < points[approach].energy - 1e-8) {
            // A nearby preceding corner may leave insufficient distance for the requested extension.
            let low = 0, high = caution;
            for (let i = 0; i < 40; i++) {
                const middle = (low + high) / 2, candidate = build(middle);
                if (candidate && energyAt(candidate, points[approach].distance) >= points[approach].energy) low = middle;
                else high = middle;
            }
            profile = build(low)!;
        }
        zones.push({ profile, start: Math.max(profile[0].distance, points[approach].distance), end });
        start = exit - 1;
    }
    if (!zones.length) return samples.map((point) => point.speed);
    return samples.map((point) => {
        let energy = (point.speed / 3.6) ** 2;
        const distance = wrap(point.distance, length);
        zones.forEach((zone) => [-1, 0, 1].forEach((offset) => {
            const at = distance - offset * length;
            if (at >= zone.start && at <= zone.end) energy = Math.min(energy, energyAt(zone.profile, at));
        }));
        return Math.sqrt(Math.max(0, energy)) * 3.6;
    });
}

/** Training owns this calculation and caches each aggression against immutable reference samples. */
export function createDrivingSpeedProfile(points: readonly DrivingSpeedPoint[], length: number, map: CircuitMapDto | null) {
    const turns = trainingTurns(points, length, map);
    const profiles = new Map<number, readonly number[]>();
    return (aggression: number): readonly number[] => {
        if (!profiles.has(aggression)) {
            const speeds = aggression < 50 ? cautiousSpeeds(points, length, turns, 1 - aggression / 50)
                : points.map((point) => point.speed * (1 + (aggression - 50) / 100));
            profiles.set(aggression, Object.freeze(speeds));
        }
        return profiles.get(aggression)!;
    };
}
