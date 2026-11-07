import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapReferenceLine } from '../live-map-reference';
import { referenceDirectionChanges } from './phrase-reference-guidance';

export const REFERENCE_DRIVING_AGGRESSION = 0;
export const MAX_DRIVING_CAUTION = 100;
export const normalizeDrivingAggression = (value: number) => Number.isFinite(value)
    ? Math.max(0, Math.min(MAX_DRIVING_CAUTION, value)) : REFERENCE_DRIVING_AGGRESSION;

export function copyPhraseMap(map: CircuitMapDto | null): CircuitMapDto | null {
    if (!map) return null;
    return {
        ...map,
        samples: Object.fromEntries(Object.entries(map.samples).map(([mode, points]) =>
            [mode, points?.map((point) => ({ ...point }))])),
        centerline_segments: map.centerline_segments?.map((segment) => ({ ...segment, tags: [...segment.tags] })),
        centerline_tags: map.centerline_tags?.map((tag) => ({ ...tag })),
    };
}

interface BrakePoint { distance: number; energy: number; brake: number }
interface BrakeZone { points: BrakePoint[]; start: number; bottom: number; exit: number }
const wrap = (value: number, length: number) => value >= 0 && value < length ? value : ((value % length) + length) % length;

function brakeAt(profile: BrakePoint[], position: number) {
    let low = 0, high = profile.length - 1;
    while (low + 1 < high) {
        const mid = (low + high) >>> 1;
        if (profile[mid].distance <= position) low = mid;
        else high = mid;
    }
    const a = profile[low], b = profile[high];
    const fraction = Math.max(0, Math.min(1, (position - a.distance) / (b.distance - a.distance)));
    return { energy: a.energy + (b.energy - a.energy) * fraction, brake: a.brake + (b.brake - a.brake) * fraction };
}

/** Phrase-owned data. Move braking earlier, preserving the minimum speed and peak deceleration. */
export function copyPhraseReference(lap: TopLapReferenceEntry | null,
    aggression = REFERENCE_DRIVING_AGGRESSION): TopLapReferenceEntry | null {
    if (!lap) return null;
    const copy = { ...lap, x: [...lap.x], y: lap.y.map((row) => [...row]), target_features: [...lap.target_features] };
    const caution = normalizeDrivingAggression(aggression) / 100;
    if (!caution) return copy;
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    if (speed < 0 || !lap.y.every((row) => Number.isFinite(row[speed]) && row[speed] >= 0)) return copy;
    const brake = lap.target_features.indexOf('expert_optimal_brake');
    const shiftBrake = brake >= 0 && lap.y.every((row) => Number.isFinite(row[brake]) && row[brake] >= 0 && row[brake] <= 1);
    const line = getLiveMapReferenceLine(lap);
    if (line.length < 2 || line.length !== lap.x.length
        || !lap.x.every((position, i) => Number.isFinite(position) && position >= 0 && position <= 1 && (!i || position > lap.x[i - 1]))
        || !line.every((point) => [point.x, point.y, point.z].every(Number.isFinite))) return copy;
    const distance = [0];
    line.forEach((point, i) => {
        const next = line[(i + 1) % line.length];
        distance.push(distance[i] + Math.hypot(next.x - point.x, next.y - point.y, next.z - point.z));
    });
    const length = distance[line.length];
    if (!Number.isFinite(length) || length <= 0) return copy;
    // Unroll cyclic sections and collapse coincident points before calculating acceleration.
    const points: { distance: number; position: number; row: number[]; energy: number }[] = [];
    [-1, 0, 1].forEach((offset) => lap.y.forEach((row, i) => {
        const at = distance[i] + offset * length;
        if (points.length && at === points[points.length - 1].distance) points.pop();
        points.push({ distance: at, position: lap.x[i] + offset, row, energy: (row[speed] / 3.6) ** 2 });
    }));
    const at = (position: number) => {
        let low = 0, high = points.length - 1;
        while (low + 1 < high) {
            const mid = (low + high) >>> 1;
            if (points[mid].distance <= position) low = mid;
            else high = mid;
        }
        const a = points[low], b = points[high];
        return { a, b, fraction: (position - a.distance) / (b.distance - a.distance) };
    };
    const referenceEnergy = (position: number) => {
        const { a, b, fraction } = at(position);
        return b.energy < a.energy ? a.energy + (b.energy - a.energy) * fraction
            : ((a.row[speed] + (b.row[speed] - a.row[speed]) * fraction) / 3.6) ** 2;
    };
    const turns = referenceDirectionChanges(lap).flatMap(({ start, end }) =>
        [-1, 0, 1].map((offset) => ({ start: start + offset * length, end: end + offset * length })))
        .sort((a, b) => a.start - b.start);
    const zones: BrakeZone[] = [];
    for (let start = 1; start + 1 < points.length; start++) {
        if (points[start + 1].energy >= points[start].energy) continue;
        let bottom = start + 1, exit = bottom;
        while (exit + 1 < points.length && points[exit + 1].energy <= points[exit].energy) {
            exit++;
            if (points[exit].energy < points[exit - 1].energy) bottom = exit;
        }
        const originalStart = points[start].distance, originalEnd = points[bottom].distance;
        const turn = turns.find(({ start: entry, end }) => end > originalStart && entry < originalEnd);
        if (originalEnd < 0 || originalEnd >= length || !turn) { start = exit; continue; }
        const advance = originalEnd - turn.start;
        const span = originalEnd - originalStart, minimum = points[bottom].energy;
        let peak = 0, peakEnd = start + 1;
        for (let i = start; i < bottom; i++) {
            const rate = (points[i].energy - points[i + 1].energy) / (2 * (points[i + 1].distance - points[i].distance));
            if (!peak || rate > peak + 1e-9) { peak = rate; peakEnd = i + 1; }
        }
        // Do not shift a later corner through the preceding corner's minimum.
        let approach = start;
        while (approach > 0 && points[approach - 1].energy <= points[approach].energy) approach--;
        while (approach < start && points[approach + 1].energy === points[approach].energy) approach++;
        const tailLength = originalEnd - points[peakEnd].distance;
        const tailArea = (points[peakEnd].energy - minimum) / 2;
        const build = (amount: number) => {
            const shift = advance * amount;
            const tailScale = Math.max(0, 1 - shift / span);
            // Shortening the release loses integral(a ds). Replace exactly that area
            // with a hold at the unchanged peak: v_end² = v_start² - 2 integral(a ds).
            const hold = (1 - tailScale) * tailArea / peak;
            const end = originalEnd - shift;
            const begin = end - (points[peakEnd].distance - originalStart + hold + tailScale * tailLength);
            const profile: BrakePoint[] = points.slice(start, peakEnd + 1).map((point) => ({
                distance: begin + point.distance - originalStart, energy: point.energy, brake: point.row[brake],
            }));
            const release = profile[profile.length - 1].distance + hold;
            if (hold > 0) profile.push({ distance: release, energy: minimum + tailScale * (points[peakEnd].energy - minimum),
                brake: points[peakEnd].row[brake] });
            if (tailScale > 0) points.slice(peakEnd + 1, bottom + 1).forEach((point) => profile.push({
                distance: release + tailScale * (point.distance - points[peakEnd].distance),
                energy: minimum + tailScale * (point.energy - minimum), brake: point.row[brake],
            }));
            profile[profile.length - 1] = { distance: end, energy: minimum, brake: 0 };
            return profile;
        };
        const feasible = (profile: BrakePoint[]) => brakeAt(profile, points[approach].distance).energy >= points[approach].energy
            && profile.some((point, i) => i > 0 && point.distance > points[approach].distance
                && Math.abs((profile[i - 1].energy - point.energy) / (2 * (point.distance - profile[i - 1].distance)) - peak) < 1e-7
                && referenceEnergy(point.distance) > point.energy + 1e-9);
        let amount = caution, profile = build(amount);
        if (!feasible(profile)) {
            // Close corners may not leave enough runway for the requested shift.
            // Keep their minima and peak rather than inventing extra braking force.
            let low = 0, high = amount;
            for (let i = 0; i < 40; i++) {
                const mid = (low + high) / 2;
                if (feasible(build(mid))) low = mid;
                else high = mid;
            }
            amount = low;
            profile = build(amount);
        }
        if (amount > 1e-8) zones.push({ points: profile, start: Math.max(profile[0].distance, points[approach].distance),
            bottom: originalEnd, exit: points[exit].distance });
        start = exit;
    }
    if (!zones.length) return copy;
    const cyclicZones = zones.flatMap((zone) => [-1, 0, 1].map((offset) => ({
        points: zone.points.map((point) => ({ ...point, distance: point.distance + offset * length })),
        start: zone.start + offset * length,
        bottom: zone.bottom + offset * length, exit: zone.exit + offset * length,
    })));
    const knots = new Set(distance);
    cyclicZones.forEach((zone) => zone.points.forEach(({ distance: position }) => {
        if (position >= zone.start && position > 0 && position < length) knots.add(position);
    }));
    // Keep the exact intersection with the preceding acceleration. Omitting it
    // would average the first braking interval and reduce the displayed peak.
    const boundaries = Array.from(knots).sort((a, b) => a - b);
    boundaries.slice(0, -1).forEach((left, i) => {
        const right = boundaries[i + 1], midpoint = (left + right) / 2;
        cyclicZones.forEach((zone) => {
            if (midpoint < zone.start || midpoint >= zone.bottom) return;
            const difference = (position: number) => referenceEnergy(position) - brakeAt(zone.points, position).energy;
            if (difference(left) * difference(right) >= 0) return;
            let low = left, high = right;
            const sign = Math.sign(difference(left));
            for (let step = 0; step < 40; step++) {
                const mid = (low + high) / 2;
                if (Math.sign(difference(mid)) === sign) low = mid;
                else high = mid;
            }
            knots.add((low + high) / 2);
        });
    });
    const adjust = (row: number[], position: number) => {
        let energy = referenceEnergy(position), pressure: number | undefined;
        cyclicZones.forEach((zone) => {
            if (position < zone.start || position > zone.exit) return;
            const target = brakeAt(zone.points, position);
            // An earlier approach may still be accelerating below the shifted curve.
            // Start its pedal rise at the actual speed intersection.
            const braking = target.energy < energy - 1e-8;
            if (position <= zone.bottom) energy = Math.min(energy, target.energy);
            pressure = Math.max(pressure ?? 0, braking ? target.brake : 0);
        });
        row[speed] = Math.sqrt(Math.max(0, energy)) * 3.6;
        if (shiftBrake && pressure !== undefined) row[brake] = pressure;
    };
    copy.y.forEach((row, i) => adjust(row, wrap(distance[i], length)));
    Array.from(knots).forEach((position) => {
        const { a, b, fraction } = at(position);
        if (position <= 0 || position >= length || position - a.distance < 1e-7 || b.distance - position < 1e-7) return;
        const row = a.row.map((value, i) => value + (b.row[i] - value) * fraction);
        adjust(row, position);
        // Avoid rounding an exact pedal-release boundary just past its lap position.
        copy.x.push(Number(wrap(a.position + (b.position - a.position) * fraction, 1).toFixed(12)));
        copy.y.push(row);
    });
    const ordered = copy.x.map((position, i) => ({ position, row: copy.y[i] })).sort((a, b) => a.position - b.position);
    copy.x = ordered.map(({ position }) => position);
    copy.y = ordered.map(({ row }) => row);
    return copy;
}
