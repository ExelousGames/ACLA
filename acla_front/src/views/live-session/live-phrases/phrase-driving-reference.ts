import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapReferenceLine } from '../live-map-reference';
import { referenceDirectionChanges, type ReferenceCornerBounds } from './phrase-reference-guidance';

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
const MAX_BRAKING_GAP_S = 0.5;
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

/** Rebuild phrase-owned speed and brake targets on the baseline's unchanged reference samples. */
export function copyPhraseReference(lap: TopLapReferenceEntry | null,
    aggression = REFERENCE_DRIVING_AGGRESSION, corners?: readonly ReferenceCornerBounds[]): TopLapReferenceEntry | null {
    if (!lap) return null;
    const copy = { ...lap, x: [...lap.x], y: lap.y.map((row) => [...row]), target_features: [...lap.target_features] };
    const caution = normalizeDrivingAggression(aggression) / 100;
    if (!caution) return copy;
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    if (speed < 0 || !lap.y.every((row) => Number.isFinite(row[speed]) && row[speed] >= 0)) return copy;
    const brake = lap.target_features.indexOf('expert_optimal_brake');
    const adjustBrake = brake >= 0 && lap.y.every((row) => Number.isFinite(row[brake]) && row[brake] >= 0 && row[brake] <= 1);
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
    const points: { distance: number; row: number[]; energy: number }[] = [];
    [-1, 0, 1].forEach((offset) => lap.y.forEach((row, i) => {
        const at = distance[i] + offset * length;
        if (points.length && at === points[points.length - 1].distance) points.pop();
        points.push({ distance: at, row, energy: (row[speed] / 3.6) ** 2 });
    }));
    const turns = referenceDirectionChanges(lap, corners).flatMap(({ start, end }) =>
        [-1, 0, 1].map((offset) => ({ start: start + offset * length, end: end + offset * length })))
        .sort((a, b) => a.start - b.start);
    const zones: BrakeZone[] = [];
    for (let start = 1; start + 1 < points.length; start++) {
        if (points[start + 1].energy >= points[start].energy) continue;
        let bottom = start + 1;
        while (bottom + 1 < points.length && points[bottom + 1].energy < points[bottom].energy) bottom++;
        const originalStart = points[start].distance;
        let turn = turns.find(({ start: entry, end }) => end > originalStart && entry < points[bottom].distance);
        // A brief coast or acceleration can separate braking inputs for the same corner.
        // Keep their signed work together and anchor the sequence at its final minimum.
        while (bottom + 1 < points.length) {
            let nextStart = bottom, gapSeconds = 0;
            while (nextStart + 1 < points.length && points[nextStart + 1].energy >= points[nextStart].energy) {
                const a = points[nextStart], b = points[nextStart + 1];
                gapSeconds += 2 * (b.distance - a.distance) / (Math.sqrt(a.energy) + Math.sqrt(b.energy));
                nextStart++;
                if (gapSeconds > MAX_BRAKING_GAP_S) break;
            }
            if (gapSeconds > MAX_BRAKING_GAP_S || nextStart + 1 >= points.length) break;
            let nextBottom = nextStart + 1;
            while (nextBottom + 1 < points.length && points[nextBottom + 1].energy < points[nextBottom].energy) nextBottom++;
            const nextTurn = turn ?? turns.find(({ start: entry, end }) => end > originalStart && entry < points[nextBottom].distance);
            if (!nextTurn || points[nextStart].distance < nextTurn.start - 1e-9 || points[nextStart].distance >= nextTurn.end
                || points[nextBottom].energy >= points[bottom].energy) break;
            bottom = nextBottom;
            turn = nextTurn;
        }
        let exit = bottom;
        while (exit + 1 < points.length && points[exit + 1].energy === points[exit].energy) exit++;
        const originalEnd = points[bottom].distance;
        if (originalEnd < 0 || originalEnd >= length || !turn) { start = exit - 1; continue; }
        const minimum = points[bottom].energy, turnStart = turn.start;
        const rateAt = (i: number) => (points[i].energy - points[i + 1].energy)
            / (2 * (points[i + 1].distance - points[i].distance));
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
        // Include every release and reapplication, including signed work through a short gap.
        const release = points.slice(peakEnd, floorStart);
        if (floor >= peak - 1e-9 || points[floorStart].distance <= turnStart + 1e-9) { start = exit - 1; continue; }
        const releaseWork = release.reduce((sum, point, i) =>
            sum + rateAt(peakEnd + i) * (points[peakEnd + i + 1].distance - point.distance), 0);
        // Move the low-rate boundary toward turn-in, shortening release distances rather
        // than reducing their rates. Integrate backward from the unchanged speed minimum.
        const build = (amount: number) => {
            const advance = amount * (points[floorStart].distance - turnStart);
            const releaseScale = Math.max(0, 1 - advance / (points[floorStart].distance - originalStart));
            const floorScale = 1 + advance / (originalEnd - points[floorStart].distance);
            const peakDistance = points[peakEnd].distance - points[peakStart].distance
                + (releaseWork * (1 - releaseScale) - floor * advance) / peak;
            if (peakDistance <= 0) return null;
            const peakScale = peakDistance / (points[peakEnd].distance - points[peakStart].distance);
            const profile: BrakePoint[] = [{ distance: originalEnd, energy: minimum, brake: 0 }];
            const prepend = (distance: number, rate: number, pressure: number) => {
                if (distance <= 0) return;
                const next = profile[profile.length - 1];
                profile.push({ distance: next.distance - distance, energy: next.energy + 2 * rate * distance, brake: pressure });
            };
            for (let i = bottom - 1; i >= start; i--) {
                const distance = points[i + 1].distance - points[i].distance;
                const scale = i >= floorStart ? floorScale : i >= peakEnd ? releaseScale : i >= peakStart ? peakScale : 1;
                prepend(distance * scale, rateAt(i), points[i].row[brake]);
            }
            return profile.reverse();
        };
        // Adjacent corners may not leave enough approach for the full extension.
        // Limit it before it would lower the preceding minimum or create a speed jump.
        let approach = start;
        while (approach > 0 && points[approach - 1].energy <= points[approach].energy) approach--;
        while (approach < start && points[approach + 1].energy === points[approach].energy) approach++;
        let profile = build(caution);
        if (!profile || brakeAt(profile, points[approach].distance).energy < points[approach].energy - 1e-8) {
            let low = 0, high = caution;
            for (let i = 0; i < 40; i++) {
                const mid = (low + high) / 2;
                const candidate = build(mid);
                if (candidate && brakeAt(candidate, points[approach].distance).energy >= points[approach].energy) low = mid;
                else high = mid;
            }
            profile = build(low)!;
        }
        // Leave the next braking input's first sample outside this sequence's release.
        const releaseEnd = exit > bottom && exit + 1 < points.length && points[exit + 1].energy < points[exit].energy ? exit - 1 : exit;
        zones.push({ points: profile, start: Math.max(profile[0].distance, points[approach].distance),
            bottom: originalEnd, exit: points[releaseEnd].distance });
        start = exit - 1;
    }
    if (!zones.length) return copy;
    const cyclicZones = zones.flatMap((zone) => [-1, 0, 1].map((offset) => ({
        points: zone.points.map((point) => ({ ...point, distance: point.distance + offset * length })),
        start: zone.start + offset * length,
        bottom: zone.bottom + offset * length, exit: zone.exit + offset * length,
    })));
    const adjust = (row: number[], position: number) => {
        let energy = (row[speed] / 3.6) ** 2, pressure: number | undefined;
        cyclicZones.forEach((zone) => {
            if (position < zone.start || position > zone.exit) return;
            const target = brakeAt(zone.points, position);
            // Join the earlier braking curve where the preceding acceleration reaches it.
            pressure = target.energy <= energy + 1e-8 ? target.brake : 0;
            if (position <= zone.bottom) energy = Math.min(energy, target.energy);
        });
        row[speed] = Math.sqrt(Math.max(0, energy)) * 3.6;
        if (adjustBrake && pressure !== undefined) row[brake] = pressure;
    };
    // Replace targets across the entire lap. Keep positions, geometry and all
    // other channels aligned with the original samples; never append profile knots.
    copy.y.forEach((row, i) => adjust(row, wrap(distance[i], length)));
    return copy;
}
