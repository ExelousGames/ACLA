import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapReferenceLine } from '../live-map-reference';

export const REFERENCE_DRIVING_AGGRESSION = 0;
export const MAX_DRIVING_CAUTION = 50;
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

/** Phrase-owned data. Added braking samples interpolate the original controls and geometry. */
export function copyPhraseReference(lap: TopLapReferenceEntry | null,
    aggression = REFERENCE_DRIVING_AGGRESSION): TopLapReferenceEntry | null {
    if (!lap) return null;
    const copy = { ...lap, x: [...lap.x], y: lap.y.map((row) => [...row]), target_features: [...lap.target_features] };
    const caution = normalizeDrivingAggression(aggression) / 100;
    if (!caution) return copy;
    const speed = lap.target_features.indexOf('expert_optimal_speed');
    if (speed < 0 || !lap.y.every((row) => Number.isFinite(row[speed]) && row[speed] >= 0)) return copy;
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
    // Three laps keep complete braking and acceleration sections across start/finish.
    // Collapse zero-distance samples before measuring local deceleration.
    const points: { distance: number; position: number; row: number[]; value: number }[] = [];
    [-1, 0, 1].forEach((offset) => lap.y.forEach((row, i) => {
        const at = distance[i] + offset * length;
        if (points.length && at === points[points.length - 1].distance) points.pop();
        points.push({ distance: at, position: lap.x[i] + offset, row, value: row[speed] });
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
    const zones: { start: number; bottom: number; end: number; stretch: number; minimum: number; loweredMinimum: number; peak: number }[] = [];
    const knots = new Set(distance);
    const rateScale = 1 - caution;
    for (let start = 0; start + 1 < points.length; start++) {
        if (points[start + 1].value >= points[start].value) continue;
        let bottom = start + 1, end = bottom;
        // Retain both edges of a flat minimum so acceleration still starts at the reference position.
        while (end + 1 < points.length && points[end + 1].value <= points[end].value) {
            end++;
            if (points[end].value < points[end - 1].value) bottom = end;
        }
        const exit = end;
        while (end + 1 < points.length && points[end + 1].value >= points[end].value) end++;
        if (end === exit) break;
        const minimum = points[bottom].value, loweredMinimum = minimum * (1 - caution);
        const brakeEnd = points[bottom].distance;
        // a = (v_start² - v_end²) / (2 ds). Scale each reference rate, then solve
        // the distance needed to dissipate the energy down to the lower minimum.
        const peak = points[start].value;
        const stretch = (peak ** 2 - loweredMinimum ** 2) / (rateScale * (peak ** 2 - minimum ** 2));
        const brakeStart = brakeEnd - (brakeEnd - points[start].distance) * stretch;
        if (points[end].distance >= 0 && brakeStart < length) {
            zones.push({ start: brakeStart, bottom: brakeEnd, end: points[end].distance, stretch,
                minimum, loweredMinimum, peak: points[end].value });
            // Keep every mapped rate boundary, including the earlier brake onset.
            // Otherwise interpolation could average a strong interval into a weak one.
            for (let i = start; i <= bottom; i++) {
                const position = brakeEnd - (brakeEnd - points[i].distance) * stretch;
                if (position <= 0 || position >= length || brakeEnd - position > length) continue;
                const { a, b } = at(position);
                if (position - a.distance > 1e-7 && b.distance - position > 1e-7) knots.add(position);
            }
        }
        start = end - 1;
    }
    if (!zones.length) return copy;
    const samples = Array.from(knots).sort((a, b) => a - b);
    const energies = samples.map((position) => {
        const { a, b, fraction } = at(position);
        let value = b.value < a.value
            ? Math.sqrt(a.value ** 2 + (b.value ** 2 - a.value ** 2) * fraction)
            : a.value + (b.value - a.value) * fraction;
        zones.forEach((zone) => {
            if (position < zone.bottom || position > zone.end) return;
            value = Math.min(value, zone.loweredMinimum + (value - zone.minimum)
                / (zone.peak - zone.minimum) * (zone.peak - zone.loweredMinimum));
        });
        return (value / 3.6) ** 2;
    });
    const caps = samples.slice(0, -1).map((position, i) => {
        const midpoint = (position + samples[i + 1]) / 2;
        let cap = Infinity;
        zones.forEach((zone) => {
            // Only the next occurrence of each corner constrains this lap position.
            if (midpoint < zone.start || midpoint >= zone.bottom || zone.bottom - midpoint > length) return;
            // Match normalized progress within this braking zone, using physical distance.
            const source = zone.bottom - (zone.bottom - midpoint) / zone.stretch;
            const { a, b } = at(source);
            const deceleration = (a.value ** 2 - b.value ** 2) / (2 * 3.6 ** 2 * (b.distance - a.distance));
            cap = Math.min(cap, rateScale * Math.max(0, deceleration));
        });
        if (!Number.isFinite(cap)) {
            const { a, b } = at(midpoint);
            cap = Math.max(0, (a.value ** 2 - b.value ** 2) / (2 * 3.6 ** 2 * (b.distance - a.distance)));
        }
        return 2 * cap * (samples[i + 1] - position);
    });
    // Backward reachability also resolves overlapping corners. Two cyclic passes
    // propagate every lower target across start/finish without a speed discontinuity.
    const count = samples.length - 1;
    for (let pass = 0; pass < 2; pass++) {
        for (let i = count - 1; i >= 0; i--) {
            energies[i] = Math.min(energies[i], energies[(i + 1) % count] + caps[i]);
        }
    }
    energies[count] = energies[0];
    const speeds = new Map(samples.map((position, i) => [position, Math.sqrt(energies[i]) * 3.6]));
    copy.y.forEach((row, i) => { row[speed] = speeds.get(distance[i])!; });
    samples.forEach((position) => {
        const { a, b, fraction } = at(position);
        if (position <= 0 || position >= length || position === a.distance || position === b.distance) return;
        const row = a.row.map((value, i) => value + (b.row[i] - value) * fraction);
        row[speed] = speeds.get(position)!;
        const normalized = a.position + (b.position - a.position) * fraction;
        copy.x.push(normalized >= 1 ? normalized - 1 : normalized);
        copy.y.push(row);
    });
    const ordered = copy.x.map((position, i) => ({ position, row: copy.y[i] })).sort((a, b) => a.position - b.position);
    copy.x = ordered.map(({ position }) => position);
    copy.y = ordered.map(({ row }) => row);
    return copy;
}
