import type { CircuitMapBinSample, CircuitMapCenterlineSegment } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapPosition } from '../live-map-data';

export interface PhraseCornerGeometry {
    shape: 'bend' | 'hairpin' | 's-bend' | 'tightening' | 'opening';
    /** Sign is only for comparing turns, not for naming left/right across games. */
    turnSign?: -1 | 1;
    headingChangeDeg: number;
    lengthM: number;
}

export const lapDistance = (from: number, to: number) => (to - from + 1) % 1;
export const segmentSpan = (segment: CircuitMapCenterlineSegment) => segment.end_position >= segment.start_position
    ? segment.end_position - segment.start_position : 1 - segment.start_position + segment.end_position;

/** Clip the actual centerline, then sample at equal distances to avoid capture-density bias. */
export function getPhraseCornerGeometry(
    line: CircuitMapBinSample[], segment: CircuitMapCenterlineSegment,
): PhraseCornerGeometry | undefined {
    const span = segmentSpan(segment);
    const start = getLiveMapPosition(line, segment.start_position);
    const end = getLiveMapPosition(line, segment.end_position);
    if (!start || !end || span <= 0) return undefined;
    const interior = line.filter((point) => {
        const offset = lapDistance(segment.start_position, point.normalized_position);
        return offset > 0 && offset < span;
    }).sort((a, b) => lapDistance(segment.start_position, a.normalized_position)
        - lapDistance(segment.start_position, b.normalized_position));
    const points: NonNullable<typeof start>[] = [];
    for (const point of [start, ...interior, end]) {
        const previous = points[points.length - 1];
        if (!previous || Math.hypot(point.x - previous.x, point.z - previous.z) > 0.01) points.push(point);
    }
    if (points.length < 3) return undefined;
    const distances = [0];
    for (let index = 1; index < points.length; index++) {
        distances.push(distances[index - 1] + Math.hypot(points[index].x - points[index - 1].x,
            points[index].z - points[index - 1].z));
    }
    const lengthM = distances[distances.length - 1];
    const steps = Math.min(64, Math.floor(lengthM / 5));
    if (steps < 2) return undefined;
    let cursor = 1;
    const sampled = Array.from({ length: steps + 1 }, (_, index) => {
        const target = lengthM * index / steps;
        while (cursor < points.length - 1 && distances[cursor] < target) cursor++;
        const fraction = (target - distances[cursor - 1]) / (distances[cursor] - distances[cursor - 1]);
        return {
            x: points[cursor - 1].x + (points[cursor].x - points[cursor - 1].x) * fraction,
            z: points[cursor - 1].z + (points[cursor].z - points[cursor - 1].z) * fraction,
        };
    });
    let positive = 0, negative = 0;
    const bendBins = [0, 0, 0, 0, 0, 0];
    for (let index = 1; index < sampled.length - 1; index++) {
        const a = sampled[index - 1], b = sampled[index], c = sampled[index + 1];
        const ux = b.x - a.x, uz = b.z - a.z, vx = c.x - b.x, vz = c.z - b.z;
        const angle = Math.atan2(ux * vz - uz * vx, ux * vx + uz * vz);
        positive += Math.max(0, angle);
        negative += Math.max(0, -angle);
        bendBins[Math.min(5, Math.floor(index / steps * 6))] += Math.abs(angle);
    }
    const total = positive + negative;
    if (total < 0.1) return undefined;
    const headingChangeDeg = (positive - negative) * 180 / Math.PI;
    // A small reversal from capture noise must not turn a single bend into an S-bend.
    const reversed = Math.min(positive, negative) >= Math.max(Math.PI / 12, total * 0.2);
    let shape: PhraseCornerGeometry['shape'] = reversed ? 's-bend'
        : Math.abs(headingChangeDeg) >= 135 - 1e-6 ? 'hairpin' : 'bend';
    const entryBend = bendBins.slice(0, 3).reduce((sum, angle) => sum + angle, 0);
    const exitBend = bendBins.slice(3).reduce((sum, angle) => sum + angle, 0);
    // Require curvature spread along the corner, not one sharp vertex in a sparse map.
    if (shape === 'bend' && bendBins.filter((angle) => angle >= 0.02).length >= 4
        && Math.min(entryBend, exitBend) >= 0.12) {
        if (exitBend > entryBend * 1.6) shape = 'tightening';
        else if (entryBend > exitBend * 1.6) shape = 'opening';
    }
    return { shape, turnSign: reversed ? undefined : headingChangeDeg > 0 ? 1 : -1, headingChangeDeg, lengthM };
}
