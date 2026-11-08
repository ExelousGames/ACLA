import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';
import { getLiveMapPosition } from '../live-map-data';
import { getLiveMapReferenceLine } from '../live-map-reference';
import { lapDistance } from './phrase-corner-geometry';
import type { PhraseMapContext } from './phrase-map-context';

export interface SteeringCornerPoint { x: number; y: number }
export interface SteeringCornerOverlaySnapshot {
    title: string;
    circuitName: string;
    referenceLine: SteeringCornerPoint[];
    leftBoundary: SteeringCornerPoint[];
    rightBoundary: SteeringCornerPoint[];
}

type MapPoint = { normalized_position: number; x: number; y: number; z: number };
const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;

/** Use the same corner as Steering, clipping each saved line in lap order (including start/finish). */
export function buildSteeringCornerOverlay(
    map: CircuitMapDto | null, reference: TopLapReferenceEntry | null, corner: PhraseMapContext,
): SteeringCornerOverlaySnapshot | null {
    const start = corner.cornerStartPosition, end = corner.cornerEndPosition;
    if (!map || !reference || map.game !== reference.game || !corner.sectionId
        || !normalized(start) || !normalized(end) || start === end) return null;
    const span = end >= start ? end - start : 1 - start + end;
    const clip = (source: MapPoint[]): SteeringCornerPoint[] => {
        const line = source.filter((point) => normalized(point.normalized_position)
            && [point.x, point.y, point.z].every(Number.isFinite))
            .sort((a, b) => a.normalized_position - b.normalized_position)
            .filter((point, index, all) => !index || point.normalized_position !== all[index - 1].normalized_position);
        const first = getLiveMapPosition(line, start), last = getLiveMapPosition(line, end);
        if (!first || !last) return [];
        const interior = line.filter((point) => {
            const offset = lapDistance(start, point.normalized_position);
            return offset > 1e-9 && offset < span - 1e-9;
        }).sort((a, b) => lapDistance(start, a.normalized_position) - lapDistance(start, b.normalized_position));
        // Match the ground plane and orientation of the analyst's trajectory view.
        return [first, ...interior, last].map((point) => ({ x: point.x, y: map.game === 'acc' ? -point.y : point.z }));
    };
    const referenceLine = clip(getLiveMapReferenceLine(reference)
        .map((point, index) => ({ ...point, normalized_position: reference.x[index] })));
    if (referenceLine.length < 2) return null;
    const segment = getCircuitMapCenterlineSegments(map).find(({ id }) => id === corner.sectionId);
    return {
        title: `Steering · ${segment?.name?.trim() || corner.sectionId}`,
        circuitName: map.circuit_name,
        referenceLine,
        leftBoundary: clip(map.samples.left_boundary ?? []),
        rightBoundary: clip(map.samples.right_boundary ?? []),
    };
}
