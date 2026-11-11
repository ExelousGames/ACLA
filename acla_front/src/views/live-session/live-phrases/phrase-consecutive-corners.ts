import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapPosition } from '../live-map-data';
import type { PhraseMapContext } from './phrase-map-context';

type TrackSide = 'left' | 'middle' | 'right';
export type ConsecutiveCornerSetup = `${TrackSide}-for-${'left' | 'right'}`;
type Point = { normalized_position: number; x: number; y: number; z: number };
const normalized = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1;
const wrap = (position: number) => ((position % 1) + 1) % 1;
const prepareLine = (source: readonly Point[] = []): Point[] => {
    if (source.length < 2 || !source.every((point) => normalized(point.normalized_position)
        && [point.x, point.y, point.z].every(Number.isFinite))) return [];
    return source.map(({ normalized_position, x, y, z }) => ({ normalized_position, x, y, z }))
        .sort((a, b) => a.normalized_position - b.normalized_position)
        .filter((point, index, all) => !index || point.normalized_position !== all[index - 1].normalized_position);
};

/** Locate the reference at the next mapped entry between the recorded track edges. */
export function createConsecutiveCornerPositioning(map: CircuitMapDto | null, reference: TopLapReferenceEntry | null):
    (corner: PhraseMapContext | undefined) => ConsecutiveCornerSetup | undefined {
    if (!map || !reference || map.game !== reference.game || reference.x.length < 2
        || reference.x.length !== reference.y.length
        || !reference.x.every((position, index) => normalized(position) && (!index || position > reference.x[index - 1]))) {
        return () => undefined;
    }
    const indices = ['x', 'y', 'z'].map((axis) => reference.target_features.indexOf(`expert_optimal_player_pos_${axis}`));
    if (indices.some((index) => index < 0)) return () => undefined;
    const line = prepareLine(reference.y.map((row, index) => ({ normalized_position: reference.x[index],
        x: row[indices[0]], y: row[indices[1]], z: row[indices[2]] })));
    const middle = prepareLine(map.samples.middle_line);
    const left = prepareLine(map.samples.left_boundary), right = prepareLine(map.samples.right_boundary);
    if ([line, middle, left, right].some((points) => points.length < 2)) return () => undefined;
    const ground = map.game === 'acc' ? 'y' : 'z';
    const targets = new Map<number, TrackSide | undefined>();
    const sideAt = (position: number): TrackSide | undefined => {
        const target = getLiveMapPosition(line, position), center = getLiveMapPosition(middle, position);
        const a = getLiveMapPosition(left, position), b = getLiveMapPosition(right, position);
        const before = getLiveMapPosition(middle, wrap(position - 0.0001));
        const after = getLiveMapPosition(middle, wrap(position + 0.0001));
        if (!target || !center || !a || !b || !before || !after) return undefined;
        const dx = b.x - a.x, dy = b[ground] - a[ground], widthSquared = dx * dx + dy * dy;
        const tx = after.x - before.x, ty = after[ground] - before[ground], tangentSquared = tx * tx + ty * ty;
        if (widthSquared <= 1e-12 || tangentSquared <= 1e-12) return undefined;
        // Edges sampled at incompatible track positions do not define a usable cross-section.
        if (Math.abs(dx * tx + dy * ty) > 0.5 * Math.sqrt(widthSquared * tangentSquared)) return undefined;
        const fractionAt = (point: { x: number; y: number; z: number }) => {
            const x = point.x - a.x, y = point[ground] - a[ground];
            const fraction = (x * dx + y * dy) / widthSquared;
            const longitudinal = (x - fraction * dx) ** 2 + (y - fraction * dy) ** 2;
            return fraction >= -1e-9 && fraction <= 1 + 1e-9 && longitudinal <= widthSquared / 4 ? fraction : undefined;
        };
        const fraction = fractionAt(target);
        if (fraction === undefined || fractionAt(center) === undefined) return undefined;
        return fraction < 1 / 3 ? 'left' : fraction > 2 / 3 ? 'right' : 'middle';
    };
    return (corner) => {
        const start = corner?.nextCornerStartPosition, direction = corner?.nextCornerDirection;
        if (!corner?.sequenceId || corner.sequenceRemaining === undefined || corner.sequenceRemaining < 1
            || start === undefined || !normalized(start)
            || (direction !== 'left' && direction !== 'right')) return undefined;
        if (!targets.has(start)) targets.set(start, sideAt(start));
        const side = targets.get(start);
        return side === undefined ? undefined : `${side}-for-${direction}`;
    };
}
