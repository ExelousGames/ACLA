import type { BirdsEyeScene } from '../track-vision/birds-eye-scene';
import type { GroundPoint } from '../track-vision/track-vision-types';
import { fitRoadPolynomial, roadCurvature } from '../track-vision/road-polynomial';

export type CornerPosition = 'inside' | 'middle' | 'outside';

/** Interpolate within a displayed boundary section, never across gaps or beyond its ends. */
function boundaryX(points: GroundPoint[], y: number): number | undefined {
    for (let index = 1; index < points.length; index++) {
        const a = points[index - 1], b = points[index];
        if (y < Math.min(a.y, b.y) || y > Math.max(a.y, b.y) || a.y === b.y) continue;
        return a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y);
    }
    return undefined;
}

/** Interpret the same calibrated flat-road scene shown in Track Vision's BEV. */
export function getPhrasePositions(scene?: BirdsEyeScene | null): {
    carAhead?: 0 | 1; playerPosition?: CornerPosition; opponentPosition?: CornerPosition;
} {
    if (!scene) return {};
    const boundary = (points: GroundPoint[]) => ({ points, curve: fitRoadPolynomial(points),
        minY: Math.min(...points.map(({ y }) => y)), maxY: Math.max(...points.map(({ y }) => y)) });
    const left = scene.leftBoundary.map(boundary), right = scene.rightBoundary.map(boundary);
    const roads = left.flatMap((left) => right.flatMap((right) => {
        const minY = Math.max(left.minY, right.minY), maxY = Math.min(left.maxY, right.maxY);
        return minY < maxY ? [{ left, right, minY, maxY }] : [];
    })).sort((a, b) => a.minY - b.minY);
    const sliceAt = (y: number) => {
        for (const road of roads) {
            if (y < road.minY || y > road.maxY) continue;
            const leftX = boundaryX(road.left.points, y), rightX = boundaryX(road.right.points, y);
            if (leftX !== undefined && rightX !== undefined && rightX > leftX) return { leftX, rightX, road };
        }
        return undefined;
    };
    const reference = roads.map(({ minY }) => ({ y: minY, slice: sliceAt(minY) })).find(({ slice }) => slice);
    const traffic = scene.cars.filter(({ position }) => position.y > 0)
        .map((car) => ({ car, slice: sliceAt(car.position.y) }));
    const onTrack = traffic.filter(({ car, slice }) => slice && car.position.x >= slice.leftX && car.position.x <= slice.rightX)
        .sort((a, b) => a.car.position.y - b.car.position.y);
    const carAhead = onTrack.length ? 1 : reference && !scene.unplacedCars && traffic.every(({ slice }) => slice) ? 0 : undefined;
    const result: ReturnType<typeof getPhrasePositions> = { carAhead };
    if (!reference?.slice) return result;
    const { road } = reference.slice;
    if (!road.left.curve || !road.right.curve) return result;
    const leftCurvature = roadCurvature(road.left.curve, reference.y);
    const rightCurvature = roadCurvature(road.right.curve, reference.y);
    const curvature = (leftCurvature + rightCurvature) / 2;
    if (!Number.isFinite(curvature) || Math.abs(curvature) < 0.0015
        || leftCurvature * Math.sign(curvature) <= -0.0015
        || rightCurvature * Math.sign(curvature) <= -0.0015) return result;
    const position = (x: number, slice: ReturnType<typeof sliceAt>): CornerPosition | undefined => {
        if (!slice || x < slice.leftX || x > slice.rightX) return undefined;
        const inside = (curvature < 0 ? x - slice.leftX : slice.rightX - x) / (slice.rightX - slice.leftX);
        return inside < 0.4 ? 'inside' : inside > 0.6 ? 'outside' : 'middle';
    };
    // The origin is compared with the nearest visible road slice, without extrapolating unseen edges.
    result.playerPosition = position(0, reference.slice);
    const opponent = onTrack.find(({ car }) => !car.pack);
    if (opponent) result.opponentPosition = position(opponent.car.position.x, opponent.slice);
    return result;
}
