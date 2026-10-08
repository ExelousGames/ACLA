import type { BirdsEyeScene } from '../track-vision/birds-eye-scene';
import type { GroundPoint } from '../track-vision/track-vision-types';
import { fitRoadPolynomial, roadCurvature } from '../track-vision/road-polynomial';

export type CornerDirection = 'left' | 'right';
export type TrackPosition = 'left' | 'middle' | 'right';

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
    carAhead?: 0 | 1; playerCorner?: CornerDirection; opponentCorner?: CornerDirection;
    playerPosition?: TrackPosition; opponentPosition?: TrackPosition;
    opponentDistanceM?: number; opponentLateralOffsetM?: number;
} {
    if (!scene) return {};
    const boundary = (points: GroundPoint[]) => ({ points, curve: fitRoadPolynomial(points),
        minY: Math.min(...points.map(({ y }) => y)), maxY: Math.max(...points.map(({ y }) => y)) });
    const left = scene.leftBoundary.map(boundary), right = scene.rightBoundary.map(boundary);
    const roads = left.flatMap((left) => right.flatMap((right) => {
        const minY = Math.max(0, left.minY, right.minY), maxY = Math.min(left.maxY, right.maxY);
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
    const opponent = onTrack.find(({ car }) => !car.pack);
    if (opponent) {
        const { x, y } = opponent.car.position;
        result.opponentDistanceM = Math.hypot(x, y);
        result.opponentLateralOffsetM = x;
    }
    const position = (x: number, slice: ReturnType<typeof sliceAt>): TrackPosition | undefined => {
        if (!slice || x < slice.leftX || x > slice.rightX) return undefined;
        const across = (x - slice.leftX) / (slice.rightX - slice.leftX);
        return across < 0.4 ? 'left' : across > 0.6 ? 'right' : 'middle';
    };
    const corner = (y: number, slice: ReturnType<typeof sliceAt>): CornerDirection | undefined => {
        if (!slice?.road.left.curve || !slice.road.right.curve) return undefined;
        const leftCurvature = roadCurvature(slice.road.left.curve, y);
        const rightCurvature = roadCurvature(slice.road.right.curve, y);
        const curvature = (leftCurvature + rightCurvature) / 2;
        if (!Number.isFinite(curvature) || Math.abs(curvature) < 0.0015
            || leftCurvature * Math.sign(curvature) <= -0.0015
            || rightCurvature * Math.sign(curvature) <= -0.0015) return undefined;
        return curvature < 0 ? 'left' : 'right';
    };
    // Prefer the origin when recent motion-aligned observations support it; never extrapolate edges.
    if (reference) {
        result.playerPosition = position(0, reference.slice);
        if (result.playerPosition) result.playerCorner = corner(reference.y, reference.slice);
    }
    if (opponent) {
        result.opponentPosition = position(opponent.car.position.x, opponent.slice);
        result.opponentCorner = corner(opponent.car.position.y, opponent.slice);
    }
    return result;
}
