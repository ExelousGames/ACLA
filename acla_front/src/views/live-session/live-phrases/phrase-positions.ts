import type { TrackBoundaryPosition, TrackGeometry, TrackVisionAnalysis } from '../track-vision/track-vision-types';
import { evaluateRoad, roadCurvature } from '../track-vision/road-polynomial';

export type CornerPosition = 'inside' | 'middle' | 'outside';

/** Live Phrases interprets the visible bend and converts boundary distances into corner positions. */
export function getPhrasePositions(vision: TrackVisionAnalysis, geometry?: TrackGeometry | null): {
    playerPosition?: CornerPosition; opponentPosition?: CornerPosition;
} {
    if (!geometry) return {};
    const curvature = geometry.curvaturePerM;
    const leftCurvature = roadCurvature(geometry.left, geometry.referenceY);
    const rightCurvature = roadCurvature(geometry.right, geometry.referenceY);
    if (Math.abs(curvature) < 0.0015 || !Number.isFinite(curvature)
        || leftCurvature * Math.sign(curvature) <= -0.0015
        || rightCurvature * Math.sign(curvature) <= -0.0015) return {};
    const position = (boundaries?: TrackBoundaryPosition): CornerPosition | undefined => {
        if (!boundaries) return undefined;
        const { leftBoundaryDistanceM: left, rightBoundaryDistanceM: right } = boundaries;
        const width = left + right;
        if (!Number.isFinite(width) || left < 0 || right < 0 || width <= 0) return undefined;
        const inside = (curvature < 0 ? left : right) / width;
        return inside < 0.4 ? 'inside' : inside > 0.6 ? 'outside' : 'middle';
    };
    const opponent = vision.opponents?.filter((car) => car.longitudinalOffsetM > 0)
        .sort((a, b) => a.longitudinalOffsetM - b.longitudinalOffsetM)[0];
    let opponentPosition: CornerPosition | undefined;
    if (opponent && opponent.longitudinalOffsetM >= Math.max(geometry.left.minY, geometry.right.minY)
        && opponent.longitudinalOffsetM <= Math.min(geometry.left.maxY, geometry.right.maxY)) {
        opponentPosition = position({
            leftBoundaryDistanceM: opponent.lateralOffsetM - evaluateRoad(geometry.left, opponent.longitudinalOffsetM),
            rightBoundaryDistanceM: evaluateRoad(geometry.right, opponent.longitudinalOffsetM) - opponent.lateralOffsetM,
        });
    }
    return { playerPosition: position(vision.driverPosition), opponentPosition };
}
