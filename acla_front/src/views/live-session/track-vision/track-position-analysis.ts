import type { CornerPosition, TrackVisionAnalysis, TrackVisionFrame } from './track-vision-types';
import { evaluateRoad, roadCurvature } from './road-polynomial';
import { createSemanticScene } from './semantic-scene';
import { reconstructTrack } from './track-reconstruction';

// Preserve the public imports used by existing scene consumers.
export { reconstructTrack } from './track-reconstruction';
export { VISION_CONFIDENCE } from './semantic-scene';

/** Positions use the depth-derived local scene and the observed extent of its road fit. */
export function analyzeTrackPositions(vision: TrackVisionFrame | null, reconstruction = reconstructTrack(vision)): TrackVisionAnalysis {
    const geometry = reconstruction?.geometry;
    if (!vision || !geometry || !reconstruction) return {};
    const scene = createSemanticScene(vision);
    if (!scene) return {};
    const curvature = geometry.curvaturePerM;
    const leftCurvature = roadCurvature(geometry.left, geometry.referenceY);
    const rightCurvature = roadCurvature(geometry.right, geometry.referenceY);
    const consistentEdges = leftCurvature * Math.sign(curvature) > -0.0015 && rightCurvature * Math.sign(curvature) > -0.0015;
    const cornerDirection = Math.abs(curvature) >= 0.0015 && consistentEdges
        ? curvature < 0 ? 'left' : 'right' : undefined;
    const position = (x: number, y: number): CornerPosition | undefined => {
        const left = evaluateRoad(geometry.left, y), right = evaluateRoad(geometry.right, y);
        if (!cornerDirection || x < left || x > right) return undefined;
        const across = (x - left) / (right - left);
        const inside = cornerDirection === 'left' ? across : 1 - across;
        return inside < 0.4 ? 'inside' : inside > 0.6 ? 'outside' : 'middle';
    };
    const result: TrackVisionAnalysis = {};
    if (cornerDirection) {
        result.cornerDirection = cornerDirection;
        result.playerPosition = position(0, geometry.referenceY);
    }
    if (!scene.hasCarLabels) return result;
    // An invalid car depth/mask is unknown, not evidence of an empty road.
    if (reconstruction.cars.length === scene.traffic.length) result.carAhead = 0;
    const traffic = [...reconstruction.cars].sort((a, b) => a.center.y - b.center.y || Number(a.pack) - Number(b.pack));
    for (const { center: point, min, max, pack, roadSupported } of traffic) {
        const minY = Math.max(geometry.left.minY, geometry.right.minY), maxY = Math.min(geometry.left.maxY, geometry.right.maxY);
        if (!roadSupported || point.y < minY || point.y > maxY
            || min.x < evaluateRoad(geometry.left, point.y) || max.x > evaluateRoad(geometry.right, point.y)) continue;
        result.carAhead = 1;
        const opponentPosition = position(point.x, point.y);
        if (!pack && opponentPosition) result.opponentPosition = opponentPosition;
        break;
    }
    return result;
}
