import type { TrackVisionAnalysis, TrackVisionFrame } from './track-vision-types';
import { evaluateRoad } from './road-polynomial';
import { createSemanticScene } from './semantic-scene';
import { reconstructTrack } from './track-reconstruction';

// Preserve the public imports used by existing scene consumers.
export { reconstructTrack } from './track-reconstruction';
export { VISION_CONFIDENCE } from './semantic-scene';

/** Positions use the depth-derived local scene and the observed extent of its road fit. */
export function analyzeTrackPositions(vision: TrackVisionFrame | null, reconstruction = reconstructTrack(vision)): TrackVisionAnalysis {
    const geometry = reconstruction?.geometry;
    if (!vision || !reconstruction) return {};
    const scene = createSemanticScene(vision);
    if (!scene) return {};
    const result: TrackVisionAnalysis = {};
    if (geometry) {
        const left = -evaluateRoad(geometry.left, geometry.referenceY), right = evaluateRoad(geometry.right, geometry.referenceY);
        if (left >= 0 && right >= 0) result.driverPosition = {
            leftBoundaryDistanceM: left, rightBoundaryDistanceM: right, referenceDistanceM: geometry.referenceY,
        };
    }
    if (!scene.hasCarLabels) return result;
    // An invalid car depth/mask is unknown, not evidence of an empty road.
    if (geometry && reconstruction.cars.length === scene.traffic.length) {
        result.carAhead = 0;
        result.opponents = [];
    }
    const traffic = [...reconstruction.cars].sort((a, b) => a.center.y - b.center.y || Number(a.pack) - Number(b.pack));
    for (const { center: point, pack, roadSupported } of traffic) {
        if (!roadSupported) continue;
        if (point.y > 0) result.carAhead = 1;
        if (!pack) (result.opponents ??= []).push({
            lateralOffsetM: point.x, longitudinalOffsetM: point.y,
        });
    }
    return result;
}
