import type { GroundPoint, TrackBoundaryPoint, TrackGeometry, TrackVisionFrame } from './track-vision-types';
import { createCameraProjection } from './camera-projection';
import { createDepthProjection } from './depth-projection';
import { evaluateRoad, fitRoadPolynomial, roadCurvature } from './road-polynomial';
import type { SemanticScene } from './semantic-scene';

export function reconstructBoundaries(vision: TrackVisionFrame, scene: SemanticScene,
    lift: NonNullable<ReturnType<typeof createDepthProjection>>) {
    const leftBoundary: GroundPoint[] = [], rightBoundary: GroundPoint[] = [], centers: GroundPoint[] = [];
    let anchor = 0.5, lastLeftRow = scene.height, lastRightRow = scene.height;
    const continuous = (a: GroundPoint, b: GroundPoint) => {
        const dy = b.y - a.y;
        return dy > 0 && dy <= 15 && Math.abs(b.x - a.x) <= 1 + dy * 0.7;
    };
    for (let row = scene.height - 1; row >= 0; row--) {
        const { v } = scene.sourcePixel(0, row + 0.5);
        if (v <= 0 || v >= 1) continue;
        const candidates: Array<{ left: GroundPoint | null; right: GroundPoint | null; center: number }> = [];
        const at = (column: number) => scene.sample(scene.sourcePixel(column + 0.5, row + 0.5).u, v);
        const edge = (column: number, outside: number, boundary: GroundPoint[], lastRow: number) => {
            // Validate each visible edge independently, including its own depth and continuity.
            if (outside < 0 || outside >= scene.width || at(column) !== 1 || at(outside) === 255 || at(outside) === 2) return null;
            // Reject the cockpit contour itself; trimming the corridor would create another false edge.
            if (scene.nearCarInterior(column, row)) return null;
            const point = lift(scene.sourcePixel(column + 0.5, row + 0.5).u, v, scene.visibleBoundaryRoad);
            if (!point) return null;
            const previous = boundary[boundary.length - 1];
            // Resume beyond gaps instead of ending the scan or comparing across missing rows.
            if (previous && lastRow - row <= 3 && !continuous(previous, point)) return null;
            return point;
        };
        for (let column = 0; column < scene.width; column++) {
            if (at(column) !== 1) continue;
            const start = column;
            while (column + 1 < scene.width && [1, 2].includes(at(column + 1))) column++;
            const end = column;
            if (end - start < 3) continue;
            const lu = scene.sourcePixel(start + 0.5, row + 0.5).u;
            const ru = scene.sourcePixel(end + 0.5, row + 0.5).u;
            const left = edge(start, start - 1, leftBoundary, lastLeftRow);
            const right = edge(end, end + 1, rightBoundary, lastRightRow);
            if (!left && !right) continue;
            candidates.push({ left, right, center: (lu + ru) / 2 });
        }
        const previousCenter = anchor;
        const best = candidates.sort((a, b) => Math.abs(a.center - previousCenter) - Math.abs(b.center - previousCenter))[0];
        if (!best) continue;
        if (best.left) {
            leftBoundary.push(best.left);
            lastLeftRow = row;
        }
        if (best.right) {
            rightBoundary.push(best.right);
            lastRightRow = row;
        }
        if (best.left && best.right) centers.push({ x: (best.left.x + best.right.x) / 2,
            y: (best.left.y + best.right.y) / 2, z: (best.left.z + best.right.z) / 2 });
        anchor = best.center;
    }
    // Fit only observations: interpolated samples must not inflate fit support or confidence.
    const geometry = fitTrackBoundaries(leftBoundary, rightBoundary, centers);
    const camera = createCameraProjection(vision.calibration!);
    const firstPixel = scene.sourcePixel(0.5, 0.5);
    const pixelWidth = scene.sourcePixel(1.5, 0.5).u - firstPixel.u;
    const inferOccluded = (observed: GroundPoint[]): TrackBoundaryPoint[] => {
        const boundary: TrackBoundaryPoint[] = [];
        for (let i = 0; i < observed.length; i++) {
            const near = observed[i - 1], far = observed[i];
            if (near && continuous(near, far) && Math.abs(far.z - near.z) <= 1 + (far.y - near.y) * 0.3) {
                const a = camera.localToImage(near)!, b = camera.localToImage(far)!;
                const rows = Math.round((a.v - b.v) / scene.pixelHeight);
                const nearDepth = camera.opticalDepth(near), farDepth = camera.opticalDepth(far);
                const estimates: TrackBoundaryPoint[] = [];
                for (let step = 1; step < rows; step++) {
                    const fraction = step / rows;
                    const u = a.u + fraction * (b.u - a.u), v = a.v + fraction * (b.v - a.v);
                    const column = Math.round((u - firstPixel.u) / pixelWidth);
                    const row = Math.round((v - firstPixel.v) / scene.pixelHeight);
                    // Require traffic across the entire gap, and never bridge cockpit/obstacles.
                    if (scene.sample(u, v) !== 2 || scene.nearCarInterior(column, row)) {
                        estimates.length = 0;
                        break;
                    }
                    // Perspective-correct interpolation on the visible endpoints' 3D segment.
                    // Linear image-row weights alone would put the road at the wrong distance.
                    const t = fraction * nearDepth / ((1 - fraction) * farDepth + fraction * nearDepth);
                    estimates.push({ x: near.x + t * (far.x - near.x), y: near.y + t * (far.y - near.y),
                        z: near.z + t * (far.z - near.z), estimated: true });
                }
                boundary.push(...estimates);
            }
            boundary.push(far);
        }
        return boundary;
    };
    const inferredLeft = inferOccluded(leftBoundary), inferredRight = inferOccluded(rightBoundary);
    return { leftBoundary: inferredLeft, rightBoundary: inferredRight,
        geometry: geometry && { ...geometry, leftBoundary: inferredLeft, rightBoundary: inferredRight } };
}

function fitTrackBoundaries(leftBoundary: GroundPoint[], rightBoundary: GroundPoint[], centers: GroundPoint[]): TrackGeometry | null {
    const left = fitRoadPolynomial(leftBoundary), right = fitRoadPolynomial(rightBoundary);
    if (!left || !right) return null;
    const referenceY = Math.max(left.minY, right.minY), maxY = Math.min(left.maxY, right.maxY);
    if (maxY - referenceY < 10) return null;
    const center = { minY: referenceY, maxY, rmseM: (left.rmseM + right.rmseM) / 2,
        coefficients: left.coefficients.map((value, i) => (value + right.coefficients[i]) / 2) as [number, number, number] };
    // A single quadratic cannot describe a chicane. Reject opposing local bends
    // when each subsection has enough metric support for its own fit.
    const nearFit = fitRoadPolynomial(centers.slice(0, Math.ceil(centers.length * 0.65)));
    const farFit = fitRoadPolynomial(centers.slice(Math.floor(centers.length * 0.35)));
    if (nearFit && farFit) {
        const nearBend = roadCurvature(nearFit, nearFit.maxY), farBend = roadCurvature(farFit, farFit.minY);
        if (nearBend * farBend < 0 && Math.min(Math.abs(nearBend), Math.abs(farBend)) > 0.003) return null;
    }
    const widths = [...leftBoundary, ...rightBoundary].filter(({ y }) => y >= referenceY && y <= maxY)
        .map(({ y }) => evaluateRoad(right, y) - evaluateRoad(left, y));
    if (Math.min(...widths) < 2.5 || Math.max(...widths) > 22 || Math.max(...widths) / Math.min(...widths) > 1.8
        || Math.max(left.rmseM, right.rmseM) > Math.min(0.65, Math.min(...widths) * 0.08)) return null;
    return { leftBoundary, rightBoundary, left, right, center, referenceY,
        trackWidthM: evaluateRoad(right, referenceY) - evaluateRoad(left, referenceY),
        lateralOffsetM: -evaluateRoad(center, referenceY),
        headingDeg: Math.atan(center.coefficients[1] + 2 * center.coefficients[2] * referenceY) * 180 / Math.PI,
        curvaturePerM: roadCurvature(center, referenceY) };
}

