import type { CornerPosition, GroundPoint, LocalTrackScene, ReconstructedCar, SegmentResult, TrackGeometry, TrackVisionAnalysis, TrackVisionFrame } from './track-vision-types';
import { createDepthProjection } from './depth-projection';
import { evaluateRoad, fitRoadPolynomial, roadCurvature } from './road-polynomial';
import { letterbox } from './yolo-segmentation';
import { createSegmentationLayers } from './segmentation-layers';

export const VISION_CONFIDENCE = 0.65;
type Box = SegmentResult['instances'][number]['box'];

/** Sample the shared, overlapping track / traffic / excluded surface layers. */
function semanticScene(vision: TrackVisionFrame) {
    const segment = vision.detections.segment;
    if (segment?.task !== 'segment') return null;
    const layers = createSegmentationLayers(segment, VISION_CONFIDENCE);
    if (!layers) return null;
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(vision.width, vision.height, 640);
    const traffic = layers.instances.filter((item) => item.kind === 'car' || item.kind === 'car pack')
        .map((item) => ({ ...item, pack: item.kind === 'car pack', box: [
            (item.box[0] * 640 - padX) / resizedWidth, (item.box[1] * 640 - padY) / resizedHeight,
            (item.box[2] * 640 - padX) / resizedWidth, (item.box[3] * 640 - padY) / resizedHeight,
        ] as Box }))
        .filter(({ box: [left, top, right, bottom], pack }) =>
            [left, top, right, bottom].every(Number.isFinite) && left >= 0.05 && right <= 0.95
            && right - left >= 0.015 && right - left <= (pack ? 0.85 : 0.4)
            && top >= 0.2 && bottom >= 0.35 && bottom <= 0.85 && bottom - top >= 0.025);
    const inMask = (mask: Uint8Array, u: number, v: number) => {
        if (u < 0 || u >= 1 || v < 0 || v >= 1 || mask.length !== segment.width * segment.height) return false;
        const x = Math.floor((padX + u * resizedWidth) / 640 * segment.width);
        const y = Math.floor((padY + v * resizedHeight) / 640 * segment.height);
        return mask[y * segment.width + x] === 1;
    };
    const road = (u: number, v: number) => inMask(layers.trackMask, u, v) && !inMask(layers.excludedMask, u, v);
    const occluded = (u: number, v: number) => inMask(layers.trafficMask, u, v);
    return {
        traffic, road, inMask, width: segment.width, height: segment.height,
        sourcePixel(x: number, y: number) {
            return { u: (x / segment.width * 640 - padX) / resizedWidth,
                v: (y / segment.height * 640 - padY) / resizedHeight };
        },
        pixelHeight: 640 / resizedHeight / segment.height,
        hasCarLabels: layers.hasCarLabels,
        // Car depth is not road depth, even though the track continues underneath.
        visibleRoad: (u: number, v: number) => road(u, v) && !occluded(u, v),
        sample(u: number, v: number) {
            if (u < 0 || u >= 1 || v < 0 || v >= 1) return 255;
            if (inMask(layers.excludedMask, u, v)) return 3;
            if (occluded(u, v)) return 2;
            return road(u, v) ? 1 : 0;
        },
    };
}

/** Distance contours on the observed road, in the same forward meters as car centers. */
export function reconstructDistanceGrid(vision: TrackVisionFrame) {
    const lift = createDepthProjection(vision), scene = semanticScene(vision);
    if (!lift || !scene) return [];
    const supports = scene.visibleRoad;
    const startY = vision.boundaryStartDistanceM === undefined ? 0 : vision.calibration!.forwardOffsetM + vision.boundaryStartDistanceM;
    const guides = [0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 40, 60, 100, 150, 200]
        .filter((distanceM) => distanceM >= startY)
        .map((distanceM) => ({ distanceM, segments: [] as Array<[GroundPoint, GroundPoint]> }));
    // Sample in mask coordinates so letterboxing and excluded surfaces match reconstruction.
    const stride = Math.max(1, Math.ceil(Math.max(scene.width, scene.height) / 80));
    const rows: Array<Array<GroundPoint | null>> = [];
    for (let row = 0; row < scene.height; row += stride) {
        const points: Array<GroundPoint | null> = [];
        for (let column = 0; column < scene.width; column += stride) {
            const { u, v } = scene.sourcePixel(column + 0.5, row + 0.5);
            points.push(supports(u, v) ? lift(u, v, supports) : null);
        }
        rows.push(points);
    }
    const contour = (a: GroundPoint | null, b: GroundPoint | null, c: GroundPoint | null) => {
        // Missing road/depth breaks the contour; do not bridge cars, cutoffs or background.
        if (!a || !b || !c) return;
        const near = Math.min(a.y, b.y, c.y), far = Math.max(a.y, b.y, c.y);
        for (const guide of guides) {
            const y = guide.distanceM;
            if (y <= near || y > far) continue;
            const crossings: GroundPoint[] = [];
            for (const [start, end] of [[a, b], [b, c], [c, a]]) {
                if ((start.y < y && end.y >= y) || (end.y < y && start.y >= y)) {
                    const t = (y - start.y) / (end.y - start.y);
                    crossings.push({ x: start.x + t * (end.x - start.x), y,
                        z: start.z + t * (end.z - start.z) });
                }
            }
            if (crossings.length === 2) guide.segments.push([crossings[0], crossings[1]]);
        }
    };
    for (let row = 1; row < rows.length; row++) {
        for (let column = 1; column < rows[row].length; column++) {
            const a = rows[row - 1][column - 1], b = rows[row - 1][column];
            const c = rows[row][column - 1], d = rows[row][column];
            contour(a, b, c);
            contour(b, d, c);
        }
    }
    return guides.filter((guide) => guide.segments.length > 0);
}

/** Same-frame segmentation identifies surfaces; depth and camera pose locate them in 3D. */
export function reconstructTrack(vision: TrackVisionFrame | null): LocalTrackScene | null {
    if (!vision) return null;
    const lift = createDepthProjection(vision), scene = semanticScene(vision);
    if (!lift || !scene) return null;
    const startY = vision.boundaryStartDistanceM === undefined ? 0 : vision.calibration!.forwardOffsetM + vision.boundaryStartDistanceM;
    const leftBoundary: GroundPoint[] = [], rightBoundary: GroundPoint[] = [], centers: GroundPoint[] = [];
    let anchor = 0.5, lastLeftRow = scene.height, lastRightRow = scene.height;
    let previousLeft: GroundPoint | null = null, previousRight: GroundPoint | null = null;
    const continuous = (a: GroundPoint, b: GroundPoint) => {
        const dy = b.y - a.y;
        return dy > 0 && dy <= 15 && Math.abs(b.x - a.x) <= 1 + dy * 0.7;
    };
    const append = (boundary: GroundPoint[], point: GroundPoint, previous: GroundPoint | null, rowGap: number) => {
        if (point.y < startY) return;
        // Intersect each observed edge with the same local Y plane. Never extrapolate across missing data.
        if (!boundary.length && previous && previous.y < startY && point.y > startY && rowGap === 1 && continuous(previous, point)) {
            const t = (startY - previous.y) / (point.y - previous.y);
            boundary.push({ x: previous.x + t * (point.x - previous.x), y: startY,
                z: previous.z + t * (point.z - previous.z) });
        }
        boundary.push(point);
    };
    for (let row = scene.height - 1; row >= 0; row--) {
        const { v } = scene.sourcePixel(0, row + 0.5);
        if (v <= 0 || v >= 1) continue;
        const candidates: Array<{ left: GroundPoint | null; right: GroundPoint | null; center: number }> = [];
        const at = (column: number) => scene.sample(scene.sourcePixel(column + 0.5, row + 0.5).u, v);
        const edge = (column: number, outside: number, boundary: GroundPoint[], lastRow: number) => {
            // Validate each visible edge independently, including its own depth and continuity.
            if (outside < 0 || outside >= scene.width || at(column) !== 1 || at(outside) === 255 || at(outside) === 2) return null;
            const point = lift(scene.sourcePixel(column + 0.5, row + 0.5).u, v, scene.visibleRoad);
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
            append(leftBoundary, best.left, previousLeft, lastLeftRow - row);
            previousLeft = best.left;
            lastLeftRow = row;
        }
        if (best.right) {
            append(rightBoundary, best.right, previousRight, lastRightRow - row);
            previousRight = best.right;
            lastRightRow = row;
        }
        if (best.left && best.right && best.left.y >= startY && best.right.y >= startY) centers.push({ x: (best.left.x + best.right.x) / 2,
            y: (best.left.y + best.right.y) / 2, z: (best.left.z + best.right.z) / 2 });
        if ((best.left && best.left.y >= startY) || (best.right && best.right.y >= startY)) anchor = best.center;
    }
    const cars: ReconstructedCar[] = [];
    for (const item of scene.traffic) {
        if (item.mask.length !== scene.width * scene.height) continue;
        const points: GroundPoint[] = [];
        const inCar = (u: number, v: number) => u >= item.box[0] && u <= item.box[2]
            && v >= item.box[1] && v <= item.box[3] && scene.inMask(item.mask, u, v);
        // Bound per-instance work while retaining the actual visible surface, including elevation.
        const stride = Math.max(1, Math.ceil(Math.sqrt(item.mask.reduce((sum, value) => sum + Number(value !== 0), 0) / 600)));
        for (let row = 0; row < scene.height; row += stride) {
            for (let column = 0; column < scene.width; column += stride) {
                const { u, v } = scene.sourcePixel(column + 0.5, row + 0.5);
                if (!inCar(u, v)) continue;
                const point = lift(u, v, inCar);
                if (point) points.push(point);
            }
        }
        if (points.length < 4) continue;
        const axes = (['x', 'y', 'z'] as const).map((key) => points.map((point) => point[key]).sort((a, b) => a - b));
        const quantile = (q: number): GroundPoint => ({ x: axes[0][Math.floor((points.length - 1) * q)],
            y: axes[1][Math.floor((points.length - 1) * q)], z: axes[2][Math.floor((points.length - 1) * q)] });
        const min = quantile(0.02), max = quantile(0.98), center = quantile(0.5);
        const supported = [-0.25, 0, 0.25].filter((offset) => scene.road((item.box[0] + item.box[2]) / 2
            + offset * (item.box[2] - item.box[0]), item.box[3] + 1.5 * scene.pixelHeight)).length;
        cars.push({ classId: item.classId, confidence: item.confidence, pack: item.pack,
            points: points.filter((point) => point.y >= min.y && point.y <= max.y), min, max, center, roadSupported: supported >= 2 });
    }
    return { leftBoundary, rightBoundary, cars, geometry: fitTrackBoundaries(leftBoundary, rightBoundary, centers) };
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

/** Positions use the depth-derived local scene and the observed extent of its road fit. */
export function analyzeTrackPositions(vision: TrackVisionFrame | null, reconstruction = reconstructTrack(vision)): TrackVisionAnalysis {
    const geometry = reconstruction?.geometry;
    if (!vision || !geometry || !reconstruction) return {};
    const scene = semanticScene(vision);
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
