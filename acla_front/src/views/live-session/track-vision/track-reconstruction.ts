import { createDepthPointCloud, DepthPointCloud, maskPointCloud, SurfacePoint } from './depth-point-cloud';
import { createDepthProjection } from './depth-projection';
import { createSemanticScene, SemanticScene } from './semantic-scene';
import { reconstructBoundaries } from './track-boundaries';
import type { GroundPoint, LocalTrackScene, ReconstructedCar, TrackVisionFrame } from './track-vision-types';

function reconstructCars(cloud: DepthPointCloud, scene: SemanticScene) {
    const traffic = scene.traffic.map((item) => {
        const contains = (u: number, v: number) => u >= item.box[0] && u <= item.box[2]
            && v >= item.box[1] && v <= item.box[3] && scene.inMask(item.mask, u, v) && !scene.excluded(u, v);
        const grid = maskPointCloud(cloud, contains, item.box, 600);
        const depths = grid.points.filter((point): point is SurfacePoint => Boolean(point)).map((point) => point.depthM).sort((a, b) => a - b);
        return { item, contains, grid, depth: depths[Math.floor(depths.length / 2)] ?? Infinity };
    });
    // Prefer unambiguous visible fragments: a large foreground overlap must not set
    // the representative depth of the mostly hidden object behind it.
    for (const candidate of traffic) {
        const depths = candidate.grid.points.filter((point): point is SurfacePoint => Boolean(point
            && !traffic.some((other) => other !== candidate && !other.item.pack && other.contains(point.u, point.v))))
            .map((point) => point.depthM).sort((a, b) => a - b);
        if (depths.length) candidate.depth = depths[Math.floor(depths.length / 2)];
    }
    // Instance ownership prevents a foreground car (or a pack mask) from becoming another car's surface.
    const owners = traffic.slice().sort((a, b) => Number(a.item.pack) - Number(b.item.pack)
        || a.depth - b.depth || b.item.confidence - a.item.confidence
        || a.item.box[0] - b.item.box[0] || a.item.box[1] - b.item.box[1]);
    const ownerAt = (u: number, v: number) => owners.find((candidate) => candidate.contains(u, v));
    const cars: ReconstructedCar[] = [];
    for (const candidate of traffic) {
        const { item, grid } = candidate;
        const points = grid.points.filter((point): point is SurfacePoint => Boolean(point && ownerAt(point.u, point.v) === candidate));
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
    return cars;
}

/** Measured geometry for coaching only; no display polygons or scene-memory cloud are constructed. */
export function reconstructTrack(vision: TrackVisionFrame | null): LocalTrackScene | null {
    if (!vision || vision.detections.segment?.task !== 'segment') return null;
    const pointCloud = createDepthPointCloud(vision);
    if (!pointCloud) return null;
    const scene = createSemanticScene(vision, true);
    if (!scene) return null;
    const lift = createDepthProjection(vision, pointCloud)!;
    return { cars: reconstructCars(pointCloud, scene),
        ...reconstructBoundaries(vision, scene, lift) };
}
