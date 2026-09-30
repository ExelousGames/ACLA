import { createCameraProjection, validCalibration } from './camera-projection';
import { letterbox } from './yolo-segmentation';
import type { CameraCalibration, GroundPoint, TrackVisionFrame } from './track-vision-types';
import { createWorldMask, resizeMask } from './world-mask';

export interface SurfacePoint extends GroundPoint {
    u: number;
    v: number;
    depthM: number;
}

/** An organized cloud: retained depth pixels keep their image topology during unprojection. */
export interface DepthPointCloud {
    calibration: CameraCalibration;
    width: number;
    height: number;
    positions: Float64Array;
    depths: Float32Array;
    columns: Float64Array;
    rows: Float64Array;
    pixelWidth: number;
    pixelHeight: number;
}

export function createDepthPointCloud(frame: TrackVisionFrame): DepthPointCloud | null {
    const depth = frame.detections.depth;
    if (!validCalibration(frame.calibration) || frame.calibration.imageWidth !== frame.width
        || frame.calibration.imageHeight !== frame.height || depth?.task !== 'depth'
        || !Number.isInteger(depth.width) || !Number.isInteger(depth.height) || depth.width <= 0 || depth.height <= 0
        || depth.values.length !== depth.width * depth.height) return null;
    const camera = createCameraProjection(frame.calibration);
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    const columns = Float64Array.from({ length: depth.width }, (_, x) => ((x + 0.5) / depth.width * 640 - padX) / resizedWidth);
    const rows = Float64Array.from({ length: depth.height }, (_, y) => ((y + 0.5) / depth.height * 640 - padY) / resizedHeight);
    const positions = new Float64Array(depth.values.length * 3);
    const depths = new Float32Array(depth.values.length);
    const segment = frame.detections.segment;
    const region = segment?.task === 'segment' ? createWorldMask(segment) : null;
    if (segment && !region) return null;
    const mask = region ? resizeMask(region, depth.width, depth.height) : null;
    for (let row = 0; row < depth.height; row++) {
        const v = rows[row];
        if (v < 0 || v >= 1) continue;
        for (let column = 0; column < depth.width; column++) {
            const u = columns[column], index = row * depth.width + column, depthM = depth.values[index];
            if ((mask && !mask[index]) || u < 0 || u >= 1 || !Number.isFinite(depthM) || depthM <= 0 || depthM > 200) continue;
            const point = camera.imageToLocal(u, v, depthM)!;
            if (point.y <= 0 || point.y > 200) continue;
            positions[index * 3] = point.x;
            positions[index * 3 + 1] = point.y;
            positions[index * 3 + 2] = point.z;
            depths[index] = depthM;
        }
    }
    return { calibration: { ...frame.calibration }, width: depth.width, height: depth.height, positions, depths, columns, rows,
        pixelWidth: 640 / depth.width / resizedWidth, pixelHeight: 640 / depth.height / resizedHeight };
}

export function cloudPoint(cloud: DepthPointCloud, column: number, row: number): SurfacePoint | null {
    if (column < 0 || column >= cloud.width || row < 0 || row >= cloud.height) return null;
    const index = row * cloud.width + column, depthM = cloud.depths[index];
    return depthM ? { x: cloud.positions[index * 3], y: cloud.positions[index * 3 + 1], z: cloud.positions[index * 3 + 2],
        u: cloud.columns[column], v: cloud.rows[row], depthM } : null;
}

/** Select measured mask samples without connecting them or filling missing pixels. */
export function maskPointCloud(cloud: DepthPointCloud, supports: (u: number, v: number) => boolean,
    bounds: readonly number[] = [0, 0, 1, 1], budget = 3000) {
    const columns = Array.from(cloud.columns.keys()).filter((x) => cloud.columns[x] >= Math.max(0, bounds[0])
        && cloud.columns[x] < Math.min(1, bounds[2]));
    const rows = Array.from(cloud.rows.keys()).filter((y) => cloud.rows[y] >= Math.max(0, bounds[1])
        && cloud.rows[y] < Math.min(1, bounds[3]));
    const stride = Math.max(1, Math.ceil(Math.sqrt(columns.length * rows.length / budget)));
    const xs = columns.filter((_, i) => i % stride === 0), ys = rows.filter((_, i) => i % stride === 0);
    const points = ys.flatMap((row) => xs.map((column) => supports(cloud.columns[column], cloud.rows[row])
        ? cloudPoint(cloud, column, row) : null));
    return { width: xs.length, height: ys.length, points };
}
