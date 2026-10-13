import type { TrackVisionFrame } from './track-vision-types';
import { createCameraProjection, sameCalibration } from './camera-projection';
import { createDepthPointCloud, DepthPointCloud } from './depth-point-cloud';

/** Sample the depth map in source-image coordinates, independently of mask resolution. */
export function createDepthProjection(frame: TrackVisionFrame, cloud = createDepthPointCloud(frame)) {
    if (cloud && !sameCalibration(frame.calibration, cloud.calibration)) cloud = createDepthPointCloud(frame);
    return cloud && frame.calibration ? samplePointCloud(cloud, createCameraProjection(frame.calibration)) : null;
}

/** Sample supported measured depth along the requested image ray for coaching boundaries. */
function samplePointCloud(cloud: DepthPointCloud, camera: ReturnType<typeof createCameraProjection>) {
    const { pixelWidth, pixelHeight } = cloud;
    return (u: number, v: number, supports: (u: number, v: number) => boolean) => {
        if (![u, v].every(Number.isFinite) || u < 0 || u >= 1 || v < 0 || v >= 1) return null;
        const x = (u - cloud.columns[0]) / pixelWidth;
        const y = (v - cloud.rows[0]) / pixelHeight;
        const x0 = Math.floor(x), y0 = Math.floor(y);
        let sum = 0, weight = 0;
        for (let row = y0; row <= y0 + 1; row++) {
            for (let column = x0; column <= x0 + 1; column++) {
                if (column < 0 || column >= cloud.width || row < 0 || row >= cloud.height) continue;
                const su = cloud.columns[column], sv = cloud.rows[row];
                const value = cloud.depths[row * cloud.width + column];
                // Never mix background depth into a car or road edge, or sample padding.
                if (su < 0 || su >= 1 || sv < 0 || sv >= 1 || !supports(su, sv)
                    || !Number.isFinite(value) || value <= 0 || value > 200) continue;
                const w = (1 - Math.abs(column - x)) * (1 - Math.abs(row - y));
                sum += value * w;
                weight += w;
            }
        }
        if (weight < 0.001) return null;
        const point = camera.imageToLocal(u, v, sum / weight);
        return point && point.y > 0 && point.y <= 200 ? point : null;
    };
}
