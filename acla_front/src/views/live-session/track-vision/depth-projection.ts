import type { TrackVisionFrame } from './track-vision-types';
import { createCameraProjection, validCalibration } from './camera-projection';
import { letterbox } from './yolo-segmentation';

/** Sample the depth map in source-image coordinates, independently of mask resolution. */
export function createDepthProjection(frame: TrackVisionFrame) {
    const depth = frame.detections.depth;
    if (!validCalibration(frame.calibration) || frame.calibration.imageWidth !== frame.width
        || frame.calibration.imageHeight !== frame.height || depth?.task !== 'depth'
        || !Number.isInteger(depth.width) || !Number.isInteger(depth.height) || depth.width <= 0 || depth.height <= 0
        || depth.values.length !== depth.width * depth.height) return null;
    const camera = createCameraProjection(frame.calibration);
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    return (u: number, v: number, supports: (u: number, v: number) => boolean) => {
        if (![u, v].every(Number.isFinite) || u < 0 || u >= 1 || v < 0 || v >= 1) return null;
        const x = (padX + u * resizedWidth) / 640 * depth.width - 0.5;
        const y = (padY + v * resizedHeight) / 640 * depth.height - 0.5;
        const x0 = Math.floor(x), y0 = Math.floor(y);
        let sum = 0, weight = 0;
        for (let row = y0; row <= y0 + 1; row++) {
            for (let column = x0; column <= x0 + 1; column++) {
                if (column < 0 || column >= depth.width || row < 0 || row >= depth.height) continue;
                const su = ((column + 0.5) / depth.width * 640 - padX) / resizedWidth;
                const sv = ((row + 0.5) / depth.height * 640 - padY) / resizedHeight;
                const value = depth.values[row * depth.width + column];
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
