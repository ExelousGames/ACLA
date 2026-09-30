import type { SegmentResult, TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';
import { createSegmentationLayers } from './segmentation-layers';
import { createTrackBoundaryMask } from './track-boundary-mask';

export const VISION_CONFIDENCE = 0.65;
type Box = SegmentResult['instances'][number]['box'];

/** Sample the shared, overlapping track / traffic / excluded surface layers. */
export function createSemanticScene(vision: TrackVisionFrame, constructBoundaries = false) {
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
    const boundaryMask = constructBoundaries ? createTrackBoundaryMask(layers, segment.width, segment.height) : layers.trackMask;
    const boundaryRoad = (u: number, v: number) => inMask(boundaryMask, u, v) && !inMask(layers.excludedMask, u, v);
    const interiorMargin = Math.max(2, Math.ceil(Math.max(segment.width, segment.height) * 0.01));
    return {
        traffic, road, inMask, width: segment.width, height: segment.height,
        sourcePixel(x: number, y: number) {
            return { u: (x / segment.width * 640 - padX) / resizedWidth,
                v: (y / segment.height * 640 - padY) / resizedHeight };
        },
        pixelHeight: 640 / resizedHeight / segment.height,
        pixelWidth: 640 / resizedWidth / segment.width,
        excluded: (u: number, v: number) => inMask(layers.excludedMask, u, v),
        occluded,
        hasCarLabels: layers.hasCarLabels,
        nearCarInterior(column: number, row: number) {
            // Check in 2D so pillars and dashboards also suppress edges across small mask gaps.
            for (let y = Math.max(0, row - interiorMargin); y <= Math.min(segment.height - 1, row + interiorMargin); y++) {
                for (let x = Math.max(0, column - interiorMargin); x <= Math.min(segment.width - 1, column + interiorMargin); x++) {
                    if (layers.carInteriorMask[y * segment.width + x]) return true;
                }
            }
            return false;
        },
        // Car depth is not road depth, even though the track continues underneath.
        visibleRoad: (u: number, v: number) => road(u, v) && !occluded(u, v),
        visibleBoundaryRoad: (u: number, v: number) => boundaryRoad(u, v) && !occluded(u, v),
        sample(u: number, v: number) {
            if (u < 0 || u >= 1 || v < 0 || v >= 1) return 255;
            if (inMask(layers.excludedMask, u, v)) return 3;
            if (occluded(u, v)) return 2;
            return boundaryRoad(u, v) ? 1 : 0;
        },
    };
}

export type SemanticScene = NonNullable<ReturnType<typeof createSemanticScene>>;
