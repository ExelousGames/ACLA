import type { SegmentResult } from './track-vision-types';
import { isCarInteriorLabel } from './vision-labels';

/** Pixel coverage in the letterboxed model input. */
export interface MaskRegion { width: number; height: number; mask: Uint8Array }

/** Keep the entire image, subtracting only accepted car-interior pixels, independent of order. */
export function createWorldMask(segment: SegmentResult & { classNames: string[] }, minimumConfidence = 0) {
    const { width, height } = segment;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) return null;
    const mask = new Uint8Array(width * height).fill(1), carInteriorMask = new Uint8Array(mask.length);
    for (const instance of segment.instances) {
        if (!Number.isFinite(instance.confidence) || instance.confidence < minimumConfidence || instance.confidence > 1
            || instance.mask.length !== mask.length || !isCarInteriorLabel(segment.classNames[instance.classId])) continue;
        instance.mask.forEach((active, i) => { if (active) carInteriorMask[i] = 1; });
    }
    carInteriorMask.forEach((active, i) => { if (active) mask[i] = 0; });
    return { width, height, mask, carInteriorMask };
}

/** Nearest pixel-center sampling preserves alignment between segmentation, model input and depth. */
export function resizeMask(region: MaskRegion, width: number, height: number) {
    if (region.width === width && region.height === height) return region.mask;
    const columns = Int32Array.from({ length: width }, (_, x) => Math.floor((x + 0.5) / width * region.width));
    const mask = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        const row = Math.floor((y + 0.5) / height * region.height) * region.width;
        for (let x = 0; x < width; x++) mask[y * width + x] = region.mask[row + columns[x]];
    }
    return mask;
}
