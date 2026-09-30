import { createSegmentationLayers } from './segmentation-layers';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';

export interface ImagePoint { x: number; y: number }
export interface ImageCar {
    classId: number;
    confidence: number;
    pack: boolean;
    /** Left, top, right and bottom in captured image pixels. */
    box: [number, number, number, number];
}
export interface ReconstructedScene {
    width: number;
    height: number;
    leftBoundary: ImagePoint[][];
    rightBoundary: ImagePoint[][];
    cars: ImageCar[];
}

/** Reconstruct visible track sides and traffic in image space, independent of coaching geometry. */
export function reconstructScene(frame: TrackVisionFrame | null): ReconstructedScene | null {
    const segment = frame?.detections.segment;
    if (!frame || frame.width <= 0 || frame.height <= 0 || segment?.task !== 'segment') return null;
    const layers = createSegmentationLayers(segment, VISION_CONFIDENCE);
    if (!layers) return null;
    const { width, height } = segment;
    const { trackMask, carInteriorMask, excludedMask, trafficMask } = layers;
    const box = letterbox(frame.width, frame.height, 640);
    const firstColumn = Math.max(0, Math.ceil(box.padX / 640 * width - 0.5));
    const endColumn = Math.min(width, Math.ceil((box.padX + box.resizedWidth) / 640 * width - 0.5));
    const firstRow = Math.max(0, Math.ceil(box.padY / 640 * height - 0.5));
    const endRow = Math.min(height, Math.ceil((box.padY + box.resizedHeight) / 640 * height - 0.5));
    const imagePoint = (x: number, y: number): ImagePoint => ({
        x: ((x + 0.5) / width * 640 - box.padX) / box.resizedWidth * frame.width,
        y: ((y + 0.5) / height * 640 - box.padY) / box.resizedHeight * frame.height,
    });
    const margin = Math.max(2, Math.ceil(Math.max(width, height) * 0.01));
    const nearInterior = (x: number, y: number) => {
        for (let row = Math.max(0, y - margin); row <= Math.min(height - 1, y + margin); row++) {
            for (let column = Math.max(0, x - margin); column <= Math.min(width - 1, x + margin); column++) {
                if (carInteriorMask[row * width + column]) return true;
            }
        }
        return false;
    };
    // Display every accepted traffic instance, including those outside the coaching analysis region.
    const cars: ImageCar[] = layers.instances.flatMap((item) => {
        if ((item.kind !== 'car' && item.kind !== 'car pack') || !item.box.every(Number.isFinite)) return [];
        const [x1, y1, x2, y2] = item.box;
        const left = Math.max(0, (x1 * 640 - box.padX) / box.resizedWidth * frame.width);
        const top = Math.max(0, (y1 * 640 - box.padY) / box.resizedHeight * frame.height);
        const right = Math.min(frame.width, (x2 * 640 - box.padX) / box.resizedWidth * frame.width);
        const bottom = Math.min(frame.height, (y2 * 640 - box.padY) / box.resizedHeight * frame.height);
        return right > left && bottom > top ? [{ classId: item.classId, confidence: item.confidence,
            pack: item.kind === 'car pack', box: [left, top, right, bottom] as ImageCar['box'] }] : [];
    });
    const result: ReconstructedScene = { width: frame.width, height: frame.height, leftBoundary: [], rightBoundary: [], cars };
    const previous: Array<{ x: number; row: number } | null> = [null, null];
    for (let row = firstRow; row < endRow; row++) {
        let left = -1, right = -1;
        for (let column = firstColumn; column < endColumn; column++) {
            if (!trackMask[row * width + column]) continue;
            if (left < 0) left = column;
            right = column;
        }
        [left, right].forEach((column, side) => {
            const point = imagePoint(column, row), outside = imagePoint(column + (side ? 1 : -1), row);
            const index = row * width + column;
            if (left < 0 || right <= left || outside.x < 0 || outside.x >= frame.width
                || excludedMask[index] || trafficMask[index] || nearInterior(column, row)) {
                previous[side] = null;
                return;
            }
            const boundary = side ? result.rightBoundary : result.leftBoundary;
            const last = previous[side];
            // Keep gaps and disconnected fragments open instead of drawing an invented connecting edge.
            if (!last || last.row !== row - 1 || Math.abs(column - last.x) > Math.max(2, width * 0.05)) boundary.push([]);
            boundary[boundary.length - 1].push(point);
            previous[side] = { x: column, row };
        });
    }
    result.leftBoundary = result.leftBoundary.filter((line) => line.length > 1);
    result.rightBoundary = result.rightBoundary.filter((line) => line.length > 1);
    return result;
}
