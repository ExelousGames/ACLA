import { VISION_INPUT_SIZE } from './vision-config';
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
    centerline: ImagePoint[][];
    cars: ImageCar[];
}

/** Reconstruct visible track sides and traffic in image space, independent of coaching geometry. */
export function reconstructScene(frame: TrackVisionFrame | null): ReconstructedScene | null {
    const segment = frame?.detections.segment;
    if (!frame || frame.width <= 0 || frame.height <= 0 || segment?.task !== 'segment') return null;
    const layers = createSegmentationLayers(segment, frame.filterConfidence ?? VISION_CONFIDENCE);
    if (!layers) return null;
    const { width, height } = segment;
    const { trackMask, carInteriorMask } = layers;
    const box = letterbox(frame.width, frame.height, VISION_INPUT_SIZE);
    const firstColumn = Math.max(0, Math.ceil(box.padX / VISION_INPUT_SIZE * width - 0.5));
    const endColumn = Math.min(width, Math.ceil((box.padX + box.resizedWidth) / VISION_INPUT_SIZE * width - 0.5));
    const firstRow = Math.max(0, Math.ceil(box.padY / VISION_INPUT_SIZE * height - 0.5));
    const endRow = Math.min(height, Math.ceil((box.padY + box.resizedHeight) / VISION_INPUT_SIZE * height - 0.5));
    const imagePoint = (x: number, y: number): ImagePoint => ({
        x: ((x + 0.5) / width * VISION_INPUT_SIZE - box.padX) / box.resizedWidth * frame.width,
        y: ((y + 0.5) / height * VISION_INPUT_SIZE - box.padY) / box.resizedHeight * frame.height,
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
        const left = Math.max(0, (x1 * VISION_INPUT_SIZE - box.padX) / box.resizedWidth * frame.width);
        const top = Math.max(0, (y1 * VISION_INPUT_SIZE - box.padY) / box.resizedHeight * frame.height);
        const right = Math.min(frame.width, (x2 * VISION_INPUT_SIZE - box.padX) / box.resizedWidth * frame.width);
        const bottom = Math.min(frame.height, (y2 * VISION_INPUT_SIZE - box.padY) / box.resizedHeight * frame.height);
        return right > left && bottom > top ? [{ classId: item.classId, confidence: item.confidence,
            pack: item.kind === 'car pack', box: [left, top, right, bottom] as ImageCar['box'] }] : [];
    });
    const result: ReconstructedScene = { width: frame.width, height: frame.height,
        leftBoundary: [], rightBoundary: [], centerline: [], cars };
    const lines = [result.leftBoundary, result.rightBoundary, result.centerline];
    const previousRows = [-1, -1, -1];
    for (let row = firstRow; row < endRow; row++) {
        let left = -1, right = -1;
        for (let column = firstColumn; column < endColumn; column++) {
            if (!trackMask[row * width + column]) continue;
            if (left < 0) left = column;
            right = column;
        }
        if (left < 0 || right <= left) continue;
        // Trace the original mask, so removing bodywork never creates a new track edge.
        const edges = [left, right].map((column, side) => {
            const outside = imagePoint(column + (side ? 1 : -1), row);
            return outside.x < 0 || outside.x >= frame.width || nearInterior(column, row)
                ? null : imagePoint(column, row);
        });
        const [leftPoint, rightPoint] = edges;
        const center = leftPoint && rightPoint ? { x: (leftPoint.x + rightPoint.x) / 2, y: leftPoint.y } : null;
        [...edges, center].forEach((point, side) => {
            if (!point) return;
            const line = lines[side];
            // Resume after missing/cockpit pixels, with no bend or width-change cutoff.
            if (!line.length || previousRows[side] !== row - 1) line.push([]);
            line[line.length - 1].push(point);
            previousRows[side] = row;
        });
    }
    result.centerline = result.centerline.filter((line) => line.length > 1);
    // Trace the full outline first, then stop both sides at the farthest supported
    // middle-line segment. An isolated pair cannot establish a prediction endpoint.
    const endY = result.centerline[0]?.[0].y ?? Infinity;
    const trim = (boundary: ImagePoint[][]) => boundary
        .map((line) => line.filter(({ y }) => y >= endY)).filter((line) => line.length > 1);
    result.leftBoundary = trim(result.leftBoundary);
    result.rightBoundary = trim(result.rightBoundary);
    return result;
}
