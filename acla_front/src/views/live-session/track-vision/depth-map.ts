import type { TrackVisionFrame } from './track-vision-types';
import { VISION_DEPTH_COLORS } from './vision-colors';
import { letterbox } from './yolo-segmentation';

const COLORS = VISION_DEPTH_COLORS.map((color) => [1, 3, 5].map((offset) => parseInt(color.slice(offset, offset + 2), 16)));
const validDepth = (value: number) => Number.isFinite(value) && value > 0;

/** Full-frame depth, independent of retained labels. Padding never contributes to the color scale. */
export function createDepthMap(frame: TrackVisionFrame | null) {
    const depth = frame?.detections.depth;
    if (!frame || depth?.task !== 'depth' || !frame.width || !frame.height || !depth.width || !depth.height) return null;
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    const crop = { x: padX / 640 * depth.width, y: padY / 640 * depth.height,
        width: resizedWidth / 640 * depth.width, height: resizedHeight / 640 * depth.height };
    let near = Infinity, far = -Infinity;
    depth.values.forEach((value, i) => {
        const x = i % depth.width + 0.5, y = Math.floor(i / depth.width) + 0.5;
        if (!validDepth(value) || x < crop.x || x >= crop.x + crop.width || y < crop.y || y >= crop.y + crop.height) return;
        near = Math.min(near, value); far = Math.max(far, value);
    });
    return { depth, crop, width: frame.width, height: frame.height,
        near: Number.isFinite(near) ? near : null, far: Number.isFinite(far) ? far : null };
}

type DepthMap = NonNullable<ReturnType<typeof createDepthMap>>;

export function drawDepthMap(context: CanvasRenderingContext2D, map: DepthMap) {
    const { depth, crop, width, height, near, far } = map;
    const layer = document.createElement('canvas');
    layer.width = depth.width; layer.height = depth.height;
    const paint = layer.getContext('2d');
    if (!paint) return;
    const pixels = paint.createImageData(layer.width, layer.height);
    depth.values.forEach((value, i) => {
        if (!validDepth(value) || near === null || far === null) return;
        const t = far > near ? Math.max(0, Math.min(1, (value - near) / (far - near))) : 0;
        const position = t * (COLORS.length - 1), lower = Math.floor(position);
        const start = COLORS[lower], end = COLORS[Math.min(lower + 1, COLORS.length - 1)];
        pixels.data.set([...start.map((channel, index) => Math.round(channel + (end[index] - channel) * (position - lower))), 255], i * 4);
    });
    paint.putImageData(pixels, 0, 0);
    context.save();
    context.fillStyle = '#090d13';
    context.fillRect(0, 0, width, height);
    context.imageSmoothingEnabled = false;
    context.drawImage(layer, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
    context.restore();
}

/** Undo CSS object-fit: contain, then map the source pixel into the letterboxed depth grid. */
export function depthAtMouse(map: DepthMap, rect: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>, clientX: number, clientY: number) {
    const scale = Math.min(rect.width / map.width, rect.height / map.height);
    if (!(scale > 0)) return null;
    const x = clientX - rect.left, y = clientY - rect.top;
    const u = (x - (rect.width - map.width * scale) / 2) / (map.width * scale);
    const v = (y - (rect.height - map.height * scale) / 2) / (map.height * scale);
    if (![u, v].every(Number.isFinite) || u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    const column = Math.floor(map.crop.x + u * map.crop.width), row = Math.floor(map.crop.y + v * map.crop.height);
    const value = map.depth.values[row * map.depth.width + column];
    return { x, y, depth: validDepth(value) ? value : null };
}
