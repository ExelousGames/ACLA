import { letterbox } from './yolo-segmentation';
import { VISION_INPUT_SIZE } from './track-vision-model';
import { DEFAULT_DEPTH_RANGE, DepthRange, TrackVisionDetection, VisionResult } from './track-vision-types';

const COLORS = [[55, 239, 172], [87, 185, 255], [255, 190, 87], [206, 135, 255], [255, 115, 137], [110, 221, 235]];

export function drawVisionOverlay(context: CanvasRenderingContext2D, result: TrackVisionDetection, depthRange: DepthRange = DEFAULT_DEPTH_RANGE) {
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(result.width, result.height, VISION_INPUT_SIZE);
    // Depth underneath segmentation masks and boxes.
    for (const task of ['depth', 'segment'] as const) {
        const detection = result.detections[task];
        if (!detection) continue;
        const layer = document.createElement('canvas');
        layer.width = detection.width;
        layer.height = detection.height;
        const layerContext = layer.getContext('2d');
        if (!layerContext) continue;
        const pixels = layerContext.createImageData(layer.width, layer.height);
        paintMask(pixels.data, detection, depthRange);
        layerContext.putImageData(pixels, 0, 0);
        context.save();
        context.imageSmoothingEnabled = task === 'depth';
        context.drawImage(layer, padX / VISION_INPUT_SIZE * layer.width, padY / VISION_INPUT_SIZE * layer.height,
            resizedWidth / VISION_INPUT_SIZE * layer.width, resizedHeight / VISION_INPUT_SIZE * layer.height,
            0, 0, result.width, result.height);
        if (detection.task === 'segment') {
            context.lineWidth = Math.max(2, result.width / 500);
            context.font = `${Math.max(12, result.width / 90)}px sans-serif`;
            for (const instance of detection.instances) {
                const [x1, y1, x2, y2] = instance.box;
                const x = Math.max(0, (x1 * VISION_INPUT_SIZE - padX) / resizedWidth * result.width);
                const y = Math.max(0, (y1 * VISION_INPUT_SIZE - padY) / resizedHeight * result.height);
                const right = Math.min(result.width, (x2 * VISION_INPUT_SIZE - padX) / resizedWidth * result.width);
                const bottom = Math.min(result.height, (y2 * VISION_INPUT_SIZE - padY) / resizedHeight * result.height);
                if (right <= x || bottom <= y) continue;
                context.strokeStyle = `rgb(${COLORS[instance.classId % COLORS.length].join(',')})`;
                context.fillStyle = context.strokeStyle;
                context.strokeRect(x, y, right - x, bottom - y);
                context.fillText(`${detection.classNames[instance.classId]} · ${Math.round(instance.confidence * 100)}%`, x + 4, Math.max(16, y + 16));
            }
        }
        context.restore();
    }
}

function paintMask(pixels: Uint8ClampedArray, result: VisionResult, depthRange: DepthRange) {
    const paint = (pixel: number, color: number[], alpha: number) => {
        pixels.set([...color, alpha], pixel * 4);
    };
    if (result.task === 'depth') {
        result.values.forEach((value, pixel) => {
            if (!Number.isFinite(value) || value <= 0) return;
            const distance = Math.max(0, Math.min(1, (value - depthRange.near) / (depthRange.far - depthRange.near)));
            paint(pixel, [Math.round(255 * (1 - distance)), 80, Math.round(255 * distance)], 100);
        });
    } else {
        result.instances.forEach((instance) => instance.mask.forEach((active, pixel) => {
            if (active) paint(pixel, COLORS[instance.classId % COLORS.length], 110);
        }));
    }
}
