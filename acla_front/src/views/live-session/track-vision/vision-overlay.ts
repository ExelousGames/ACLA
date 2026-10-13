import { letterbox } from './yolo-segmentation';
import { VISION_INPUT_SIZE } from './vision-config';
import { SegmentResult, TrackVisionFrame } from './track-vision-types';
import { VISION_LABEL_COLORS } from './vision-colors';

const COLORS = VISION_LABEL_COLORS.map((color) => [1, 3, 5].map((offset) => parseInt(color.slice(offset, offset + 2), 16)));

export function drawVisionOverlay(context: CanvasRenderingContext2D, result: TrackVisionFrame, displayLabel = '') {
    const detection = result.detections.segment;
    if (detection?.task !== 'segment') return;
    // Capture shows raw detections; interior exclusions and amodal completion belong downstream.
    const instances = detection.instances.filter((instance) => !displayLabel || detection.classNames[instance.classId] === displayLabel);
    if (!instances.length) return;
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(result.width, result.height, VISION_INPUT_SIZE);
    const layer = document.createElement('canvas');
    layer.width = detection.width;
    layer.height = detection.height;
    const layerContext = layer.getContext('2d');
    if (!layerContext) return;
    const pixels = layerContext.createImageData(layer.width, layer.height);
    paintMask(pixels.data, instances);
    layerContext.putImageData(pixels, 0, 0);
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(layer, padX / VISION_INPUT_SIZE * layer.width, padY / VISION_INPUT_SIZE * layer.height,
        resizedWidth / VISION_INPUT_SIZE * layer.width, resizedHeight / VISION_INPUT_SIZE * layer.height,
        0, 0, result.width, result.height);
    context.lineWidth = Math.max(2, result.width / 500);
    context.font = `${Math.max(12, result.width / 90)}px sans-serif`;
    for (const instance of instances) {
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
    context.restore();
}

function paintMask(pixels: Uint8ClampedArray, instances: SegmentResult['instances']) {
    const paint = (pixel: number, color: number[], alpha: number) => {
        const offset = pixel * 4;
        // Composite overlapping raw masks without subtracting either detection.
        const backgroundAlpha = pixels[offset + 3] * (1 - alpha / 255);
        const combinedAlpha = alpha + backgroundAlpha;
        for (let channel = 0; channel < 3; channel++) {
            pixels[offset + channel] = (color[channel] * alpha + pixels[offset + channel] * backgroundAlpha) / combinedAlpha;
        }
        pixels[offset + 3] = combinedAlpha;
    };
    instances.forEach((instance) => {
        if (instance.mask.length !== pixels.length / 4) return;
        instance.mask.forEach((active, pixel) => {
            if (active) paint(pixel, COLORS[instance.classId % COLORS.length], 110);
        });
    });
}
