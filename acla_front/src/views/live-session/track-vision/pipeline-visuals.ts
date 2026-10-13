import { VISION_INPUT_SIZE } from './vision-config';
import { AmodalMask, predictAmodalMasks } from './amodal-masks';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';
import { VISION_DEPTH_COLORS } from './vision-colors';
import { letterbox } from './yolo-segmentation';
import { formatDepth } from './depth-map';

const DEPTH_COLORS = VISION_DEPTH_COLORS.map((color) => [1, 3, 5].map((offset) => parseInt(color.slice(offset, offset + 2), 16)));

export const PIPELINE_STEPS = [
    { id: 'capture', label: 'Capture', title: 'Captured window' },
    { id: 'calibration', label: 'Camera position', title: 'Set the camera position' },
    { id: 'segmentation', label: 'Segmentation', title: 'Every label in the camera view' },
    { id: 'filtering', label: 'Filtering', title: 'Masks after filtering' },
    { id: 'depth-map', label: 'Depth map', title: 'Depth across the entire frame' },
    { id: 'depth', label: 'Label depths', title: 'Depth of each retained mask' },
    { id: 'scene', label: 'Reconstructed scene', title: 'Reconstructed scene' },
    { id: 'birds-eye', label: "Bird's-eye view", title: 'Top-down track boundaries and cars' },
] as const;
export type PipelineStep = typeof PIPELINE_STEPS[number]['id'];

export function filteredMasks(frame: TrackVisionFrame | null) {
    return frame ? predictAmodalMasks(frame, frame.filterConfidence ?? VISION_CONFIDENCE).filter((mask) => mask.bounds) : [];
}

export function filteredFrame(frame: TrackVisionFrame, masks: AmodalMask[]): TrackVisionFrame {
    const segment = frame.detections.segment;
    return segment?.task === 'segment' ? { ...frame, detections: { ...frame.detections,
        segment: { ...segment, instances: masks.map((mask) => ({ ...mask,
            box: mask.bounds ? [mask.bounds[0] / segment.width, mask.bounds[1] / segment.height,
                mask.bounds[2] / segment.width, mask.bounds[3] / segment.height] : mask.box,
        })) },
    } } : frame;
}

export function labelDepths(masks: AmodalMask[]) {
    const instances = new Map<number, number>();
    return masks.map((mask, maskIndex) => {
        const instance = (instances.get(mask.classId) ?? 0) + 1;
        instances.set(mask.classId, instance);
        const values: number[] = [];
        let estimated = 0;
        mask.depths.forEach((depth, i) => {
            if (!mask.mask[i] || !Number.isFinite(depth) || depth <= 0 || depth > 200) return;
            if (mask.hiddenMask[i]) estimated++;
            else if (mask.visibleMask[i]) values.push(depth);
        });
        values.sort((a, b) => a - b);
        const middle = Math.floor(values.length / 2);
        return { classId: mask.classId, label: mask.label, maskIndex, instance, estimated,
            samples: values.length, near: values[0] ?? null, far: values[values.length - 1] ?? null,
            median: values.length ? values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2 : null };
    }).sort((a, b) => a.classId - b.classId || a.instance - b.instance);
}

/** Include observed and predicted pixels in the shared depth color scale. */
export function labelDepthRange(masks: AmodalMask[]) {
    let near = Infinity, far = 0;
    masks.forEach((mask) => mask.depths.forEach((depth, i) => {
        if (!mask.mask[i] || !Number.isFinite(depth) || depth <= 0 || depth > 200) return;
        near = Math.min(near, depth); far = Math.max(far, depth);
    }));
    return { near: far ? near : null, far: far || null };
}

/** Paint retained masks together, with per-mask names and depths matching the table. */
export function drawLabelDepths(context: CanvasRenderingContext2D, frame: TrackVisionFrame, masks: AmodalMask[]) {
    const segment = frame.detections.segment;
    if (segment?.task !== 'segment' || !masks.length) return;
    const depths = new Float32Array(segment.width * segment.height);
    for (const mask of masks) mask.depths.forEach((depth, i) => {
        if (mask.mask[i] && Number.isFinite(depth) && depth > 0 && depth <= 200 && (!depths[i] || depth < depths[i])) depths[i] = depth;
    });
    const { near, far } = labelDepthRange(masks);
    const layer = document.createElement('canvas');
    layer.width = segment.width; layer.height = segment.height;
    const paint = layer.getContext('2d');
    if (!paint) return;
    const pixels = paint.createImageData(layer.width, layer.height);
    depths.forEach((depth, i) => {
        if (!depth || near === null || far === null) return;
        const t = far > near ? Math.max(0, Math.min(1, (depth - near) / (far - near))) : 0;
        const position = t * (DEPTH_COLORS.length - 1), lower = Math.floor(position);
        const start = DEPTH_COLORS[lower], end = DEPTH_COLORS[Math.min(lower + 1, DEPTH_COLORS.length - 1)];
        const color = start.map((channel, index) => Math.round(channel + (end[index] - channel) * (position - lower)));
        pixels.data.set([...color, 190], i * 4);
    });
    paint.putImageData(pixels, 0, 0);
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, VISION_INPUT_SIZE);
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(layer, padX / VISION_INPUT_SIZE * layer.width, padY / VISION_INPUT_SIZE * layer.height,
        resizedWidth / VISION_INPUT_SIZE * layer.width, resizedHeight / VISION_INPUT_SIZE * layer.height, 0, 0, frame.width, frame.height);
    const fontSize = Math.max(12, frame.width / 90);
    context.font = `${fontSize}px sans-serif`;
    labelDepths(masks).forEach((row) => {
        const mask = masks[row.maskIndex];
        if (!mask.bounds) return;
        const x = Math.max(4, (mask.bounds[0] / segment.width * VISION_INPUT_SIZE - padX) / resizedWidth * frame.width + 4);
        const y = Math.max(fontSize + 4, (mask.bounds[1] / segment.height * VISION_INPUT_SIZE - padY) / resizedHeight * frame.height + fontSize + 4);
        const depth = frame.detections.depth;
        const text = `${row.label} #${row.instance} · ${row.median === null ? 'No depth' : formatDepth(row.median, depth?.task === 'depth' ? depth.scale : undefined)}`;
        context.fillStyle = '#090d13dd';
        context.fillRect(x - 3, y - fontSize - 2, context.measureText(text).width + 6, fontSize + 6);
        context.fillStyle = '#ffffff';
        context.fillText(text, x, y);
    });
    context.restore();
}
