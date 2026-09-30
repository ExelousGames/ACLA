import { AmodalMask, predictAmodalMasks } from './amodal-masks';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';

export const PIPELINE_STEPS = [
    { id: 'capture', label: 'Capture', title: 'Captured window', description: 'The original frame from your simulator window.' },
    { id: 'calibration', label: 'Camera position', title: 'Set the camera position', description: 'Align the reference grid with your driving view, then apply the camera settings.' },
    { id: 'segmentation', label: 'Segmentation', title: 'Every label in the camera view', description: 'Inspect the detected masks, label names and confidence over the captured frame.' },
    { id: 'filtering', label: 'Filtering', title: 'Masks after filtering', description: 'Inspect confidence filtering and depth ordering. Car interior masks are retained for downstream boundary filtering.' },
    { id: 'depth', label: 'Label depths', title: 'Depth of each retained label', description: 'Compare estimated distances in the filtered masks. Empty areas have no supported label depth.' },
    { id: 'scene', label: 'Reconstructed scene', title: 'Reconstructed scene', description: 'Track boundaries, cars and car packs in 2D, with cockpit outlines removed using the car interior mask.' },
] as const;
export type PipelineStep = typeof PIPELINE_STEPS[number]['id'];

export function filteredMasks(frame: TrackVisionFrame | null) {
    return frame ? predictAmodalMasks(frame, VISION_CONFIDENCE).filter((mask) => mask.bounds) : [];
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
    const labels = new Map<number, { classId: number; label: string; instances: number; values: number[]; estimated: number }>();
    for (const mask of masks) {
        const row = labels.get(mask.classId) ?? { classId: mask.classId, label: mask.label, instances: 0, values: [], estimated: 0 };
        row.instances++;
        mask.depths.forEach((depth, i) => {
            if (!mask.mask[i] || !Number.isFinite(depth) || depth <= 0 || depth > 200) return;
            if (mask.hiddenMask[i]) row.estimated++;
            else if (mask.visibleMask[i]) row.values.push(depth);
        });
        labels.set(mask.classId, row);
    }
    return Array.from(labels.values()).sort((a, b) => a.classId - b.classId).map(({ values, ...row }) => {
        values.sort((a, b) => a - b);
        const middle = Math.floor(values.length / 2);
        return { ...row, samples: values.length, near: values[0] ?? null, far: values[values.length - 1] ?? null,
            median: values.length ? values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2 : null };
    });
}

/** Paint only retained mask depths; camera-space meters share one scale across every label. */
export function drawLabelDepths(context: CanvasRenderingContext2D, frame: TrackVisionFrame, masks: AmodalMask[]) {
    const segment = frame.detections.segment;
    if (segment?.task !== 'segment' || !masks.length) return;
    const depths = new Float32Array(segment.width * segment.height);
    for (const mask of masks) mask.depths.forEach((depth, i) => {
        if (mask.mask[i] && depth > 0 && depth <= 200 && (!depths[i] || depth < depths[i])) depths[i] = depth;
    });
    const values = depths.filter((value) => value > 0);
    if (!values.length) return;
    const rows = labelDepths(masks);
    let near = Infinity, far = 0;
    values.forEach((value) => { near = Math.min(near, value); far = Math.max(far, value); });
    const layer = document.createElement('canvas');
    layer.width = segment.width; layer.height = segment.height;
    const paint = layer.getContext('2d');
    if (!paint) return;
    const pixels = paint.createImageData(layer.width, layer.height);
    depths.forEach((depth, i) => {
        if (!depth) return;
        const t = far > near ? (depth - near) / (far - near) : 0;
        pixels.data.set([Math.round(255 - 168 * t), Math.round(190 - 5 * t), Math.round(87 + 168 * t), 190], i * 4);
    });
    paint.putImageData(pixels, 0, 0);
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(layer, padX / 640 * layer.width, padY / 640 * layer.height,
        resizedWidth / 640 * layer.width, resizedHeight / 640 * layer.height, 0, 0, frame.width, frame.height);
    const fontSize = Math.max(12, frame.width / 90);
    context.font = `${fontSize}px sans-serif`;
    masks.forEach((mask) => {
        if (!mask.bounds) return;
        const row = rows.find((item) => item.classId === mask.classId)!;
        const x = Math.max(4, (mask.bounds[0] / segment.width * 640 - padX) / resizedWidth * frame.width + 4);
        const y = Math.max(fontSize + 4, (mask.bounds[1] / segment.height * 640 - padY) / resizedHeight * frame.height + fontSize + 4);
        const text = `${mask.label} · ${row.median === null ? 'No depth' : `${row.median.toFixed(1)} m`}`;
        context.fillStyle = '#090d13dd';
        context.fillRect(x - 3, y - fontSize - 2, context.measureText(text).width + 6, fontSize + 6);
        context.fillStyle = '#ffffff';
        context.fillText(text, x, y);
    });
    context.restore();
}
