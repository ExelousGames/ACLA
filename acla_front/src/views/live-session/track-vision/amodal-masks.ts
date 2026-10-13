import { VISION_INPUT_SIZE } from './vision-config';
import type { SegmentResult, TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';
import { isCarInteriorLabel } from './vision-labels';

export type AmodalMask = SegmentResult['instances'][number] & {
    label: string;
    visibleMask: Uint8Array;
    hiddenMask: Uint8Array;
    /** Observed depth on visible pixels; interpolated depth on hidden pixels. */
    depths: Float32Array;
    /** Completed mask bounds on the segmentation grid, with exclusive right/bottom edges. Null when empty. */
    bounds: [number, number, number, number] | null;
};

const median = (values: number[]) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;

/** Raw detections stay intact. The 2D pipeline retains cockpit masks for downstream edge filtering. */
export function predictAmodalMasks(frame: TrackVisionFrame, minimumConfidence = 0): AmodalMask[] {
    const segment = frame.detections.segment, depth = frame.detections.depth;
    const nearer = (observed: number, predicted: number) => observed > 0
        && predicted - observed > Math.max(depth?.task === 'depth' && depth.scale === 'relative' ? 0 : 0.1, predicted * 0.02);
    if (segment?.task !== 'segment' || !Number.isInteger(segment.width) || !Number.isInteger(segment.height)
        || segment.width <= 0 || segment.height <= 0 || frame.width <= 0 || frame.height <= 0) return [];
    const { width, height } = segment, size = width * height;
    const box = letterbox(frame.width, frame.height, VISION_INPUT_SIZE);
    const inCapture = (x: number, y: number) => (x + 0.5) / width * VISION_INPUT_SIZE >= box.padX
        && (x + 0.5) / width * VISION_INPUT_SIZE < box.padX + box.resizedWidth
        && (y + 0.5) / height * VISION_INPUT_SIZE >= box.padY
        && (y + 0.5) / height * VISION_INPUT_SIZE < box.padY + box.resizedHeight;
    const observed = new Float32Array(size);
    const validDepth = depth?.task === 'depth' && Number.isInteger(depth.width) && Number.isInteger(depth.height)
        && depth.width > 0 && depth.height > 0 && depth.values.length === depth.width * depth.height;
    if (validDepth) for (let i = 0; i < size; i++) {
        const x = i % width, y = Math.floor(i / width);
        const value = depth.values[Math.floor((y + 0.5) / height * depth.height) * depth.width
            + Math.floor((x + 0.5) / width * depth.width)];
        if (inCapture(x, y) && Number.isFinite(value) && value > 0 && value <= 200) observed[i] = value;
    }
    const coverage = new Uint16Array(size);
    const candidates = segment.instances.filter((item) => Number.isFinite(item.confidence)
        && item.confidence >= 0 && (item.confidence >= minimumConfidence || isCarInteriorLabel(segment.classNames[item.classId]))
        && item.confidence <= 1 && item.mask.length === size
        && item.box.every(Number.isFinite)).map((item) => {
        const mask = item.mask.map((active, i) => Number(Boolean(active) && inCapture(i % width, Math.floor(i / width))));
        let area = 0;
        mask.forEach((active, i) => { if (active) { coverage[i]++; area++; } });
        return { ...item, mask, area, depth: 0 };
    });
    for (const candidate of candidates) {
        const exclusive: number[] = [], all: number[] = [];
        candidate.mask.forEach((active, i) => {
            if (!active || !observed[i]) return;
            all.push(observed[i]);
            if (coverage[i] === 1) exclusive.push(observed[i]);
        });
        candidate.depth = median(exclusive.length ? exclusive : all);
    }
    // Small silhouettes win equal-depth overlaps, independently of label and input order.
    candidates.sort((a, b) => a.area - b.area || b.confidence - a.confidence || a.classId - b.classId
        || a.box[0] - b.box[0] || a.box[1] - b.box[1] || a.box[2] - b.box[2] || a.box[3] - b.box[3]);
    const owners = new Int32Array(size).fill(-1);
    candidates.forEach((candidate, index) => candidate.mask.forEach((active, i) => {
        // Keep cockpit masks as an independent downstream input, without cutting other labels here.
        if (!active || isCarInteriorLabel(segment.classNames[candidate.classId])) return;
        const previous = owners[i];
        if (previous < 0 || (observed[i] && candidate.depth && candidates[previous].depth
            && Math.abs(candidate.depth - observed[i]) < Math.abs(candidates[previous].depth - observed[i]))) owners[i] = index;
    }));
    return candidates.map((candidate, index) => {
        let left = width, top = height, right = 0, bottom = 0;
        const depths = new Float32Array(size), mask = new Uint8Array(size);
        const visibleMask = candidate.mask.map((active, i) => {
            if (!active || (owners[i] !== index && !isCarInteriorLabel(segment.classNames[candidate.classId]))) return 0;
            const x = i % width, y = Math.floor(i / width);
            left = Math.min(left, x); top = Math.min(top, y);
            right = Math.max(right, x + 1); bottom = Math.max(bottom, y + 1);
            depths[i] = observed[i]; mask[i] = 1;
            return 1;
        });
        // Completion only bridges visible endpoints, so it cannot extend these bounds.
        const bounds: AmodalMask['bounds'] = right > left ? [left, top, right, bottom] : null;
        const hiddenMask = new Uint8Array(size);
        const boundsWidth = Math.max(0, right - left), boundsSize = boundsWidth * Math.max(0, bottom - top);
        const sums = new Float64Array(boundsSize), counts = new Uint8Array(boundsSize);
        // Bridge only nearer, labelled occluders bracketed by original visible samples.
        // Inverse depth is linear along a projected planar surface; predictions never become seeds.
        const bridge = (start: number, stride: number, length: number) => {
            let previous = -1;
            for (let offset = 0; offset < length; offset++) {
                const i = start + offset * stride;
                if (!visibleMask[i] || !observed[i]) continue;
                if (previous >= 0 && offset - previous > 1) {
                    const a = observed[start + previous * stride], b = observed[i];
                    let supported = true;
                    for (let step = previous + 1; step < offset; step++) {
                        const j = start + step * stride, t = (step - previous) / (offset - previous);
                        const prediction = 1 / ((1 - t) / a + t / b);
                        if (owners[j] < 0 || owners[j] === index || !nearer(observed[j], prediction)) { supported = false; break; }
                    }
                    if (supported) for (let step = previous + 1; step < offset; step++) {
                        const j = start + step * stride, t = (step - previous) / (offset - previous);
                        const local = (Math.floor(j / width) - top) * boundsWidth + j % width - left;
                        sums[local] += (1 - t) / a + t / b;
                        counts[local]++;
                    }
                }
                previous = offset;
            }
        };
        for (let y = top; y < bottom; y++) bridge(y * width + left, 1, right - left);
        for (let x = left; x < right; x++) bridge(top * width + x, width, bottom - top);
        for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) {
            const i = y * width + x, local = (y - top) * boundsWidth + x - left, count = counts[local];
            if (!count) continue;
            mask[i] = hiddenMask[i] = 1;
            depths[i] = count / sums[local];
        }
        return { classId: candidate.classId, confidence: candidate.confidence, box: candidate.box,
            label: segment.classNames[candidate.classId] ?? `Class ${candidate.classId}`,
            mask, visibleMask, hiddenMask, depths, bounds };
    });
}
